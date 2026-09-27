"""Out-of-core, multiprocess featurisation over the read-only memmapped pool.

Why this module exists
----------------------
The pipeline originally featurised *every* candidate pair into one giant on-disk
matrix: 300,537,790 x 35 float32 = **39.19 GiB**. That cannot be done on this
machine and no amount of flush scheduling changes that. The Windows commit
limit is 24.11 GiB (15.61 GiB RAM + 8.50 GiB pagefile), and a *writeable* file
mapping is charged to commit as its pages are dirtied, so writing a 39.19 GiB
matrix requires more commit than exists. The run died at 3 of 89 chunks, six
times, for exactly this reason.

So the full matrix is never built. Nothing downstream wanted it:

* the classifier only ever consumed a bounded subsample of pairs, and
* inference scores the candidate set in chunks and keeps only the winners.

Both consumers are streaming, and this module is what makes streaming cheap
enough to be practical. The key change is that the Source-2/3 pool is a
**read-only** memmap (``preprocessing.load_normalised(..., memmap=True)``), so
every worker maps the same 5.69 GiB of columns through the OS page cache and
none of it is charged to private commit. That is what turns a
GIL-bound serial loop into a process pool:

===========================  ===========  ======  ==============
mode                         pairs/s      speedup  commit/worker
===========================  ===========  ======  ==============
serial                        5,303       1.00x      0.61 GiB
2 processes                  30,527       5.76x      0.57 GiB
4 processes                  47,120       8.89x      0.57 GiB
6 processes                  56,364      10.63x      0.57 GiB
8 processes                  46,375       8.75x      0.57 GiB
===========================  ===========  ======  ==============

(measured by ``artifacts/bench_featurise.py``; 16 logical cores, so 6 is the
measured optimum and 8 regresses.)

Two costs set the block size
----------------------------
:func:`features.featurise_arrays` builds one ``record_cache`` per *unique* row
on each side of a block -- frozensets of character n-grams, content tokens and
soundex codes, at roughly 600 bytes to a few KB per record. Those caches are what
makes the featuriser fast (per-record work is done once, not once per pair), and
they are also its peak memory. A block of 12,500 pairs costs ~0.3 GiB of caches
above interpreter baseline; a block of 50,000 costs ~1.3 GiB.

That is the second bug the old design had. ``FEATURE_CHUNK_ENTITIES`` was 25,000
Source-1 entities, which at a measured 136 candidates per entity is **3.4 million
pairs per block** -- tens of gigabytes of transient cache, freed on return but
never returned to the OS. Blocks here are sized in *pairs*, not entities, and
kept small enough that N workers x block fits the commit budget with room to
spare.

The asymmetry that makes this pay
--------------------------------
Caches are reused when one record appears in several pairs of the same block.
A Source-1 entity has ~136 candidates, so a block holding a whole number of
entities gets good reuse on the Source-1 side for free. The pool side gets no
reuse at all -- pool rows within a block are effectively distinct -- which is
exactly why the block has to stay small.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np

from . import features as ft
from . import preprocessing as pp

#: Measured optimum on this box (see the module docstring). Six processes, not
#: sixteen: each worker is CPU-bound on the same 16 logical cores, and 8 already
#: regressed.
DEFAULT_WORKERS = 6

#: Pairs per block. Sized from the measurement above: ~12,500 pairs cost ~0.3 GiB
#: of per-record caches per worker, so 16,000 keeps six workers near 4 GiB total.
DEFAULT_BLOCK_PAIRS = 16_000

#: Cache files the candidate stage writes, keyed by the name used in the filename.
_PAIR_CONTEXT_FILES = (
    "pairs", "entity_counts", "candidate_degree",
    "ctx_pool_source", "ctx_pool_frequency",
)

_STATE: Dict[str, object] = {}


# --------------------------------------------------------------------------- #
# Worker side
# --------------------------------------------------------------------------- #


def _init_worker(
    cache_dir: str,
    split: str,
    projection: Tuple[str, ...],
    known_countries: Tuple[str, ...],
    model_path: Optional[str] = None,
) -> None:
    """Open every input read-only. Called once per worker process.

    Nothing private is duplicated: the pool projection, the pair array and the
    context arrays are all mapped from the same on-disk files in every worker,
    so the OS shares one physical copy of each page.
    """

    cache = Path(cache_dir)
    _STATE.clear()
    for name in _PAIR_CONTEXT_FILES:
        _STATE[name] = np.load(cache / f"{split}_{name}.npy", mmap_mode="r")
    _STATE["s1"] = pp.load_normalised(
        cache, split, sources=(1,), columns=projection, memmap=True
    )
    _STATE["pool"] = pp.load_normalised(
        cache, split, sources=(2, 3), columns=projection, memmap=True
    )
    _STATE["columns"] = list(ft.feature_columns())
    _STATE["known"] = set(known_countries)
    if model_path:
        from . import model as md

        booster, _cols, _meta = md.load_model(model_path)
        _STATE["booster"] = booster


def featurise_block(task: Tuple[int, int]) -> np.ndarray:
    """Featurise ``pairs[lo:hi]`` into a fresh float32 block."""

    lo, hi = task
    pairs = _STATE["pairs"]
    block = np.asarray(pairs[lo:hi])
    out = np.empty((hi - lo, len(_STATE["columns"])), dtype=np.float32)
    ft.featurise_arrays(
        _STATE["s1"], _STATE["pool"],
        block[:, 0], block[:, 1], block[:, 2],
        _STATE["ctx_pool_source"][lo:hi], _STATE["ctx_pool_frequency"][lo:hi],
        _STATE["candidate_degree"][block[:, 1]],
        _STATE["entity_counts"][block[:, 0]],
        columns=_STATE["columns"],
        known_countries=_STATE["known"],
        out=out,
    )
    return out


def score_block(task: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Featurise ``pairs[lo:hi]`` and return ``(s1_row, pool_row, score)``.

    Scoring inside the worker is deliberate. Returning the feature block to the
    parent would push 16,000 x 35 float32 (~2.2 MB, and 39 GiB over a full test
    run) through a pipe for no reason: the parent only ever wants the winners.
    Thresholding in the worker and returning the ~0.1% of pairs that clear it
    keeps the inter-process traffic at a few hundred rows per block.

    The threshold is applied here but contention is *not* resolved here -- a
    single pool record can be claimed from blocks that no one worker can see, so
    the global contest has to run in the parent over the union of survivors.
    """

    lo, hi, threshold = task
    block = featurise_block((lo, hi))
    scores = _STATE["booster"].predict(block, num_threads=1)
    pairs = np.asarray(_STATE["pairs"][lo:hi])
    keep = scores >= threshold
    return (
        np.asarray(pairs[keep, 0], dtype=np.int64),
        np.asarray(pairs[keep, 1], dtype=np.int64),
        scores[keep].astype(np.float64),
    )


# --------------------------------------------------------------------------- #
# Parent side
# --------------------------------------------------------------------------- #


def plan_blocks(
    starts: np.ndarray, n_s1: int, block_pairs: int = DEFAULT_BLOCK_PAIRS,
    lo_row: int = 0, hi_row: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """Split the Source-1 row range ``[lo_row, hi_row)`` into pair blocks.

    Blocks never straddle an entity: a Source-1 entity's candidates are always
    contiguous in the pair array and are always featurised together, which is
    what lets the per-record caches be reused across an entity's ~136 pairs.
    Blocks are therefore aligned to entity boundaries and are *at most*
    ``block_pairs`` long, so the worst case is one very wide entity.

    ``starts`` is the ``len(n_s1) + 1`` searchsorted boundary array, so
    ``starts[i]:starts[i+1]`` is entity ``i``'s slice of the pair array.
    """

    hi_row = n_s1 if hi_row is None else hi_row
    blocks: List[Tuple[int, int]] = []
    row = lo_row
    while row < hi_row:
        lo = int(starts[row])
        if lo == int(starts[row + 1]) and row + 1 < hi_row:
            # No candidates for this entity; nothing to featurise.
            row += 1
            continue
        take = 0
        stop = row
        while stop < hi_row and take < block_pairs:
            width = int(starts[stop + 1]) - int(starts[stop])
            if take and take + width > block_pairs:
                break
            take += width
            stop += 1
        if stop == row:  # a single entity wider than block_pairs
            stop = row + 1
        blocks.append((lo, int(starts[stop])))
        row = stop
    return blocks


class Featuriser:
    """A process pool that maps blocks of candidate pairs to features.

    Usable as a context manager. ``n_proc=1`` runs in-process with no pool, which
    is what the unit tests and the smoke pipeline use -- it makes the worker and
    the caller share one code path, so a bug cannot hide behind the fork.
    """

    def __init__(
        self,
        cache_dir: Path,
        split: str,
        projection: Sequence[str],
        known_countries: Iterable[str],
        n_proc: int = DEFAULT_WORKERS,
        model_path: Optional[str] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.projection = tuple(projection)
        self.known_countries = tuple(sorted(known_countries))
        self.n_proc = max(1, int(n_proc))
        self.model_path = model_path
        self._pool: Optional[mp.pool.Pool] = None
        self._local = False

    # -- lifecycle -------------------------------------------------------- #

    def __enter__(self) -> "Featuriser":
        if self.n_proc == 1:
            _init_worker(
                str(self.cache_dir), self.split, self.projection,
                self.known_countries, self.model_path,
            )
            self._local = True
            return self
        # "spawn", not the Windows default of "fork" (which does not exist) and
        # not "forkserver": every worker opens the same read-only memmaps in its
        # own initializer, so there is no parent state to inherit and no
        # pickle-through-parent bandwidth.
        ctx = mp.get_context("spawn")
        self._pool = ctx.Pool(
            self.n_proc, initializer=_init_worker,
            initargs=(
                str(self.cache_dir), self.split, self.projection,
                self.known_countries, self.model_path,
            ),
        )
        return self

    def __exit__(self, *exc) -> None:
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None
        if self._local:
            _STATE.clear()
            self._local = False

    # -- work ------------------------------------------------------------- #

    def map_features(
        self, blocks: Sequence[Tuple[int, int]], chunksize: int = 1,
    ) -> Iterator[np.ndarray]:
        """Yield one feature block per input block, in submission order.

        ``imap`` rather than ``map`` so a 6-worker pool is never six blocks
        resident at once; with a small ``chunksize`` the in-flight blocks are
        what bounds memory, not the pool size.
        """

        if self._pool is None:
            for block in blocks:
                yield featurise_block(block)
            return
        for out in self._pool.imap(featurise_block, blocks, chunksize=chunksize):
            yield out

    def map_scores(
        self, blocks: Sequence[Tuple[int, int]], threshold: float, chunksize: int = 1,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Yield the threshold-clearing claims in each block."""

        tasks = [(lo, hi, threshold) for lo, hi in blocks]
        if self._pool is None:
            for task in tasks:
                yield score_block(task)
            return
        for out in self._pool.imap(score_block, tasks, chunksize=chunksize):
            yield out


__all__ = [
    "DEFAULT_WORKERS", "DEFAULT_BLOCK_PAIRS", "Featuriser", "plan_blocks",
    "featurise_block", "score_block",
]
