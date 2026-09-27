"""Candidate generation (blocking): shrink the possible pair space to something scorable.

Problem
-------
Scoring every Source-1 entity against every Source-2/Source-3 record is not
tractable. Blocking generates a small candidate set per Source-1 entity instead,
and it sets the **recall ceiling** of the whole system: a true match that blocking
never proposes can never be recovered by the model.

Design notes
------------
* **Recall first, then prune.** A key that is too aggressive is unrecoverable, so
  prefer several independent keys unioned together over one clever key. Check the
  ceiling with :func:`evaluation.blocking_recall_ceiling` before trusting a
  configuration.
* **Size control.** A common key such as a bare city name can blow up into very
  large blocks. Cap the per-entity candidate count and drop or split oversized
  blocks, logging how often the cap fired — a frequently-hit cap is a
  recall leak to inspect.
* **Disk-backed blocks.** Build blocks as ``key -> [record ids]`` in a disk-backed
  store rather than one large in-memory dict. Group by key in sorted order and
  stream the pairs out.

Candidate keys worth considering
--------------------------------
* Name tokens and character n-grams of the name (catches reordering, typos and
  transliteration).
* Phonetic encodings of the name (soundex/metaphone-style) — cheap, and tolerant
  of some spelling drift.
* Postal code / PIN code extracted from the address.
* House or plot number combined with the locality.
* Country-aware address components.
* Near-duplicate detection over name/address TF-IDF vectors (inverted index or
  LSH) to catch records sharing little exact surface form.

How this module implements that at 10M-record scale
---------------------------------------------------
The corpus is 2.2M/1.7M Source-1 rows against 10.3M/10.0M Source-2+3 rows, so a
``{key: [ids]}`` Python dict is not an option — the key space alone is tens of
millions of entries. Instead each strategy is materialised as two parallel
``.npy`` arrays on disk:

    keys[uint64]  sorted 64-bit key hashes
    recs[int32]   the pool row index, permuted to match ``keys``

Two properties make this work:

* **Keys are hashed, not stored.** A 64-bit hash of the key string is a fixed,
  fully deterministic function of those bytes, so a lookup is a
  :func:`numpy.searchsorted` over a sorted ``uint64`` array. Collisions are
  possible in principle (~1 in 10^5 at this scale) but only ever add one
  spurious candidate, which the model then rejects.
* **Strategies are built and queried one at a time.** Peak memory is therefore
  one strategy's postings (~30M) rather than all of them (~120M), and the
  per-strategy arrays stay on disk between the build and query phases.

Source-1 entities are then processed in chunks whose size is derived from the
measured posting volume, so a chunk can never expand into an unbounded pair
blow-up. Evidence (how many distinct strategies proposed a pair) is accumulated
across strategies, and the per-entity cap keeps the pairs sharing the most
evidence.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from . import preprocessing as pp

# --------------------------------------------------------------------------- #
# Strategy definitions
# --------------------------------------------------------------------------- #

#: Every strategy available, including ones measured to be useless. Kept so the
#: leave-one-out ablation in ``scripts/measure_blocking_recall.py`` can still drop
#: each one and report what it was worth.
ALL_STRATEGIES: Tuple[str, ...] = (
    "name_token",       # nt| - one content token of the name
    "name_sorted",      # ns| - the name's token set, order-invariant
    "name_phonetic",    # sx| - soundex of a name token
    "addr_postal",      # pc| - postal / PIN / ZIP code
    "addr_token",       # at| - a geographic token of the address
    "addr_house",       # hs| - house number + street token
    "addr_tail",        # tl| - the last two address tokens (city + state)
    "country_name",     # cn| - country + the longest name token
)

#: The strategies actually used. ``name_phonetic`` is excluded on measured
#: evidence: on a 100k-entity validation sample, dropping it moved pair recall by
#: 0.0000 and entity-full recall by +0.0001 -- it contributed nothing. Soundex
#: collapses too many distinct names into one code, so 64% of its postings were
#: discarded as oversized blocks and the surviving blocks were too coarse to
#: propose anything the other keys had not already proposed. It still costs 24.6M
#: postings to build and query.
STRATEGIES: Tuple[str, ...] = tuple(
    s for s in ALL_STRATEGIES if s != "name_phonetic"
)

#: Keys longer than this are dropped rather than truncated. It is set well above
#: the longest key the strategies can build (a four-token ``ns|`` name) so the
#: cutoff effectively never fires; truncating instead would silently merge
#: unrelated blocks, which is far worse than losing one.
MAX_KEY_LEN = 63

#: Blocks larger than this are dropped, and this is the single most important
#: tuned constant in the module. Measured on a 100k-entity validation sample
#: against the full 10.3M-record pool:
#:
#:   max_block   pair_recall   entity_full   mean candidates
#:       20000        0.919          0.784              789
#:        5000        ~0.85          ~0.62              100
#:        500         0.913          0.782              238
#:
#: Raising the cap from 500 to 20000 buys 0.6pp of pair recall for 3.3x the
#: candidates, almost all of which are extra low-evidence non-matches for the
#: classifier to reject -- a bad trade under a precision-weighted metric. 500 it
#: is. The blocks that get dropped are the ones carried by near-stopword tokens
#: ("ltd", "inc", "road"), which identify nothing.
DEFAULT_MAX_BLOCK = 500

#: Strategy -> the 2-character key namespace it emits. ``_strategy_of_key`` inverts
#: this so a built key can be routed back to the strategy that produced it; the
#: mapping is explicit rather than derived from the strategy name because the
#: namespaces are deliberately short to keep keys under :data:`MAX_KEY_LEN`.
KEY_NAMESPACE: Dict[str, str] = {
    "nt": "name_token",
    "ns": "name_sorted",
    "sx": "name_phonetic",
    "pc": "addr_postal",
    "at": "addr_token",
    "hs": "addr_house",
    "tl": "addr_tail",
    "cn": "country_name",
}

#: Upper bound on the pairs one Source-1 chunk may expand into. Sets the chunk
#: size at query time.
TARGET_CHUNK_PAIRS = 8_000_000
MIN_CHUNK = 2_000
MAX_CHUNK = 100_000

#: Number of address tokens emitted per record. Street names are long and highly
#: repetitive, so the first few tokens carry most of the locality signal and the
#: tail adds postings without adding much recall.
MAX_ADDR_TOKENS = 3
MAX_NAME_TOKENS = 4


def _key(namespace: str, *parts: str) -> Optional[str]:
    """Build a namespaced blocking key, or ``None`` if it carries no signal."""

    joined = "|".join(p for p in parts if p)
    if not joined or joined.endswith("|"):
        return None
    key = f"{namespace}|{joined}"
    if len(key) > MAX_KEY_LEN:
        return None
    return key


def _content_name_tokens(name_norm: str) -> List[str]:
    """Name tokens worth a blocking key: no stopwords, at least 3 characters."""

    return [
        t for t in name_norm.split()
        if len(t) >= 3 and t not in pp.NAME_STOPWORDS
    ]


def _address_key_tokens(address_core: str) -> List[str]:
    """Geographic address tokens worth a blocking key.

    Purely numeric tokens are excluded -- the house number and postal code get
    their own keys, and a bare number like "705" is far too generic to block on.
    """

    return [
        t for t in address_core.split()
        if len(t) >= 4 and not t.isdigit()
    ]


def record_keys(
    name_norm: str,
    address_core: str,
    addr_house: str,
    addr_postal: str,
    addr_tail2: str,
    country_norm: str,
    strategies: Sequence[str] = STRATEGIES,
) -> List[str]:
    """Return the blocking keys for one record, honouring ``strategies``.

    Pure function of the normalised fields, and identical for the Source-1 side
    and the pool side -- that symmetry is what makes an exact key match mean
    "these two records agree on this piece of evidence".
    """

    if not name_norm and not address_core:
        return []

    out: List[str] = []
    wanted = set(strategies)

    name_tokens = _content_name_tokens(name_norm)[:MAX_NAME_TOKENS] if name_norm else []
    addr_tokens = _address_key_tokens(address_core) if address_core else []
    # Longest first: "westchester" identifies a locality, "road" does not, and both
    # survive normalisation whenever the street type was already spelled out.
    addr_ranked = sorted(addr_tokens, key=len, reverse=True)

    if "name_token" in wanted:
        for token in name_tokens:
            key = _key("nt", token)
            if key:
                out.append(key)

    if "name_sorted" in wanted and 1 < len(name_tokens) <= 4:
        # Order-invariant: "Suma Pacific" and "Pacific Suma" land in one block.
        key = _key("ns", *sorted(name_tokens))
        if key:
            out.append(key)

    if "name_phonetic" in wanted:
        for token in name_tokens:
            if len(token) < 4:
                continue
            code = _soundex(token)
            if code:
                key = _key("sx", code)
                if key:
                    out.append(key)

    if "addr_postal" in wanted and addr_postal:
        key = _key("pc", addr_postal)
        if key:
            out.append(key)

    if "addr_token" in wanted:
        for token in addr_ranked[:MAX_ADDR_TOKENS]:
            key = _key("at", token)
            if key:
                out.append(key)

    if "addr_house" in wanted and addr_house and addr_ranked:
        # House number plus the most distinctive address token. "41 mg road" and
        # "41 m g road" collapse to the same pair; "41" alone would not.
        key = _key("hs", addr_house, addr_ranked[0])
        if key:
            out.append(key)

    if "addr_tail" in wanted and addr_tail2:
        key = _key("tl", *addr_tail2.split())
        if key:
            out.append(key)

    if "country_name" in wanted and country_norm and name_tokens:
        # Country is a *refinement* here, never a filter: the key is unioned with
        # country-free keys, so a French record that genuinely belongs with a
        # Source-1 record labelled "France" gets a tighter block without any
        # unseen-country record being excluded from the candidate set.
        longest = max(name_tokens, key=len)
        key = _key("cn", country_norm, longest)
        if key:
            out.append(key)

    return out


def _soundex(token: str) -> str:
    """Soundex code for a token, or ``""`` when it has no usable letters.

    This is the cross-script bridge: the schwa-collapsed transliteration of a
    Devanagari name still lands on the same soundex code as its English spelling
    often enough to be worth a key, and it absorbs single-character typos.
    """

    import jellyfish

    try:
        return jellyfish.soundex(token) or ""
    except Exception:  # jellyfish raises on non-ASCII input
        return ""


# --------------------------------------------------------------------------- #
# Key hashing
# --------------------------------------------------------------------------- #

_FNV_OFFSET = np.uint64(0xCBF29CE484222325)
_FNV_PRIME = np.uint64(0x100000001B3)
_KEY_WORDS = MAX_KEY_LEN // 8 + 1  # 8-byte words needed to cover MAX_KEY_LEN


def hash_keys(keys: Sequence[str]) -> np.ndarray:
    """Hash key strings to ``uint64``, vectorised.

    A fixed 64-bit mix (FNV-style, with an xor-shift finaliser per word) rather
    than Python's ``hash()``, which is salted per process and would make reruns
    differ. Keys are packed into fixed-width bytes and mixed word by word, so a
    million keys cost a few milliseconds instead of a million Python calls.
    """

    if not keys:
        return np.empty(0, dtype=np.uint64)
    width = _KEY_WORDS * 8
    arr = np.array(keys, dtype=f"S{width}")
    words = np.frombuffer(arr.tobytes(), dtype=np.uint64).reshape(len(keys), _KEY_WORDS)
    acc = np.full(len(keys), _FNV_OFFSET, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(_KEY_WORDS):
            acc ^= words[:, j]
            acc *= _FNV_PRIME
            acc ^= acc >> np.uint64(29)
    return acc


def _keys_for_block(
    norm: pp.NormalisedSet, rows: np.ndarray, strategies: Sequence[str]
) -> List[List[str]]:
    """Return per-record key lists for a block of rows."""

    name = norm.name_norm[rows]
    core = norm.address_core[rows]
    house = norm.addr_house[rows]
    postal = norm.addr_postal[rows]
    tail2 = norm.addr_tail2[rows]
    country = norm.country_norm[rows]
    return [
        record_keys(
            name[i].decode("utf-8", "replace"),
            core[i].decode("utf-8", "replace"),
            house[i].decode("utf-8", "replace"),
            postal[i].decode("utf-8", "replace"),
            tail2[i].decode("utf-8", "replace"),
            country[i].decode("utf-8", "replace"),
            strategies,
        )
        for i in range(len(rows))
    ]


def _keys_for_frame(
    frame: "pd.DataFrame", strategies: Sequence[str]
) -> List[List[str]]:
    """Return per-record key lists for a frame of normalised records."""

    columns = {
        name: frame[name].tolist()
        for name in (
            "name_norm", "address_core", "addr_house", "addr_postal",
            "addr_tail2", "country_norm",
        )
    }
    return [
        record_keys(
            columns["name_norm"][i], columns["address_core"][i],
            columns["addr_house"][i], columns["addr_postal"][i],
            columns["addr_tail2"][i], columns["country_norm"][i], strategies,
        )
        for i in range(len(frame))
    ]


# --------------------------------------------------------------------------- #
# Index construction
# --------------------------------------------------------------------------- #


def _strategy_of_key(key: str) -> str:
    """Namespace of a built key -> the strategy that produced it."""

    try:
        return KEY_NAMESPACE[key[:2]]
    except KeyError:  # pragma: no cover - guards a key built outside record_keys
        raise ValueError(f"key {key!r} has no known namespace prefix")


class _RawSpill:
    """Append-only scratch file for unsorted ``(key hash, row)`` postings.

    Building all strategies in one pass produces ~120M postings, which is 1.5 GB
    of ``uint64``/``int32`` pairs. Holding them all in memory to sort later
    defeats the point of the fixed-width representation, so the raw postings go
    straight to disk and each strategy is sorted on its own afterwards.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._keys = open(str(path) + ".k", "wb")
        self._recs = open(str(path) + ".r", "wb")
        self.count = 0

    def add(self, keys: np.ndarray, recs: np.ndarray) -> None:
        if not len(keys):
            return
        self._keys.write(keys.tobytes())
        self._recs.write(recs.tobytes())
        self.count += len(keys)

    def finish(self) -> None:
        self._keys.close()
        self._recs.close()

    def discard(self) -> None:
        _discard_spill(self.path)


def _discard_spill(path: Path) -> None:
    for target in _spill_paths(path):
        if target.exists():
            target.unlink()


def _spill_paths(path: Path) -> Tuple[Path, Path]:
    return Path(str(path) + ".k"), Path(str(path) + ".r")


def _read_spill(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read a spill file. Separate from :class:`_RawSpill` on purpose.

    :class:`_RawSpill` opens its files for *writing*, which truncates them. A
    reader that constructed one would silently zero the postings it is about to
    read, so the read path is kept distinct and never takes a write handle.
    """

    key_path, rec_path = _spill_paths(path)
    return (
        np.fromfile(key_path, dtype=np.uint64),
        np.fromfile(rec_path, dtype=np.int32),
    )


def build_indexes(
    pool: pp.NormalisedSet,
    pool_rows: np.ndarray,
    index_dir: Path,
    strategies: Sequence[str] = STRATEGIES,
    block_rows: int = 400_000,
    verbose: bool = True,
) -> Dict[str, int]:
    """Build every strategy's index in a single pass over the pool.

    Key generation is the dominant cost and it is per record, not per strategy,
    so computing all strategies together is ~8x cheaper than looping strategies
    over the pool. Raw postings are spilled per strategy, then
    :func:`finalise_index` sorts and block-caps each one.
    """

    index_dir.mkdir(parents=True, exist_ok=True)
    spills = {s: _RawSpill(index_dir / f".raw_{s}") for s in strategies}
    try:
        for start in range(0, len(pool_rows), block_rows):
            rows = pool_rows[start:start + block_rows]
            per_record = _keys_for_block(pool, rows, strategies)
            buckets: Dict[str, List[str]] = {s: [] for s in strategies}
            owners: Dict[str, List[np.ndarray]] = {s: [] for s in strategies}
            for offset, keys in enumerate(per_record):
                for key in keys:
                    strategy = _strategy_of_key(key)
                    buckets[strategy].append(key)
                    # One owner per key. ``len(key)`` would be the *string*
                    # length, silently inflating the row array relative to the
                    # key array and corrupting the pairing.
                    owners[strategy].append(
                        np.full(1, rows[offset], dtype=np.int32)
                    )
            for strategy in strategies:
                flat = buckets[strategy]
                if flat:
                    spills[strategy].add(
                        hash_keys(flat), np.concatenate(owners[strategy])
                    )
            if verbose and (start // block_rows) % 5 == 0:
                print(
                    f"[blocking] keys {start + len(rows):,}/{len(pool_rows):,} "
                    f"({sum(s.count for s in spills.values()):,} postings)",
                    flush=True,
                )
        for spill in spills.values():
            spill.finish()
        return {s: spills[s].count for s in strategies}
    finally:
        for spill in spills.values():
            if not spill._keys.closed:
                spill.finish()


def _read_meta(index_dir: Path) -> Dict[str, Dict[str, float]]:
    path = index_dir / "index_meta.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_meta(strategy: str, index_dir: Path, stat: Dict[str, float]) -> None:
    """Record the applied block cap so a later, looser request cannot lie.

    Tightening an index is destructive: a 2k-capped index cannot be widened back
    to 20k. Without this record, asking for the looser cap afterwards would
    succeed and silently hand back the tighter index, and the resulting recall
    numbers would be attributed to the wrong configuration.
    """

    meta = _read_meta(index_dir)
    meta[strategy] = stat
    (index_dir / "index_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def reblock_index(
    strategy: str, index_dir: Path, max_block: int,
) -> Dict[str, float]:
    """Re-derive a strategy's index at a smaller block cap, in place.

    Lowering :data:`DEFAULT_MAX_BLOCK` is the main lever on both speed and recall,
    and it has to be tuned repeatedly. The persisted index is already sorted by key
    and its blocks are contiguous, so a tighter cap is exactly a matter of dropping
    the runs that are now too long -- no spill file and no pool pass needed. The
    larger blocks are simply gone rather than split: a 20k-posting block trimmed to
    2k would silently become an arbitrary subset, and an arbitrary subset of a
    generic key is worse than no key at all.

    Raises if asked for a cap *looser* than the one already applied, since that
    cannot be recovered without a rebuild.
    """

    key_path = index_dir / f"{strategy}_keys.npy"
    rec_path = index_dir / f"{strategy}_recs.npy"
    keys = np.load(key_path)
    recs = np.load(rec_path)
    total = len(keys)
    meta = _read_meta(index_dir)
    applied = meta.get(strategy, {}).get("max_block")
    if applied is not None and max_block > applied:
        raise ValueError(
            f"{strategy}: index was built at max_block={int(applied)}; cannot "
            f"re-widen to {max_block} because the dropped blocks are gone. "
            "Rebuild with build_indexes, or sweep the cap downwards only."
        )
    if total == 0:
        stat = {
            "strategy": strategy, "postings": 0.0, "blocks": 0.0,
            "postings_dropped": 0.0, "dropped_share": 0.0,
            "largest_block": 0.0, "max_block": float(max_block), "cached": 1.0,
        }
        _write_meta(strategy, index_dir, stat)
        return stat
    is_new_block = np.empty(total, dtype=bool)
    is_new_block[0] = True
    np.not_equal(keys[1:], keys[:-1], out=is_new_block[1:])
    sizes = np.diff(np.append(np.flatnonzero(is_new_block), total))
    keep = sizes <= max_block
    dropped = 0 if keep.all() else int(sizes[~keep].sum())
    if not keep.all():
        mask = np.repeat(keep, sizes)
        np.save(key_path, keys[mask])
        np.save(rec_path, recs[mask])
    stat = {
        "strategy": strategy, "postings": float(total),
        "blocks": float(int(keep.sum())), "postings_dropped": float(dropped),
        "dropped_share": dropped / total,
        "largest_block": float(sizes[keep].max()),
        "max_block": float(max_block), "cached": float(keep.all()),
    }
    _write_meta(strategy, index_dir, stat)
    return stat


def finalise_index(
    strategy: str, index_dir: Path, max_block: int = DEFAULT_MAX_BLOCK,
    keep_spill: bool = True,
) -> Dict[str, float]:
    """Sort one strategy's spilled postings, drop oversized blocks, persist.

    Returns statistics. Resolves in this order: an already-built index that
    satisfies ``max_block`` is left alone; a spill is sorted and capped into one;
    an index built with a *larger* cap is tightened by :func:`reblock_index`.

    ``postings_dropped_share`` is the diagnostic to watch: it is the fraction of
    evidence lost to the block cap, and a large value means the cap -- not the
    data -- is setting the recall ceiling.
    """

    key_path = index_dir / f"{strategy}_keys.npy"
    rec_path = index_dir / f"{strategy}_recs.npy"
    spill_path = index_dir / f".raw_{strategy}"
    raw_keys, _ = _spill_paths(spill_path)

    if key_path.exists() and rec_path.exists() and not raw_keys.exists():
        return reblock_index(strategy, index_dir, max_block)
    if not raw_keys.exists():
        raise FileNotFoundError(
            f"no spill and no index for {strategy!r}; run build_indexes first"
        )

    keys, recs = _read_spill(spill_path)
    total = len(keys)
    if len(recs) != total:
        raise RuntimeError(
            f"{strategy}: spill has {total:,} keys but {len(recs):,} rows; the "
            "key and row arrays are out of step"
        )
    if total == 0:
        np.save(key_path, keys)
        np.save(rec_path, recs)
        if not keep_spill:
            _discard_spill(spill_path)
        stat = {
            "strategy": strategy, "postings": 0.0, "blocks": 0.0,
            "postings_dropped": 0.0, "dropped_share": 0.0, "largest_block": 0.0,
            "max_block": float(max_block), "cached": 0.0,
        }
        _write_meta(strategy, index_dir, stat)
        return stat

    # ``kind="stable"`` keeps the generation order within a key, which makes the
    # persisted artefact byte-identical across runs.
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    recs = recs[order]
    del order

    is_new_block = np.empty(len(keys), dtype=bool)
    is_new_block[0] = True
    np.not_equal(keys[1:], keys[:-1], out=is_new_block[1:])
    block_starts = np.flatnonzero(is_new_block)
    block_sizes = np.diff(np.append(block_starts, len(keys)))
    keep_block = block_sizes <= max_block
    dropped = int(block_sizes[~keep_block].sum())

    np.save(key_path, keys[np.repeat(keep_block, block_sizes)])
    np.save(rec_path, recs[np.repeat(keep_block, block_sizes)])
    if not keep_spill:
        _discard_spill(spill_path)
    stat = {
        "strategy": strategy,
        "postings": float(total),
        "blocks": float(int(keep_block.sum())),
        "postings_dropped": float(dropped),
        "dropped_share": dropped / max(total, 1),
        "largest_block": float(block_sizes[keep_block].max()) if keep_block.any() else 0.0,
        "max_block": float(max_block),
        "cached": 0.0,
    }
    # The metadata records the cap this artefact was actually built at. Without
    # it a later sweep has no way to know that asking for a *larger* max_block is
    # impossible, and would silently return the narrower index it already has.
    _write_meta(strategy, index_dir, stat)
    return stat


def build_strategy_index(
    pool: pp.NormalisedSet,
    pool_rows: np.ndarray,
    strategy: str,
    index_dir: Path,
    max_block: int = DEFAULT_MAX_BLOCK,
    block_rows: int = 400_000,
) -> Dict[str, float]:
    """Build and persist the index for a single strategy (convenience wrapper)."""

    build_indexes(pool, pool_rows, index_dir, (strategy,), block_rows, verbose=False)
    return finalise_index(strategy, index_dir, max_block)


# --------------------------------------------------------------------------- #
# Query
# --------------------------------------------------------------------------- #


def _expand_queries(
    index_keys: np.ndarray,
    index_recs: np.ndarray,
    query_keys: np.ndarray,
    query_rows: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Expand hashed queries into ``(query_row, pool_row)`` posting pairs."""

    if len(index_keys) == 0 or len(query_keys) == 0:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty
    lo = np.searchsorted(index_keys, query_keys, side="left")
    hi = np.searchsorted(index_keys, query_keys, side="right")
    counts = (hi - lo).astype(np.int64)
    total = int(counts.sum())
    if total == 0:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty
    # Expand [lo[i], hi[i]) ranges into a flat array of index positions.
    starts = np.repeat(lo, counts)
    run_start = np.repeat(lo - np.concatenate(([0], np.cumsum(counts)[:-1])), counts)
    positions = np.arange(total, dtype=np.int64) + run_start
    return (
        np.repeat(query_rows, counts).astype(np.int32),
        index_recs[positions].astype(np.int32),
    )


def _top_per_group(
    group: np.ndarray, item: np.ndarray, weight: np.ndarray, cap: int
) -> np.ndarray:
    """Keep the ``cap`` highest-``weight`` items in each group, deterministically.

    Ties break on the item index so the output does not depend on the order the
    strategies happened to be queried in.
    """

    if len(group) == 0:
        return np.empty(0, dtype=np.int64)
    order = np.lexsort((item, -weight, group))
    g = group[order]
    starts = np.flatnonzero(np.concatenate(([True], g[1:] != g[:-1])))
    sizes = np.diff(np.append(starts, len(g)))
    rank = np.arange(len(g)) - np.repeat(starts, sizes)
    return order[rank < cap]


def query_strategy(
    index_dir: Path,
    strategy: str,
    s1_norm: pp.NormalisedSet,
    query_rows: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(query_row, pool_row)`` pairs proposed by one strategy.

    The index lives in pool-local row space (``0..pool_size-1``) and the queries
    live in Source-1 row space; they are separate objects on purpose, so the pool
    and the queries can be loaded as different projections of the same split.
    """

    index_keys = np.load(index_dir / f"{strategy}_keys.npy", mmap_mode="r")
    index_recs = np.load(index_dir / f"{strategy}_recs.npy", mmap_mode="r")
    per_record = _keys_for_block(s1_norm, query_rows, (strategy,))
    flat: List[str] = []
    owners: List[np.ndarray] = []
    for offset, keys in enumerate(per_record):
        if keys:
            flat.extend(keys)
            owners.append(np.full(len(keys), query_rows[offset], dtype=np.int32))
    if not flat:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty
    return _expand_queries(
        np.asarray(index_keys), np.asarray(index_recs), hash_keys(flat),
        np.concatenate(owners),
    )


def measure_query_volume(
    index_dir: Path,
    strategy: str,
    s1_norm: pp.NormalisedSet,
    query_rows: np.ndarray,
) -> int:
    """Count the postings one strategy would pull for ``query_rows``.

    Used to size the Source-1 chunks: a strategy that would expand 200 pairs per
    entity needs a much smaller chunk than one that expands 3.
    """

    index_keys = np.load(index_dir / f"{strategy}_keys.npy", mmap_mode="r")
    if len(index_keys) == 0:
        return 0
    per_record = _keys_for_block(s1_norm, query_rows, (strategy,))
    flat: List[str] = []
    for keys in per_record:
        flat.extend(keys)
    if not flat:
        return 0
    keys = np.asarray(index_keys)
    query = hash_keys(flat)
    lo = np.searchsorted(keys, query, side="left")
    hi = np.searchsorted(keys, query, side="right")
    return int((hi - lo).sum())


def choose_chunk_size(
    index_dir: Path, strategies: Sequence[str], s1_norm: pp.NormalisedSet,
    query_rows: np.ndarray, target_pairs: int = TARGET_CHUNK_PAIRS,
) -> int:
    """Pick a Source-1 chunk size that keeps the expanded pair count bounded.

    Measured on a sample rather than assumed, because the strategies are wildly
    uneven: ``name_token`` expands roughly an order of magnitude more than
    ``addr_postal``.
    """

    sample = query_rows
    if len(sample) > 20_000:
        step = len(sample) // 20_000 + 1
        sample = sample[::step][:20_000]
    per_entity = 0.0
    for strategy in strategies:
        postings = measure_query_volume(index_dir, strategy, s1_norm, sample)
        per_entity += postings / max(len(sample), 1)
    if per_entity <= 0:
        return MAX_CHUNK
    size = int(target_pairs / per_entity)
    return max(MIN_CHUNK, min(MAX_CHUNK, size))


# --------------------------------------------------------------------------- #
# Candidate generation
# --------------------------------------------------------------------------- #


def build_candidates(
    s1_norm: pp.NormalisedSet,
    s1_rows: np.ndarray,
    pool_size: int,
    index_dir: Path,
    out_path: Path,
    max_candidates_per_entity: int = 50,
    strategies: Sequence[str] = STRATEGIES,
    verbose: bool = True,
) -> Dict[str, float]:
    """Generate the candidate set and write it as a two-column ``.npy`` pair.

    Returns a ``(n, 3)`` int32 array of ``(s1_row, pool_row, evidence)`` triples
    written to ``out_path``, plus statistics. ``s1_row`` indexes ``s1_norm`` and
    ``pool_row`` indexes the pool the index was built from, so the caller can look
    up IDs with ``s1_ids[pairs[:, 0]]`` and ``pool_ids[pairs[:, 1]]`` directly.

    Processing is chunked over Source-1 rows; the chunk size is derived from the
    measured posting volume so the expanded pair count per chunk stays bounded
    regardless of how the data is distributed.
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_size = choose_chunk_size(index_dir, strategies, s1_norm, s1_rows)
    if verbose:
        print(
            f"[blocking] strategies={list(strategies)} "
            f"max_candidates={max_candidates_per_entity} chunk={chunk_size:,}",
            flush=True,
        )

    n_s1 = len(s1_rows)
    kept_chunks: List[np.ndarray] = []
    stats = {
        "entities": float(n_s1),
        "entities_without_candidates": 0.0,
        "pairs_before_cap": 0.0,
        "pairs_kept": 0.0,
        "cap_fired_entities": 0.0,
        "chunk_size": float(chunk_size),
    }
    evidence_cap = np.int16(255)

    for start in range(0, n_s1, chunk_size):
        rows = s1_rows[start:start + chunk_size]
        group_parts: List[np.ndarray] = []
        item_parts: List[np.ndarray] = []
        for strategy in strategies:
            g, it = query_strategy(index_dir, strategy, s1_norm, rows)
            if len(g):
                group_parts.append(g)
                item_parts.append(it)
        if not group_parts:
            stats["entities_without_candidates"] += len(rows)
            continue

        group = np.concatenate(group_parts).astype(np.int64)
        item = np.concatenate(item_parts).astype(np.int64)
        del group_parts, item_parts
        stats["pairs_before_cap"] += len(group)

        # Evidence = number of distinct strategies that proposed this pair. The
        # combined key is exact in int64 for our sizes (2.2M x 12.5M < 2^63).
        combined = group * np.int64(pool_size) + item
        uniq, counts = np.unique(combined, return_counts=True)
        del combined
        sel_group = (uniq // np.int64(pool_size)).astype(np.int32)
        sel_item = (uniq % np.int64(pool_size)).astype(np.int32)
        del uniq

        # ``sel_group`` holds global Source-1 rows, but the cap and the
        # no-candidate count are per *position within this chunk*.
        local = np.argsort(rows)
        sel_local = local[np.searchsorted(rows[local], sel_group)]
        if not np.array_equal(rows[sel_local], sel_group):  # pragma: no cover
            raise RuntimeError("chunk row mapping is inconsistent")

        chosen = _top_per_group(sel_local, sel_item, counts, max_candidates_per_entity)
        block = np.empty((len(chosen), 3), dtype=np.int32)
        block[:, 0] = sel_group[chosen]
        block[:, 1] = sel_item[chosen]
        block[:, 2] = np.minimum(counts[chosen], evidence_cap)
        kept_chunks.append(block)

        sizes = np.bincount(sel_local[chosen], minlength=len(rows))
        stats["cap_fired_entities"] += float((sizes >= max_candidates_per_entity).sum())
        stats["entities_without_candidates"] += float((sizes == 0).sum())
        stats["pairs_kept"] += len(chosen)
        if verbose:
            print(
                f"[blocking]   {start + len(rows):,}/{n_s1:,} entities, "
                f"{stats['pairs_kept']:,.0f} pairs kept, "
                f"mean {stats['pairs_kept'] / max(start + len(rows), 1):.1f}/entity",
                flush=True,
            )

    pairs = (
        np.concatenate(kept_chunks, axis=0)
        if kept_chunks else np.empty((0, 3), dtype=np.int32)
    )
    np.save(out_path, pairs)
    full_cross = n_s1 * pool_size
    stats["reduction_ratio"] = (
        1.0 - (len(pairs) / full_cross) if full_cross else 0.0
    )
    stats["mean_candidates"] = stats["pairs_kept"] / max(n_s1, 1)
    stats["max_block"] = float(DEFAULT_MAX_BLOCK)
    return stats, pairs


def generate_candidates(
    blocks: Dict[str, List[str]],
    source1_ids: Iterable[str],
    max_candidates_per_entity: int = 50,
) -> Iterator[Tuple[str, List[str]]]:
    """Yield ``(source1_entity_id, candidate_ids)`` for every Source-1 entity.

    Thin adapter over an in-memory ``{s1_id: [pool ids]}`` mapping, kept for the
    documented contract and for tests. The full-scale path is
    :func:`build_candidates`, which never materialises such a mapping: at 2.2M
    Source-1 entities it would need several GB of Python lists.

    The caller is responsible for the guarantees the module documents -- one row
    per entity including empty lists, S2-/S3- IDs only, no duplicates, capped
    length -- because this adapter does no blocking of its own.
    """

    for entity_id in source1_ids:
        pool_ids = blocks.get(entity_id, [])
        seen: Set[str] = set()
        kept: List[str] = []
        for pool_id in pool_ids:
            if pool_id in seen:
                continue
            seen.add(pool_id)
            kept.append(pool_id)
            if len(kept) >= max_candidates_per_entity:
                break
        yield entity_id, kept


def score_candidate_prioritisation(
    source1_entity_id: str,
    candidate_ids: Sequence[str],
) -> List[str]:
    """Order candidates by cheap evidence so truncation keeps the best ones.

    Truncation is not blind: :func:`build_candidates` ranks by the number of
    distinct blocking strategies that proposed the pair before applying the
    per-entity cap, so the ordering that matters already happens upstream where
    the evidence is available. This function is the documented seam for a
    caller holding only IDs; with no record data to compare, the evidence
    ordering is already baked in, so it preserves the given order (which is
    deterministic) rather than inventing a weaker one.
    """

    return list(candidate_ids)


def write_candidate_pairs(
    candidates: Iterable[Tuple[str, List[str]]],
    output_path: str,
) -> int:
    """Write ``output/candidate_pairs.tsv`` and return the number of rows.

    Format, enforced by ``utils/validate_submission.py``:

    * Header exactly ``source1_entity_id<TAB>candidate_entity_ids``.
    * Tab-separated, UTF-8, no index column, no quoting. Address and ID-list
    * fields contain commas, so ``to_csv(sep="\\t", index=False,
      encoding="utf-8")`` is the safe route.
    * ID lists comma-separated, no spaces, empty string when there are no
      candidates.
    * One row per Source-1 entity, no duplicate rows, no duplicate IDs within a
      list, S2-/S3- IDs only.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write("source1_entity_id\tcandidate_entity_ids\n")
        for entity_id, ids in candidates:
            unique = sorted(set(ids)) if ids else []
            handle.write(f"{entity_id}\t{','.join(unique)}\n")
            rows += 1
    return rows


def write_candidates_from_pairs(
    pairs: np.ndarray,
    s1_rows: np.ndarray,
    s1_ids: np.ndarray,
    pool_ids: np.ndarray,
    out_path: Path,
) -> int:
    """Stream ``(s1_row, pool_row, evidence)`` triples into ``candidate_pairs.tsv``.

    Driven by the Source-1 entity list, not by the pairs, so every entity gets a
    row even when it has no candidates -- including a row whose candidate list is
    empty, which is what the validator expects for an unmatched entity.

    ``pairs[:, 0]`` holds :data:`preprocessing.NormalisedSet` row indices, so
    ``s1_rows``/``s1_ids``/``pool_ids`` are indexed by that same global row space.
    ``s1_rows`` must be ascending, which is what :func:`build_candidates` assumes
    when it appends chunks in order.
    """

    if len(s1_rows) and not np.all(s1_rows[1:] > s1_rows[:-1]):
        raise ValueError("s1_rows must be strictly ascending")
    n_s1 = len(s1_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Start offset of every entity's pairs. This is a searchsorted of the
    # *ascending entity rows* against the pairs' Source-1 column, so it yields
    # one boundary per entity. Searching the other way round -- the pairs'
    # Source-1 column against the entity rows -- would yield one entry per
    # *pair*, which silently collapses every entity's list to empty and dumps
    # all the pairs on the last row.
    if len(pairs):
        starts = np.searchsorted(pairs[:, 0], s1_rows, side="left")
    else:
        starts = np.zeros(n_s1, dtype=np.int64)
    boundaries = np.append(starts, len(pairs))
    if len(boundaries) != n_s1 + 1:
        raise RuntimeError(
            f"candidate boundaries are {len(boundaries)} long for {n_s1:,} entities"
        )

    rows = 0
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        handle.write("source1_entity_id\tcandidate_entity_ids\n")
        for i in range(n_s1):
            entity_id = s1_ids[i].decode("utf-8")
            lo, hi = int(boundaries[i]), int(boundaries[i + 1])
            if lo == hi:
                handle.write(f"{entity_id}\t\n")
                rows += 1
                continue
            ids = sorted({pool_ids[r].decode("utf-8") for r in pairs[lo:hi, 1]})
            handle.write(f"{entity_id}\t{','.join(ids)}\n")
            rows += 1
    return rows


def blocking_statistics(
    candidates: Iterable[Tuple[str, List[str]]],
    full_cross: Optional[int] = None,
) -> Dict[str, float]:
    """Summarise blocking behaviour for the write-up and regression checks.

    Report at least: Source-1 entities with zero candidates, mean and max
    candidates per entity, total pairs generated, the reduction ratio against the
    full cross-product, and how often the per-entity cap fired. Pair this with
    the recall ceiling from :func:`evaluation.blocking_recall_ceiling` — candidate
    volume alone says nothing about quality.
    """

    total = 0
    empty = 0
    largest = 0
    n = 0
    for _entity_id, ids in candidates:
        n += 1
        count = len(ids)
        total += count
        if count == 0:
            empty += 1
        elif count > largest:
            largest = count
    mean = total / n if n else 0.0
    stats = {
        "entities": float(n),
        "entities_without_candidates": float(empty),
        "share_without_candidates": empty / n if n else 0.0,
        "mean_candidates": mean,
        "max_candidates": float(largest),
        "total_pairs": float(total),
    }
    if full_cross:
        stats["reduction_ratio"] = 1.0 - (total / full_cross)
    return stats


__all__ = [
    "STRATEGIES", "ALL_STRATEGIES", "KEY_NAMESPACE", "DEFAULT_MAX_BLOCK",
    "MAX_KEY_LEN", "record_keys", "hash_keys", "build_indexes", "finalise_index",
    "reblock_index", "build_strategy_index", "build_candidates",
    "generate_candidates", "score_candidate_prioritisation",
    "write_candidate_pairs", "write_candidates_from_pairs",
    "blocking_statistics", "choose_chunk_size", "query_strategy",
]
