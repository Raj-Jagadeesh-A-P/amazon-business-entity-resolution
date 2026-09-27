"""End-to-end orchestration: blocking -> features -> model -> submission.

Pipeline position
-----------------
``preprocessing`` -> ``blocking`` -> ``features`` -> ``model`` -> **this module**
-> ``output/matching_results.tsv``.

Stages
------
Each stage is separately runnable and restartable, writing its artefact to
``cache/`` so a long run can be resumed rather than restarted:

    index       build the blocking index for a split
    candidates  query the index, write candidate_pairs.tsv + a row-index array
    features    featurise the candidate pairs into a float32 matrix
    train       fit the classifier and tune the threshold on a validation split
    predict     score the test candidates, resolve contention, write the output

Memory and parallelism
----------------------
The single largest object is the pool projection: 592 bytes per record, or
6.1 GB for the 10.3M training records. That is why the stages are separate. One
copy of the pool plus a feature matrix chunked to disk is the peak, and no stage
tries to hold the pool *and* the whole feature matrix at once.

Featurisation is threaded rather than multiprocessed, and the reason is
measured, not assumed: rapidfuzz and numpy release the GIL, but the
set-intersection features (token/character-n-gram Jaccard, soundex overlap) are
pure Python and hold it. Benchmarked at 16.1k pairs/s serial, threads gave
1.00x on 16 workers -- no gain at all. Multiprocessing would need one pool copy
per worker, which does not fit in 15.6 GB. Chunks are therefore sized so the
feature matrix stays bounded and the run streams to disk.

The known way to fix this properly is to replace the per-pair set intersections
with fixed-width MinHash sketches, which turn each similarity into a vectorised
``bitwise_and`` plus a popcount over the whole chunk at once. That is the single
highest-value optimisation left in this pipeline and is recorded as future work
rather than done here.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from . import blocking, evaluation, features as ft, model as md
from . import preprocessing as pp
from . import streaming as sm

#: Columns :mod:`features` needs. Narrower than the full record on purpose: the
#: pool projection is the peak memory object in the pipeline.
FEATURE_COLUMNS: Tuple[str, ...] = (
    "name_norm", "name_legal", "address_core", "addr_house", "addr_postal",
    "addr_state", "addr_city", "addr_tail2", "landmark_tokens", "country_norm",
)

#: Columns the blocking keys are built from. Even narrower -- blocking never reads
#: the legal form or the landmark tokens.
BLOCK_COLUMNS: Tuple[str, ...] = (
    "name_norm", "address_core", "addr_house", "addr_postal",
    "addr_tail2", "country_norm",
)

#: Source-1 entities per feature block. The chunk's matrix is
#: ``chunk * candidates_per_entity * 35 * 4`` bytes, so 25k entities at 200
#: candidates is 700 MB.
FEATURE_CHUNK_ENTITIES = 25_000

#: Validation fraction of Source-1 entities, split at the *entity* level.
VALID_FRACTION = 0.15

#: Pair budgets for training and threshold tuning. The pair matrix is
#: memory-mapped on disk, but the *sampled* copy handed to LightGBM has to fit in
#: RAM alongside the 6.1 GB pool projection, so each side is capped. 12M rows of
#: 35 float32 features is 1.7 GB; at a mean of ~200 candidates per entity that
#: covers ~60k training entities, which is far more than LightGBM needs to fit a
#: 35-feature model without overfitting.
MAX_TRAIN_PAIRS = 12_000_000
MAX_VALID_PAIRS = 4_000_000

#: Source-1 rows per chunk when matching pairs against the ground truth. Each
#: chunk materialises one ``searchsorted`` per side over its own slice, so this
#: only has to be small enough that a chunk's candidate keys stay in cache. The
#: truth side is ~3.5 keys per row, so 250k rows is well under a megabyte.
TRUTH_CHUNK_ROWS = 250_000

#: How many Source-1 entities to featurise for fitting and for threshold tuning.
#:
#: These replace the old design, which featurised all 300,537,790 candidate pairs
#: into a 39.19 GiB matrix. Nothing ever needed that matrix: the classifier wants
#: a bounded sample, and inference streams. Measured candidate counts are 136 per
#: entity, so these give ~13.6M fitting rows and ~6.8M validation rows, against a
#: ceiling of ~540k positive rows. That is far more than a 35-feature GBDT needs
#: to saturate, and it costs 1.9 GB and 0.95 GB instead of 39.19 GiB.
FIT_ENTITIES = 100_000
VALID_ENTITIES = 50_000

#: Seed for the entity-level train/validation partition and for the subsample of
#: each side. Recorded in ``artifacts/split.json`` so the split is reproducible.
SPLIT_SEED = 42


def _log(message: str) -> None:
    print(message, flush=True)


# --------------------------------------------------------------------------- #
# Stage: index
# --------------------------------------------------------------------------- #


def run_index(split: str, cache_dir: Path, max_block: int, rebuild: bool) -> None:
    """Build the blocking index over the split's Source-2 + Source-3 pool."""

    index_dir = cache_dir / f"index_{split}"
    sources = (2, 3) if split == "train" else (2, 3)
    pool = pp.load_normalised(cache_dir, split, sources=sources, columns=BLOCK_COLUMNS)
    _log(f"[index] {split} pool rows={len(pool):,}")
    # A missing or partial index has to be generated; ``--rebuild`` forces
    # regeneration even when a previous run left one behind.
    have_index = index_dir.is_dir() and any(
        (index_dir / f"{s}_keys.npy").exists() or (index_dir / f".raw_{s}.keys.npy").exists()
        for s in blocking.STRATEGIES
    )
    if rebuild or not have_index:
        if rebuild:
            _log("[index] --rebuild: discarding any existing index")
        blocking.build_indexes(
            pool, np.arange(len(pool), dtype=np.int32), index_dir,
            blocking.STRATEGIES,
        )
    for strategy in blocking.STRATEGIES:
        stat = blocking.finalise_index(
            strategy, index_dir, max_block, keep_spill=False
        )
        _log(
            f"[index]   {stat['strategy']:15s} blocks={stat['blocks']:>9,.0f} "
            f"postings={stat['postings']:>12,.0f} dropped={stat['dropped_share']:6.1%} "
            f"largest={stat['largest_block']:>8,.0f}"
        )


# --------------------------------------------------------------------------- #
# Stage: candidates
# --------------------------------------------------------------------------- #


def run_candidates(
    split: str, cache_dir: Path, artifacts: Path, output_dir: Path,
    max_candidates: int,
) -> Tuple[Path, Path]:
    """Query the index for every Source-1 entity and write the candidate set."""

    index_dir = cache_dir / f"index_{split}"
    s1 = pp.load_normalised(cache_dir, split, sources=(1,), columns=BLOCK_COLUMNS)
    pool = pp.load_normalised(cache_dir, split, sources=(2, 3), columns=BLOCK_COLUMNS)
    _log(f"[candidates] {split} s1={len(s1):,} pool={len(pool):,}")

    pairs_path = cache_dir / f"{split}_pairs.npy"
    stats, pairs = blocking.build_candidates(
        s1, np.arange(len(s1), dtype=np.int32), len(pool), index_dir, pairs_path,
        max_candidates_per_entity=max_candidates,
        strategies=blocking.STRATEGIES, verbose=True,
    )
    for key in ("entities", "mean_candidates", "pairs_before_cap", "pairs_kept",
                "cap_fired_entities", "entities_without_candidates",
                "reduction_ratio", "chunk_size"):
        _log(f"[candidates]   {key} = {stats[key]:,.2f}")

    # The blocking context features need per-record statistics computed once.
    pool_ids = pool.entity_id
    pool_source = pp.source_column(pool)
    # Name frequency is keyed on the blocking name tokens, which is what
    # "this name is shared" actually means for a match; the entity ID is unique by
    # construction and would make the count a constant 1.
    context = _pool_context(pool_ids, pool_source, pairs[:, 1], pool.name_norm)
    # int32, not bincount's native int64: this is one value per Source-1 entity
    # broadcast across its pairs, and the int64 array would be 2.4 GB on disk at
    # the default cap while the largest possible count is a few hundred.
    entity_counts = np.bincount(
        pairs[:, 0], minlength=len(s1)
    ).astype(np.int32)
    candidate_degree = _candidate_degree(pairs[:, 1], len(pool))
    for name, array in context.items():
        np.save(cache_dir / f"{split}_ctx_{name}.npy", array)
    np.save(cache_dir / f"{split}_entity_counts.npy", entity_counts)
    np.save(cache_dir / f"{split}_candidate_degree.npy", candidate_degree)
    # Persist the ID arrays so later stages need not reload the whole pool.
    np.save(cache_dir / f"{split}_pool_ids.npy", pool_ids)
    np.save(cache_dir / f"{split}_s1_ids.npy", s1.entity_id)
    (cache_dir / f"{split}_max_candidates.txt").write_text(
        str(max_candidates), encoding="utf-8"
    )
    _log(f"[candidates] context written: {sorted(context)}")

    # The recall ceiling is deliberately not computed here. It needs a pass over
    # the whole pair array against the ground truth, which is the same pass that
    # produces the training labels, so it is reported by the features stage
    # instead of walking ~3x10^8 pairs twice. To see the ceiling without
    # featurising, run scripts/measure_blocking_recall.py against the index.

    tsv = output_dir / "candidate_pairs.tsv"
    rows = blocking.write_candidates_from_pairs(
        pairs, np.arange(len(s1), dtype=np.int32), s1.entity_id, pool_ids, tsv,
    )
    _log(f"[candidates] wrote {tsv} with {rows:,} rows")
    return pairs_path, tsv


def _pool_context(
    pool_ids: np.ndarray, pool_source: np.ndarray, pool_rows: np.ndarray,
    pool_name_key: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """Per-pair blocking context arrays, aligned to the pair order."""

    if pool_name_key is None:
        frequency = np.ones(len(pool_ids), dtype=np.int32)
    else:
        frequency = _name_frequency(pool_name_key)
    return {
        "pool_source": pool_source[pool_rows].astype(np.float32),
        "pool_frequency": frequency[pool_rows].astype(np.float32),
    }


def _name_frequency(name_key: np.ndarray) -> np.ndarray:
    """How many pool records share each record's blocking name key.

    A useful singleton prior: a name key that occurs once is very unlikely to be
    a match, while one shared by 40 records is probably a common chain whose real
    matches are correspondingly many.

    Computed with a single ``np.unique`` over the key array rather than per-record
    lookups, because the pool has 10.3M rows and the result is aligned to pool row
    order, not to pair order -- the caller indexes it with ``pool_rows``.
    """

    if not len(name_key):
        return np.zeros(0, dtype=np.int32)
    _, inverse, counts = np.unique(
        np.asarray(name_key), return_inverse=True, return_counts=True
    )
    return counts[inverse].astype(np.int32)


def _candidate_degree(pool_rows: np.ndarray, pool_size: int) -> np.ndarray:
    """How many candidate pairs reference each pool record.

    Computed with ``bincount`` over the pool rows rather than by unique-ing the
    pair array: this is a per-pool-record counter over ~5x10^8 pairs, so it has
    to be a single linear pass.
    """

    return np.bincount(pool_rows, minlength=pool_size).astype(np.int32)


def _truth_row_map(
    s1_ids: np.ndarray, pool_ids: np.ndarray, pool_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Ground-truth ``(s1_row, pool_row)`` pairs, sorted by ``s1_row``.

    Both the labels and the recall ceiling need the same mapping, and both need
    it in ``s1_row`` order so it can be walked alongside the pair array, which is
    grouped by ``s1_row``.

    The lookups are ``searchsorted`` over sorted copies rather than dicts. A dict
    over the 10.3M pool IDs costs about a gigabyte by itself, and the labels are
    only ever consulted per Source-1 row range.
    """

    truth = pp.load_ground_truth(pp.ground_truth_path(pp.resolve_data_dir()))

    order = np.argsort(s1_ids)
    sorted_s1 = s1_ids[order]
    gt_s1 = np.asarray(sorted(truth), dtype=s1_ids.dtype)
    slot = np.searchsorted(sorted_s1, gt_s1)
    found = slot < len(order)
    found[found] &= sorted_s1[slot[found]] == gt_s1[found]
    if not found.all():
        _log(
            f"[truth] {int((~found).sum()):,} labelled entities are absent from the "
            "cached Source-1 rows and cannot be scored"
        )
    s1_row = np.where(found, order[np.clip(slot, 0, len(order) - 1)], -1)

    counts = [len(matches) for matches in truth.values()]
    flat_pool = np.fromiter(
        (pid for matches in truth.values() for pid in matches),
        dtype=pool_ids.dtype, count=sum(counts),
    )
    flat_s1 = np.repeat(s1_row, counts)

    keep = flat_s1 >= 0
    if not keep.all():
        flat_pool, flat_s1 = flat_pool[keep], flat_s1[keep]

    uniq_pool, inverse = np.unique(flat_pool, return_inverse=True)
    pool_order = np.argsort(pool_ids)
    slot = np.searchsorted(pool_ids[pool_order], uniq_pool)
    in_pool = slot < len(pool_order)
    if not in_pool.all():
        _log(
            f"[truth] {int((~in_pool).sum()):,} labelled pool IDs are absent from "
            "the cached pool and cannot be matched"
        )
    pool_row = np.where(
        in_pool, pool_order[np.clip(slot, 0, len(pool_order) - 1)], -1
    )
    keep = in_pool[np.clip(inverse, 0, len(in_pool) - 1)]
    truth_s1 = flat_s1[keep]
    truth_pool = pool_row[inverse][keep]

    order = np.lexsort((truth_pool, truth_s1))
    return truth_s1[order].astype(np.int64), truth_pool[order].astype(np.int64)


def _label_and_ceiling(
    pairs: np.ndarray, truth_s1: np.ndarray, truth_pool: np.ndarray,
    pool_size: int, n_s1: int, chunk_rows: int = TRUTH_CHUNK_ROWS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Match candidate pairs against ground truth, and score blocking recall.

    Returns ``(labels, truth_hit)`` where ``labels[i]`` is 1 when pair ``i`` is a
    true match, and ``truth_hit[j]`` is True when truth pair ``j`` was offered as
    a candidate.

    Walked in Source-1 row ranges rather than as one flat ``np.isin`` over the
    whole pair array: the pair array is ~3x10^8 rows, so a whole-array
    ``isin`` would build a multi-gigabyte sorted copy of it, whereas each chunk
    only holds the truth keys for its own row range -- a few thousand entries.
    """
    labels = np.empty(len(pairs), dtype=np.int8)
    truth_hit = np.empty(len(truth_s1), dtype=bool)
    if not len(truth_s1):
        return np.zeros(len(pairs), dtype=np.int8), truth_hit

    # The pair array is grouped by s1_row, so one searchsorted gives the slice
    # boundaries for every entity, and per-chunk slices follow from that.
    starts = np.searchsorted(pairs[:, 0], np.arange(n_s1 + 1), side="left")
    for begin in range(0, n_s1, chunk_rows):
        end = min(begin + chunk_rows, n_s1)
        pair_lo, pair_hi = int(starts[begin]), int(starts[end])
        truth_lo = int(np.searchsorted(truth_s1, begin, side="left"))
        truth_hi = int(np.searchsorted(truth_s1, end, side="left"))
        cand_key = (
            pairs[pair_lo:pair_hi, 0].astype(np.int64) * np.int64(pool_size)
            + pairs[pair_lo:pair_hi, 1]
        )
        true_key = (
            truth_s1[truth_lo:truth_hi] * np.int64(pool_size) + truth_pool[truth_lo:truth_hi]
        )
        labels[pair_lo:pair_hi] = np.isin(cand_key, true_key)
        truth_hit[truth_lo:truth_hi] = np.isin(true_key, cand_key)
    return labels, truth_hit


def _recall_from_hits(truth_s1: np.ndarray, truth_hit: np.ndarray) -> Dict[str, float]:
    """Blocking recall, from one boolean per ground-truth pair.

    Two rates, and the second is the one that matters: ``pair_recall`` is the
    fraction of true pairs retrieved, while ``entity_full_recall`` is the
    fraction of matched entities with *every* true match retrieved. Macro F_0.5
    scores whole entities, so an entity that retrieved 3 of its 4 matches scores
    zero on that entity no matter how good its other predictions are.
    """

    if not len(truth_s1):
        return {
            "pair_recall": 0.0, "entity_full_recall": 0.0,
            "true_pairs": 0.0, "entities_with_matches": 0.0,
        }
    # All of an entity's truth pairs must be hits for that entity to be full.
    edges = np.flatnonzero(truth_s1[1:] != truth_s1[:-1]) + 1
    per_entity = np.logical_and.reduceat(truth_hit, np.concatenate(([0], edges)))
    return {
        "pair_recall": float(truth_hit.sum()) / len(truth_hit),
        "entity_full_recall": float(per_entity.sum()) / len(per_entity),
        "true_pairs": float(len(truth_hit)),
        "entities_with_matches": float(len(per_entity)),
        "truth_pairs_recovered": float(truth_hit.sum()),
    }


# --------------------------------------------------------------------------- #
# Stage: features
# --------------------------------------------------------------------------- #


# Stage: labels
# --------------------------------------------------------------------------- #


# Stage: labels
# --------------------------------------------------------------------------- #


def run_labels(split: str, cache_dir: Path, artifacts: Path) -> Path:
    """Label every candidate pair against the ground truth; measure the ceiling.

    Separate from featurisation because it needs no features at all: a pair is
    positive iff its pool record appears in the ground truth list for its
    Source-1 entity. That is one ``searchsorted`` pass over the pair array, so
    the labels are available in minutes instead of after a multi-hour
    featurisation -- and the blocking recall ceiling falls out of the same walk.

    Writes ``train_labels.npy``, one int8 per row of ``train_pairs.npy``, in the
    pair array's own order. The alignment is the whole contract: a permuted
    label vector trains a model that scores well and predicts nonsense, and
    nothing downstream can detect it. So it is asserted here rather than trusted.
    """

    if split != "train":
        raise ValueError("labels only exist for the training split")

    mm = dict(mmap_mode="r", allow_pickle=False)
    pairs = np.load(cache_dir / "train_pairs.npy", **mm)
    s1_ids = np.load(cache_dir / "train_s1_ids.npy", **mm)
    pool_ids = pool_ids_of(cache_dir, "train")
    n_s1 = len(s1_ids)
    _log(f"[labels] pairs={len(pairs):,} s1={n_s1:,} pool={len(pool_ids):,}")

    truth = pp.load_ground_truth(pp.ground_truth_path(pp.resolve_data_dir()))
    truth_s1, truth_pool = _truth_row_map(s1_ids, pool_ids, len(pool_ids))
    labels, truth_hit = _label_and_ceiling(
        pairs, truth_s1, truth_pool, len(pool_ids), n_s1
    )

    # 1:1 with the feature rows, asserted rather than assumed.
    if len(labels) != len(pairs):
        raise RuntimeError(
            f"label vector has {len(labels):,} rows but the pair array has "
            f"{len(pairs):,}; they must be row-aligned or training is corrupted"
        )
    path = cache_dir / "train_labels.npy"
    np.save(path, labels)

    positives = int(labels.sum())
    negatives = len(labels) - positives
    _log(
        f"[labels] wrote {path.name}: {positives:,} positive / {negatives:,} "
        f"negative ({positives / max(len(labels), 1):.4%} positive)"
    )

    ceiling = _recall_from_hits(truth_s1, truth_hit)
    _log(
        f"[labels] blocking recall ceiling: "
        f"pair={ceiling['pair_recall']:.4f} "
        f"entity_full={ceiling['entity_full_recall']:.4f} "
        f"over {int(ceiling['true_pairs']):,} true pairs"
    )
    _spot_check_labels(pairs, labels, truth, s1_ids, pool_ids)

    artifacts.mkdir(parents=True, exist_ok=True)
    report = {
        "pairs": int(len(labels)),
        "positives": positives,
        "negatives": negatives,
        "positive_rate": positives / max(len(labels), 1),
        "ceiling": ceiling,
    }
    (artifacts / "train_labels.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (artifacts / "blocking_ceiling_train.json").write_text(
        json.dumps(ceiling, indent=2), encoding="utf-8"
    )
    return path


def _spot_check_labels(
    pairs: np.ndarray, labels: np.ndarray, truth: Dict[str, List[str]],
    s1_ids: np.ndarray, pool_ids: np.ndarray, sample: int = 3, seed: int = 0,
) -> None:
    """Re-derive a few labels from the raw ground-truth file and compare.

    A mislabelled or row-permuted label vector is invisible to every downstream
    check: the model still trains, the probabilities still look calibrated, and
    the threshold still tunes. So a handful of rows are re-derived independently
    here -- from ``train_ground_truth.tsv`` via the ID strings, not from the
    searchsorted machinery that produced the labels -- on both sides of the
    decision. Row count is checked by the caller.
    """

    rng = np.random.default_rng(seed)
    for name, want_positive in (("positive", True), ("negative", False)):
        rows = np.flatnonzero(labels == 1 if want_positive else labels == 0)
        if not len(rows):
            raise RuntimeError(
                f"no {name} candidate pairs at all; the labels cannot be trusted"
            )
        pick = rows[rng.choice(len(rows), size=min(sample, len(rows)), replace=False)]
        for row in pick:
            s1_id = s1_ids[pairs[row, 0]].decode("utf-8")
            pool_id = pool_ids[pairs[row, 1]].decode("utf-8")
            truth_list = truth.get(s1_id, [])
            derived = pool_id in truth_list
            if derived != bool(labels[row]):
                raise RuntimeError(
                    f"label check failed at pair row {row:,}: {s1_id} vs {pool_id} "
                    f"is {'a' if derived else 'not a'} ground-truth match, but the "
                    f"label says {'positive' if labels[row] else 'negative'}"
                )
        _log(
            f"[labels]   {name} spot check: {len(pick)} rows re-derived from "
            "train_ground_truth.tsv and consistent"
        )


# --------------------------------------------------------------------------- #
# Stage: entity split
# --------------------------------------------------------------------------- #


def build_split(
    cache_dir: Path, artifacts: Path, seed: int = SPLIT_SEED,
    fraction: float = VALID_FRACTION,
    fit_entities: int = FIT_ENTITIES,
    valid_entities: int = VALID_ENTITIES,
) -> Dict[str, object]:
    """Partition Source-1 entities into fit and validation, and persist it.

    Split at the **entity** level, stratified by country, and persisted to
    ``artifacts/split.json`` plus two ID lists. A pair-level split leaks: a
    Source-2 record that truly matches a validation Source-1 entity frequently
    also matches a training one, so the model would be tuned against an entity
    whose pool records it had already partly seen.

    The partition is taken first, at the full :data:`VALID_FRACTION`, and only
    then is each side cut down to the number of entities that will actually be
    featurised. Doing it in that order matters: the subsample is drawn from
    inside the held-out partition, so a fitted model has never seen a validation
    entity even in a subsampled form, and the *fraction* is the documented,
    canonical 15% regardless of how much of it gets used.

    Stratification is by ``country_norm`` so the US/India mix is represented in
    both halves at the same proportions rather than at whatever the sample
    happens to draw.
    """

    mm = dict(mmap_mode="r", allow_pickle=False)
    s1 = pp.load_normalised(
        cache_dir, "train", sources=(1,), columns=("country_norm",)
    )
    s1_ids = s1.entity_id
    n_s1 = len(s1_ids)
    countries = np.char.decode(np.asarray(s1.country_norm))
    decoded = np.char.decode(s1_ids)

    strata = {entity: country for entity, country in zip(decoded, countries)}
    fit_ids, valid_ids = pp.split_train_validation(
        decoded, validation_fraction=fraction, seed=seed, strata=strata,
    )
    _log(
        f"[split] entity-level partition ({fraction:.0%} held out, seed {seed}, "
        f"stratified by country): {len(fit_ids):,} fit / {len(valid_ids):,} valid"
    )

    # ID list -> Source-1 row index, via a sorted view rather than a dict (a dict
    # over 2.2M keys is ~500 MB on its own).
    order = np.argsort(s1_ids)
    sorted_ids = s1_ids[order]

    def rows_of(ids: List[str]) -> np.ndarray:
        keys = np.array([i.encode("utf-8") for i in ids], dtype=sorted_ids.dtype)
        slot = np.searchsorted(sorted_ids, keys)
        if len(slot) and (slot.max() >= len(order) or
                           (sorted_ids[np.minimum(slot, len(order) - 1)] != keys).any()):
            raise RuntimeError("entity->row mapping failed; cache and split disagree")
        return order[slot].astype(np.int64)

    fit_rows = rows_of(fit_ids)
    valid_rows = rows_of(valid_ids)
    fit_rows = _stratified_subsample(fit_rows, countries, fit_entities, seed)
    valid_rows = _stratified_subsample(valid_rows, countries, valid_entities, seed + 1)
    overlap = np.intersect1d(fit_rows, valid_rows, assume_unique=False)
    if len(overlap):
        raise RuntimeError(
            f"{len(overlap):,} entities appear on both sides of the split; "
            "threshold tuning would be measured on seen entities"
        )
    _log(
        f"[split] featurised subsample: {len(fit_rows):,} fit / "
        f"{len(valid_rows):,} valid entities "
        f"({len(fit_ids):,} / {len(valid_ids):,} in the full partition)"
    )

    artifacts.mkdir(parents=True, exist_ok=True)
    for name, rows in (("fit", fit_rows), ("validation", valid_rows)):
        (artifacts / f"{name}_entities.txt").write_text(
            "\n".join(np.char.decode(np.asarray(s1_ids)[rows])) + "\n",
            encoding="utf-8",
        )
    (artifacts / "split.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "validation_fraction": fraction,
                "fit_entities_requested": fit_entities,
                "valid_entities_requested": valid_entities,
                "fit_entities": int(len(fit_rows)),
                "validation_entities": int(len(valid_rows)),
                "full_partition_fit": int(len(fit_ids)),
                "full_partition_valid": int(len(valid_ids)),
                "stratified_by": "country_norm",
                "overlap": 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    np.save(artifacts / "fit_rows.npy", fit_rows)
    np.save(artifacts / "valid_rows.npy", valid_rows)
    return {"fit_rows": fit_rows, "valid_rows": valid_rows, "n_s1": n_s1}


def _stratified_subsample(
    rows: np.ndarray, countries: np.ndarray, target: int, seed: int,
) -> np.ndarray:
    """Cut ``rows`` down to ``target``, keeping each country's share intact.

    A plain random subsample would represent the countries in expectation but
    not by construction, and with a validation set this small that is a real
    source of noise in the tuned threshold. Allocating the quota per country
    first makes the retained mix match the parent population up to one entity.
    """

    rows = np.sort(np.asarray(rows, dtype=np.int64))
    if target <= 0 or len(rows) <= target:
        return rows
    rng = np.random.default_rng(seed)
    country_of_row = countries[rows]
    # Stable sort groups the rows by country, so one argsort plus a
    # change-point detection splits them into per-country runs.
    order = np.argsort(country_of_row, kind="stable")
    grouped = country_of_row[order]
    starts_ = np.flatnonzero(
        np.concatenate(([True], grouped[1:] != grouped[:-1]))
    )
    groups = np.split(np.arange(len(sorted_rows := rows[order])), starts_[1:])
    keep: List[np.ndarray] = []
    for group in groups:
        quota = min(int(round(len(group) * target / len(rows))), len(group))
        if quota <= 0:
            continue
        pick = rng.choice(len(group), size=quota, replace=False)
        keep.append(sorted_rows[group[np.sort(pick)]])
    if not keep:
        # Every country rounded down to zero (target far below the number of
        # countries): fall back to a uniform draw so the split is never empty.
        pick = rng.choice(len(rows), size=target, replace=False)
        return np.sort(rows[np.sort(pick)])
    chosen = np.sort(np.concatenate(keep))
    # Rounding can leave the total a little under target; top up uniformly.
    if len(chosen) < target:
        extra = np.setdiff1d(rows, chosen, assume_unique=True)
        take = min(target - len(chosen), len(extra))
        if take:
            pick = rng.choice(len(extra), size=take, replace=False)
            chosen = np.sort(np.concatenate([chosen, extra[np.sort(pick)]]))
    return chosen


# --------------------------------------------------------------------------- #
# Stage: features (bounded sample)
# --------------------------------------------------------------------------- #


def _blocks_for_rows(
    starts: np.ndarray, rows: np.ndarray, block_pairs: int,
) -> List[Tuple[int, int]]:
    """Pair blocks covering ``rows``, in runs of consecutive entities.

    Consecutive chosen entities share a block so the per-record caches are reused
    across an entity's ~136 candidates. Entities with no candidates contribute
    nothing and are skipped -- they are still counted by the metric, as a correct
    abstention, because scoring iterates the entity list rather than the pairs.
    """

    blocks: List[Tuple[int, int]] = []
    i = 0
    while i < len(rows):
        lo = int(starts[rows[i]])
        j = i + 1
        while j < len(rows) and rows[j] == rows[j - 1] + 1:
            if int(starts[rows[j] + 1]) - lo > block_pairs:
                break
            j += 1
        hi = int(starts[rows[j - 1] + 1])
        if hi > lo:
            blocks.append((lo, hi))
        i = j
    return blocks


def run_sample_features(
    cache_dir: Path, artifacts: Path, n_proc: int = sm.DEFAULT_WORKERS,
    block_pairs: int = sm.DEFAULT_BLOCK_PAIRS,
) -> Dict[str, Path]:
    """Featurise the fitting and validation entity samples into bounded matrices.

    This replaces the stage that tried to featurise all 300,537,790 candidate
    pairs into a 39.19 GiB matrix. That was not slow, it was impossible: the
    Windows commit limit on this box is 24.11 GiB, and a writeable file mapping
    is charged to commit as it dirties, so a 39.19 GiB output cannot be written
    at all. The run died at 3 of 89 chunks, six times.

    Nothing downstream needed the full matrix, so this materialises only the
    ~13.6M fitting rows and ~6.8M validation rows that the classifier and the
    threshold sweep actually consume: 1.9 GB and 0.95 GB, resumed per block.
    """

    split = build_split(cache_dir, artifacts)
    mm = dict(mmap_mode="r", allow_pickle=False)
    pairs = np.load(cache_dir / "train_pairs.npy", **mm)
    s1_ids = np.load(cache_dir / "train_s1_ids.npy", **mm)
    labels = np.load(cache_dir / "train_labels.npy", **mm)
    n_s1 = len(s1_ids)
    starts = np.searchsorted(np.asarray(pairs[:, 0]), np.arange(n_s1 + 1), side="left")
    known = _known_countries(cache_dir, "train")
    _log(f"[features] known countries ({len(known)}): {sorted(known)}")

    out: Dict[str, Path] = {}
    with sm.Featuriser(
        cache_dir, "train", FEATURE_COLUMNS, known, n_proc=n_proc
    ) as featuriser:
        for side_key, side_name in [("valid_rows", "validation"), ("fit_rows", "fit")]:
            out[side_name] = _featurise_side(
                side_name, split[side_key], pairs, labels, starts, cache_dir,
                featuriser, block_pairs,
            )
    return out


def _featurise_side(
    side: str, rows: np.ndarray, pairs: np.ndarray, labels: np.ndarray,
    starts: np.ndarray, cache_dir: Path, featuriser: "sm.Featuriser",
    block_pairs: int,
) -> Path:
    """Featurise one side of the split into a preallocated on-disk matrix."""

    n_columns = len(ft.feature_columns())
    blocks = _blocks_for_rows(starts, rows, block_pairs)
    total = sum(hi - lo for lo, hi in blocks)
    matrix_path = cache_dir / f"train_{side}_features.npy"
    progress_path = cache_dir / f"train_{side}_features.progress.json"
    _log(
        f"[features] {side}: {len(rows):,} entities -> {len(blocks):,} blocks, "
        f"{total:,} pairs ({total * n_columns * 4 / 1024 ** 3:.2f} GiB)"
    )

    done = 0
    if matrix_path.exists() and progress_path.exists():
        try:
            state = json.loads(progress_path.read_text(encoding="utf-8"))
            if (int(state.get("total_rows", -1)) == total
                    and list(state.get("shape", ())) == [total, n_columns]):
                done = max(0, min(int(state.get("blocks_done", 0)), len(blocks)))
                if done:
                    _log(f"[features] {side}: resuming at block {done}/{len(blocks)}")
        except (OSError, ValueError, TypeError):
            done = 0
        if not done:
            progress_path.unlink(missing_ok=True)

    out = np.lib.format.open_memmap(
        matrix_path, mode="r+" if done else "w+", dtype=np.float32,
        shape=(total, n_columns),
    )
    started = time.time()
    written = 0
    for index, block in enumerate(featuriser.map_features(blocks[done:]), start=done):
        lo, hi = blocks[index]
        out[written:written + (hi - lo)] = block
        written += hi - lo
        out.flush()
        progress_path.write_text(
            json.dumps(
                {
                    "blocks_done": index + 1, "blocks_total": len(blocks),
                    "rows_done": written, "total_rows": total,
                    "shape": [total, n_columns],
                }
            ),
            encoding="utf-8",
        )
        if not index % 200:
            rate = written / max(time.time() - started, 1e-9)
            _log(
                f"[features]   {side} block {index + 1}/{len(blocks)} "
                f"{written:,}/{total:,} rows ({rate:,.0f}/s)"
            )
    if out is not None:
        out.flush()
        del out
    progress_path.unlink(missing_ok=True)

    # Row indices back into the pair array. Saving the pair row rather than the
    # resolved (s1_row, pool_row) pair halves the bytes and keeps a single source
    # of truth for row alignment.
    pair_rows = np.concatenate([
        np.arange(lo, hi, dtype=np.int64) for lo, hi in blocks
    ]) if blocks else np.empty(0, np.int64)
    np.save(cache_dir / f"train_{side}_pair_rows.npy", pair_rows)
    np.save(cache_dir / f"train_{side}_labels.npy", labels[pair_rows])
    positives = int(labels[pair_rows].sum())
    _log(
        f"[features] {side}: wrote {matrix_path.name} "
        f"({total:,} x {n_columns}), {positives:,} positive "
        f"({positives / max(total, 1):.3%})"
    )
    return matrix_path


def pool_ids_of(cache_dir: Path, split: str) -> np.ndarray:
    """Pool entity IDs, read back from the candidate stage's context arrays."""

    return np.load(cache_dir / f"{split}_pool_ids.npy")


def _known_countries(cache_dir: Path, split: str) -> Set[str]:
    """Countries seen in training, used only to flag *unseen* ones.

    The flag is a feature, not a filter: ``France`` is absent from training and
    must still be able to match. Deriving the set from the split being processed
    would leak test information into inference, so for anything but ``train`` the
    recorded training set is used.
    """

    path = cache_dir / "known_countries.json"
    if split == "train":
        countries = _collect_countries(cache_dir, split)
        path.write_text(json.dumps(sorted(countries)), encoding="utf-8")
        return countries
    if path.exists():
        return set(json.loads(path.read_text(encoding="utf-8")))
    return _collect_countries(cache_dir, split)


def _collect_countries(cache_dir: Path, split: str) -> Set[str]:
    found: Set[str] = set()
    for source in (1, 2, 3):
        norm = pp.load_normalised(
            cache_dir, split, sources=(source,), columns=("country_norm",)
        )
        values = {v.decode("utf-8", "replace") for v in np.unique(norm.country_norm)}
        found |= {v for v in values if v}
    return found



# --------------------------------------------------------------------------- #
# Stage: train
# --------------------------------------------------------------------------- #


def _load_side(cache_dir: Path, side: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one side's featurised sample as ``(features, labels, pair_rows)``."""

    features = np.load(cache_dir / f"train_{side}_features.npy", mmap_mode="r")
    labels = np.load(cache_dir / f"train_{side}_labels.npy")
    pair_rows = np.load(cache_dir / f"train_{side}_pair_rows.npy")
    if len(labels) != len(pair_rows) or features.shape[0] != len(pair_rows):
        raise RuntimeError(
            f"{side} sample is misaligned: features {features.shape[0]:,}, "
            f"labels {len(labels):,}, pair rows {len(pair_rows):,}"
        )
    return features, labels, pair_rows


def _truth_counts(
    valid_rows: np.ndarray, s1_ids: np.ndarray,
    truth: Dict[str, List[str]],
) -> np.ndarray:
    """``|T|`` per validation entity, from the **full** ground truth.

    Two deliberate choices here, both of which flatter the model if done the
    other way:

    * the denominator is every true match, not just the ones blocking proposed.
      A true match that was never a candidate is an unreachable miss, and folding
      the candidate set into the denominator would score the classifier against a
      ceiling it was never asked to reach.
    * entities come from the persisted validation ID list, not from the pairs. An
      entity with no candidates, or whose candidates were all dropped, still
      appears -- as a correct abstention if it is a true singleton.
    """

    return np.array(
        [len(truth.get(s1_ids[row].decode("utf-8"), ())) for row in valid_rows],
        dtype=np.int64,
    )


def run_train(
    cache_dir: Path, artifacts: Path, seed: int = SPLIT_SEED,
    n_jobs: int = 8, num_boost_round: int = 2000,
) -> Dict[str, object]:
    """Fit the classifier and tune the threshold against macro F_0.5.

    Consumes the bounded samples written by :func:`run_sample_features`. The
    validation side is scored **exhaustively** -- every candidate of every
    validation entity -- because the metric averages over entities, and
    subsampling candidates within an entity would make an entity look like a
    singleton it is not.
    """

    mm = dict(mmap_mode="r", allow_pickle=False)
    pairs = np.load(cache_dir / "train_pairs.npy", **mm)
    s1_ids = np.load(cache_dir / "train_s1_ids.npy", **mm)
    pool_ids = pool_ids_of(cache_dir, "train")
    valid_rows = np.load(artifacts / "valid_rows.npy")
    n_s1 = len(s1_ids)

    X_fit, y_fit, rows_fit = _load_side(cache_dir, "fit")
    X_valid, y_valid, rows_valid = _load_side(cache_dir, "validation")
    columns = list(ft.feature_columns())
    _log(
        f"[train] fit X={X_fit.shape} positives={int(y_fit.sum()):,} "
        f"({y_fit.mean():.3%}) | validation X={X_valid.shape} "
        f"positives={int(y_valid.sum()):,} ({y_valid.mean():.3%})"
    )
    if X_fit.shape[1] != len(columns):
        raise RuntimeError(
            f"sample has {X_fit.shape[1]} feature columns, expected {len(columns)}; "
            "refeaturise rather than align silently"
        )

    # Early stopping on a slice of the *fitting* entities. The validation side is
    # the operating-point set, so using it to pick the number of boosting rounds
    # would tune two things on one measurement and make the reported F_0.5
    # optimistic.
    inner = _inner_early_stop_mask(pairs[rows_fit][:, 0], seed=seed)
    X_inner, y_inner = (
        np.asarray(X_fit[~inner]), y_fit[~inner]
    )
    _log(
        f"[train] early-stopping slice: {int(inner.sum()):,} rows held out of the "
        f"{len(y_fit):,} fitting rows"
    )
    model = md.train_classifier(
        np.asarray(X_fit[inner]), y_fit[inner], columns, seed=seed, n_jobs=n_jobs,
        valid=(X_inner, y_inner), num_boost_round=num_boost_round,
    )
    del X_inner, y_inner, X_fit, inner

    scores = md.predict_proba(model, np.asarray(X_valid), n_jobs=n_jobs)
    valid_pairs = np.asarray(pairs[rows_valid])
    valid_s1 = valid_pairs[:, 0].astype(np.int64)
    truth = pp.load_ground_truth(pp.ground_truth_path(pp.resolve_data_dir()))

    # The entity index the metric averages over: the persisted validation rows,
    # mapped to a dense 0..n-1 range. Using row indices directly would work too,
    # but the count arrays are then 2.2M long per threshold.
    n_true = _truth_counts(valid_rows, s1_ids, truth)
    slot_of_pair = _slot_of_rows(valid_s1, valid_rows)
    singleton_slots = int((n_true == 0).sum())
    _log(
        f"[validation] {len(valid_rows):,} entities "
        f"({singleton_slots:,} true singletons, {singleton_slots / max(len(valid_rows), 1):.2%}), "
        f"{len(valid_s1):,} candidate pairs"
    )

    best_t, best_score, sweep = md.tune_threshold_arrays(
        slot_of_pair, valid_pairs[:, 1].astype(np.int64), scores, y_valid,
        len(valid_rows), n_true, resolve=False, pool_size=len(pool_ids),
    )
    best_t_r, best_score_r, sweep_r = md.tune_threshold_arrays(
        slot_of_pair, valid_pairs[:, 1].astype(np.int64), scores, y_valid,
        len(valid_rows), n_true, resolve=True, pool_size=len(pool_ids),
    )
    _log(
        f"[train] threshold={best_t:.4f} macro_f05={best_score:.4f} (no resolution)"
    )
    _log(
        f"[train] threshold={best_t_r:.4f} macro_f05={best_score_r:.4f} (with resolution)"
    )
    resolve = best_score_r > best_score
    threshold = best_t_r if resolve else best_t
    final_score = best_score_r if resolve else best_score
    chosen_sweep = sweep_r if resolve else sweep

    metrics = _validation_metrics(
        slot_of_pair, valid_pairs[:, 1].astype(np.int64), scores, y_valid,
        len(valid_rows), n_true, threshold, pool_size=len(pool_ids), resolve=resolve,
    )
    ceiling_path = artifacts / "blocking_ceiling_train.json"
    ceiling = json.loads(ceiling_path.read_text("utf-8")) if ceiling_path.exists() else {}
    _log(
        f"[validation] macro_f05={metrics['macro_f05']:.4f} "
        f"pair_precision={metrics['pair_precision']:.4f} "
        f"pair_recall={metrics['pair_recall']:.4f} "
        f"singleton_accuracy={metrics['singleton_accuracy']:.4f} "
        f"predicted_matches={metrics['predicted_pairs']:,}"
    )
    _log(
        f"[validation] ceiling: pair={ceiling.get('pair_recall', float('nan')):.4f} "
        f"entity_full={ceiling.get('entity_full_recall', float('nan')):.4f}"
    )

    importances = _feature_importances(model, columns, artifacts)
    top = ", ".join(
        f"{name}={value:.0f}" for name, value in importances[:8]
    )
    _log(f"[train] top feature importances: {top}")

    (artifacts / "threshold_sweep.json").write_text(
        json.dumps(
            {
                "best_threshold": best_t, "best_score": best_score,
                "best_threshold_resolved": best_t_r, "best_score_resolved": best_score_r,
                "selected_threshold": threshold, "resolve_competition": resolve,
                "sweep": chosen_sweep, "sweep_unresolved": sweep,
                "validation_entities": int(len(valid_rows)),
                "validation_pairs": int(len(valid_s1)),
                "validation_singletons": int(singleton_slots),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (artifacts / "validation_metrics.json").write_text(
        json.dumps(
            {
                **metrics,
                "blocking_recall_ceiling": ceiling,
                "threshold": threshold,
                "resolve_competition": resolve,
                "validation_entities": int(len(valid_rows)),
                "validation_pairs": int(len(valid_s1)),
                "validation_singletons": int(singleton_slots),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    md.save_model(model, columns, str(artifacts / "matcher.txt"))
    cap_path = cache_dir / "train_max_candidates.txt"
    config = {
        "threshold": threshold,
        "validation_macro_f05": final_score,
        "resolve_competition": resolve,
        "max_candidates_per_entity": (
            int(cap_path.read_text(encoding="utf-8").strip())
            if cap_path.exists() else None
        ),
        "strategies": list(blocking.STRATEGIES),
        "max_block": blocking.DEFAULT_MAX_BLOCK,
        "seed": seed,
        "feature_columns": columns,
        "n_boost_round": int(
            getattr(model, "best_iteration", 0) or num_boost_round
        ),
    }
    (artifacts / "model_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    _log(f"[train] saved model and config (threshold={threshold:.4f})")
    return config


def _slot_of_rows(s1_rows: np.ndarray, valid_rows: np.ndarray) -> np.ndarray:
    """Map Source-1 row indices to a dense 0..len(valid_rows)-1 slot index.

    The metric averages over entities, so the count arrays are indexed by entity
    and are only as long as the validation set rather than the full 2.2M-row
    Source-1 table. Rows not in ``valid_rows`` cannot occur: the pair rows come
    from the same side of the same split.
    """

    if len(s1_rows) and (s1_rows.min() < valid_rows.min()
                         or s1_rows.max() > valid_rows.max()):
        raise RuntimeError("validation pairs reference entities outside the split")
    lookup = np.zeros(int(s1_rows.max()) + 1 if len(s1_rows) else 1, dtype=np.int64)
    lookup[valid_rows] = np.arange(len(valid_rows), dtype=np.int64)
    slot = lookup[s1_rows]
    if (lookup[valid_rows] != np.arange(len(valid_rows))).any():
        raise RuntimeError("validation split contains duplicate entities")
    return slot


def _inner_early_stop_mask(fit_s1_rows: np.ndarray, seed: int, fraction: float = 0.05):
    """Entity-level mask over the fitting rows, for early stopping only."""

    import hashlib

    rows = np.unique(fit_s1_rows)
    flags = np.fromiter(
        (
            int.from_bytes(
                hashlib.blake2b(f"{seed}:es:{int(r)}".encode(), digest_size=8).digest(),
                "big",
            ) / float(1 << 64) < fraction
            for r in rows
        ),
        dtype=bool,
        count=len(rows),
    )
    lookup = np.zeros(int(fit_s1_rows.max()) + 1, dtype=bool)
    lookup[rows] = flags
    return lookup[fit_s1_rows]


def _validation_metrics(
    slot_of_pair: np.ndarray, pool_of_pair: np.ndarray, scores: np.ndarray,
    is_true: np.ndarray, n_entities: int, n_true: np.ndarray,
    threshold: float, pool_size: int, resolve: bool,
) -> Dict[str, float]:
    """Precision / recall / F_0.5 / singleton accuracy at one operating point.

    All four are entity-level except the pair-level pair, which is reported
    alongside because it is the number that says *where* the score is lost: a
    high pair precision with a low F_0.5 means entities are being lost whole,
    which is a recall or singleton problem rather than a threshold problem.
    """

    from . import evaluation as ev

    keep = np.flatnonzero(scores >= threshold)
    if resolve:
        keep = keep[md.resolve_competition_indices(
            slot_of_pair[keep], pool_of_pair[keep], scores[keep], 0.0, pool_size,
        )]
    n_pred, n_hit = ev.per_entity_counts(
        slot_of_pair[keep], is_true[keep], n_entities
    )
    true_positive = int(n_hit.sum())
    predicted = int(n_pred.sum())
    actual = int(n_true.sum())
    singletons = n_true == 0
    return {
        "threshold": float(threshold),
        "macro_f05": ev.macro_f05_counts(n_true, n_pred, n_hit),
        "pair_precision": true_positive / predicted if predicted else 0.0,
        "pair_recall": true_positive / actual if actual else 0.0,
        "true_positive": float(true_positive),
        "predicted_pairs": float(predicted),
        "true_pairs": float(actual),
        "singleton_entities": int(singletons.sum()),
        "singleton_accuracy": (
            float((n_pred[singletons] == 0).mean()) if singletons.any() else 0.0
        ),
        "entities_predicted_empty": int((n_pred == 0).sum()),
    }


def _feature_importances(
    model, columns: Sequence[str], artifacts: Path, top: int = 35,
) -> List[Tuple[str, float]]:
    """Gain-based importances, persisted for the write-up."""

    booster = getattr(model, "booster_", model)
    gains = booster.feature_importance(importance_type="gain")
    ranked = sorted(zip(columns, gains.tolist()), key=lambda kv: -kv[1])
    (artifacts / "feature_importance.json").write_text(
        json.dumps(
            [{"feature": name, "gain": value, "rank": i + 1}
             for i, (name, value) in enumerate(ranked[:top])],
            indent=2,
        ),
        encoding="utf-8",
    )
    return [(name, value) for name, value in ranked[:top]]


# Stage: predict
# --------------------------------------------------------------------------- #


def run_predict(
    cache_dir: Path, artifacts: Path, output_dir: Path,
    n_proc: int = sm.DEFAULT_WORKERS, block_pairs: int = sm.DEFAULT_BLOCK_PAIRS,
    shard_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Score the test candidates, resolve contention, and write both outputs.

    Streams, and does not build a feature matrix at all. Each block of candidate
    pairs is featurised in a worker, scored there, thresholded there, and only
    the survivors come back to the parent -- a few hundred rows per block instead
    of 2.2 MB of features, which over the full test split is the difference
    between 33 GiB of disk traffic and none. The pipeline used to materialise a
    39.19 GiB ``test_features.npy`` first, which is the same commit-limit
    impossibility the training stage hit and which nothing read more than once.

    Resumable through per-block shards. This is the longest single stage in the
    run, so a kill at 90% must not cost the whole thing: each block's claims are
    written to their own shard and a restart continues from the first block
    without one.

    Returns ``(matching_results.tsv, candidate_pairs.tsv)``.
    """

    config = json.loads((artifacts / "model_config.json").read_text(encoding="utf-8"))
    _booster, columns, _meta = md.load_model(str(artifacts / "matcher.txt"))
    if list(columns) != list(ft.feature_columns()):
        raise RuntimeError(
            "model feature columns do not match the current feature set; "
            "retrain rather than reorder silently"
        )
    threshold = float(config["threshold"])
    resolve = bool(config.get("resolve_competition", True))

    mm = dict(mmap_mode="r", allow_pickle=False)
    pairs = np.load(cache_dir / "test_pairs.npy", **mm)
    s1_ids = np.load(cache_dir / "test_s1_ids.npy", **mm)
    pool_ids = pool_ids_of(cache_dir, "test")
    n_s1 = len(s1_ids)
    starts = np.searchsorted(
        np.asarray(pairs[:, 0]), np.arange(n_s1 + 1), side="left"
    )
    blocks = sm.plan_blocks(starts, n_s1, block_pairs)
    _log(
        f"[predict] {n_s1:,} test entities, {len(pairs):,} candidate pairs, "
        f"{len(blocks):,} blocks, threshold={threshold:.4f} resolve={resolve}"
    )

    # The *training* country set, not the test set's own. Deriving it from the
    # split being processed would let the test split tell the model which
    # countries it is looking at; ``ctx_country_known`` is meant to mean "seen
    # during training", and that is the only thing that makes the flag
    # informative for an unseen country like France.
    known = set(json.loads((cache_dir / "known_countries.json").read_text("utf-8")))
    _log(f"[predict] known (training) countries: {sorted(known)}")

    shard_dir = Path(shard_dir) if shard_dir else cache_dir / "predict_shards"
    claim_s1, claim_pool, claim_score = _collect_claims(
        shard_dir, blocks, cache_dir, known, n_proc, threshold, resolve,
        len(pool_ids), str(artifacts / "matcher.txt"),
    )
    _log(f"[predict] {len(claim_s1):,} claims before the global contest")

    if resolve and len(claim_s1):
        # A pool record can be claimed from two different blocks, and no single
        # worker sees both, so the global pass is not optional. It is cheap: the
        # survivor set is a few million rows, not the 2.4x10^8 candidate pairs.
        claim_s1, claim_pool, claim_score = md.resolve_competition_arrays(
            claim_s1, claim_pool, claim_score, 0.0, len(pool_ids)
        )
    _log(f"[predict] {len(claim_s1):,} final matches")

    order = np.argsort(claim_s1, kind="stable")
    claim_s1, claim_pool = claim_s1[order], claim_pool[order]
    claim_starts = np.searchsorted(claim_s1, np.arange(n_s1 + 1), side="left")

    matching_path = _write_matching_results(
        output_dir / "matching_results.tsv", s1_ids, pool_ids, claim_pool,
        claim_starts, n_s1,
    )
    candidate_path = _write_candidate_pairs(
        output_dir / "candidate_pairs.tsv", s1_ids, pool_ids, pairs, starts, n_s1,
    )
    return matching_path, candidate_path


def _collect_claims(
    shard_dir: Path, blocks: Sequence[Tuple[int, int]], cache_dir: Path,
    known: Set[str], n_proc: int, threshold: float, resolve: bool,
    pool_size: int, model_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score every block, persisting each block's claims to a shard.

    A shard exists only for a finished block, so a kill costs at most the blocks
    in flight. Resolution runs *within* each block before the shard is written,
    which is safe to do incrementally: a claim dropped there is dominated by a
    surviving claim on the same pool record inside the same block, so the global
    winner for that record is never among the casualties. The global pass over the
    union still runs afterwards, to settle records claimed from different blocks.
    """

    s1_out: List[np.ndarray] = []
    pool_out: List[np.ndarray] = []
    score_out: List[np.ndarray] = []
    pending: List[Tuple[int, int, int, Path]] = []
    for index, (lo, hi) in enumerate(blocks):
        shard = shard_dir / f"block_{index:07d}.npz"
        data = _read_shard(shard)
        if data is not None:
            s1_out.append(data[0])
            pool_out.append(data[1])
            score_out.append(data[2])
            continue
        pending.append((index, lo, hi, shard))
    reused = len(blocks) - len(pending)
    if reused:
        _log(f"[predict] reusing {reused:,} shards from a previous run")

    if pending:
        _log(
            f"[predict] scoring {len(pending):,} blocks on {n_proc} workers "
            f"({sum(hi - lo for _i, lo, hi, _s in pending):,} pairs)"
        )
        started = time.time()
        with sm.Featuriser(
            cache_dir, "test", FEATURE_COLUMNS, known, n_proc=n_proc,
            model_path=model_path,
        ) as featuriser:
            stream = featuriser.map_scores(
                [(lo, hi) for _i, lo, hi, _s in pending], threshold
            )
            for offset, (index, lo, hi, shard) in enumerate(pending):
                s1, pool, score = next(stream)
                if resolve:
                    s1, pool, score = md.resolve_competition_arrays(
                        s1, pool, score, 0.0, pool_size
                    )
                _write_shard(shard, s1, pool, score)
                s1_out.append(s1)
                pool_out.append(pool)
                score_out.append(score)
                if not offset % 200:
                    done_pairs = offset + 1
                    rate = done_pairs / max(time.time() - started, 1e-9)
                    _log(
                        f"[predict]   block {offset + 1:,}/{len(pending):,}, "
                        f"{sum(len(x) for x in s1_out):,} matches so far "
                        f"({rate:.1f} blocks/s)"
                    )
    if not s1_out:
        empty = np.empty(0, np.int64)
        return empty, empty.copy(), np.empty(0, np.float64)
    return (
        np.concatenate(s1_out), np.concatenate(pool_out), np.concatenate(score_out)
    )


def _read_shard(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load a block's claims, or ``None`` if the shard is absent or truncated.

    A shard killed mid-write would otherwise be read as a short block and
    silently drop claims, so anything unreadable is treated as absent and the
    block is recomputed.
    """

    if not path.exists():
        return None
    try:
        data = np.load(path)
        s1, pool, score = data["s1"], data["pool"], data["score"]
    except (OSError, ValueError, KeyError):
        path.unlink(missing_ok=True)
        return None
    if not (len(s1) == len(pool) == len(score)):
        path.unlink(missing_ok=True)
        return None
    return s1, pool, score


def _write_shard(
    path: Path, s1: np.ndarray, pool: np.ndarray, score: np.ndarray,
) -> None:
    """Write a shard atomically, so a kill never leaves a half-file behind."""

    staging = path.with_suffix(".npz.building")
    np.savez(staging, s1=s1, pool=pool, score=score)
    os.replace(staging, path)


def _write_matching_results(
    path: Path, s1_ids: np.ndarray, pool_ids: np.ndarray, claim_pool: np.ndarray,
    claim_starts: np.ndarray, n_s1: int,
) -> Path:
    """One row per Source-1 entity; an empty list for a predicted singleton.

    Every entity gets a row. A missing row is a submission rejection, and an
    empty list is a *correct* answer for a singleton worth a full point on that
    entity, so abstaining has to stay expressible. IDs within a row are sorted and
    de-duplicated, both of which the validator enforces.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    matched = 0
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write("source1_entity_id\tmatched_entity_ids\n")
        for i in range(n_s1):
            lo, hi = int(claim_starts[i]), int(claim_starts[i + 1])
            entity = s1_ids[i].decode("utf-8")
            if lo == hi:
                handle.write(f"{entity}\t\n")
                continue
            ids = sorted({pool_ids[r].decode("utf-8") for r in claim_pool[lo:hi]})
            matched += 1
            handle.write(f"{entity}\t{','.join(ids)}\n")
    _log(f"[predict] wrote {path.name}: {n_s1:,} rows, {matched:,} with matches")
    return path


def _write_candidate_pairs(
    path: Path, s1_ids: np.ndarray, pool_ids: np.ndarray, pairs: np.ndarray,
    starts: np.ndarray, n_s1: int,
) -> Path:
    """One row per Source-1 entity, listing the candidates actually scored.

    Written from ``test_pairs.npy`` -- the exact set the model saw in this run --
    rather than re-derived from the blocking index, so the file cannot disagree
    with ``matching_results.tsv`` about what was considered. Every ID in
    ``matching_results.tsv`` must therefore appear here, which the validator
    checks.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write("source1_entity_id\tcandidate_entity_ids\n")
        for i in range(n_s1):
            lo, hi = int(starts[i]), int(starts[i + 1])
            entity = s1_ids[i].decode("utf-8")
            if hi <= lo:
                handle.write(f"{entity}\t\n")
                continue
            rows = np.asarray(pairs[lo:hi, 1])
            ids = sorted({pool_ids[r].decode("utf-8") for r in rows})
            total += len(ids)
            handle.write(f"{entity}\t{','.join(ids)}\n")
    _log(f"[predict] wrote {path.name}: {n_s1:,} rows, {total:,} candidate IDs")
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[3]
    parser.add_argument("--stage", required=True, choices=[
        "index", "candidates", "features", "train", "predict",
    ])
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--cache-dir", default=str(repo / "cache"))
    parser.add_argument("--artifacts", default=str(repo / "artifacts"))
    parser.add_argument("--output-dir", default=str(repo / "output"))
    parser.add_argument("--max-block", type=int, default=blocking.DEFAULT_MAX_BLOCK)
    parser.add_argument("--max-candidates", type=int, default=200)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--max-train-pairs", type=int, default=MAX_TRAIN_PAIRS,
        help="pair budget for fitting (0 = use every training pair)",
    )
    parser.add_argument(
        "--max-valid-pairs", type=int, default=MAX_VALID_PAIRS,
        help="pair budget for threshold tuning (0 = use every validation pair)",
    )
    parser.add_argument(
        "--pool-memmap", type=int, default=1, choices=[0, 1],
        help="back the Source-2+3 pool with on-disk memmaps (1) or in-RAM "
             "arrays (0). Exists so the two paths can be A/B'd for byte equality; "
             "0 is the pre-memmap behaviour and is only useful for that.",
    )
    args = parser.parse_args(argv)

    cache_dir = Path(args.cache_dir)
    artifacts = Path(args.artifacts)
    output_dir = Path(args.output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    if args.stage == "index":
        run_index(args.split, cache_dir, args.max_block, args.rebuild)
    elif args.stage == "candidates":
        run_candidates(
            args.split, cache_dir, artifacts, output_dir, args.max_candidates
        )
    elif args.stage == "features":
        if args.split == "train":
            run_labels(args.split, cache_dir, artifacts)
        run_sample_features(cache_dir, artifacts, args.threads, bool(args.pool_memmap))
    elif args.stage == "labels":
        run_labels(args.split, cache_dir, artifacts)
    elif args.stage == "train":
        run_train(
            cache_dir, artifacts, seed=args.seed,
            max_train_pairs=args.max_train_pairs,
            max_valid_pairs=args.max_valid_pairs,
        )
    elif args.stage == "predict":
        run_predict(
            cache_dir, artifacts, output_dir, args.threads,
            pool_memmap=bool(args.pool_memmap),
        )
    _log(f"[{args.stage}] done in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
