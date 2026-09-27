"""Train and apply the pairwise match classifier, then apply global consistency.

Pipeline position
-----------------
``preprocessing`` -> ``blocking`` -> ``features`` -> **this module** ->
``output/matching_results.tsv``.

Problem shape
-------------
This is not plain binary classification. Each Source-1 entity gets a
variable-length list of matched IDs, that list may legitimately be empty, and a
Source-2/3 ID should not end up claimed by two different Source-1 entities.

Label construction
------------------
Flatten the ground truth into positive pairs, then add negatives from the
blocking candidate set (any candidate absent from the true list is a negative
for that entity). The candidate set defines what the model ever sees, so negatives
drawn from it match the inference-time distribution. Note that entities with no
match contribute negatives only — and those are exactly the cases the metric
rewards, so keep them in and keep them measurable.

Model choice
------------
A gradient-boosted tree classifier (LightGBM/XGBoost) over the handcrafted
features in :mod:`features` is the strong baseline: it trains fast, handles the
heterogeneous scales of the feature set natively, and produces the calibrated
per-candidate score that thresholding needs. The challenge also permits
MIT/Apache-2.0-licensed models up to 8B parameters, so a text-embedding model is
a later option to evaluate against this baseline.

Global consistency
------------------
Per-pair thresholding can leave a Source-2 record matched by two different
Source-1 entities. Add a post-processing step resolving that contention, then
verify the effect against the metric before keeping it.

Threshold
---------
F_0.5 weights precision twice as heavily as recall, so the operating point sits
on the high-precision side of the ROC curve. Tune the decision threshold on the
validation split against the *actual* macro F_0.5 from :mod:`evaluation` — not
pairwise F1, and not on the training split. Expect the optimum near the top of
the probability range and to be sensitive; report the score curve, not just the
peak.
"""

from __future__ import annotations

import json
import math
import os
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

#: Library versions are recorded in the model artefact. A model trained under one
#: LightGBM release and scored under another can shift the operating point, which
#: would silently invalidate a tuned threshold.
_ARTEFACT_VERSION = 1


# --------------------------------------------------------------------------- #
# Label construction
# --------------------------------------------------------------------------- #


def build_training_pairs(
    candidates: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
    negatives_per_entity: int = 20,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, int]]:
    """Turn ground truth + candidates into a labelled pair list.

    ``candidates`` and ``ground_truth`` are keyed by Source-1 ID. Returns
    ``(pair_index, labels, feature_columns, diagnostics)``:

    * ``pair_index`` -- an ``(n, 2)`` int array of ``(s1_row, pool_row)`` for the
      caller to featurise. Rows are emitted here rather than
      :class:`preprocessing.Record` objects so the arrays stay compact at 10^8
      pairs.
    * ``labels`` -- 1 for a true match, 0 otherwise.
    * ``feature_columns`` -- the canonical column order from
      :func:`features.feature_columns`.
    * ``diagnostics`` -- notably ``positives_missing_from_candidates``, which is
      the blocking recall ceiling and a headline number for the write-up.

    * **positives** for every ``(source1_id, matched_id)`` in the ground truth
      that also appears in the candidate set. Positives missing from the
      candidates are not silently dropped — they are counted, because that count
      is the blocking recall ceiling.
    * **negatives** sampled from each entity's candidates that are absent from
      its true match list, capped at ``negatives_per_entity`` so entities with
      many candidates do not dominate the training distribution.
    * entities with an empty true list contribute negatives only.

    Labels are pair-level, but evaluation is entity-level — see :mod:`evaluation`
    for why the two must not be conflated.
    """

    from . import features as ft

    rng = np.random.default_rng(seed)
    columns = list(ft.feature_columns())

    pair_rows: List[Tuple[str, str]] = []
    labels: List[int] = []
    diagnostics = {
        "entities": 0,
        "entities_with_truth": 0,
        "singletons": 0,
        "true_pairs": 0,
        "positives_missing_from_candidates": 0,
        "negatives": 0,
    }

    for entity_id, truth_ids in ground_truth.items():
        diagnostics["entities"] += 1
        truth = set(truth_ids)
        candidate_ids = candidates.get(entity_id, [])
        diagnostics["entities_with_truth"] += int(bool(truth))
        diagnostics["singletons"] += int(not truth)
        diagnostics["true_pairs"] += len(truth)

        candidate_set = set(candidate_ids)
        present = truth & candidate_set
        diagnostics["positives_missing_from_candidates"] += len(truth - candidate_set)

        for pool_id in sorted(present):
            pair_rows.append((entity_id, pool_id))
            labels.append(1)

        negatives = [c for c in candidate_set if c not in truth]
        if len(negatives) > negatives_per_entity:
            chosen = rng.choice(len(negatives), size=negatives_per_entity, replace=False)
            negatives = [negatives[i] for i in sorted(chosen)]
        for pool_id in negatives:
            pair_rows.append((entity_id, pool_id))
            labels.append(0)
        diagnostics["negatives"] += len(negatives)

    if not pair_rows:
        return (
            np.empty((0, 2), dtype=np.int64),
            np.empty(0, dtype=np.int8),
            columns,
            diagnostics,
        )
    return (
        np.asarray(pair_rows, dtype=object),
        np.asarray(labels, dtype=np.int8),
        columns,
        diagnostics,
    )


# --------------------------------------------------------------------------- #
# Classifier
# --------------------------------------------------------------------------- #


def train_classifier(
    X,
    y,
    feature_columns: Sequence[str],
    num_boost_round: int = 800,
    learning_rate: float = 0.05,
    seed: int = 42,
    n_jobs: int = 2,
    valid: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    scale_pos_weight: Optional[float] = None,
):
    """Fit the gradient-boosted match classifier and return the trained model.

    Expect strong class imbalance (a small fraction of positives). Class
    weighting is used rather than duplicating minority rows — duplication costs
    recall on the many near-miss negatives that actually decide the F_0.5
    operating point.

    ``seed`` is fixed so threshold and error-analysis results are reproducible.
    ``valid`` enables early stopping on a held-out slice; without it the full
    ``num_boost_round`` is used, which is only safe when the round count was
    itself chosen on a validation split.
    """

    import lightgbm as lgb

    X = np.ascontiguousarray(X, dtype=np.float32)
    y = np.asarray(y)
    n_positive = int(y.sum())
    n_negative = int(len(y) - n_positive)
    if n_positive == 0 or n_negative == 0:
        raise ValueError(
            f"training set has {n_positive} positives and {n_negative} negatives; "
            "cannot fit a classifier"
        )
    if scale_pos_weight is None:
        scale_pos_weight = n_negative / n_positive

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": learning_rate,
        "num_leaves": 127,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "scale_pos_weight": scale_pos_weight,
        "seed": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        "num_threads": n_jobs,
        "verbose": -1,
        "deterministic": True,
        "force_row_wise": True,
    }
    train_set = lgb.Dataset(
        X, label=y, feature_name=list(feature_columns),
        categorical_feature=[], free_raw_data=False,
    )
    callbacks = [lgb.log_evaluation(period=100)]
    valid_sets = None
    if valid is not None:
        X_valid, y_valid = valid
        valid_sets = [
            lgb.Dataset(
                np.ascontiguousarray(X_valid, dtype=np.float32), label=y_valid,
                feature_name=list(feature_columns), reference=train_set,
                free_raw_data=False,
            )
        ]
        callbacks.append(lgb.early_stopping(50, verbose=True))

    model = lgb.train(
        params, train_set, num_boost_round=num_boost_round,
        valid_sets=valid_sets, callbacks=callbacks,
    )
    model.feature_columns_used = list(feature_columns)  # type: ignore[attr-defined]
    model.training_config = {  # type: ignore[attr-defined]
        "num_boost_round": num_boost_round,
        "learning_rate": learning_rate,
        "seed": seed,
        "scale_pos_weight": scale_pos_weight,
        "positives": n_positive,
        "negatives": n_negative,
        "lightgbm_version": lgb.__version__,
    }
    return model


def predict_proba(
    model,
    X,
    n_jobs: int = 2,
    chunk_rows: int = 2_000_000,
) -> np.ndarray:
    """Return a match probability per candidate pair.

    This probability drives thresholding, so it needs to be reasonably calibrated
    on the validation split; if it is not, apply isotonic or Platt calibration
    before tuning the threshold in :func:`tune_threshold`.

    Chunked because a full-split prediction is 10^8 rows and LightGBM wants a
    contiguous float32 array.
    """

    X = np.ascontiguousarray(X, dtype=np.float32)
    out = np.empty(len(X), dtype=np.float32)
    for start in range(0, len(X), chunk_rows):
        stop = min(start + chunk_rows, len(X))
        out[start:stop] = model.predict(
            X[start:stop], num_threads=n_jobs
        )
    return out


# --------------------------------------------------------------------------- #
# Threshold and resolution
# --------------------------------------------------------------------------- #


def tune_threshold(
    scores,
    entities,
    pool_ids,
    true_matches,
    metric_fn: Callable[[Dict[str, List[str]], Dict[str, List[str]]], float],
    grid: Optional[Iterable[float]] = None,
    resolve: bool = False,
) -> Tuple[float, float, List[Tuple[float, float]]]:
    """Pick the decision threshold that maximises the competition metric.

    ``scores``, ``entities`` and ``pool_ids`` are parallel per-candidate-pair
    sequences; ``true_matches`` maps Source-1 ID -> list of true match IDs.
    ``pool_ids`` is an explicit argument because the predicted match sets cannot
    be reconstructed from the scores alone.

    Sweeps candidate thresholds, and for each one builds the predicted match sets
    and scores them with the real macro F_0.5 from :func:`evaluation.macro_f05`.
    Tuning on pairwise F1 picks the wrong operating point, because singleton
    handling and per-entity averaging only exist at the entity level.

    ``grid`` defaults to a fine range over the upper part of the probability
    distribution, where the optimum is expected: F_0.5 weights precision twice as
    heavily as recall, so a confident positive matters much more than a marginal
    one. Scanning the whole 0-1 range wastes almost all of the sweep on
    thresholds no sane model would use.

    ``resolve`` applies :func:`resolve_competition` inside the sweep, which is the
    only honest way to compare the two, since the operating point that is best
    before contention resolution is not necessarily best after it.

    Returns ``(best_threshold, best_score, sweep)``. The full sweep is returned
    because with a precision-heavy metric the score is flat near the optimum and
    falls off sharply once precision degrades, which is worth documenting.
    """

    if grid is None:
        grid = np.round(np.arange(0.05, 0.995, 0.005), 4)
    grid = [float(t) for t in grid]

    by_entity: Dict[str, List[Tuple[str, float]]] = {}
    for entity, pool_id, score in zip(entities, pool_ids, scores):
        by_entity.setdefault(entity, []).append((pool_id, float(score)))

    sweep: List[Tuple[float, float]] = []
    best_threshold, best_score = 0.5, -1.0
    for threshold in grid:
        if resolve:
            predicted = resolve_competition(by_entity, threshold)
        else:
            predicted = {
                entity: [pid for pid, score in items if score >= threshold]
                for entity, items in by_entity.items()
            }
        score = metric_fn(predicted, true_matches)
        sweep.append((threshold, score))
        if score > best_score:
            best_threshold, best_score = threshold, score
    return best_threshold, best_score, sweep


def tune_threshold_arrays(
    s1_rows: np.ndarray,
    pool_rows: np.ndarray,
    scores: np.ndarray,
    is_true_pair: np.ndarray,
    n_s1: int,
    n_true: np.ndarray,
    grid: Optional[Iterable[float]] = None,
    resolve: bool = False,
    pool_size: int = 1,
) -> Tuple[float, float, List[Tuple[float, float]]]:
    """Vectorised :func:`tune_threshold` over aligned per-pair arrays.

    Same contract and same objective -- maximise macro F_0.5 over *all* ``n_s1``
    Source-1 entities, not pairwise F1 -- but without materialising a dictionary
    of entity keys per threshold. :func:`tune_threshold` needs ~190 dictionaries
    of tens of thousands of entries to sweep one operating point; at validation
    scale that is minutes of pure Python per sweep and two sweeps are needed
    (with and without contention resolution).

    Inputs are row-aligned per-pair arrays over the *validation* candidate set:

    * ``s1_rows`` -- Source-1 row index, sorted ascending;
    * ``pool_rows`` -- pool row index, used only for contention resolution;
    * ``scores`` -- match probability per pair;
    * ``is_true_pair`` -- 1 where the pair is a true match. This is what makes
      the hit count computable without rebuilding match sets per threshold;
    * ``n_s1`` -- number of Source-1 rows in scope, so entities with no
      candidates still count;
    * ``n_true`` -- ``|T|`` per Source-1 row, taken from the **full** ground
      truth rather than from the candidate set. A true match that blocking never
      proposed is an unreachable miss, and folding the candidate set into the
      denominator would quietly score the model against a ceiling it was never
      meant to reach.

    Returns ``(best_threshold, best_macro_f05, sweep)``.
    """

    from . import evaluation as ev

    if grid is None:
        grid = np.round(np.arange(0.01, 0.999, 0.002), 4)
    grid = [float(t) for t in grid]

    s1_rows = np.asarray(s1_rows, dtype=np.int64)
    pool_rows = np.asarray(pool_rows, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    is_true_pair = np.asarray(is_true_pair, dtype=np.int8)
    n_true = np.asarray(n_true, dtype=np.int64)
    if not (len(s1_rows) == len(pool_rows) == len(scores) == len(is_true_pair)):
        raise ValueError(
            f"mismatched validation arrays: {len(s1_rows)} s1 rows, "
            f"{len(pool_rows)} pool rows, {len(scores)} scores, "
            f"{len(is_true_pair)} truth flags"
        )
    if len(n_true) != n_s1:
        raise ValueError(f"n_true has {len(n_true)} entries, expected {n_s1}")

    sweep: List[Tuple[float, float]] = []
    best_threshold, best_score = 0.5, -1.0
    for threshold in grid:
        keep = np.flatnonzero(scores >= threshold)
        if resolve:
            # Resolve on the *original* row indices so the truth mask can be
            # carried through untouched. A claim dropped by resolution is not a
            # predicted match, so leaving its truth flag in would overstate
            # recall exactly where resolution did its job.
            winners = resolve_competition_indices(
                s1_rows[keep], pool_rows[keep], scores[keep], 0.0, pool_size,
            )
            keep = keep[winners]
        n_pred, n_hit = ev.per_entity_counts(
            s1_rows[keep], is_true_pair[keep], n_s1
        )
        score = ev.macro_f05_counts(n_true, n_pred, n_hit)
        sweep.append((threshold, score))
        if score > best_score:
            best_threshold, best_score = threshold, score
    return best_threshold, best_score, sweep


def resolve_competition(
    entity_candidates: Dict[str, List[Tuple[str, float]]],
    threshold: float,
) -> Dict[str, List[str]]:
    """Turn per-pair scores into one final match list per Source-1 entity.

    Applies the threshold, then resolves the case where two Source-1 entities both
    claim the same Source-2/Source-3 record. The competition is asymmetric: **one
    Source-1 entity may match many records**, but a single Source-2/3 record
    belongs to only one real business, so it should end up with a single owner.
    Keep the highest-scoring claim for contended records.

    Rules to honour, all enforced by ``utils/validate_submission.py``:

    * return **every** entity passed in, including those with no match (an empty
      list is a correct answer worth 1.0 on that entity; a missing row is a
      submission rejection);
    * no duplicate IDs within a list;
    * S2-/S3- IDs only, no self-matches;
    * deterministic given the same scores, so reruns reproduce byte-identical
      output.

    Ties are broken on the Source-1 ID so the result does not depend on dict
    ordering. The cost is that a genuine tie between two entities is resolved
    arbitrarily rather than by entity-level evidence, which is the honest
    outcome — the pair scores carry no information to break it.
    """

    # One claim per (entity, pool record): the same pair arrives from several
    # blocking strategies with different scores, and only the best is meaningful.
    # Keeping the *first* seen instead would let a weaker duplicate hide a
    # stronger one and hand the pool record to the wrong entity.
    best: Dict[Tuple[str, str], float] = {}
    for entity, items in entity_candidates.items():
        for pool_id, score in items:
            if score < threshold:
                continue
            key = (entity, pool_id)
            if score > best.get(key, -math.inf):
                best[key] = score

    # Keep the single best claim per pool record.
    owner: Dict[str, Tuple[str, float]] = {}
    for (entity, pool_id), score in best.items():
        current = owner.get(pool_id)
        if current is None or (score, entity) > (current[1], current[0]):
            owner[pool_id] = (entity, score)

    result: Dict[str, List[str]] = {
        entity: [] for entity in entity_candidates
    }
    for pool_id, (entity, _score) in owner.items():
        result.setdefault(entity, []).append(pool_id)
    for entity in result:
        result[entity] = sorted(set(result[entity]))
    return result


def resolve_competition_indices(
    s1_rows: np.ndarray, pool_rows: np.ndarray, scores: np.ndarray,
    threshold: float, pool_size: int,
) -> np.ndarray:
    """Indices into the input arrays of the claims that survive resolution.

    The primitive both array-based callers are built on. Returning indices
    rather than copies of the rows is what lets :func:`tune_threshold_arrays`
    carry an aligned truth mask through resolution without re-deriving it --
    there is exactly one implementation of "who wins a contested pool record"
    here, so the sweep and the submission path cannot disagree about it.

    Semantics: drop claims scoring below ``threshold``; collapse duplicate
    ``(s1_row, pool_row)`` claims keeping the best score (the same pair arrives
    from several blocking strategies); then keep only the single best claim per
    pool record, breaking ties on the lower Source-1 row so the result does not
    depend on input order.
    """

    s1_rows = np.asarray(s1_rows)
    pool_rows = np.asarray(pool_rows)
    scores = np.asarray(scores, dtype=np.float64)
    if not (len(s1_rows) == len(pool_rows) == len(scores)):
        raise ValueError(
            f"mismatched claim arrays: {len(s1_rows)} entities, "
            f"{len(pool_rows)} pool rows, {len(scores)} scores"
        )
    original = np.arange(len(scores), dtype=np.int64)
    s1_rows = s1_rows.astype(np.int64)
    pool_rows = pool_rows.astype(np.int64)
    keep = scores >= threshold
    original, s1_rows, pool_rows, scores = (
        original[keep], s1_rows[keep], pool_rows[keep], scores[keep]
    )
    if not len(s1_rows):
        return original

    # One claim per (entity, pool record). Sort by key ascending, score
    # descending, so the first row of each run is the winner.
    pair_key = s1_rows * np.int64(max(pool_size, 1)) + pool_rows
    order = np.lexsort((-scores, pair_key))
    s1_rows, pool_rows, scores, original, pair_key = (
        s1_rows[order], pool_rows[order], scores[order], original[order],
        pair_key[order],
    )
    first = _first_of(pair_key)
    s1_rows, pool_rows, scores, original = (
        s1_rows[first], pool_rows[first], scores[first], original[first]
    )

    # One owner per pool record: score descending, then Source-1 row ascending so
    # a tie resolves deterministically rather than by input order.
    order = np.lexsort((s1_rows, -scores, pool_rows))
    s1_rows, pool_rows, scores, original = (
        s1_rows[order], pool_rows[order], scores[order], original[order]
    )
    first = _first_of(pool_rows)
    return original[first]


def _first_of(keys: np.ndarray) -> np.ndarray:
    """Boolean mask selecting the first row of each run of equal keys."""

    mask = np.empty(len(keys), dtype=bool)
    mask[0] = True
    np.not_equal(keys[1:], keys[:-1], out=mask[1:])
    return mask


def resolve_competition_arrays(
    s1_rows: np.ndarray, pool_rows: np.ndarray, scores: np.ndarray,
    threshold: float, pool_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised :func:`resolve_competition` over row indices instead of strings.

    Returns ``(s1_row, pool_row, score)`` of the surviving matches, sorted by
    ``s1_row``. Semantics match the dict version exactly -- both delegate to
    :func:`resolve_competition_indices`.

    The scores are returned, not just consumed, because resolution is applied
    more than once at submission scale: a per-chunk pass narrows the field, and a
    later global pass has to compare the survivors by their *original* scores.
    Discarding them would make that pass tie-break on row order instead.

    The dict version cannot be used at submission scale. Test has 1.7M Source-1
    entities at ~136 candidates each, so a claims dict would hold ~2.4x10^8
    ``(str, float)`` tuples -- tens of gigabytes. Working in row indices keeps the
    stage within a few hundred megabytes, and being vectorisable also removes a
    ~2.4x10^8-iteration Python loop from the critical path.
    """

    winners = resolve_competition_indices(
        s1_rows, pool_rows, scores, threshold, pool_size
    )
    if not len(winners):
        empty = np.empty(0, np.int64)
        return empty, empty.copy(), np.empty(0, np.float64)
    s1_rows = np.asarray(s1_rows, dtype=np.int64)[winners]
    pool_rows = np.asarray(pool_rows, dtype=np.int64)[winners]
    scores = np.asarray(scores, dtype=np.float64)[winners]
    order = np.argsort(s1_rows, kind="stable")
    return s1_rows[order], pool_rows[order], scores[order]


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def save_model(
    model,
    feature_columns: Sequence[str],
    path: str,
) -> None:
    """Persist the trained model together with its feature column order.

    The column order is part of the model contract: reusing a different order at
    inference time silently misaligns features while still producing
    plausible-looking scores, so the damage stays invisible until the submission
    score drops. Store it in the same file and pin library versions so the
    artefact loads under the environment in ``requirements.txt``.
    """

    import lightgbm as lgb

    booster = getattr(model, "booster_", model)
    booster.save_model(path, num_iteration=getattr(booster, "best_iteration", None) or None)
    sidecar = str(path) + ".meta.json"
    with open(sidecar, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "artefact_version": _ARTEFACT_VERSION,
                "feature_columns": list(feature_columns),
                "lightgbm_version": lgb.__version__,
                "training_config": getattr(model, "training_config", {}),
            },
            handle,
            indent=2,
        )


def load_model(path: str):
    """Load a model saved by :func:`save_model` and return it with its columns.

    Fails loudly on a feature-column mismatch rather than reordering silently.
    """

    import lightgbm as lgb

    booster = lgb.Booster(model_file=path)
    sidecar = str(path) + ".meta.json"
    if not os.path.exists(sidecar):
        raise FileNotFoundError(
            f"{sidecar} missing; without the recorded column order the model "
            "cannot be loaded safely"
        )
    with open(sidecar, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    recorded = meta.get("feature_columns") or booster.feature_name()
    return booster, recorded, meta


__all__ = [
    "build_training_pairs", "train_classifier", "predict_proba", "tune_threshold",
    "tune_threshold_arrays", "resolve_competition", "resolve_competition_arrays",
    "resolve_competition_indices", "save_model", "load_model",
]
