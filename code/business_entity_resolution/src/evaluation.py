"""Scoring and diagnostics, reproducing the leaderboard metric exactly.

The metric
----------
F_beta with beta = 0.5, precision-heavy:

    F_0.5 = (1.25 * P * R) / (0.25 * P + R)

computed as a **macro average over Source-1 entities**: F_0.5 is calculated per
Source-1 entity, then averaged across *all* Source-1 entities in the evaluation
set. For one entity with true set ``T`` and predicted set ``H``:

* both empty -> 1.0  (correctly identifying a singleton is a full point)
* ``T`` empty, ``H`` non-empty -> 0.0  (a false merge on a singleton scores zero)
* ``H`` empty, ``T`` non-empty -> 0.0
* otherwise ``P = |T n H| / |H|``, ``R = |T n H| / |T|``

Worked example from the problem statement: truth ``{S2-00047, S3-00812}``,
prediction ``{S2-00047, S2-00193, S3-00812}`` gives ``P = 2/3``, ``R = 1.0``,
``F_0.5 = 0.714``.

Two consequences that shape the approach
----------------------------------------
* **Singleton accuracy is a first-class term.** Entities with no true match are
  worth the same as any other entity, so a model that never predicts an empty
  list forfeits those points *and* generates false merges. 5.58% of the
  training entities are singletons, so this is ~5.6 points of macro score.
* **The score is entity-level, not pair-level.** A model can look strong on
  pairwise F1 and lose badly here. Always tune and report the entity-level
  number.

Everything here runs against a validation split of the *training* data, since
the test set ships without labels. Use the entity-level split from
:func:`preprocessing.split_train_validation` — a pair-level split leaks, because
a Source-2 record that truly matches a validation Source-1 entity is often also
a true match of a training one.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

BETA = 0.5
_BETA_SQ = BETA * BETA


def f_beta(precision: float, recall: float, beta: float = BETA) -> float:
    """Return the F_beta score for one entity, with the degenerate cases handled.

    The four cases are explicit because a naive formula divides by zero or
    returns the wrong thing when a set is empty:

    * ``P + R == 0`` (no true matches and nothing predicted) -> **1.0**, the
      correct-singleton case a plain formula would score as 0 or NaN;
    * predicted empty, truth non-empty -> 0.0;
    * truth empty, prediction non-empty -> 0.0;
    * otherwise the standard weighted harmonic mean.
    """

    if precision + recall <= 0.0:
        # Both zero: either a correct singleton, or nothing at all. Only the
        # former is reachable once the caller has already excluded the
        # "empty prediction, non-empty truth" case, which is handled below.
        return 1.0
    denominator = _BETA_SQ * precision + recall
    if denominator <= 0.0:
        return 0.0
    return (1.0 + _BETA_SQ) * precision * recall / denominator


def entity_f05(true_set: Set[str], pred_set: Set[str], beta: float = BETA) -> float:
    """Return the F_beta score for a single entity from its two match sets."""

    beta_sq = beta * beta
    if not true_set:
        # A singleton: an empty prediction is a full point, any prediction zero.
        return 1.0 if not pred_set else 0.0
    if not pred_set:
        return 0.0
    hits = len(true_set & pred_set)
    if hits == 0:
        return 0.0
    precision = hits / len(pred_set)
    recall = hits / len(true_set)
    return (1.0 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def entity_f05_counts(
    n_true: np.ndarray, n_pred: np.ndarray, n_hit: np.ndarray,
    beta: float = BETA,
) -> np.ndarray:
    """Vectorised :func:`entity_f05` over per-entity counts.

    ``n_true[i]``, ``n_pred[i]`` and ``n_hit[i]`` are the sizes of entity ``i``'s
    true set, predicted set, and their intersection. Returns one F_beta per
    entity, in the same edge-case order as the scalar version:

    * both empty -> **1.0**, the correctly-abstained singleton. This is the case
      a naive closed form gets wrong: ``P`` and ``R`` are both ``0/0`` there, so
      the formula divides by zero and returns NaN, and an implementation that
      clamps NaN to 0 silently forfeits ~5.6 points of macro score.
    * exactly one side empty -> **0.0**.
    * no hits -> **0.0**.
    * otherwise the standard weighted harmonic mean.

    This exists because the threshold sweep needs the metric evaluated ~190
    times over ~10^7 pairs. Doing that through :func:`macro_f05` means building
    190 dictionaries of 40,000 entity keys; counting into three arrays first
    turns each evaluation into a handful of ``np.add.reduceat`` calls.
    :func:`tests.test_evaluation` asserts the two agree exactly on randomised
    inputs, so the vectorised form cannot drift from the definition.
    """

    n_true = np.asarray(n_true, dtype=np.float64)
    n_pred = np.asarray(n_pred, dtype=np.float64)
    n_hit = np.asarray(n_hit, dtype=np.float64)
    out = np.zeros(len(n_true), dtype=np.float64)

    both_empty = (n_true == 0) & (n_pred == 0)
    out[both_empty] = 1.0

    scored = (n_true > 0) & (n_pred > 0) & (n_hit > 0)
    if scored.any():
        precision = n_hit[scored] / n_pred[scored]
        recall = n_hit[scored] / n_true[scored]
        denominator = beta * beta * precision + recall
        # denominator > 0 by construction (recall > 0 here), so no guard needed.
        out[scored] = (1.0 + beta * beta) * precision * recall / denominator
    return out


def macro_f05_counts(
    n_true: np.ndarray, n_pred: np.ndarray, n_hit: np.ndarray,
) -> float:
    """Macro F_0.5 over per-entity counts; see :func:`entity_f05_counts`."""

    if not len(n_true):
        return 0.0
    return float(entity_f05_counts(n_true, n_pred, n_hit).mean())


def per_entity_counts(
    s1_rows: np.ndarray, is_true_pair: np.ndarray, n_s1: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce aligned per-pair arrays to ``(n_pred, n_hit)`` per Source-1 row.

    ``s1_rows`` must be sorted ascending, which is how the candidate pair array
    is built. ``n_s1`` rows are always counted, so an entity with no candidate
    pairs reports ``(0, 0)`` and is scored as a correct abstention rather than
    being dropped from the average.
    """

    n_pred = np.bincount(s1_rows, minlength=n_s1).astype(np.int64)
    if len(is_true_pair):
        n_hit = np.bincount(
            s1_rows, weights=is_true_pair.astype(np.float64), minlength=n_s1
        ).astype(np.int64)
    else:
        n_hit = np.zeros(n_s1, dtype=np.int64)
    return n_pred, n_hit


def _entity_keys(
    predictions: Dict[str, List[str]], ground_truth: Dict[str, List[str]]
) -> List[str]:
    """Return the union of entity keys, each appearing exactly once.

    The union is what :func:`macro_f05` averages over. Restricting it to the
    intersection would silently reward a model for abstaining on the hard cases,
    which is exactly the behaviour the metric exists to penalise.
    """

    keys = set(predictions) | set(ground_truth)
    return sorted(keys)


def macro_f05(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
) -> float:
    """Return the macro-averaged F_0.5 over every Source-1 entity."""

    keys = _entity_keys(predictions, ground_truth)
    if not keys:
        return 0.0
    total = 0.0
    for key in keys:
        total += entity_f05(set(ground_truth.get(key, ())), set(predictions.get(key, ())))
    return total / len(keys)


def per_entity_scores(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
) -> Dict[str, float]:
    """Return the per-entity F_0.5, for slicing the macro score by subset."""

    return {
        key: entity_f05(set(ground_truth.get(key, ())), set(predictions.get(key, ())))
        for key in _entity_keys(predictions, ground_truth)
    }


def subset_macro_f05(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
    subset: Iterable[str],
) -> float:
    """Macro F_0.5 restricted to ``subset`` of entity IDs."""

    scores = [
        entity_f05(set(ground_truth.get(k, ())), set(predictions.get(k, ())))
        for k in subset
    ]
    return sum(scores) / len(scores) if scores else 0.0


def blocking_recall_ceiling(
    candidates: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
) -> float:
    """Fraction of true matches the blocking stage actually proposed.

    The hard upper bound on recall for the whole system: no model can recover a
    true match that was never proposed. Report it alongside the realised score —
    if the realised F_0.5 sits well below the ceiling, the model is the
    bottleneck; if the ceiling is itself too low, the fix is in :mod:`blocking`.
    """

    found = 0
    total = 0
    for entity_id, true_matches in ground_truth.items():
        if not true_matches:
            continue
        proposed = set(candidates.get(entity_id, ()))
        found += len(proposed & set(true_matches))
        total += len(true_matches)
    return found / total if total else 0.0


def blocking_recall_detail(
    candidates: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
) -> Dict[str, float]:
    """Break the blocking ceiling down by entity type, for the write-up.

    Separates entities that lost *some* matches from entities that lost *all* of
    them, and reports the singleton-safe share of the entity population. The
    second number matters: an entity with no candidates can never score above
    zero, so a high share of empty candidate lists caps the achievable macro
    score regardless of how good the model is.
    """

    total_pairs = 0
    found_pairs = 0
    entities_with_matches = 0
    entities_fully_covered = 0
    entities_no_candidates = 0
    entities_empty_truth = 0

    for entity_id, true_matches in ground_truth.items():
        proposed = set(candidates.get(entity_id, ()))
        if not proposed:
            entities_no_candidates += 1
        if not true_matches:
            entities_empty_truth += 1
            continue
        entities_with_matches += 1
        truth = set(true_matches)
        hits = len(truth & proposed)
        found_pairs += hits
        total_pairs += len(truth)
        if hits == len(truth):
            entities_fully_covered += 1

    n_entities = len(ground_truth)
    return {
        "pair_recall": found_pairs / total_pairs if total_pairs else 0.0,
        "pair_recall_all_pairs": found_pairs,
        "true_pairs": total_pairs,
        "entity_full_recall": (
            entities_fully_covered / entities_with_matches if entities_with_matches else 0.0
        ),
        "entities_with_matches": entities_with_matches,
        "singleton_entities": entities_empty_truth,
        "entities_without_candidates": entities_no_candidates,
        "entities_without_candidates_share": (
            entities_no_candidates / n_entities if n_entities else 0.0
        ),
    }


def threshold_sweep(
    scores: Dict[str, List[Tuple[str, float]]],
    ground_truth: Dict[str, List[str]],
    grid: Iterable[float],
    resolve: Optional[Callable[[Dict[str, List[Tuple[str, float]]], float], Dict[str, List[str]]]] = None,
) -> List[Tuple[float, float]]:
    """Return ``(threshold, macro_f05)`` for each threshold on the grid.

    Feeds :func:`model.tune_threshold`. With a precision-heavy metric the curve is
    usually flat near its peak and then drops sharply, so plot the whole curve in
    the documentation rather than a single optimum.
    """

    if resolve is None:
        from .model import resolve_competition

        resolve = resolve_competition
    out: List[Tuple[float, float]] = []
    for threshold in grid:
        predictions = resolve(scores, threshold)
        out.append((float(threshold), macro_f05(predictions, ground_truth)))
    return out


def count_matched_against_truth(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
) -> Dict[str, float]:
    """Pair-level precision/recall summary, for diagnosing where F_0.5 comes from.

    Not the competition metric — reported only alongside it, because a model can
    have good pair-level recall and still lose macro F_0.5 to singleton errors.
    """

    true_positive = false_positive = false_negative = 0
    for entity_id, truth_list in ground_truth.items():
        truth = set(truth_list)
        pred = set(predictions.get(entity_id, ()))
        true_positive += len(truth & pred)
        false_positive += len(pred - truth)
        false_negative += len(truth - pred)
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative else 0.0
    )
    return {
        "pair_precision": precision,
        "pair_recall": recall,
        "true_positive": float(true_positive),
        "false_positive": float(false_positive),
        "false_negative": float(false_negative),
    }


def error_analysis(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
    records: Optional[Dict[str, object]] = None,
    top_n: int = 20,
) -> Dict[str, List[Tuple[str, str, str]]]:
    """Return the worst-scoring entities with the records behind them.

    Separate the two error directions, since they have different fixes:

    * **false merges** (predicted but not true) — a precision problem, and the
      more damaging one under F_0.5. Usually generic names, landmark-only address
      agreement, or duplicates in the source data.
    * **missed matches** (true but not predicted) — a recall problem. Usually
      severe transliteration drift, a missing or contradictory address, or a true
      match blocking never proposed.

    With ``records`` supplied, return the raw name/address of both sides for each
    case. The pattern across examples is what drives the next round of feature
    work, so keep the raw strings, not just the scores.
    """

    false_merges: List[Tuple[float, str, List[str]]] = []
    missed: List[Tuple[float, str, List[str]]] = []
    singleton_errors: List[Tuple[str, List[str]]] = []

    for entity_id in _entity_keys(predictions, ground_truth):
        truth = set(ground_truth.get(entity_id, ()))
        pred = set(predictions.get(entity_id, ()))
        extra = pred - truth
        missing = truth - pred
        if extra:
            false_merges.append((len(extra) / max(len(pred), 1), entity_id, sorted(extra)))
        if missing:
            missed.append((len(missing) / max(len(truth), 1), entity_id, sorted(missing)))
        if not truth and pred:
            singleton_errors.append((entity_id, sorted(pred)))

    false_merges.sort(key=lambda x: (-x[0], x[1]))
    missed.sort(key=lambda x: (-x[0], x[1]))
    singleton_errors.sort()

    def describe(entity_id: str, ids: Sequence[str]) -> Tuple[str, str, str]:
        if records is None:
            return entity_id, ",".join(ids), ""
        left = records.get(entity_id)
        right = [
            records.get(i) for i in ids[:3]
        ]
        left_text = _record_text(left)
        right_text = " || ".join(_record_text(r) for r in right)
        return entity_id, left_text, right_text

    return {
        "false_merges": [describe(eid, ids) for _score, eid, ids in false_merges[:top_n]],
        "missed_matches": [describe(eid, ids) for _score, eid, ids in missed[:top_n]],
        "singleton_false_merges": [
            describe(eid, ids) for eid, ids in singleton_errors[:top_n]
        ],
    }


def _record_text(record: object) -> str:
    """Render a record's identifying fields for the error report."""

    if record is None:
        return "<no record>"
    name = getattr(record, "name_raw", None) or getattr(record, "name_norm", "")
    address = getattr(record, "address_raw", None) or getattr(record, "address_norm", "")
    country = getattr(record, "country", "") or getattr(record, "country_norm", "")
    return f"{name!r} | {address!r} | {country}"


__all__ = [
    "BETA", "f_beta", "entity_f05", "macro_f05", "per_entity_scores",
    "subset_macro_f05", "blocking_recall_ceiling", "blocking_recall_detail",
    "threshold_sweep", "error_analysis", "count_matched_against_truth",
    "entity_f05_counts", "macro_f05_counts", "per_entity_counts",
]
