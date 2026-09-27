"""Parity and correctness tests for the streaming/vectorised additions.

Two claims are load-bearing for the rest of the pipeline and neither is
self-evident from reading the code, so both are tested against the *existing*
reference implementation rather than against hand-written expectations:

1. :func:`evaluation.macro_f05_counts` agrees exactly with
   :func:`evaluation.macro_f05`, the dict-based original that states the metric
   as the problem statement does. If the vectorised form drifts, every threshold
   in the sweep is tuned against a different objective than the one that is
   reported.
2. :func:`model.resolve_competition_arrays` agrees with
   :func:`model.resolve_competition`, the dict-based original. The threshold
   sweep and the submission writer both go through the array path, so it has to
   mean the same thing as the readable one.

Also covers the streaming block planner, because a block that straddles an
entity boundary silently doubles the per-record cache cost, and a planner that
drops a row silently loses training pairs.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import evaluation as ev  # noqa: E402
from src import model as md  # noqa: E402
from src import streaming as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Metric parity
# --------------------------------------------------------------------------- #


def test_counts_metric_matches_dict_metric_on_randomised_truth():
    """The vectorised metric must equal the reference, singleton case included."""

    rng = random.Random(20260927)
    for _ in range(60):
        n = rng.randint(1, 40)
        predictions: dict[str, list[str]] = {}
        ground_truth: dict[str, list[str]] = {}
        n_true, n_pred, n_hit = [], [], []
        for i in range(n):
            key = f"S1-{i:05d}"
            truth = {f"S2-{rng.randrange(50):05d}" for _ in range(rng.randint(0, 4))}
            # Overlap the prediction with the truth sometimes and not others, so
            # the four edge cases (both empty / either empty / disjoint) all
            # actually occur.
            pred = {f"S2-{rng.randrange(50):05d}" for _ in range(rng.randint(0, 4))}
            predictions[key] = sorted(pred)
            ground_truth[key] = sorted(truth)
            n_true.append(len(truth))
            n_pred.append(len(pred))
            n_hit.append(len(truth & pred))
        expected = ev.macro_f05(predictions, ground_truth)
        actual = ev.macro_f05_counts(
            np.array(n_true), np.array(n_pred), np.array(n_hit)
        )
        assert actual == pytest.approx(expected, abs=1e-12), (n_true, n_pred, n_hit)


def test_correctly_empty_singleton_scores_one():
    """The 0/0 case is the one a closed-form F_beta gets wrong."""

    assert ev.entity_f05_counts(
        np.array([0]), np.array([0]), np.array([0])
    )[0] == 1.0
    assert ev.entity_f05(set(), set()) == 1.0


def test_singleton_with_any_prediction_scores_zero():
    assert ev.entity_f05_counts(
        np.array([0]), np.array([1]), np.array([0])
    )[0] == 0.0
    assert ev.entity_f05(set(), {"S2-1"}) == 0.0


def test_matched_entity_with_empty_prediction_scores_zero():
    assert ev.entity_f05_counts(
        np.array([2]), np.array([0]), np.array([0])
    )[0] == 0.0


def test_f05_matches_the_worked_example_from_the_problem_statement():
    # truth {S2-00047, S3-00812}, prediction {S2-00047, S2-00193, S3-00812}
    # -> P = 2/3, R = 1, F_0.5 = 0.714
    truth = {"S2-00047", "S3-00812"}
    pred = {"S2-00047", "S2-00193", "S3-00812"}
    assert ev.entity_f05(truth, pred) == pytest.approx(0.714, abs=0.001)
    assert ev.entity_f05_counts(
        np.array([2]), np.array([3]), np.array([2])
    )[0] == pytest.approx(0.714, abs=0.001)


def test_per_entity_counts_includes_entities_with_no_pairs():
    """An entity absent from the candidate set must still be scored, as 1.0 if
    it is a true singleton -- dropping it would inflate the macro average."""

    s1_rows = np.array([0, 0, 2], dtype=np.int64)
    is_true = np.array([1, 0, 1], dtype=np.int8)
    n_pred, n_hit = ev.per_entity_counts(s1_rows, is_true, n_s1=4)
    assert n_pred.tolist() == [2, 0, 1, 0]
    assert n_hit.tolist() == [1, 0, 1, 0]
    # Entity 1 has no candidates; if it is a singleton, that is a full point.
    score = ev.macro_f05_counts(
        np.array([2, 0, 1, 0]), n_pred, n_hit
    )
    assert score == pytest.approx((0.5 + 1.0 + 1.0 + 1.0) / 4)


# --------------------------------------------------------------------------- #
# Contention-resolution parity
# --------------------------------------------------------------------------- #


def test_array_resolution_matches_dict_resolution():
    rng = random.Random(4242)
    for _ in range(40):
        n_entities = rng.randint(1, 12)
        n_pool = rng.randint(1, 10)
        pool_ids = [f"S2-{i:05d}" for i in range(n_pool)]
        entity_candidates: dict[str, list[tuple[str, float]]] = {}
        s1_rows, pool_rows, scores = [], [], []
        for e in range(n_entities):
            entity = f"S1-{e:05d}"
            items = []
            for _ in range(rng.randint(0, 6)):
                pid = pool_ids[rng.randrange(n_pool)]
                score = round(rng.random(), 3)
                items.append((pid, score))
                s1_rows.append(e)
                pool_rows.append(int(pid.split("-")[1]))
                scores.append(score)
            entity_candidates[entity] = items
        expected = md.resolve_competition(entity_candidates, 0.5)
        got_s1, got_pool, _got_score = md.resolve_competition_arrays(
            np.array(s1_rows, dtype=np.int64),
            np.array(pool_rows, dtype=np.int64),
            np.array(scores, dtype=np.float64),
            0.5, n_pool,
        )
        actual: dict[str, list[str]] = {f"S1-{e:05d}": [] for e in range(n_entities)}
        for s1, pool in zip(got_s1, got_pool):
            actual[f"S1-{s1:05d}"].append(pool_ids[int(pool)])
        for key in actual:
            actual[key] = sorted(set(actual[key]))
        assert actual == expected


def test_resolution_keeps_one_owner_per_pool_record():
    got_s1, got_pool, _ = md.resolve_competition_arrays(
        np.array([0, 1, 1], dtype=np.int64),
        np.array([7, 7, 9], dtype=np.int64),
        np.array([0.9, 0.8, 0.7]),
        0.5, 100,
    )
    # Record 7 is claimed by both entities; the higher score (0.9) keeps it.
    assert sorted(zip(got_s1.tolist(), got_pool.tolist())) == [(0, 7), (1, 9)]


def test_resolution_collapses_duplicate_pairs_from_multiple_blocking_keys():
    got_s1, got_pool, got_score = md.resolve_competition_arrays(
        np.array([0, 0], dtype=np.int64),
        np.array([3, 3], dtype=np.int64),
        np.array([0.4, 0.95]),
        0.0, 100,
    )
    # Same (entity, pool) pair from two blocking strategies: the best score wins
    # rather than whichever arrived first.
    assert got_s1.tolist() == [0] and got_pool.tolist() == [3]
    assert got_score.tolist() == [pytest.approx(0.95)]


def test_resolution_indices_index_into_the_original_arrays():
    s1 = np.array([0, 0, 1], dtype=np.int64)
    pool = np.array([2, 2, 2], dtype=np.int64)
    scores = np.array([0.9, 0.2, 0.7])
    winners = md.resolve_competition_indices(s1, pool, scores, 0.0, 10)
    # Record 2 is claimed by entity 0 twice and by entity 1 once. The duplicate
    # pair collapses to its best score, then the record itself goes to the
    # highest-scoring claimant, which is entity 0's 0.9 at original index 0.
    assert winners.tolist() == [0]


# --------------------------------------------------------------------------- #
# Threshold sweep
# --------------------------------------------------------------------------- #


def test_threshold_sweep_agrees_with_the_dict_sweep():
    """The sweep is the thing that picks the operating point; if the fast path
    disagrees with the reference sweep it is tuning the wrong objective."""

    rng = random.Random(7)
    n_s1, n_pool = 30, 40
    s1_rows, pool_rows, scores, is_true = [], [], [], []
    by_entity: dict[str, list[tuple[str, float]]] = {}
    truth: dict[str, list[str]] = {}
    for e in range(n_s1):
        entity = f"S1-{e:05d}"
        truth_ids = {f"S2-{rng.randrange(n_pool):05d}" for _ in range(rng.randint(0, 3))}
        truth[entity] = sorted(truth_ids)
        items = []
        for _ in range(rng.randint(0, 8)):
            pid = rng.randrange(n_pool)
            score = round(rng.random(), 3)
            is_true.append(int(f"S2-{pid:05d}" in truth_ids))
            s1_rows.append(e)
            pool_rows.append(pid)
            scores.append(score)
            items.append((f"S2-{pid:05d}", score))
        by_entity[entity] = items
    s1_rows = np.array(s1_rows, dtype=np.int64)
    pool_rows = np.array(pool_rows, dtype=np.int64)
    scores = np.array(scores, dtype=np.float64)
    is_true = np.array(is_true, dtype=np.int8)
    n_true = np.array([len(truth[f"S1-{e:05d}"]) for e in range(n_s1)], dtype=np.int64)

    grid = [0.2, 0.5, 0.8]
    for resolve in (False, True):
        fast_t, fast_score, _ = md.tune_threshold_arrays(
            s1_rows, pool_rows, scores, is_true, n_s1, n_true, grid=grid,
            resolve=resolve, pool_size=n_pool,
        )
        slow_t, slow_score, _ = md.tune_threshold(
            scores,
            [f"S1-{e:05d}" for e in s1_rows],
            [f"S2-{p:05d}" for p in pool_rows],
            truth, ev.macro_f05, grid=grid, resolve=resolve,
        )
        assert fast_t == pytest.approx(slow_t)
        assert fast_score == pytest.approx(slow_score, abs=1e-12), resolve


def test_sweep_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="mismatched validation arrays"):
        md.tune_threshold_arrays(
            np.array([0, 1]), np.array([0]), np.array([0.5, 0.5]),
            np.array([1]), 2, np.array([1, 1]),
        )
    with pytest.raises(ValueError, match="n_true has"):
        md.tune_threshold_arrays(
            np.array([0]), np.array([0]), np.array([0.5]), np.array([1]),
            2, np.array([1]),
        )


# --------------------------------------------------------------------------- #
# Block planner
# --------------------------------------------------------------------------- #


def _starts(widths: list[int]) -> np.ndarray:
    return np.concatenate(([0], np.cumsum(widths))).astype(np.int64)


def test_plan_blocks_covers_every_pair_exactly_once():
    widths = [0, 3, 0, 10, 5, 0, 7]
    starts = _starts(widths)
    blocks = st.plan_blocks(starts, len(widths), block_pairs=6)
    covered: list[int] = []
    for lo, hi in blocks:
        assert hi > lo
        covered.extend(range(lo, hi))
    assert covered == list(range(int(starts[-1])))


def test_plan_blocks_never_straddles_an_entity():
    """A block spanning two entities duplicates the boundary entity's per-record
    caches, which is the cost the block size exists to bound."""

    widths = [4, 4, 4, 4]
    starts = _starts(widths)
    for lo, hi in st.plan_blocks(starts, len(widths), block_pairs=6):
        # Every entity boundary strictly inside the block would be a straddle.
        for boundary in starts[1:-1]:
            assert not (lo < boundary < hi), (lo, hi, boundary)


def test_plan_blocks_emits_one_block_for_an_entity_wider_than_the_limit():
    """A single very wide entity must still be featurised, not skipped."""

    starts = _starts([2, 500, 1])
    blocks = st.plan_blocks(starts, 3, block_pairs=10)
    assert blocks == [(0, 2), (2, 502), (502, 503)]


def test_plan_blocks_skips_entities_with_no_candidates():
    starts = _starts([0, 0, 3, 0])
    assert st.plan_blocks(starts, 4, block_pairs=10) == [(0, 3)]


def test_plan_blocks_honours_a_row_range():
    starts = _starts([2, 2, 2, 2])
    blocks = st.plan_blocks(starts, 4, block_pairs=100, lo_row=1, hi_row=3)
    assert blocks == [(2, 6)]
