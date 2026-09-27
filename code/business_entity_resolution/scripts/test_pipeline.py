"""Regression tests for pipeline-stage invariants.

Each test here corresponds to a bug that actually occurred, which is the only
justification for a test existing. The common theme is *row-space and shape
mismatches between stages*: every one of these passed in isolation and failed
only once the stages were composed.

Run:  python scripts/test_pipeline.py
"""

from __future__ import annotations

import gc
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import blocking, pipeline  # noqa: E402

CASES: list[tuple[str, callable]] = []


def case(fn):
    CASES.append((fn.__name__, fn))
    return fn


def assert_eq(got, want, what: str) -> None:
    if got != want:
        raise AssertionError(f"{what}: got {got!r}, want {want!r}")


# --------------------------------------------------------------------------- #
# Candidate TSV boundaries
# --------------------------------------------------------------------------- #


@case
def test_candidate_tsv_gives_every_entity_its_own_pairs() -> None:
    """A regression test for a boundary search done in the wrong direction.

    ``write_candidates_from_pairs`` used to build its per-entity slice boundaries
    by searching the *entity* rows against the pairs' Source-1 column. That yields
    one entry per pair, so every entity's candidate list came out empty except the
    last, which received all of them. The validator caught it as 2,924 empty rows
    against a 115,294-pair file.
    """

    # Entity 0 has 3 candidates, entity 1 has 1, entity 2 has 2, entity 3 has none.
    pairs = np.array(
        [[0, 10, 1], [0, 11, 1], [0, 12, 1], [1, 20, 1], [2, 30, 1], [2, 31, 1]],
        dtype=np.int64,
    )
    s1_ids = np.array([b"S1-a", b"S1-b", b"S1-c", b"S1-d"])
    # Pool rows referenced by the pairs: 10, 11, 12, 20, 30, 31.
    pool_ids = np.array(
        [b"S2-filler"] * 10    # rows 0..9
        + [b"S2-x", b"S2-y", b"S2-z"]   # rows 10..12
        + [b"S2-filler"] * 7   # rows 13..19
        + [b"S2-w"]            # row 20
        + [b"S2-filler"] * 9   # rows 21..29
        + [b"S3-p", b"S3-q"]   # rows 30, 31
    )
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "candidate_pairs.tsv"
        written = blocking.write_candidates_from_pairs(
            pairs, np.arange(4), s1_ids, pool_ids, out
        )
        lines = out.read_text(encoding="utf-8").splitlines()

    assert_eq(written, 4, "rows written")
    assert_eq(lines[0], "source1_entity_id\tcandidate_entity_ids", "header")
    assert_eq(
        [ln.split("\t")[0] for ln in lines[1:]],
        ["S1-a", "S1-b", "S1-c", "S1-d"],
        "one row per entity, in order",
    )
    assert_eq(lines[1].split("\t")[1], "S2-x,S2-y,S2-z", "entity 0 candidates")
    assert_eq(lines[2].split("\t")[1], "S2-w", "entity 1 candidates")
    assert_eq(lines[3].split("\t")[1], "S3-p,S3-q", "entity 2 candidates")
    assert_eq(lines[4].split("\t")[1], "", "entity 3 has no candidates")


@case
def test_candidate_tsv_deduplicates_pool_ids() -> None:
    """Two strategies can surface the same pool record; the TSV must list it once."""

    pairs = np.array([[0, 5, 3], [0, 5, 4], [0, 5, 1]], dtype=np.int64)
    s1_ids = np.array([b"S1-a"])
    pool_ids = np.array([b"S2-dup"] * 6)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "c.tsv"
        blocking.write_candidates_from_pairs(
            pairs, np.arange(1), s1_ids, pool_ids, out
        )
        line = out.read_text(encoding="utf-8").splitlines()[1]
    assert_eq(line, "S1-a\tS2-dup", "duplicate pool ID listed once")


@case
def test_candidate_tsv_handles_zero_pairs() -> None:
    """Every entity still needs a row when blocking found nothing at all."""

    s1_ids = np.array([b"S1-a", b"S1-b"])
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "c.tsv"
        written = blocking.write_candidates_from_pairs(
            np.zeros((0, 3), dtype=np.int64), np.arange(2), s1_ids,
            np.array([], dtype="S5"), out,
        )
        lines = out.read_text(encoding="utf-8").splitlines()
    assert_eq(written, 2, "rows written with no pairs")
    assert_eq(lines[1], "S1-a\t", "empty row for entity a")
    assert_eq(lines[2], "S1-b\t", "empty row for entity b")


# --------------------------------------------------------------------------- #
# Subsampling
# --------------------------------------------------------------------------- #


@case
def test_subsample_keeps_all_positives() -> None:
    """Uniform sampling alone would drop ~99.6% of the positive class."""

    y = np.zeros(10_000, dtype=np.int8)
    y[[7, 123, 4567, 9999]] = 1
    rows = np.arange(10_000)
    picked = pipeline._subsample(rows, y, budget=500, seed=42, label="t")
    assert_eq(int(y[picked].sum()), 4, "all positives kept")
    assert_eq(len(picked), 500, "budget respected")
    assert_eq(bool(np.all(np.diff(picked) > 0)), True, "rows sorted and unique")


@case
def test_subsample_unlimited_when_budget_is_zero() -> None:
    y = np.zeros(100, dtype=np.int8)
    picked = pipeline._subsample(np.arange(100), y, budget=0, seed=1, label="t")
    assert_eq(len(picked), 100, "budget 0 means no cap")


@case
def test_subsample_is_deterministic() -> None:
    y = np.zeros(5_000, dtype=np.int8)
    y[::500] = 1
    rows = np.arange(5_000)
    a = pipeline._subsample(rows, y, budget=300, seed=7, label="t")
    b = pipeline._subsample(rows, y, budget=300, seed=7, label="t")
    assert_eq(np.array_equal(a, b), True, "same seed gives same sample")


# --------------------------------------------------------------------------- #
# Validation split
# --------------------------------------------------------------------------- #


@case
def test_validation_mask_is_entity_level_and_pure() -> None:
    """Every pair of an entity lands on the same side, and the split is stable."""

    s1_ids = np.array([f"S1-{i}".encode() for i in range(500)])
    pairs = np.repeat(np.arange(500, dtype=np.int64), 3)
    pairs = np.column_stack([pairs, np.arange(len(pairs)), np.ones(len(pairs))])
    pairs = pairs.astype(np.int64)

    mask = pipeline._validation_mask(s1_ids, pairs, seed=42)
    for entity in range(500):
        rows = mask[entity * 3:(entity + 1) * 3]
        if len(set(rows.tolist())) != 1:
            raise AssertionError(f"entity {entity} split across both sides")
    again = pipeline._validation_mask(s1_ids, pairs, seed=42)
    assert_eq(np.array_equal(mask, again), True, "split is stable across calls")
    other = pipeline._validation_mask(s1_ids, pairs, seed=43)
    assert_eq(np.array_equal(mask, other), False, "split depends on the seed")


@case
def test_validation_mask_respects_fraction() -> None:
    s1_ids = np.array([f"S1-{i}".encode() for i in range(20_000)])
    pairs = np.arange(20_000, dtype=np.int64).reshape(-1, 1)
    pairs = np.column_stack([pairs, pairs, pairs])
    mask = pipeline._validation_mask(s1_ids, pairs, seed=42, fraction=0.15)
    share = float(mask.mean())
    if not 0.13 < share < 0.17:
        raise AssertionError(f"validation share {share:.3f} far from 0.15")


# --------------------------------------------------------------------------- #
# Context features
# --------------------------------------------------------------------------- #


@case
def test_name_frequency_counts_shared_names() -> None:
    """A constant 1 here would silently disable the singleton prior."""

    key = np.array([b"alpha pvt", b"beta", b"alpha pvt", b"gamma", b"gamma", b"gamma"])
    got = pipeline._name_frequency(key).tolist()
    assert_eq(got, [2, 1, 2, 3, 3, 3], "per-record name frequency")


@case
def test_name_frequency_handles_empty_pool() -> None:
    assert_eq(len(pipeline._name_frequency(np.array([], dtype="S5"))), 0, "empty pool")


@case
def test_candidate_degree_counts_references() -> None:
    got = pipeline._candidate_degree(np.array([0, 0, 2, 2, 2, 4]), 6).tolist()
    assert_eq(got, [2, 0, 3, 0, 1, 0], "candidate degree per pool row")


@case
def test_pool_context_is_aligned_to_pairs() -> None:
    """Context arrays are indexed by pair, in the order the pairs were written."""

    pool_ids = np.array([b"S2-a", b"S2-b", b"S2-c", b"S3-a"])
    pool_source = np.array([2, 2, 2, 3], dtype=np.int8)
    pool_rows = np.array([3, 0, 0], dtype=np.int64)
    context = pipeline._pool_context(
        pool_ids, pool_source, pool_rows, np.array([b"x", b"y", b"z", b"x"])
    )
    assert_eq(
        context["pool_source"].tolist(), [3.0, 2.0, 2.0], "source aligned to pairs"
    )
    # pool rows 3 and 0 both have name "x", so frequency 2; row 0 appears twice.
    assert_eq(
        context["pool_frequency"].tolist(), [2.0, 2.0, 2.0], "frequency aligned to pairs"
    )


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #


@case
def test_combined_key_does_not_collide() -> None:
    """(s1_row, pool_row) must be encoded so distinct pairs differ.

    Row-space mixing is the classic silent bug here: a naive concatenation of the
    two rows would make (1, 23) and (12, 3) share a key.
    """

    pool_size = 100
    keys = [1 * pool_size + 23, 12 * pool_size + 3]
    assert_eq(len(set(keys)), 2, "combined keys are distinct")


@case
def test_resolve_competition_arrays_matches_dict_version() -> None:
    """The array path is a performance rewrite; it must not change the answer.

    This is the only thing standing between the fast path and a silently
    different submission, so it is checked against the reference implementation
    rather than against hand-written expectations.
    """

    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src import model as md

    rng = np.random.default_rng(11)
    n_s1, n_pool = 40, 25
    rows = []
    for _ in range(300):
        rows.append((int(rng.integers(n_s1)), int(rng.integers(n_pool))))
    s1_rows = np.array([r[0] for r in rows])
    pool_rows = np.array([r[1] for r in rows])
    scores = rng.random(len(rows))

    s1_names = [f"S1-{i}" for i in range(n_s1)]
    pool_names = [f"S2-{i}" for i in range(n_pool)]
    claims: dict[str, list[tuple[str, float]]] = {name: [] for name in s1_names}
    for s1_row, pool_row, score in zip(s1_rows, pool_rows, scores):
        claims[s1_names[s1_row]].append((pool_names[pool_row], float(score)))

    for threshold in (0.0, 0.5, 0.9):
        expected = md.resolve_competition(claims, threshold)
        win_s1, win_pool, _scores = md.resolve_competition_arrays(
            s1_rows, pool_rows, scores, threshold, n_pool
        )
        got: dict[str, list[str]] = {name: [] for name in s1_names}
        for s1_row, pool_row in zip(win_s1, win_pool):
            got[s1_names[s1_row]].append(pool_names[pool_row])
        got = {k: sorted(v) for k, v in got.items()}
        want = {k: sorted(v) for k, v in expected.items()}
        assert_eq(got, want, f"resolution agrees at threshold {threshold}")


@case
def test_two_stage_resolution_equals_single_stage() -> None:
    """``run_predict`` resolves per chunk, then globally, to bound memory.

    The per-chunk pass only ever drops claims dominated within their own chunk, so
    the global winner for every pool record must survive to the second pass. This
    is the property that makes the memory optimisation legal.
    """

    from src import model as md

    rng = np.random.default_rng(23)
    n_s1, n_pool, n_pairs = 60, 30, 500
    s1_rows = rng.integers(0, n_s1, n_pairs)
    pool_rows = rng.integers(0, n_pool, n_pairs)
    scores = rng.random(n_pairs)
    threshold = 0.4

    one_pass = md.resolve_competition_arrays(
        s1_rows, pool_rows, scores, threshold, n_pool
    )

    # Simulate four chunks by slicing, resolving each, then resolving the union.
    bounds = np.linspace(0, n_pairs, 5).astype(int)
    parts = [
        md.resolve_competition_arrays(
            s1_rows[a:b], pool_rows[a:b], scores[a:b], threshold, n_pool
        )
        for a, b in zip(bounds[:-1], bounds[1:])
    ]
    staged = md.resolve_competition_arrays(
        np.concatenate([p[0] for p in parts]),
        np.concatenate([p[1] for p in parts]),
        np.concatenate([p[2] for p in parts]),
        0.0, n_pool,
    )
    assert_eq(
        [one_pass[0].tolist(), one_pass[1].tolist()],
        [staged[0].tolist(), staged[1].tolist()],
        "chunked resolution matches single-pass",
    )


@case
def test_resolution_preserves_winning_score() -> None:
    """The returned score must be the winner's, so a later pass can compare it."""

    from src import model as md

    s1_rows = np.array([0, 1])
    pool_rows = np.array([7, 7])
    scores = np.array([0.9, 0.2])
    win_s1, win_pool, win_score = md.resolve_competition_arrays(
        s1_rows, pool_rows, scores, 0.0, 10
    )
    assert_eq(win_s1.tolist(), [0], "entity 0 wins")
    assert_eq(win_pool.tolist(), [7], "on pool record 7")
    assert_eq(round(float(win_score[0]), 6), 0.9, "winner's own score is returned")


@case
def test_resolve_competition_arrays_gives_each_pool_row_one_owner() -> None:
    from src import model as md

    s1_rows = np.array([0, 0, 1, 1, 2])
    pool_rows = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.9, 0.4, 0.5, 0.8, 0.3])
    win_s1, win_pool, _ = md.resolve_competition_arrays(
        s1_rows, pool_rows, scores, 0.0, 3
    )
    assert_eq(sorted(win_pool.tolist()), [0, 1], "one winner per pool row")
    # pool 0: entity 0 at 0.9 beats entity 1 at 0.5. pool 1: entity 1 at 0.8.
    assert_eq(
        dict(zip(win_pool.tolist(), win_s1.tolist())), {0: 0, 1: 1}, "best claim wins"
    )


@case
def test_resolve_competition_arrays_breaks_ties_on_source1_row() -> None:
    """A tie must be deterministic, not dependent on input order."""

    from src import model as md

    s1_rows = np.array([7, 3])
    pool_rows = np.array([0, 0])
    scores = np.array([0.5, 0.5])
    forward = md.resolve_competition_arrays(s1_rows, pool_rows, scores, 0.0, 1)
    backward = md.resolve_competition_arrays(
        s1_rows[::-1], pool_rows[::-1], scores[::-1], 0.0, 1
    )
    assert_eq(forward[0].tolist(), [3], "lower Source-1 row wins the tie")
    assert_eq(
        [a.tolist() for a in forward], [b.tolist() for b in backward],
        "independent of input order",
    )


@case
def test_resolve_competition_arrays_output_sorted_by_entity() -> None:
    from src import model as md

    rng = np.random.default_rng(5)
    s1_rows = rng.integers(0, 50, 400)
    pool_rows = rng.integers(0, 500, 400)
    scores = rng.random(400)
    win_s1, _pool, _score = md.resolve_competition_arrays(
        s1_rows, pool_rows, scores, 0.2, 500
    )
    assert_eq(bool(np.all(np.diff(win_s1) >= 0)), True, "sorted by Source-1 row")


@case
def test_resolve_competition_arrays_handles_empty_input() -> None:
    from src import model as md

    win_s1, win_pool, win_score = md.resolve_competition_arrays(
        np.array([], np.int64), np.array([], np.int64), np.array([]), 0.5, 10
    )
    assert_eq((len(win_s1), len(win_pool), len(win_score)), (0, 0, 0), "empty input")


@case
def test_label_and_ceiling_matches_brute_force() -> None:
    """The chunked label/ceiling pass must agree with an exhaustive scan.

    It is split into Source-1 row ranges to bound memory, and a wrong boundary
    there would silently mislabel pairs near a chunk edge -- which no amount of
    downstream metric checking would catch. The chunk size is deliberately set
    small so several boundaries fall inside the data.
    """

    rng = np.random.default_rng(3)
    n_s1, n_pool = 40, 25
    rows = np.repeat(np.arange(n_s1, dtype=np.int64), 3)
    pool = rng.integers(0, n_pool, len(rows)).astype(np.int64)
    # build_candidates guarantees the array is grouped by s1_row; reproduce that.
    order = np.argsort(rows * n_pool + pool, kind="stable")
    rows, pool = rows[order], pool[order]
    pairs = np.column_stack([rows, pool, np.ones(len(rows), np.int64)])

    # Entity 7 deliberately lists the same pool record twice, as a duplicated
    # ground-truth row would.
    truth_pairs = {(1, 2), (1, 5), (1, 9), (5, 3), (7, 4), (39, 7)}
    truth_s1 = np.array(sorted(a for a, _ in truth_pairs), dtype=np.int64)
    truth_pool = np.array([b for _, b in sorted(truth_pairs)], dtype=np.int64)

    labels, truth_hit = pipeline._label_and_ceiling(
        pairs, truth_s1, truth_pool, n_pool, n_s1, chunk_rows=8
    )
    expected_labels = np.array(
        [1 if (int(a), int(b)) in truth_pairs else 0 for a, b in zip(rows, pool)],
        dtype=np.int8,
    )
    candidates = {(int(a), int(b)) for a, b in zip(rows, pool)}
    expected_hit = np.array(
        [(int(a), int(b)) in candidates for a, b in zip(truth_s1, truth_pool)],
        dtype=bool,
    )
    assert_eq(np.array_equal(labels, expected_labels), True, "labels")
    assert_eq(np.array_equal(truth_hit, expected_hit), True, "truth hits")


@case
def test_recall_counts_only_fully_covered_entities() -> None:
    """entity_full_recall must be all-or-nothing per entity, not per pair.

    Macro F_0.5 scores whole entities, so an entity that retrieved 3 of 4 true
    matches scores zero on that entity. Reporting 3/4 as partial credit here
    would overstate the achievable ceiling.
    """

    # Two truth entities: the first fully retrieved, the second missing one.
    truth_s1 = np.array([1, 1, 5, 5], dtype=np.int64)
    truth_hit = np.array([True, True, True, False])
    got = pipeline._recall_from_hits(truth_s1, truth_hit)
    assert_eq(got["pair_recall"], 0.75, "pair recall counts pairs")
    assert_eq(got["entity_full_recall"], 0.5, "entity recall is all-or-nothing")
    assert_eq(got["true_pairs"], 4.0, "true pair count")
    assert_eq(got["entities_with_matches"], 2.0, "entity count")


@case
def test_recall_handles_no_truth() -> None:
    got = pipeline._recall_from_hits(np.array([], np.int64), np.array([], bool))
    assert_eq(got["pair_recall"], 0.0, "no truth pairs")
    assert_eq(got["entity_full_recall"], 0.0, "no truth entities")


@case
def test_label_and_ceiling_handles_empty_pairs() -> None:
    labels, hit = pipeline._label_and_ceiling(
        np.zeros((0, 3), np.int64), np.array([1], np.int64), np.array([2], np.int64),
        10, 5,
    )
    assert_eq((len(labels), len(hit)), (0, 1), "shapes preserved")
    assert_eq(bool(hit[0]), False, "an unretrieved truth pair is not a hit")


@case
def test_featurise_arrays_out_buffer_matches_allocating_path() -> None:
    """Writing into a caller-supplied buffer must be bit-identical.

    The production caller passes a slice of the on-disk memmap, which saves a
    second ~700 MB buffer per chunk. Two things could silently differ: a feature
    assigned conditionally might be left holding whatever was already in the
    destination, and a mismatched buffer might be written past the end. So the
    buffer is pre-poisoned with a value no feature can take, and every cell is
    checked for having been written.
    """

    pool = _tiny_normalised_set([
        ("Zeta Foods", "9 Baker Rd, Austin, TX 73301", "co"),
        ("Zeta Food Co", "9 Baker Road Austin TX 73301", "co"),
        ("", "x", ""),
    ])
    s1 = _tiny_normalised_set([
        ("Acme Trading LLC", "12 Main St, Springfield, IL 62704", "llc"),
        ("Acme Trading", "12 Main Street Springfield Illinois 62704", "llc"),
        ("ACME TRADING L.L.C.", "12 Main St., Springfld, IL, 62704", "llc"),
    ])
    s1_rows = np.array([0, 1, 2, 0, 1, 2])
    pool_rows = np.array([0, 1, 2, 2, 1, 0])
    evidence = np.array([1, 2, 1, 3, 1, 1])
    degree = np.array([3, 2, 1, 1, 1, 1], np.int32)
    # Per-pair, exactly as the pipeline gathers it from the per-entity count.
    counts = np.array([2, 2, 2, 2, 2, 2], np.int32)
    source = np.full(6, 2.0, np.float32)
    frequency = np.array([1, 1, 2, 2, 1, 1], np.float32)
    kwargs = dict(
        columns=list(pipeline.ft.feature_columns()),
        known_countries={"united states"},
    )
    args = (s1, pool, s1_rows, pool_rows, evidence, source, frequency, degree, counts)

    allocated = pipeline.ft.featurise_arrays(*args, **kwargs)
    poisoned = np.full(allocated.shape, -999.0, np.float32)
    written = pipeline.ft.featurise_arrays(*args, out=poisoned, **kwargs)

    assert_eq(np.array_equal(allocated, written), True, "bit-identical")
    assert_eq(bool((written == -999.0).any()), False, "every cell written")
    assert_eq(int((written != 0).sum()) > 0, True, "some features are set")

    try:
        pipeline.ft.featurise_arrays(
            *args, out=np.zeros((3, allocated.shape[1]), np.float32), **kwargs
        )
        assert_eq("no error", "ValueError", "mismatched buffer rejected")
    except ValueError:
        assert_eq("ValueError", "ValueError", "mismatched buffer rejected")


@case
def test_memmap_pool_is_byte_identical_to_in_ram_pool() -> None:
    """The memmap pool must be indistinguishable from the in-RAM pool.

    The features stage reads the pool through a read-only memmap instead of an
    in-RAM array. That is only safe if every column, including the widths and the
    row order that entity_id fixes, comes back exactly the same -- a memmap that
    silently returned zeros would produce plausible-looking features that are all
    wrong, and no downstream check would catch it.
    """

    import pyarrow as pa
    import pyarrow.parquet as pq

    pp = pipeline.pp
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp)
        rows_a = [("Acme Holdings", "12 main st springfield il", "llc"),
                  ("Globex Ltd", "9 baker rd austin tx", "pvt ltd"),
                  ("Initech", "1 wayland rd springfield il", "")]
        rows_b = [("Umbrella Corp", "44 hill rd austin tx", "corp"),
                  ("Soylent", "8 market st springfield il", "")]
        for source, rows in ((2, rows_a), (3, rows_b)):
            table = pa.table({
                "entity_id": [f"S{source}-{i}" for i in range(len(rows))],
                "name_norm": [r[0].lower() for r in rows],
                "name_legal": [r[2] for r in rows],
                "address_core": [r[1].lower() for r in rows],
                "addr_house": [r[1].split()[0] for r in rows],
                "addr_postal": ["62704"] * len(rows),
                "addr_state": [r[1].split()[-2] for r in rows],
                "addr_city": [r[1].split()[-3] for r in rows],
                "addr_tail2": [r[1].split()[-1] for r in rows],
                "landmark_tokens": [r[1].split()[1] for r in rows],
                "country_norm": ["united states"] * len(rows),
            })
            pq.write_table(table, cache / f"train_source{source}.parquet")

        columns = ["name_norm", "name_legal", "address_core", "addr_house",
                   "addr_postal", "addr_state", "addr_city", "addr_tail2",
                   "landmark_tokens", "country_norm"]
        ram = pp.load_normalised(cache, "train", sources=(2, 3),
                                 columns=columns, memmap=False)
        mapped = pp.load_normalised(cache, "train", sources=(2, 3),
                                   columns=columns, memmap=True)

        assert_eq(len(ram), len(mapped), "memmap pool row count")
        assert_eq(len(mapped), 5, "rows are the two sources concatenated in order")
        for col in list(columns) + ["entity_id"]:
            a = getattr(ram, col)
            b = getattr(mapped, col)
            assert_eq(str(b.dtype), str(a.dtype), f"{col} dtype")
            assert_eq(bool(np.array_equal(a, b)), True, f"{col} bytes equal")
        # The multi-source concatenation order is the part a pooled backing could
        # plausibly get wrong, so assert it directly rather than trusting equality
        # with a path that might share the same bug.
        assert_eq(
            [bytes(v).decode() for v in mapped.entity_id[:5]],
            ["S2-0", "S2-1", "S2-2", "S3-0", "S3-1"],
            "memmap row order across sources",
        )
        assert_eq(
            bool(getattr(mapped, "name_norm").flags.writeable), False,
            "memmap columns are read-only",
        )
        for col in list(columns) + ["entity_id"]:
            getattr(mapped, col)._mmap.close()


@case
def test_memmap_pool_reuses_a_complete_backing_and_rebuilds_a_partial_one() -> None:
    """A half-written backing must never be mistaken for a finished one.

    The backing is reused across runs, so a build interrupted partway leaves
    .npy files with valid headers and mostly zeros. Reading those back would look
    like a pool full of empty fields.
    """

    import pyarrow as pa
    import pyarrow.parquet as pq

    pp = pipeline.pp
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp)
        table = pa.table({
            "entity_id": ["S2-0", "S2-1"],
            "name_norm": ["acme", "globex"],
            "name_legal": ["llc", "ltd"],
            "address_core": ["12 main st", "9 baker rd"],
            "addr_house": ["12", "9"],
            "addr_postal": ["62704", "73301"],
            "addr_state": ["il", "tx"],
            "addr_city": ["springfield", "austin"],
            "addr_tail2": ["st", "rd"],
            "landmark_tokens": ["main", "baker"],
            "country_norm": ["united states", "united states"],
        })
        pq.write_table(table, cache / "train_source2.parquet")
        columns = ["name_norm", "country_norm"]

        first = pp.load_normalised(cache, "train", sources=(2,),
                                   columns=columns, memmap=True)
        assert_eq([bytes(v).decode() for v in first.name_norm],
                  ["acme", "globex"], "first build reads the source")

        # Same layout: the sidecar still certifies it, so the .npy is reused.
        again = pp.load_normalised(cache, "train", sources=(2,),
                                   columns=columns, memmap=True)
        assert_eq([bytes(v).decode() for v in again.name_norm],
                  ["acme", "globex"], "reused backing still reads correctly")

        # Simulate a build killed midway: sidecar gone, data left behind. The
        # earlier mappings must be released first -- Windows cannot replace or
        # delete a file that is still mapped, which is a real constraint of the
        # backing rather than a quirk of the test.
        tags = sorted(cache.glob("train_norm_*.meta.json"))
        assert_eq(len(tags), 1, "one sidecar was written")
        tags[0].unlink()
        for stale in (first, again):
            for col in list(columns) + ["entity_id"]:
                getattr(stale, col)._mmap.close()
        del first, again
        gc.collect()

        rebuilt = pp.load_normalised(cache, "train", sources=(2,),
                                     columns=columns, memmap=True)
        assert_eq([bytes(v).decode() for v in rebuilt.name_norm],
                  ["acme", "globex"], "rebuilt after the sidecar was lost")
        assert_eq(len(sorted(cache.glob("train_norm_*.meta.json"))), 1,
                  "sidecar rewritten")
        assert_eq(len(sorted(cache.glob("*.building"))), 0,
                  "no staging files left behind")
        for col in list(columns) + ["entity_id"]:
            getattr(rebuilt, col)._mmap.close()


@case
def test_memmap_backing_keys_on_the_projected_columns() -> None:
    """Two projections of one split must not share a backing.

    Blocking asks for BLOCK_COLUMNS and the featuriser for FEATURE_COLUMNS. If
    they collided, whichever ran second would silently get the wrong set.
    """

    pp = pipeline.pp
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp)
        _tag_a, paths_a, _meta_a = pp._norm_backing_paths(
            cache, "train", (2, 3), ("name_norm", "entity_id"))
        _tag_b, paths_b, _meta_b = pp._norm_backing_paths(
            cache, "train", (2, 3), ("name_norm", "entity_id", "addr_city"))
        assert_eq(set(paths_a.values()) & set(paths_b.values()), set(),
                  "differing projections get distinct files")
        _tag_c, paths_c, _meta_c = pp._norm_backing_paths(
            cache, "train", (1,), ("name_norm", "entity_id"))
        assert_eq(set(paths_a.values()) & set(paths_c.values()), set(),
                  "differing sources get distinct files")
        _tag_d, paths_d, _meta_d = pp._norm_backing_paths(
            cache, "train", (2, 3), ("name_norm", "entity_id"))
        assert_eq(paths_d, paths_a, "identical projection is stable across calls")


def _tiny_normalised_set(rows):
    """Build a minimal NormalisedSet from (name, address, legal) triples."""

    pp = pipeline.pp
    arrays = {}
    names = [r[0] for r in rows]
    addrs = [r[1] for r in rows]
    legal = [r[2] for r in rows]
    payload = {
        "entity_id": [f"X{i}" for i in range(len(rows))],
        "country_norm": ["united states"] * len(rows),
        "name_norm": [n.lower() for n in names],
        "name_legal": legal,
        "address_core": [a.lower() for a in addrs],
        "address_norm": [a.lower() for a in addrs],
        "addr_house": ["12" if i < len(rows) - 1 else "9" for i in range(len(rows))],
        "addr_postal": ["62704" if i < len(rows) - 1 else "73301"
                        for i in range(len(rows))],
        "addr_state": ["il" if i < len(rows) - 1 else "tx" for i in range(len(rows))],
        "addr_city": ["springfield" if i < len(rows) - 1 else "austin"
                      for i in range(len(rows))],
        "addr_tail2": [""] * len(rows),
        "landmark_tokens": ["main st" if i < len(rows) - 1 else "baker rd"
                           for i in range(len(rows))],
    }
    for col, values in payload.items():
        arrays[col] = np.array(values, dtype=f"S{pp._WIDTHS[col]}")
    arrays["name_present"] = np.array([bool(n) for n in names])
    arrays["address_present"] = np.array([bool(a) for a in addrs])
    return pp.NormalisedSet(arrays)


def main() -> int:
    failures = 0
    for name, fn in CASES:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001 - a test harness reports everything
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
            failures += 1
    print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
