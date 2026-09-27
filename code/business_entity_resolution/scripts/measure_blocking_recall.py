"""Measure the blocking recall ceiling on a sample of training Source-1 entities.

Recall is the whole point of this script. A candidate set can be arbitrarily
small and still useless, so every configuration is scored against
``evaluation.blocking_recall_ceiling`` before it is accepted:

* ``pair_recall``           -- of all true matches, how many were proposed
* ``pair_recall_all_pairs`` -- same, counting singletons as "no candidate needed"
* ``entity_full_recall``    -- of entities that DO have matches, how many had
                               *all* of them proposed. This is the number that
                               actually caps macro F0.5: one missing match on a
                               3-match entity is already a lost entity.

Run:  python scripts/measure_blocking_recall.py --sample 100000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import blocking, evaluation, preprocessing as pp

# Blocking needs only these; the full record is ~640 bytes and a three-source
# train load would not fit in memory.
PROJECTION = (
    "name_norm", "address_core", "addr_house", "addr_postal",
    "addr_tail2", "country_norm",
)


def recall_for(
    s1, pool, s1_ids, pool_ids, sample_rows, truth_subset, index_dir, strategies,
    max_block, max_candidates, out_dir, label,
):
    """Configure the index at ``max_block`` and measure one recall ceiling."""

    index_stats = [
        blocking.finalise_index(strategy, index_dir, max_block)
        for strategy in strategies
    ]
    dropped = sum(s["postings_dropped"] for s in index_stats)
    total = sum(s["postings"] for s in index_stats)
    print(
        f"[sweep] {label}: max_block={max_block} -> dropped "
        f"{dropped / max(total, 1):.1%} of postings", flush=True,
    )

    stats, pairs = blocking.build_candidates(
        s1, sample_rows, len(pool_ids), index_dir, out_dir / "pairs.npy",
        max_candidates_per_entity=max_candidates,
        strategies=strategies, verbose=False,
    )
    by_row: dict[int, list[str]] = {}
    for s1_row, pool_row in pairs[:, :2]:
        by_row.setdefault(int(s1_row), []).append(pool_ids[pool_row].decode("utf-8"))
    candidate_ids = {
        s1_ids[row].decode("utf-8"): sorted(by_row.get(int(row), []))
        for row in sample_rows
    }
    detail = evaluation.blocking_recall_detail(
        candidate_ids, {e: truth_subset[e] for e in candidate_ids}
    )
    detail["mean_candidates"] = stats["mean_candidates"]
    detail["reduction_ratio"] = stats["reduction_ratio"]
    detail["entities_without_candidates"] = stats["entities_without_candidates"]
    detail["cap_fired_entities"] = stats["cap_fired_entities"]
    detail["chunk_size"] = stats["chunk_size"]
    detail["pairs_before_cap"] = stats["pairs_before_cap"]
    detail["postings_dropped_share"] = dropped / max(total, 1)
    return detail, index_stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="../../cache")
    parser.add_argument("--artifacts", default="../../artifacts")
    parser.add_argument("--sample", type=int, default=100_000,
                        help="Source-1 entities to measure")
    parser.add_argument("--max-block", type=int, default=blocking.DEFAULT_MAX_BLOCK)
    parser.add_argument("--max-candidates", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--sweep", action="store_true",
                        help="sweep max_block and ablate individual strategies")
    parser.add_argument("--sweep-cap", action="store_true",
                        help="sweep max_candidates_per_entity at a fixed max_block")
    parser.add_argument(
        "--strategies", default=",".join(blocking.STRATEGIES),
        help="comma-separated subset, to attribute recall to individual keys",
    )
    args = parser.parse_args()

    cache = Path(args.cache)
    artifacts = Path(args.artifacts)
    index_dir = cache / "index_train"
    artifacts.mkdir(parents=True, exist_ok=True)
    strategies = [s for s in args.strategies.split(",") if s]
    for strategy in strategies:
        if strategy not in blocking.STRATEGIES:
            raise SystemExit(f"unknown strategy {strategy!r}")

    t0 = time.time()
    print("[load] pool (source2+3)", flush=True)
    pool = pp.load_normalised(cache, "train", sources=(2, 3), columns=PROJECTION)
    print(f"[load] pool rows={len(pool):,} in {time.time() - t0:.0f}s", flush=True)
    pool_rows = np.arange(len(pool), dtype=np.int32)

    if args.rebuild:
        print("[index] building", flush=True)
        t1 = time.time()
        counts = blocking.build_indexes(pool, pool_rows, index_dir, strategies)
        for strategy, count in counts.items():
            print(f"[index]   {strategy}: {count:,} raw postings", flush=True)
        print(f"[index] key generation took {time.time() - t1:.0f}s", flush=True)
    else:
        missing = [
            s for s in strategies
            if not (index_dir / f"{s}_keys.npy").exists()
        ]
        if missing:
            raise SystemExit(f"index missing for {missing} - pass --rebuild")

    print("[load] source1", flush=True)
    s1 = pp.load_normalised(cache, "train", sources=(1,), columns=PROJECTION)
    s1_ids = s1.entity_id
    pool_ids = pool.entity_id

    rng = np.random.default_rng(args.seed)
    # Stratify nothing here: entities are sampled uniformly so the measured mean
    # candidates per entity is the number the model will actually face.
    take = min(args.sample, len(s1))
    chosen = np.sort(rng.choice(len(s1), size=take, replace=False)).astype(np.int32)
    sample_rows = chosen
    print(f"[sample] {take:,} Source-1 entities", flush=True)

    # --- ground truth restricted to the sample -------------------------------
    gt_path = pp.ground_truth_path(pp.resolve_data_dir())
    full_truth = pp.load_ground_truth(gt_path)
    wanted = {s1_ids[i].decode("utf-8") for i in sample_rows}
    truth_subset = {eid: full_truth.get(eid, []) for eid in wanted}
    del full_truth
    matched = sum(1 for v in truth_subset.values() if v)
    print(
        f"[truth] {matched:,} of {len(wanted):,} sampled entities have matches "
        f"({sum(len(v) for v in truth_subset.values()):,} true pairs)",
        flush=True,
    )

    out_dir = artifacts / "blocking_probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sweep_cap:
        results = []
        print(f"\n[sweep] === max_candidates at max_block={args.max_block} ===",
              flush=True)
        for max_candidates in (50, 100, 200, 400, 800):
            detail, _ = recall_for(
                s1, pool, s1_ids, pool_ids, sample_rows, truth_subset, index_dir,
                strategies, args.max_block, max_candidates, out_dir,
                f"max_candidates={max_candidates}",
            )
            print(
                f"   cap={max_candidates:4d}  pair_recall={detail['pair_recall']:.4f} "
                f"full={detail['entity_full_recall']:.4f} "
                f"mean_cand={detail['mean_candidates']:.1f} "
                f"cap_fired={detail['cap_fired_entities']/take:.1%}",
                flush=True,
            )
            results.append({"config": f"max_candidates={max_candidates}", **detail})
        out = artifacts / "blocking_cap_sweep.json"
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[report] wrote {out}")
        return 0

    if args.sweep:
        results = []
        # The cap sweep must be monotonically decreasing: tightening an index is
        # destructive, so re-widening would silently return the tighter index.
        caps = sorted({20_000, 5_000, 2_000, 500}, reverse=True)
        print("\n[sweep] === max_block (descending) ===", flush=True)
        best_cap, best_full = None, -1.0
        for max_block in caps:
            detail, _ = recall_for(
                s1, pool, s1_ids, pool_ids, sample_rows, truth_subset, index_dir,
                strategies, max_block, args.max_candidates, out_dir,
                f"max_block={max_block}",
            )
            print(
                f"   pair_recall={detail['pair_recall']:.4f} "
                f"full={detail['entity_full_recall']:.4f} "
                f"mean_cand={detail['mean_candidates']:.1f} "
                f"cap_fired={detail['cap_fired_entities']/take:.1%} "
                f"pre_cap={detail['pairs_before_cap']/1e6:.1f}M",
                flush=True,
            )
            results.append({"config": f"max_block={max_block}", **detail})
            if detail["entity_full_recall"] > best_full:
                best_cap, best_full = max_block, detail["entity_full_recall"]

        # Leave-one-out runs at the tightest cap applied so far, so the index is
        # never asked to widen.
        tightest = caps[-1]
        print(
            f"\n[sweep] === leave-one-strategy-out (at max_block={tightest}) ===",
            flush=True,
        )
        for dropped in strategies:
            subset = [s for s in strategies if s != dropped]
            detail, _ = recall_for(
                s1, pool, s1_ids, pool_ids, sample_rows, truth_subset, index_dir,
                subset, tightest, args.max_candidates, out_dir,
                f"without {dropped}",
            )
            print(
                f"   without {dropped:15s} pair_recall={detail['pair_recall']:.4f} "
                f"full={detail['entity_full_recall']:.4f} "
                f"mean_cand={detail['mean_candidates']:.1f}",
                flush=True,
            )
            results.append({"config": f"without_{dropped}", **detail})

        print(f"\n[sweep] best entity_full_recall={best_full:.4f} at max_block={best_cap}")
        out = artifacts / "blocking_sweep.json"
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[report] wrote {out}")
        return 0

    detail, index_stats = recall_for(
        s1, pool, s1_ids, pool_ids, sample_rows, truth_subset, index_dir,
        strategies, args.max_block, args.max_candidates, out_dir, "default",
    )
    print("\n[recall ceiling] " + json.dumps(detail, indent=2))

    report = {
        "sample_entities": take,
        "max_block": args.max_block,
        "max_candidates_per_entity": args.max_candidates,
        "strategies": strategies,
        "index_stats": index_stats,
        "recall": detail,
    }
    out = artifacts / f"blocking_recall_{len(strategies)}strat.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[report] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
