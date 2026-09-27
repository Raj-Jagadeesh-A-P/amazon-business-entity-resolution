"""Stage 1b: ground-truth structure and blocking-feasibility probes.

Read-only. Run:  python scripts/eda_gt.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA_CANDIDATES = [ROOT / "dataset" / "student_resource" / "dataset", ROOT / "dataset"]


def find_data_dir() -> Path:
    for cand in DATA_CANDIDATES:
        if (cand / "train").is_dir():
            return cand
    raise SystemExit("no dataset dir")


def main() -> int:
    data_dir = find_data_dir()
    gt = pd.read_csv(
        data_dir / "train" / "train_ground_truth.tsv",
        sep="\t", dtype=str, keep_default_na=False, encoding="utf-8",
    )
    lists = [
        [t for t in s.split(",") if t] if s.strip() else []
        for s in gt["matched_entity_ids"]
    ]
    sizes = pd.Series([len(x) for x in lists])
    total = sizes.sum()
    print("########## GROUND TRUTH (correctly parsed) ##########")
    print(f"  S1 train entities: {len(gt):,}")
    print(f"  total true pairs:  {total:,}")
    print(f"  singletons (size 0): {(sizes == 0).sum():,} ({(sizes == 0).mean():.2%})")
    print(f"  mean matches per entity: {total / len(gt):.2f}")
    print("\n  match-list size distribution:")
    print(sizes.value_counts().sort_index().to_string())
    s2 = sum(1 for l in lists for t in l if t.startswith("S2-"))
    s3 = sum(1 for l in lists for t in l if t.startswith("S3-"))
    print(f"\n  S2 pairs: {s2:,} ({s2 / total:.1%})   S3 pairs: {s3:,} ({s3 / total:.1%})")
    owners = Counter(t for l in lists for t in l)
    multi = {k: v for k, v in owners.items() if v > 1}
    print(f"  distinct matched pool records: {len(owners):,}")
    print(f"  pool records claimed by >1 S1 entity: {len(multi):,} "
          f"(max claims {max(owners.values())})")
    print(f"  pool coverage: {len(owners):,}/10,320,219 = {len(owners) / 10320219:.1%} "
          f"of the train pool is a true match of some S1 entity")

    # per-entity: how many S1 entities share a pool record -> contention
    contested = sum(v for v in owners.values() if v > 1)
    print(f"  true pairs lost to unavoidable contention: {contested - len(multi):,}")

    print("\n########## LENGTH DISTRIBUTIONS (train source1 + source2 sample) ##########")
    for src in (1, 2):
        df = pd.read_csv(
            data_dir / "train" / f"train_source{src}.tsv",
            sep="\t", dtype=str, keep_default_na=False, encoding="utf-8", nrows=400_000,
        )
        for col in ("business_name", "business_address"):
            lens = df[col].str.len()
            print(f"  source{src} {col}: p50={lens.quantile(.5):.0f} "
                  f"p90={lens.quantile(.9):.0f} p99={lens.quantile(.99):.0f} "
                  f"p99.9={lens.quantile(.999):.0f} max={lens.max()}")

    print("\n########## NAME-TOKEN DOCUMENT FREQUENCY (train pool sample) ##########")
    pool = pd.concat(
        [
            pd.read_csv(data_dir / "train" / f"train_source{s}.tsv", sep="\t", dtype=str,
                        keep_default_na=False, encoding="utf-8", nrows=1_500_000)
            for s in (2, 3)
        ],
        ignore_index=True,
    )
    print(f"  pool sample rows: {len(pool):,}")
    toks = (
        pool["business_name"].str.lower()
        .str.replace(r"[^a-z0-9 ]", " ", regex=True)
        .str.split()
        .explode()
    )
    toks = toks[toks.str.len() >= 3]
    dfc = toks.value_counts()
    n = len(pool)
    print(f"  distinct name tokens (len>=3): {len(dfc):,}")
    for thr in (10, 100, 1_000, 5_000, 20_000, 100_000):
        print(f"    tokens with df > {thr:>7,}: {(dfc > thr).sum():>9,} "
              f"covering {(dfc[dfc > thr].sum() / n):.1%} of rows")
    print("  most common name tokens:", dfc.head(20).to_dict())

    atoks = (
        pool["business_address"].str.lower()
        .str.replace(r"[^a-z0-9 ]", " ", regex=True)
        .str.split()
        .explode()
    )
    atoks = atoks[atoks.str.len() >= 4]
    adfc = atoks.value_counts()
    print(f"  distinct address tokens (len>=4): {len(adfc):,}")
    for thr in (1_000, 20_000, 100_000):
        print(f"    address tokens with df > {thr:>7,}: {(adfc > thr).sum():>9,}")
    print("  most common address tokens:", adfc.head(20).to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
