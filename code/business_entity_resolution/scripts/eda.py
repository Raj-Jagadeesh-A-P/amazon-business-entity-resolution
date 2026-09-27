"""Stage 1 EDA: profile the raw TSVs and inspect match/no-match structure.

Read-only. Writes nothing. Run:  python scripts/eda.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA_CANDIDATES = [
    ROOT / "dataset" / "student_resource" / "dataset",
    ROOT / "dataset",
]


def find_data_dir() -> Path:
    for cand in DATA_CANDIDATES:
        if (cand / "train").is_dir():
            return cand
    raise SystemExit("could not locate a dataset directory containing train/")


def profile_source(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
        na_filter=False,
    )
    print(f"\n=== {path.name} ===")
    print(f"  rows: {len(df):,}  columns: {list(df.columns)}")
    print(f"  unique entity_id: {df['entity_id'].nunique():,}")
    bad_prefix = ~df["entity_id"].str.startswith(path.name.split("_")[1].upper() + "-")
    print(f"  rows with unexpected id prefix: {int(bad_prefix.sum()):,}")
    for col in ("business_name", "business_address"):
        empty = (df[col].str.strip() == "").mean()
        print(f"  empty {col}: {empty:.1%}   mean len: {df[col].str.len().mean():.1f}")
    cc = df["country"].value_counts(dropna=False)
    print(f"  countries ({len(cc)} distinct):")
    for country, n in cc.head(12).items():
        print(f"      {country!r:24s} {n:>10,}  ({n / len(df):.1%})")
    if len(cc) > 12:
        print(f"      ... {len(cc) - 12} more")
    return df


def main() -> int:
    pd.set_option("display.width", 200)
    data_dir = find_data_dir()
    print(f"data dir: {data_dir}")

    frames = {}
    for split in ("train", "test"):
        for source in (1, 2, 3):
            path = data_dir / split / f"{split}_source{source}.tsv"
            frames[(split, source)] = profile_source(path)

    print("\n\n########## GROUND TRUTH ##########")
    gt = pd.read_csv(
        data_dir / "train" / "train_ground_truth.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    )
    print(f"  rows: {len(gt):,}  unique s1: {gt['source1_entity_id'].nunique():,}")
    lists = gt["matched_entity_ids"].str.split(",")
    sizes = lists.str.len()
    print(f"  match-list size distribution:\n{sizes.value_counts().sort_index().to_string()}")
    print(f"  singletons (size 0): {(sizes == 0).mean():.1%}")
    print(f"  S2 share of all matches: "
          f"{sum(l.count('S2-') for l in lists) / max(1, sizes.sum()):.1%}")
    print(f"  S3 share of all matches: "
          f"{sum(l.count('S3-') for l in lists) / max(1, sizes.sum()):.1%}")
    print(f"  entries with empty-string artefacts: "
          f"{sum(1 for l in lists if any(not t for t in l)):,}")

    # every S1 train entity present in ground truth?
    s1_train = set(frames[("train", 1)]["entity_id"])
    gt_ids = set(gt["source1_entity_id"])
    print(f"  S1 train entities: {len(s1_train):,}; in GT: {len(gt_ids):,}; "
          f"GT not in S1: {len(gt_ids - s1_train):,}; S1 not in GT: {len(s1_train - gt_ids):,}")

    # do matched ids exist in the source files?
    pool = set(frames[("train", 2)]["entity_id"]) | set(frames[("train", 3)]["entity_id"])
    all_matched = {mid for l in lists for mid in l}
    print(f"  distinct matched ids: {len(all_matched):,}; missing from pool: "
          f"{len(all_matched - pool):,}")

    # reverse direction: does a matched S2/S3 record serve >1 S1 entity?
    from collections import Counter

    owners = Counter(mid for l in lists for mid in l)
    multi = [v for v in owners.values() if v > 1]
    print(f"  matched records claimed by >1 S1 entity: {len(multi):,} "
          f"(max {max(owners.values()) if owners else 0})")

    # contention among S2/S3 themselves
    print("\n  within-source duplicate normalized names (first 20k rows of S2 train):")
    s2 = frames[("train", 2)].head(20000)
    norm = s2["business_name"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.strip()
    print(f"    {int(norm.duplicated().sum()):,} of {len(norm):,} share a name with another row")

    print("\n\n########## MATCHED PAIR SAMPLES ##########")
    s1 = frames[("train", 1)].set_index("entity_id")
    s2 = frames[("train", 2)].set_index("entity_id")
    s3 = frames[("train", 3)].set_index("entity_id")
    lookup = {**s2.to_dict("index"), **s3.to_dict("index")}

    rng = pd.Series(range(len(gt))).sample(400, random_state=7)
    for idx in rng[:14]:
        row = gt.iloc[idx]
        s1_id = row["source1_entity_id"]
        left = s1.loc[s1_id] if s1_id in s1.index else None
        print(f"\n--- {s1_id} ---")
        print(f"  S1: {left['business_name']!r} | {left['business_address']!r} | {left['country']!r}"
              if left is not None else "  S1: <missing>")
        for mid in row["matched_entity_ids"].split(",") if row["matched_entity_ids"] else []:
            r = lookup.get(mid)
            if r is None:
                print(f"  {mid}: <missing>")
            else:
                print(f"  {mid}: {r['business_name']!r} | {r['business_address']!r} | {r['country']!r}")

    print("\n\n########## SINGLETON SAMPLES ##########")
    singles = gt[sizes == 0]
    for idx in singles.sample(6, random_state=3).index:
        s1_id = gt.loc[idx, "source1_entity_id"]
        left = s1.loc[s1_id]
        print(f"  {s1_id}: {left['business_name']!r} | {left['business_address']!r} | {left['country']!r}")

    print("\n\n########## CROSS-SOURCE COUNTRY x MATCH RATE ##########")
    for source in (1, 2, 3):
        print(f"  train source{source} country mix: "
              f"{dict(frames[('train', source)]['country'].value_counts().head(5))}")

    print("\n\n########## NEAR-DUPLICATE SAMPLES (S2 train, blocked by name token) ##########")
    s2_full = frames[("train", 2)]
    norm2 = s2_full["business_name"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.strip()
    dupes = norm2[norm2.duplicated(keep=False) & (norm2 != "")]
    print(f"  duplicate-name S2 rows: {len(dupes):,}")
    for value in dupes.value_counts().head(5).index:
        sub = s2_full[norm2 == value]
        print(f"  name={value!r} -> {len(sub)} rows; sample: {sub['entity_id'].head(3).tolist()}")
        for _, r in sub.head(3).iterrows():
            print(f"      {r['entity_id']}: {r['business_address']!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
