"""Stage: normalise a split into the Parquet cache.

Delegates to :func:`preprocessing.cache_split`, which owns the single-writer
Parquet session, the ``.partial`` staging file and the row-count verification. An
existing cache whose footer row count disagrees with the raw source is rebuilt
automatically, so a truncated file can never be mistaken for a finished one.

Run:  python scripts/build_cache.py --split train
      python scripts/build_cache.py --split test
      python scripts/build_cache.py --split train --force
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import preprocessing as pp  # noqa: E402

REPO = Path(__file__).resolve().parents[3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--cache-dir", default=str(REPO / "cache"))
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunksize", type=int, default=400_000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    paths = pp.cache_split(
        args.split,
        Path(args.cache_dir),
        sources=(1, 2, 3),
        data_dir=args.data_dir,
        chunksize=args.chunksize,
        limit=args.limit,
        force=args.force,
    )
    for path in paths:
        rows = pp.parquet_row_count(path)
        print(f"[cache] {path.name}: {rows:,} rows verified")
    print(f"[cache] {args.split} cache ready in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
