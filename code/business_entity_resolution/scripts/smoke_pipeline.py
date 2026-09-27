"""End-to-end smoke test of the whole chain on a small slice of real data.

Runs index -> candidates -> features -> train -> predict over the first few
thousand rows of every source, then validates both TSVs with the competition
validator. This is the check that the stages *compose*: it catches row-space
mismatches between stages, chunk-boundary errors, and output-format violations,
none of which any single-stage unit test would see.

The slice is built by normalising the head of each raw source, so the smoke test
exercises the real normalisation path rather than hand-written fixtures.

Run:  python scripts/smoke_pipeline.py [--s1 N] [--pool N]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import pipeline, preprocessing as pp  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
WORK = REPO / "artifacts" / "smoke"
S1_LIMIT = 3_000
POOL_LIMIT = 40_000
STAGES = (
    ("index", "train"), ("candidates", "train"), ("features", "train"),
    ("train", "train"),
    ("index", "test"), ("candidates", "test"), ("features", "test"),
    ("predict", "test"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1", type=int, default=S1_LIMIT,
                        help="Source-1 rows per split")
    parser.add_argument("--pool", type=int, default=POOL_LIMIT,
                        help="Source-2/3 rows per split")
    parser.add_argument("--max-candidates", type=int, default=50)
    parser.add_argument(
        "--pool-memmap", type=int, default=1, choices=[0, 1],
        help="passed through to the pipeline; 0 runs the pre-memmap pool path so "
             "the two can be diffed for identical outputs",
    )
    parser.add_argument(
        "--work", default=None,
        help="override the scratch directory, so two runs can be kept side by "
             "side for a diff (default artifacts/smoke)",
    )
    args = parser.parse_args()
    work = Path(args.work) if args.work else WORK
    if work.exists():
        shutil.rmtree(work)
    for sub in ("cache", "artifacts", "output"):
        (work / sub).mkdir(parents=True, exist_ok=True)

    test_dir = _build_slice(work, args.s1, args.pool)
    _install_trimmed_truth(work, args.s1, args.pool)

    try:
        for stage, split in STAGES:
            print(f"\n{'=' * 70}\n[smoke] stage {stage} (split={split})\n{'=' * 70}",
                  flush=True)
            started = time.time()
            code = pipeline.main([
                "--stage", stage, "--split", split,
                "--cache-dir", str(work / "cache"),
                "--artifacts", str(work / "artifacts"),
                "--output-dir", str(work / "output"),
                "--max-candidates", str(args.max_candidates),
                "--threads", "2",
                "--pool-memmap", str(args.pool_memmap),
            ])
            print(f"[smoke] stage {stage} rc={code} in {time.time() - started:.1f}s",
                  flush=True)
            if code != 0:
                return code
    finally:
        pp.override_ground_truth(None)

    return _validate_outputs(work, test_dir)


def _build_slice(work: Path, s1_limit: int, pool_limit: int) -> Path:
    """Normalise the head of each raw source into the smoke cache directory.

    Also writes the same rows back out as raw-shaped TSVs, because the competition
    validator reads ``test_source{1,2,3}.tsv`` from a test directory. Pointing it
    at the real directory would report every entity outside the slice as missing,
    which is noise rather than signal.
    """

    raw_dir = work / "test_dir"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        for source, limit in ((1, s1_limit), (2, pool_limit), (3, pool_limit)):
            chunks = list(
                pp.iter_source_chunks(split, source, chunksize=1000, limit=limit)
            )
            path = work / "cache" / f"{split}_source{source}.parquet"
            rows = pp._write_parquet_split(
                path, [pp._records_to_frame(c) for c in chunks]
            )
            print(f"[smoke] cache {path.name}: {rows:,} rows", flush=True)
            if rows == 0:
                raise SystemExit(f"[smoke] empty slice for {split} source {source}")
            if split == "test":
                _write_raw(raw_dir / f"test_source{source}.tsv", chunks)
    return raw_dir


def _write_raw(path: Path, chunks: list) -> None:
    """Write the slice back in the raw file's own column shape.

    Column names and order match the real ``test_source{n}.tsv`` so the
    competition validator can read this directory unchanged.
    """

    import pandas as pd

    rows = [
        {
            "entity_id": record.entity_id,
            "business_name": record.name_raw,
            "business_address": record.address_raw,
            "country": record.country,
        }
        for chunk in chunks for record in chunk
    ]
    frame = pd.DataFrame(
        rows, columns=["entity_id", "business_name", "business_address", "country"]
    ).drop_duplicates(subset="entity_id", keep="first")
    frame.to_csv(path, sep="\t", index=False, encoding="utf-8")
    print(f"[smoke] raw   {path.name}: {len(frame):,} rows", flush=True)


def _install_trimmed_truth(work: Path, s1_limit: int, pool_limit: int) -> None:
    """Restrict the labels to the slice, so recall and labels stay meaningful.

    Without this the sliced run would be scored against a pool that no longer
    contains the matched records, and every entity would look like a singleton.
    """

    import pandas as pd

    pool_ids: set[str] = set()
    for source, limit in ((2, pool_limit), (3, pool_limit)):
        for chunk in pp.iter_source_chunks("train", source, chunksize=5000, limit=limit):
            pool_ids.update(r.entity_id for r in chunk)
    s1_ids: set[str] = set()
    for chunk in pp.iter_source_chunks("train", 1, chunksize=5000, limit=s1_limit):
        s1_ids.update(r.entity_id for r in chunk)

    truth = pp.load_ground_truth()
    rows = []
    for entity_id in sorted(s1_ids):
        matches = [m for m in truth.get(entity_id, []) if m in pool_ids]
        rows.append((entity_id, ",".join(matches)))
    path = work / "trimmed_ground_truth.tsv"
    pd.DataFrame(
        rows, columns=["source1_entity_id", "matched_entity_ids"]
    ).to_csv(path, sep="\t", index=False, encoding="utf-8")
    kept = sum(1 for _, m in rows if m)
    print(
        f"[smoke] trimmed labels: {len(rows):,} entities, {kept:,} with a match "
        f"inside the {len(pool_ids):,}-record slice -> {path.name}",
        flush=True,
    )
    pp.override_ground_truth(path)


def _validate_outputs(work: Path, test_dir: Path) -> int:
    print(f"\n{'=' * 70}\n[smoke] validating outputs against {test_dir}\n{'=' * 70}")
    failures = 0
    validator = REPO / "utils" / "validate_submission.py"
    candidates = work / "output" / "candidate_pairs.tsv"
    matching = work / "output" / "matching_results.tsv"
    for name, path in (("candidate_pairs.tsv", candidates),
                       ("matching_results.tsv", matching)):
        if not path.exists():
            print(f"[smoke] MISSING {name}")
            failures += 1
            continue
        result = subprocess.run(
            [sys.executable, str(validator),
             "--candidate", str(candidates), "--matching", str(matching),
             "--test-dir", str(test_dir), "--check-ids"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        print(f"--- {name}: rc={result.returncode}")
        print((result.stdout or "").strip()[:3000])
        if result.stderr.strip():
            print("STDERR:", result.stderr.strip()[:2000])
        failures += result.returncode != 0
    if failures:
        print(f"\n[smoke] FAILED: {failures} output(s) invalid")
        return 1
    print("\n[smoke] PASSED: full chain green and both outputs valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
