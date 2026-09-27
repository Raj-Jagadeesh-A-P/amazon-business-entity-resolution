"""Drive the whole pipeline end-to-end over the full train and test splits.

Runs the stages in the only order that works, logging each one and failing fast:

    train: index -> candidates -> features -> train
    test:  index -> candidates -> features -> predict

Two things make this a script rather than a shell one-liner:

* **Disk staging.** The feature matrices are the largest artefacts in the project
  -- about 42 GB for train and 33 GB for test at the default candidate cap. They
  are not both needed at once, because nothing reads the train matrix after the
  model is fitted, so the train matrix is deleted once ``train`` finishes. Without
  that the two would have to coexist and the run would not fit on disk.
* **Restartability.** Each stage is skipped if its outputs are already present,
  so an interrupted run resumes instead of restarting. ``--force`` re-runs a
  stage regardless, and ``--from`` resumes the sequence at a named stage.

Run:  python scripts/run_full.py [--max-candidates 200] [--max-block 500]
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

REPO = Path(__file__).resolve().parents[3]
CODE = REPO / "code" / "business_entity_resolution"
CACHE = REPO / "cache"
ARTIFACTS = REPO / "artifacts"
OUTPUT = REPO / "output"
LOGS = ARTIFACTS / "logs"

#: (stage, split, sentinel file that proves the stage completed, freeme_after)
#: ``freeme_after`` names a file to delete once the stage is done -- only ever a
#: large regenerable intermediate.
PLAN: List[Tuple[str, str, Sequence[Path], Sequence[Path]]] = [
    ("index",      "train", [CACHE / "index_train" / "index_meta.json"], []),
    ("candidates", "train", [CACHE / "train_pairs.npy",
                             CACHE / "train_pool_ids.npy"], []),
    ("features",   "train", [CACHE / "train_features.npy",
                             CACHE / "train_labels.npy"], []),
    ("train",      "train", [ARTIFACTS / "matcher.txt",
                             ARTIFACTS / "model_config.json"],
                  [CACHE / "train_features.npy"]),
    ("index",      "test",  [CACHE / "index_test" / "index_meta.json"], []),
    ("candidates", "test",  [CACHE / "test_pairs.npy",
                             CACHE / "test_pool_ids.npy"], []),
    ("features",   "test",  [CACHE / "test_features.npy"], []),
    ("predict",    "test",  [OUTPUT / "matching_results.tsv",
                             OUTPUT / "candidate_pairs.tsv"], []),
]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-candidates", type=int, default=200,
                        help="candidates kept per Source-1 entity (default 200)")
    parser.add_argument("--max-block", type=int, default=500,
                        help="postings per blocking block (default 500)")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--from", dest="start_at", default=None,
                        help="resume the plan at this stage name")
    parser.add_argument("--force", action="store_true",
                        help="re-run stages even if their outputs exist")
    args = parser.parse_args(argv)

    LOGS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    begin = 0
    if args.start_at:
        names = [f"{s}:{sp}" for s, sp, _, _ in PLAN]
        if args.start_at not in names:
            parser.error(f"--from must be one of: {', '.join(names)}")
        begin = names.index(args.start_at)
        print(f"[run] resuming at {args.start_at}", flush=True)

    overall = time.time()
    for stage, split, sentinels, cleanup in PLAN[begin:]:
        tag = f"{stage}:{split}"
        if not args.force and all(p.exists() for p in sentinels):
            print(f"[run] {tag}: already complete, skipping", flush=True)
        else:
            _run_stage(stage, split, args)
        for path in cleanup:
            _free(path)

    print(
        f"[run] all stages complete in {(time.time() - overall) / 60:.1f} min",
        flush=True,
    )
    return _report()


def _run_stage(stage: str, split: str, args: argparse.Namespace) -> None:
    """Run one pipeline stage as a subprocess, streaming its log to disk.

    A subprocess rather than an in-process call so that a stage which exhausts
    memory takes only itself down, and so the log is a complete record even if the
    stage is killed.
    """

    log = LOGS / f"{stage}_{split}.log"
    command = [
        sys.executable, "-m", "src.pipeline",
        "--stage", stage, "--split", split,
        "--cache-dir", str(CACHE),
        "--artifacts", str(ARTIFACTS),
        "--output-dir", str(OUTPUT),
        "--max-block", str(args.max_block),
        "--max-candidates", str(args.max_candidates),
        "--threads", str(args.threads),
        "--seed", str(args.seed),
    ]
    print(f"\n[run] {' '.join(command)}\n[run] logging to {log}", flush=True)
    started = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as handle:
        result = subprocess.run(
            command, cwd=str(CODE), stdout=handle,
            stderr=subprocess.STDOUT, env=_env(),
        )
    elapsed = (time.time() - started) / 60
    if result.returncode != 0:
        tail = "\n".join(log.read_text(
            encoding="utf-8", errors="replace").splitlines()[-25:])
        raise SystemExit(
            f"[run] stage {stage} ({split}) failed after {elapsed:.1f} min "
            f"(exit {result.returncode}). Tail of {log}:\n{tail}"
        )
    print(f"[run] {stage}:{split} finished in {elapsed:.1f} min", flush=True)


def _env() -> dict:
    return {**os.environ, "PYTHONIOENCODING": "utf-8"}


def _free(path: Path) -> None:
    """Delete a large regenerable intermediate, reporting the space reclaimed."""

    if not path.exists():
        return
    size = path.stat().st_size / 1e9
    path.unlink()
    gc.collect()
    print(f"[run] freed {size:.1f} GB: {path.name}", flush=True)


def _disk_free() -> float:
    usage = shutil.disk_usage(REPO)
    return usage.free / 1e9


def _report() -> int:
    print("\n" + "=" * 70, flush=True)
    for path in sorted(OUTPUT.glob("*.tsv")):
        rows = sum(1 for _ in path.open(encoding="utf-8")) - 1
        print(f"[run] {path.name}: {rows:,} rows, "
              f"{path.stat().st_size / 1e6:.0f} MB", flush=True)
    config = ARTIFACTS / "model_config.json"
    if config.exists():
        print(f"\n[run] model config:\n{config.read_text(encoding='utf-8')}",
              flush=True)
    ceiling = ARTIFACTS / "blocking_ceiling_train.json"
    if ceiling.exists():
        print(f"\n[run] blocking ceiling:\n{ceiling.read_text(encoding='utf-8')}",
              flush=True)
    print(f"\n[run] disk free: {_disk_free():.1f} GB", flush=True)

    print("\n[run] validating against the real test set ...", flush=True)
    result = subprocess.run(
        [sys.executable, str(REPO / "utils" / "validate_submission.py"),
         "--matching", str(OUTPUT / "matching_results.tsv"),
         "--candidate", str(OUTPUT / "candidate_pairs.tsv"),
         "--test-dir", str(REPO / "dataset" / "student_resource" / "dataset" / "test"),
         "--check-ids"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=_env(),
    )
    print(result.stdout or result.stderr, flush=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
