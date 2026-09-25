"""Command-line entry point: raw TSVs in, scored submission files out.

Stages
------
``--stage all`` (default) runs the full chain:

    dataset/test/*.tsv
        -> preprocessing   normalise names/addresses, stream records
        -> blocking        candidate pairs (sets the recall ceiling)
        -> features        pairwise similarity features
        -> model           train on train data, score candidates, threshold
        -> output          output/matching_results.tsv
                           output/candidate_pairs.tsv

``--stage block`` stops after candidate generation and writes
``output/candidate_pairs.tsv`` — the useful checkpoint for iterating on blocking
quality before paying for model training. ``--stage train`` fits the model and
caches it. ``--stage infer`` scores an existing model against the test candidates
and writes ``output/matching_results.tsv``.

Output contract (enforced by ``utils/validate_submission.py``)
---------------------------------------------------------------
Both files are tab-separated UTF-8, no index column, no quoting:

* ``output/matching_results.tsv`` — columns ``source1_entity_id`` and
  ``matched_entity_ids``; the only file scored on the leaderboard.
* ``output/candidate_pairs.tsv`` — columns ``source1_entity_id`` and
  ``candidate_entity_ids``; the **last** stage of candidate generation, i.e.
  exactly the set the model scored.

Both need one row per Source-1 entity in the test set, including entities with no
match (empty ID list). Missing rows, duplicate rows, duplicate IDs inside a list,
Source-1 self-matches, or IDs absent from the test Source-2/3 files are
rejections, not merely lost points. Final matches must be a subset of the
candidates.

Conventions
-----------
* Stream rather than load. Make each stage independently runnable and
  restartable, caching artefacts to disk between stages.
* The test set contains ``France``, which never appears in training. Nothing here
  may filter, hard-code, or one-hot on a fixed country list.
* Keep the run deterministic — fixed seeds, sorted iteration — so a rerun
  reproduces byte-identical output and any score change is attributable to a real
  change.

Usage
-----
    python -m src.pipeline --stage all
    python -m src.pipeline --stage block --max-candidates 50
    python -m src.pipeline --stage infer --model artifacts/model.txt

TODO: wire the stages together, then add argument parsing, artefact paths and
progress logging.
"""

from __future__ import annotations

# TODO: replace these placeholders with imports from the sibling modules once the
# implementation lands, e.g. `from . import preprocessing, blocking, features,
# model, evaluation`. Keep this entry point importable from the repository root
# without side effects.

from typing import Dict, List, Optional

DEFAULT_TRAIN_DIR = "dataset/train"
DEFAULT_TEST_DIR = "dataset/test"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_MODEL_DIR = "artifacts"

MATCHING_FILENAME = "matching_results.tsv"
CANDIDATE_FILENAME = "candidate_pairs.tsv"

STAGES = ("all", "block", "train", "infer", "evaluate")


def run_preprocessing(split: str, data_dir: str) -> Dict[str, object]:
    """Normalise the records for one split and cache them to disk.

    Returns whatever handle the later stages consume — a path to the cached
    normalised records plus basic counts, not the full in-memory dataset.

    TODO: call :func:`preprocessing.load_all_sources` in chunks and write the
    normalised output to a cache file so later stages do not re-normalise on
    every run.
    """

    raise NotImplementedError


def run_blocking(
    split: str,
    normalised: Dict[str, object],
    max_candidates_per_entity: int = 50,
    keys: Optional[List[str]] = None,
) -> Dict[str, object]:
    """Generate the candidate set and write ``output/candidate_pairs.tsv``.

    Also logs the statistics from :func:`blocking.blocking_statistics` and, when
    run against the training split, the recall ceiling from
    :func:`evaluation.blocking_recall_ceiling` — that ceiling is the number that
    says whether the model is worth tuning yet.

    TODO: call :func:`blocking.generate_candidates` and
    :func:`blocking.write_candidate_pairs` using a disk-backed block store.
    """

    raise NotImplementedError


def run_training(
    train_dir: str,
    validation_fraction: float = 0.1,
    model_path: Optional[str] = None,
) -> Dict[str, object]:
    """Fit the classifier on the training split and cache the model.

    Builds labelled pairs from the training ground truth restricted to the
    training partition, tunes the decision threshold on the held-out validation
    entities, and reports the validation macro F_0.5. Everything downstream
    inherits this threshold, so report the threshold itself and not just a score.

    TODO: call :func:`model.build_training_pairs`, :func:`model.train_classifier`
    and :func:`model.tune_threshold`, then :func:`model.save_model`.
    """

    raise NotImplementedError


def run_inference(
    split: str,
    model_path: str,
    data_dir: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> str:
    """Score the test candidates and write ``output/matching_results.tsv``.

    Returns the output path. Guarantees one row per Source-1 entity in the test
    set, empty ID list included — so iterate the *test* entity list as the driver
    and fill in predictions, rather than iterating the predictions and hoping
    nothing is missing.

    TODO: featurise the test candidates, score with :func:`model.predict_proba`,
    resolve per-entity lists with :func:`model.resolve_competition`, and write the
    file tab-separated with ``index=False``.
    """

    raise NotImplementedError


def main(argv: Optional[List[str]] = None) -> int:
    """Parse arguments, run the requested stages, and return a process exit code.

    Arguments: ``--stage`` (one of :data:`STAGES`), ``--data-dir``,
    ``--output-dir``, ``--model``, ``--max-candidates``,
    ``--validation-fraction``, ``--seed``, and ``--limit`` for a quick smoke run
    over a subset of the data.

    Return 0 on success and non-zero on failure, so the pipeline can be driven
    from a shell script.

    TODO: implement argument parsing, stage dispatch, and progress logging with
    elapsed time per stage.
    """

    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
