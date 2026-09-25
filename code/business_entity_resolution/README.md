# Business Entity Resolution — pipeline code

Source code for the business entity resolution solution: given business records
from three independent, noisy sources with no shared identifiers, decide which
records refer to the same real-world business.

This folder is self-contained. With the `dataset/` folder alongside it, it
regenerates both submission files from scratch.

## Layout

```
business_entity_resolution/
├── src/
│   ├── preprocessing.py   load + normalise the raw TSVs
│   ├── blocking.py        candidate generation (sets the recall ceiling)
│   ├── features.py        pairwise similarity features
│   ├── model.py           train / score / threshold / resolve competition
│   ├── evaluation.py      macro F_0.5, blocking recall, error analysis
│   └── pipeline.py        CLI entry point
├── README.md
└── requirements.txt
```

## Data flow

```
dataset/{train,test}/*_source{1,2,3}.tsv
    │
    ├─ preprocessing  normalise names + addresses, stream records
    ├─ blocking       candidate pairs per Source-1 entity
    ├─ features       pairwise similarity features
    ├─ model          train on train split, score candidates, threshold
    └─ output         output/matching_results.tsv
                      output/candidate_pairs.tsv
```

`pipeline.py` is the only entry point; the other modules are library code and
are safe to import individually.

## Status

**Skeleton only.** Every function is declared with its signature, contract and
`TODO` markers — no logic is implemented yet. The docstrings carry the
constraints the implementation has to satisfy (dataset facts, the scoring
metric, the output format rules), so they are worth reading before writing code.

## Requirements

Python 3.10+. Install with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r code/business_entity_resolution/requirements.txt
```

The version pins are a starting point rather than a tested lockfile — verify the
set resolves for your interpreter before relying on it:
    pip install --dry-run -r requirements.txt
Re-pin from the versions that actually install, then commit the result.

## Running

```bash
# full chain: normalise -> block -> features -> train -> infer -> output
python -m src.pipeline --stage all

# candidate generation only (fast; good for iterating on blocking quality)
python -m src.pipeline --stage block --max-candidates 50

# reuse a cached model to score the test candidates
python -m src.pipeline --stage infer --model artifacts/model.txt

# quick smoke run over a subset
python -m src.pipeline --stage all --limit 20000
```

Flags to support: `--stage`, `--data-dir`, `--output-dir`, `--model`,
`--max-candidates`, `--validation-fraction`, `--seed`, `--limit`.

## Outputs

Both files are tab-separated UTF-8, written with
`to_csv(sep="\t", index=False, encoding="utf-8")`. Note that the ID-list columns
contain commas and the address field contains commas, so the tab separator and
`quoting` behaviour matter.

| File | Columns | Scored |
| --- | --- | --- |
| `output/matching_results.tsv` | `source1_entity_id`, `matched_entity_ids` | yes, on the leaderboard |
| `output/candidate_pairs.tsv` | `source1_entity_id`, `candidate_entity_ids` | no; used to audit blocking |

Every Source-1 entity in the test set needs exactly one row in each file, with an
empty ID list when it has no match. Final matches must be a subset of the
candidates.

## Validating output

From the repository root (this is where `dataset/` and `utils/` live):

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Exit code 0 means the files are safe to submit; 1 lists the issues to fix. Pass
`--check-ids` to also verify that every matched/candidate ID exists in the test
Source-2/3 files; that check is off by default because it loads all Source-2/3
IDs at once.

## Evaluation

There is no test-set ground truth, so performance is measured on a validation
split of the training data, held out **by Source-1 entity** (never by pair — a
pair-level split leaks, since a Source-2 record that truly matches a validation
entity is often also a true match of a training one).

The leaderboard metric is **macro F_0.5** over Source-1 entities:

```
F_0.5 = (1.25 * P * R) / (0.25 * P + R)
```

computed per entity, then averaged across all entities. Two things follow, and
they drive most of the design:

- **Singletons count fully.** An entity with no true match scores 1.0 for an
  empty prediction and 0.0 for any prediction, and such entities are a meaningful
  share of the training set — so correctly abstaining is worth real points.
- **The score is entity-level.** Pairwise F1 is not a proxy for it. Tune the
  threshold against macro F_0.5, not against pairwise metrics.

## Performance notes

- Stream or spill to disk. No in-memory cross-joins, and no step that assumes a
  whole source file can be loaded at once.
- Use a disk-backed index for blocking blocks rather than one large dict.
- Keep edit-distance features on a C implementation (`rapidfuzz`) and precompute
  per-record vectors so per-pair work stays cheap.
- Make each stage independently runnable and restartable, with cached artefacts
  between stages.

## Reproducibility

Fixed seeds throughout, sorted iteration order, and a fixed feature column order
shared between training and inference (a mismatch there silently misaligns
features while still producing plausible-looking scores). Same input should
produce byte-identical output.
