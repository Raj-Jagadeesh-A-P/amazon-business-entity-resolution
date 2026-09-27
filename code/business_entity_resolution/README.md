# Business Entity Resolution — pipeline code

Given business records from three independent, noisy sources with no shared
identifiers, decide which records refer to the same real-world business.

## Layout

```
business_entity_resolution/
├── src/
│   ├── preprocessing.py   normalise + cache the raw TSVs
│   ├── blocking.py        candidate generation (sets the recall ceiling)
│   ├── features.py        pairwise similarity features
│   ├── model.py           train / score / threshold / resolve contention
│   ├── evaluation.py      macro F_0.5, blocking recall, error analysis
│   └── pipeline.py        CLI entry point
├── scripts/
│   ├── build_cache.py           rebuild the normalised caches
│   ├── measure_blocking_recall.py   blocking recall / cap sweeps
│   ├── test_normalisation.py     normalisation regression cases
│   ├── test_pipeline.py          stage-invariant regression tests
│   └── smoke_pipeline.py         end-to-end run over a real data slice
├── README.md
└── requirements.txt
```

`pipeline.py` is the only entry point; the other modules are library code and
are safe to import individually.

## Data flow

```
dataset/student_resource/dataset/{train,test}/*_source{1,2,3}.tsv
    │
    ├─ preprocessing  normalise names + addresses -> cache/*.parquet
    ├─ blocking       candidate pairs per Source-1 entity -> index_{split}/
    ├─ candidates     per-entity candidate cap + context stats
    ├─ features       35 pairwise features -> {split}_features.npy (memmap)
    ├─ model          LightGBM + entity-level threshold tuning
    └─ predict        score test candidates, resolve, write both TSVs
```

Every stage writes its outputs to `cache/` and `artifacts/` and reads them back,
so stages are independently runnable and restartable. Re-running a stage reuses
whatever is already on disk unless `--rebuild` is passed.

## Running

Run stages individually from `code/business_entity_resolution/`:

```bash
# 1. normalise + cache (only needed once; verifies row counts on write)
python scripts/build_cache.py --split train
python scripts/build_cache.py --split test

# 2. blocking index over the Source-2 + Source-3 pool
python -m src.pipeline --stage index --split train --max-block 500
python -m src.pipeline --stage index --split test  --max-block 500

# 3. candidates + context features
python -m src.pipeline --stage candidates --split train --max-candidates 200
python -m src.pipeline --stage candidates --split test  --max-candidates 200

# 4. featurise
python -m src.pipeline --stage features --split train
python -m src.pipeline --stage features --split test

# 5. fit and tune the threshold, 6. write the submission
python -m src.pipeline --stage train
python -m src.pipeline --stage predict --output-dir ../../output
```

`build_cache.py` also takes `--data-dir`, `--cache-dir`, `--limit`, `--chunksize`
and `--force`. `--limit` normalises only the head of each source, which is the
quickest way to test a normalisation change without a full 30-minute pass.

### Flags

| Flag | Default | Notes |
| --- | --- | --- |
| `--stage` | required | `index`, `candidates`, `features`, `train`, `predict` |
| `--split` | `train` | `train` or `test` |
| `--cache-dir` | `<repo>/cache` | normalised caches, indexes, pair arrays |
| `--artifacts` | `<repo>/artifacts` | reports and the fitted model |
| `--output-dir` | `<repo>/output` | the two submission TSVs |
| `--max-block` | `500` | postings per blocking block; **lowering is destructive** |
| `--max-candidates` | `200` | candidates kept per Source-1 entity |
| `--max-train-pairs` | `12,000,000` | fit budget; `0` uses every pair |
| `--max-valid-pairs` | `4,000,000` | threshold-tuning budget; `0` uses every pair |
| `--threads` | `4` | LightGBM `num_threads` |
| `--seed` | `42` | also selects the entity-level validation split |
| `--rebuild` | off | regenerate the blocking index even if one exists |

The data directory is located by search, not by flag: `preprocessing.resolve_data_dir()`
probes `dataset/student_resource/dataset` (the real layout) before the flatter
`dataset/` the top-level README describes. Both work; the nested one is correct.

## Tests

```bash
python scripts/test_normalisation.py     # name/address normalisation cases
python scripts/test_pipeline.py          # stage-invariant regression tests
python scripts/smoke_pipeline.py         # full chain on a real slice + validator
```

`smoke_pipeline.py` is the one that matters. It normalises the head of every real
source, runs all six stages, and passes both outputs to the competition
validator — which is what catches the bugs that unit tests cannot see, because
they are all row-space mismatches *between* stages. Override the slice size with
`--s1` / `--pool`.

## Tuning blocking

`scripts/measure_blocking_recall.py` measures the recall ceiling against the
training labels for a sample of Source-1 entities against the full pool:

```bash
python scripts/measure_blocking_recall.py --sample 100000 --sweep
python scripts/measure_blocking_recall.py --sample 100000 --sweep-cap
```

Measured on 100k entities against the full 10.3M-record pool (7 strategies,
`name_phonetic` excluded — leave-one-out showed it changed pair recall by 0.0000
while costing 24.6M postings):

| `max_block` | cap 50 | cap 100 | cap 200 | cap 400 | cap 800 |
| --- | --- | --- | --- | --- | --- |
| 500 — pair / full recall | .814 / .589 | .849 / .649 | .881 / .712 | **.905 / .763** | .913 / .782 |
| 20,000 — pair / full recall | .847 / .640 | .869 / .681 | .890 / .722 | .906 / .756 | .919 / .784 |

At `max_block=20,000` the cap-800 setting costs a mean of 789 candidates per
entity against 238 at `max_block=500`, for 0.4% more recall. The large blocks are
almost entirely extra noise, so `max_block=500` is the default.

Note that tightening is one-way: a 500-capped index cannot be widened back to
20,000, because the dropped postings are gone rather than merely hidden. The index
records the cap it was built at in `cache/index_*/index_meta.json` and raises if
asked to widen, so a sweep cannot silently report recall for the wrong
configuration.

## Outputs

Both files are tab-separated UTF-8 written with `index=False`. The ID-list columns
contain commas and the address field contains commas, so the tab separator matters.

| File | Columns | Scored |
| --- | --- | --- |
| `output/matching_results.tsv` | `source1_entity_id`, `matched_entity_ids` | yes, on the leaderboard |
| `output/candidate_pairs.tsv` | `source1_entity_id`, `candidate_entity_ids` | no; used to audit blocking |

Every Source-1 entity in the test set needs exactly one row in each file, with an
empty ID list when it has no match. Final matches must be a subset of the
candidates.

## Validating output

From the repository root:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/student_resource/dataset/test \
    --check-ids
```

Exit code 0 means the files are safe to submit; 1 lists the issues to fix.
`--check-ids` also verifies that every matched/candidate ID exists in the test
Source-2/3 files; it is off by default because it loads all 10M IDs at once.

## Evaluation

There is no test-set ground truth, so performance is measured on a validation
split of the training data, held out **by Source-1 entity** (never by pair — a
pair-level split leaks, since a Source-2 record that truly matches a validation
entity is often also a true match of a training one). The split is hashed from
the entity ID, so it is stable across runs and independent of pair ordering.

The leaderboard metric is **macro F_0.5** over Source-1 entities:

```
F_0.5 = (1.25 * P * R) / (0.25 * P + R)
```

computed per entity, then averaged across all entities. Two things follow, and
they drive most of the design:

- **Singletons count fully.** An entity with no true match scores 1.0 for an
  empty prediction and 0.0 for any prediction. 5.58% of training entities are
  singletons, so correctly abstaining is worth real points.
- **The score is entity-level.** Pairwise F1 is not a proxy for it. The threshold
  is tuned against macro F_0.5 directly, never against pairwise metrics.

## Design notes worth knowing before changing things

**Memory is the binding constraint, not CPU.** The pool is 10.3M records and the
candidate set runs to hundreds of millions of pairs. Three things follow, all of
which look like over-engineering until you hit them:

- The normalised records live in fixed-width byte arrays, not Python objects.
  As objects they cost several GB of per-string overhead; as `S` arrays, about
  one byte per character.
- The feature matrix is written through `np.lib.format.open_memmap`, never
  allocated in RAM. At 35 float32 columns and a mean of 200 candidates per
  entity it is tens of gigabytes.
- `run_predict` filters to threshold-passing pairs *before* building any
  container, and resolves contention with a pair of vectorised sorts. A claims
  dict of every candidate would be ~3.5x10^8 `(str, float)` tuples.

**Featurisation does not parallelise with threads.** The set-intersection features
are pure Python and hold the GIL; measured at 16.1k pairs/s serial and 1.00x with
16 threads. Process workers would each need their own copy of the ~6.1 GB pool
projection, which does not fit either. The real fix is replacing the exact set
intersections with fixed-width MinHash sketches, which is not implemented.

**Training samples pairs, but never samples away a positive.** Positives are
~0.4% of candidate pairs, so uniform sampling would starve the model. Negatives
are drawn uniformly, which is a fair picture of what inference actually sees.

**The `country` field is open-set.** Test contains `France`, which does not appear
in training. Country is never used to filter or hard-code; features record what
is known and what is not.

## Reproducibility

Fixed seeds throughout, sorted iteration order, and a fixed feature column order
shared between training and inference — a mismatch there silently misaligns
features while still producing plausible-looking scores, so `predict` refuses to
run if the stored column order differs from the current feature set. The same
input produces byte-identical output.
