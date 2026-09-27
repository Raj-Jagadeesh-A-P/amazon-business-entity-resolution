# amazon-business-entity-resolution

A machine learning project to match business records from three independent,
noisy sources with no shared identifiers, and decide which records refer to the
same real-world business.

## Folder structure

```
amazon-business-entity-resolution/
├── output/
│   ├── matching_results.tsv        # final matches (the scored file)
│   └── candidate_pairs.tsv         # blocking candidate set fed to the model
│
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── preprocessing.py    # normalise + cache the raw TSVs
│       │   ├── blocking.py         # candidate generation (sets the recall ceiling)
│       │   ├── features.py         # 35 pairwise similarity features
│       │   ├── model.py            # train / score / threshold / resolve contention
│       │   ├── evaluation.py       # macro F_0.5, blocking recall, error analysis
│       │   └── pipeline.py         # CLI entry point
│       │
│       ├── scripts/                # cache builder, blocking sweeps, tests
│       ├── README.md               # full reproduction guide
│       └── requirements.txt        # pinned dependencies
│
├── dataset/student_resource/dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   │
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
│
├── cache/                          # generated: normalised records + blocking indexes
├── artifacts/                      # generated: reports, fitted model, sweep results
├── utils/
│   └── validate_submission.py      # local format check before submitting
│
├── README_dataset.md               # full problem statement
├── Documentation_template.md       # methodology write-up
└── .gitignore
```

`cache/` and `artifacts/` are generated. Everything in them can be rebuilt from
the raw TSVs; see `code/business_entity_resolution/README.md` for the commands and
the order.

## Approach in one paragraph

Records are normalised (transliteration, legal-form stripping, structured
address parsing into house number / street / city / state / postal code), then
**blocked** with seven key families over those normalised fields so that only
plausible pairs are compared — blocking is what makes a 10.3M-record pool
tractable and it sets the recall ceiling, so it is measured rather than assumed.
Surviving candidate pairs are scored by a LightGBM model over 35 features
(weighted name, address and cross-field similarities, plus blocking context), and
the decision threshold is tuned directly against **entity-level macro F₀.₅** on a
validation split held out by Source-1 entity. Because a single Source-2/3 record
can only belong to one business, matches are made one-to-many from Source-1 but
contended pool records are resolved to a single winning claimant.

## Status

Implemented and validated end-to-end on a data slice
(`python scripts/smoke_pipeline.py` runs all six stages and passes both outputs
to the competition validator). The full-scale run is not yet complete: see
*Next steps* below.

The scoring metric is macro F₀.₅ over Source-1 entities, where a singleton
entity scores 1.0 for an empty prediction. 5.58% of training entities are
singletons, so abstaining is worth real points and threshold tuning targets the
entity-level metric directly.

## Measured so far

Blocking recall on 100,000 sampled Source-1 entities against the full 10.3M-record
training pool, 7 strategies:

| `max_block` | cap 50 | cap 200 | cap 400 | cap 800 |
| --- | --- | --- | --- | --- |
| 500 — pair / entity-full recall | .814 / .589 | .881 / .712 | .905 / .763 | .913 / .782 |
| 20,000 — pair / entity-full recall | .847 / .640 | .890 / .722 | .906 / .756 | .919 / .784 |

`max_block=500` is the default: the 20,000 blocks buy 0.4% more recall for 3.3x
the candidates per entity, which is almost entirely noise.

## Next steps

1. Run the full-scale pipeline on the complete train and test splits.
2. Fit the model, tune the threshold, and record validation macro F₀.₅.
3. Validate both outputs with `utils/validate_submission.py --check-ids`.
4. Replace the exact set-intersection features with fixed-width MinHash sketches.
   Featurisation measures 16.1k pairs/s and does not thread-scale, so this is the
   binding constraint on candidate volume rather than on model quality.
5. Complete `Documentation_template.md` with the full-scale results.
