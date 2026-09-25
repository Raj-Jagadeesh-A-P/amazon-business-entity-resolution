# amazon-business-entity-resolution
A machine learning project to match business records from different sources and identify which records belong to the same business.

## Folder structure

```
business-entity-resolution/
├── output/
│   ├── matching_results.tsv        # final matches (the only scored file)
│   └── candidate_pairs.tsv         # blocking candidate set fed to the model
│
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── preprocessing.py    # load + normalise the raw TSVs
│       │   ├── blocking.py         # candidate generation (sets the recall ceiling)
│       │   ├── features.py         # pairwise similarity features
│       │   ├── model.py            # train / score / threshold
│       │   ├── evaluation.py       # macro F_0.5, blocking recall, error analysis
│       │   └── pipeline.py         # CLI entry point
│       │
│       ├── README.md               # how to reproduce end-to-end
│       └── requirements.txt        # pinned dependencies
│
├── dataset/
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
├── utils/
│   └── validate_submission.py      # local format check before submitting
│
├── README_dataset.md               # full problem statement
├── Documentation_template.md       # methodology write-up template
└── .gitignore
```

`src/` is currently skeleton-only: signatures, contracts and TODOs, no logic yet.
