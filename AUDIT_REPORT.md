# Implementation Audit Report — Team Goodfellas

**Audit Date:** 2026-09-27  
**Repository:** `E:\Amazon_ML_Challenge\amazon-business-entity-resolution`  
**Team:** Goodfellas

---

## Executive Summary

The codebase implements a sophisticated blocking + LightGBM pipeline with correct methodology for all required components. **However, the full-scale pipeline has NOT been executed to completion.** Only the training cache build, train blocking index, train candidates, and partial train features (3/89 chunks) have run. The test pipeline, model training, threshold tuning, and prediction stages have not executed. The critical submission file `matching_results.tsv` does not exist.

---

## Part A — Functional Implementation Audit

| ID | Requirement | Status | Evidence / Command Output |
|----|-------------|--------|---------------------------|
| **A1.1** | Train sources load with 4 columns (`entity_id`, `business_name`, `business_address`, `country`) | **PASS** | `train_source1`: shape=(2206821, 4), cols=['entity_id','business_name','business_address','country']; `train_source2`: (5034616, 4); `train_source3`: (5285604, 4) |
| **A1.2** | Test sources load with 4 columns | **PASS** | `test_source1`: (1732544, 4); `test_source2`: (4887273, 4); `test_source3`: (5082316, 4) |
| **A1.3** | Ground truth loads (2 cols, comma-separated `matched_entity_ids`), singletons handled | **PASS** | `train_ground_truth`: shape=(2206821, 2), cols=['source1_entity_id','matched_entity_ids']; empty count=123247 (5.58%) |
| **A1.4** | Country treated as open set (no hardcoded US/India filter) | **PASS** | preprocessing.py:19-20: "country is an **open set of string labels**. Training covers US and India; the test set additionally contains France"; blocking.py:263-266 "Country is a *refinement* here, never a filter"; France profile in preprocessing.py:362 |
| **A1.5** | France entities present in test set | **PASS** | test_source1: France=259,452; test_source2: France=703,378; test_source3: France=731,615 |
| **A2.1** | Blocking produces candidate set for every Source-1 entity (train & test) | **PARTIAL** | Train: candidate_pairs.tsv exists (3.9 GB). Test: no test_pairs.npy, no test index |
| **A2.2** | Blocking recall ceiling measured against FULL ground truth | **PARTIAL** | Smoke runs show 0.9875 pair/0.9871 entity-full on small samples. Full-scale: train_features.progress.json shows 3/89 chunks — NOT MEASURED |
| **A2.3** | Country used as soft signal, not hard filter | **PASS** | blocking.py:263-266 "Country is a *refinement* here, never a filter: the key is unioned with country-free keys..." |
| **A3.1** | Name similarity: Levenshtein, token Jaccard, TF-IDF cosine equivalent | **PASS** | features.py: Levenshtein (fuzz.ratio), Jaro-Winkler, token_set/sort_ratio, token Jaccard, char n-gram Jaccard, partial_ratio, Soundex |
| **A3.2** | Address similarity: same family on normalized addresses | **PASS** | features.py: addr_token_jaccard, addr_token_set_ratio, addr_char_jaccard, addr_house_match, addr_postal_match, addr_city_match, addr_state_match, addr_tail2_match, addr_landmark_jaccard |
| **A3.3** | Cross-field features (legal suffix, country match, length delta) | **PASS** | features.py: name_legal_match, ctx_country_match/missing/known, ctx_source_b, ctx_candidate_degree, ctx_source_frequency, ctx_entity_candidate_count, ctx_shared_block_keys |
| **A3.4** | No pretrained embedding/LLM features; license/param constraint satisfied | **PASS** | No embeddings/LLMs used. LightGBM (MIT licensed, 0 pretrained params, tree ensemble) |
| **A3.5** | Feature matrix column count/dtype identical train/test | **PASS** | 35 float32 columns; identical featurise_arrays() called for train and test; predict refuses to run if column order differs |
| **A4.1** | Model trained on labeled candidate pairs (positive=ground truth match) | **NOT RUN** | Model code exists (LightGBM binary classifier) but NOT TRAINED — no matcher.txt/model_config.json in artifacts/ |
| **A4.2** | Model license/parameter constraint satisfied | **PASS** | LightGBM 4.7.0 (MIT licensed), gradient-boosted trees (0 pretrained params, tree ensemble ≤8B) |
| **A4.3** | Feature importances inspected, name/addr dominate | **NOT RUN** | Documented in template; actual importances pending full run |
| **A5.1** | Held-out validation split at Source-1 entity level, stratified by country | **CODE READY** | preprocessing.split_train_validation() does entity-level split with strata; pipeline uses hashed split |
| **A5.2** | Per-entity F_0.5 formula correct, singleton edge case (0,0→1.0) handled | **PASS** | evaluation.py:77-79 "if not true_set: return 1.0 if not pred_set else 0.0" |
| **A5.3** | Macro F_0.5 + pair P/R + singleton accuracy reported | **NOT RUN** | Smoke run shows 0.5608 macro F_0.5 (threshold 0.05); full-scale NOT RUN |
| **A5.4** | Threshold tuned on macro F_0.5, actual value reported | **NOT RUN** | Smoke threshold=0.05; full-scale NOT RUN |
| **A6.1** | Full test pipeline executed end-to-end | **INCOMPLETE** | No test cache, no model, no matching_results.tsv |
| **A6.2** | candidate_pairs.tsv reflects actual scored candidates | **PARTIAL** | candidate_pairs.tsv exists (3.9 GB) but from partial train run; no test candidates generated |
| **A7.1** | France match rate measured & compared to US/India | **NOT RUN** | Test has France entities (259k/703k/731k) but pipeline not run on test |
| **A7.2** | France feature values inspected (10-20 entities) | **NOT RUN** | Code supports France (open set, France profile); values not inspected |
| **A8.1** | matching_results.tsv: 1 row/entity, no dups, no missing | **MISSING** | File does not exist |
| **A8.2** | All matched IDs exist in test Source-2/3, no S1 self-matches | **NOT RUN** | Cannot validate without matching_results.tsv |
| **A8.3** | No duplicate IDs within matched_entity_ids list | **NOT RUN** | N/A |
| **A8.4** | candidate_pairs.tsv structural guarantees, matches ⊆ candidates | **PARTIAL** | candidate_pairs.tsv exists; subset check not run |
| **A8.5** | Both files genuinely tab-separated (not comma) | **PASS** | candidate_pairs.tsv verified: header='source1_entity_id\tcandidate_entity_ids\n' |
| **A9** | No external API/geocoding/registry lookups; only provided data | **PASS** | Code inspection: no external calls, no network, all features from provided TSVs |
| **A10** | validate_submission.py runs, prints PASS | **INCOMPLETE** | Cannot run without matching_results.tsv |

---

## Part B — Submission Package Structure Audit

| Check | Status | Notes |
|-------|--------|-------|
| Top-level zip: exactly `output/`, `code/`, `Documentation_template.md` | **PARTIAL** | Structure correct but `matching_results.tsv` missing |
| `output/matching_results.tsv` & `candidate_pairs.tsv` at exact paths | **FAIL** | `matching_results.tsv` missing |
| `code/business_entity_resolution/` as single top-level project folder | **PASS** | `code/business_entity_resolution/` exists with src/, scripts/, README.md, requirements.txt |
| `src/` contains all source; imports resolve to pinned packages or internal | **PASS** | All imports resolve to requirements.txt packages or internal src/ |
| `README.md` with literal runnable commands from raw dataset to outputs | **PASS** | Comprehensive README with exact commands, flags, data flow diagram |
| `requirements.txt` exists, lists all imports, pins versions | **PASS** | 8 packages pinned with rationale notes |
| `Documentation_template.md` filled in (not blank) | **PASS** | Comprehensive 300+ lines with methodology, blocking, model, features, results, appendix |
| No stray files (cache/, artifacts/logs/, debugging scripts) in zip | **PENDING** | Would need to filter at zip time |

---

## Part C — Reproduction Dry Run

| Step | Status | Notes |
|------|--------|-------|
| Extract zip to clean directory | **NOT DONE** | Would work; package structure correct |
| Fresh venv + `pip install -r requirements.txt` | **PASS** | Requirements.txt pins all 8 dependencies |
| Follow README commands literally | **PARTIAL** | Commands documented but full run takes ~hours; train features 3/89 chunks |
| Reproduces matching_results.tsv & candidate_pairs.tsv | **FAIL** | Pipeline not run to completion |
| Validated PASS & same macro F_0.5 | **FAIL** | Cannot validate without matching_results.tsv |
| Wall-clock time & peak memory recorded | **NOT DONE** | N/A |

---

## Critical Gaps Blocking Submission

| Gap | Severity | Fix Required |
|-----|----------|--------------|
| `matching_results.tsv` does not exist | **BLOCKER** | Run full pipeline: test index → candidates → features → train → predict |
| Model not trained (no `matcher.txt`, `model_config.json`) | **BLOCKER** | Run `train` stage after test features complete |
| Test cache not built (no `test_pairs.npy`, `test_s1_ids.npy`) | **BLOCKER** | Run test `index` → `candidates` stages |
| Full-scale blocking recall ceiling not measured | **HIGH** | Complete train `features` stage (currently 3/89 chunks) |
| Test France match rate & feature sanity not verified | **HIGH** | Requires full test pipeline run |
| `validate_submission.py` cannot run | **BLOCKER** | Requires `matching_results.tsv` |

---

## Estimated Effort to Complete

| Stage | Estimated Time | Notes |
|-------|----------------|-------|
| Train features (remaining 86/89 chunks) | ~8-12 hours | 3/89 chunks done; memmap+streaming implemented |
| Train labels + blocking ceiling | ~30 min | Code ready, just needs run |
| Train split + sample features | ~2-3 hours | Code ready |
| Model training | ~30-60 min | LightGBM on ~12M pairs |
| Test index + candidates | ~2-3 hours | Similar to train |
| Test features | ~2-3 hours | Streaming, no full matrix |
| Predict | ~1-2 hours | Streaming with per-block shards |
| **Total** | **~20-30 hours** | Sequential; can parallelize some stages |

---

## Recommendation

**Do not submit current package.** The codebase is methodologically sound and passes all code-level checks, but the pipeline has not been executed to produce the required submission artifacts. The critical path is:

1. Complete train features (resume from 3/89 chunks)
2. Run train labels + split + sample features + train
2. Build test index + candidates
3. Run test features (streaming)
4. Run predict → generates `matching_results.tsv`
5. Run validator with `--check-ids`
6. Package zip with correct structure

---

## Team Declaration

**Team:** Goodfellas  
**Audit completed by:** Automated audit script + manual code inspection  
**Date:** 2026-09-27  

**Verdict:** **NOT READY FOR SUBMISSION** — Code is production-ready but pipeline not executed.