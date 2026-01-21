# Celiac ML Pipeline Test Suite

This test suite validates the integrity of the Celiac Disease ML pipeline (release 1.1.0). With only **148 incident cases** in the dataset, even small bugs could compromise the entire study.

## Test Results

```
================== 72 passed, 1 skipped in 4.77s ==================
```

| Status | Count |
|--------|-------|
| Passed | 72 |
| Skipped | 1 |
| **Total** | **73** |

The single skipped test (`test_real_data_zero_missing_claim`) requires the actual dataset file and can be run manually when needed.

---

## Why These Tests Exist

### The Core Problem

Building ML models for rare disease prediction (0.34% prevalence) presents unique challenges:

1. **Limited positive samples**: Only 148 incident cases total
2. **High cost of errors**: Incorrect splits or data leakage invalidates the study
3. **Silent failures**: Many bugs produce plausible but wrong results
4. **Reproducibility**: Clinical research must be reproducible

These tests act as guardrails against subtle bugs that could silently corrupt results.

---

## Test Categories

### 1. Data Quality (`test_data_quality.py`) - 15 tests

**Purpose:** Prevent reverse causality and validate outcome encoding.

| Test | What It Catches |
|------|-----------------|
| `test_incident_only_excludes_prevalent` | Including prevalent cases would detect current disease, not predict future risk |
| `test_binary_encoding_correct` | Wrong encoding (Controls=1, Incident=0) would invert the model |
| `test_uncertain_controls_removed_after_filtering` | Controls with diagnosis dates are ambiguous and must be removed |
| `test_prevalence_realistic` | Unrealistic prevalence suggests data corruption |

**Why it matters:** Prevalent cases have biomarkers measured after diagnosis. Including them would create models that detect current disease instead of predicting future risk.

---

### 2. Missing Data (`test_missing_data.py`) - 12 tests

**Purpose:** Ensure the proteomic matrix stays fully observed and demographic missingness is handled deterministically.

| Test | What It Catches |
|------|-----------------|
| `test_zero_missing_in_proteins_synthetic` | Missing proteins would require imputation, introducing bias |
| `test_protein_zscored` | Non-standardized proteins would dominate the model |
| `test_metadata_complete_after_filtering` | Missing age/BMI after filtering indicates broken row filters |
| `test_eid_unique_and_complete` | Duplicate participant IDs would cause data leakage |

**Why it matters:** The pipeline assumes zero missing entries across the 2,920 protein columns. If that contract breaks, downstream models silently produce biased predictions.

---

### 3. Split Generation (`test_save_splits.py`) - 18 tests

**Purpose:** Ensure valid, reproducible train/test/holdout splits.

| Test | What It Catches |
|------|-----------------|
| `test_no_data_leakage_all_splits` | Same sample in train and test = inflated performance |
| `test_split_coverage_complete` | Missing samples = lost statistical power |
| `test_stratification_preserves_prevalence` | With 148 cases, unstratified splits could have 0 cases in test set |
| `test_same_seed_same_splits` | Non-reproducible splits = non-reproducible science |
| `test_holdout_in_full_space` | Index misalignment = wrong samples retrieved |

**Why it matters:** With 148 incident cases split 70/30, each fold has ~44 cases. Any split error significantly impacts statistical power.

---

### 4. Categorical Encoding (`test_categorical_encoding.py`) - 10 tests

**Purpose:** Validate handling of missing ethnicity as "Missing" category.

| Test | What It Catches |
|------|-----------------|
| `test_missing_becomes_own_category` | Dropping 17% of subjects loses 7,674 samples |
| `test_missing_category_signal_detection` | If missingness correlates with outcome (MNAR), it carries predictive signal |
| `test_train_test_consistency_with_missing` | Different encoding in train vs test = features don't match |
| `test_onehot_encoder_with_missing_category` | Encoder must handle "Missing" like any other category |

**Why it matters:** 17% of subjects have missing ethnicity. Treating "Missing" as its own category keeps those subjects and lets the model learn whether missingness itself is informative.

---

### 5. Row Filter Sync (`test_row_filter_sync.py`) - 11 tests

**Purpose:** Ensure `save_splits.py` and `celiacML_faith.py` use identical filtering.

| Test | What It Catches |
|------|-----------------|
| `test_shared_filter_import` | Different filter functions = split indices reference WRONG subjects |
| `test_filter_produces_same_counts` | Different row counts = silent data corruption |
| `test_index_space_consistency` | Non-sequential indices after filtering = indexing errors |
| `test_scenario_definitions_match` | IncidentOnly including Prevalent = reverse causality |

**Why it matters:** If split generation and model training use different row filters, train/test indices point to wrong subjects. This is nearly impossible to detect without explicit tests.

---

### 6. Prevalence Adjusted Model (`test_prevalence_adjusted_model.py`) - 3 tests

**Purpose:** Validate the prevalence adjustment wrapper for deployment.

| Test | What It Catches |
|------|-----------------|
| `test_prevalence_adjusted_model_matches_adjust_function` | Wrapper must match standalone adjustment function |
| `test_predict_uses_adjusted_probabilities` | `predict()` must use calibrated probabilities, not raw |
| `test_joblib_roundtrip_preserves_behavior` | Serialized model must produce same predictions |

**Why it matters:** Training prevalence (~0.34%) differs from real-world prevalence. The wrapper adjusts predictions so a "2% predicted risk" truly means ~2% probability.

---

## Running the Tests

### Standard Run
```bash
cd analysis
python3 -m pytest tests/ -v
```

### With Timing
```bash
python3 -m pytest tests/ -v --durations=10
```

### Specific Test File
```bash
python3 -m pytest tests/test_data_quality.py -v
```

### Single Test
```bash
python3 -m pytest tests/test_data_quality.py::TestOutcomeLabels::test_incident_only_excludes_prevalent -v
```

### Real Data Test (manual)
```bash
python3 -m pytest tests/test_missing_data.py::TestRealDataMissingness -v --no-header -rN
```

---

## Test Fixtures

Shared fixtures are defined in `conftest.py`:

| Fixture | Description |
|---------|-------------|
| `synthetic_celiac_data` | 1,010 samples (1000 controls, 5 incident, 5 prevalent), 100 proteins |
| `synthetic_celiac_data_with_missing` | Same as above with 5% missing introduced |
| `sample_splits` | Pre-computed train/test/holdout splits with filtering applied |
| `project_root`, `data_dir` | Path helpers |

The synthetic data mimics real data properties:
- Realistic prevalence (~0.5%)
- 3 "uncertain controls" (Controls with CeD_date)
- Top 10 proteins have Cohen's d ~ 1.5 signal in cases
- Z-scored protein values (mean ~0, std ~1 in controls)

---

## Critical Invariants

These properties MUST always hold:

1. **No prevalent cases in IncidentOnly analysis** - Prevents reverse causality
2. **Train/test/holdout are mutually exclusive** - Prevents data leakage
3. **Split indices are valid for filtered dataset** - Prevents indexing errors
4. **Row filters match between split generation and training** - Prevents silent corruption
5. **Proteins have zero missing values** - Imputation would introduce bias
6. **Model serialization preserves predictions** - Deployment reliability

---

## Adding New Tests

Guidelines:

1. **Test critical assumptions** - What would silently break if wrong?
2. **Use descriptive names** - `test_incident_only_excludes_prevalent` tells you what it tests
3. **Add docstrings** - Explain WHY the test matters, not just what it does
4. **Use fixtures** - Don't duplicate test data setup
5. **Test edge cases** - What happens with minimal data? Empty data?

---

## Dependencies

```
pytest>=7.0.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
statsmodels>=0.13.0
joblib>=1.0.0
```

Install:
```bash
pip install pytest numpy pandas scikit-learn statsmodels joblib
```
