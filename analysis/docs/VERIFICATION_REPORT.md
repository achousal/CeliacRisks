# Feature Report Bug Fix - Verification Report

**Date:** 2026-01-21
**Issue:** Feature reports not being generated
**Status:** ✓ Fixed

## Problem Summary

Feature report artifacts were not being generated despite `evaluation.feature_reports: true`. The output directory `results/split_seed*/reports/feature_reports/` was created but remained empty.

## Root Cause

### Feature Naming Mismatch

**File:** src/ced_ml/models/training.py:642

The code checked for `num__` prefix but actual feature names use `_resid` suffix:
- Expected: `num__{protein}` (from sklearn ColumnTransformer)
- Actual: `{protein}_resid` (from ResidualTransformer)

Result: Zero proteins extracted → empty feature reports

## Fix Applied

Updated 3 functions in training.py to handle both naming patterns:
1. `_extract_from_model_coefficients()` (lines 639-655)
2. `_extract_from_kbest_transformed()` (lines 575-593)
3. `_extract_from_rf_permutation()` (lines 708-724)

### Code Pattern

```python
# Before
if name.startswith("num__"):
    orig = name[len("num__") :]

# After
if name.startswith("num__"):
    orig = name[len("num__") :]
elif name.endswith("_resid"):
    orig = name[: -len("_resid")]
else:
    continue
```

## Verification

### Test Results

**Trained model (split_seed0):**
- Before: 0 proteins extracted
- After: 478 proteins extracted ✓

**Mock test:**
- ✓ Extracted 7/10 proteins with `_resid` suffix

**Unit tests:**
- ✓ All 4 extraction tests pass

## Next Steps

Re-run training to generate feature reports:
```bash
./run_local.sh
```

Expected outputs:
- `results/split_seed*/reports/feature_reports/{scenario}__{model}__feature_report_train.csv`
- `results/split_seed*/reports/stable_panel/{scenario}__stable_panel__KBest.csv`
- `results/split_seed*/reports/panels/{scenario}__{model}__final_test_panel.json`

---

**Status:** ✓ Ready for deployment
