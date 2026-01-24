# Refactor Analysis Report

**Date**: 2026-01-23
**Analyzer**: refactor-cleaner agent
**Codebase**: CeliacRisks ML Pipeline (analysis/src/ced_ml/)
**Version**: 1.1.0

---

## Executive Summary

Analysis of the CeliacRisks codebase reveals opportunities for cleanup across **4 major categories**:

1. **Dead Code**: Unused functions, dependencies, and parameters
2. **High Complexity**: 44 functions exceeding complexity threshold (C901 > 10)
3. **Code Duplication**: Redundant patterns in CLI wrappers and evaluation logic
4. **Unused Parameters**: 34 function arguments that are never used

**Codebase Stats**:
- 61 Python files
- 311 function definitions
- 25,596 total lines of code
- 921 tests (65% coverage)
- 44 high-complexity functions
- 34 unused parameters

**Estimated Impact**:
- Remove ~500 lines of code
- Eliminate 4 dependencies (~80MB install size)
- Reduce 44 → <20 functions over complexity threshold
- Improve average file size from 420 → 320 lines

---

## 1. Dead Code

### 1.1 Unused Functions

#### `build_holdout_metadata` (utils/metadata.py:259-294)
**Status**: Exported but never called
**Evidence**: Only referenced in `utils/__init__.py` exports, never used in codebase
**Note in code**: "Currently unused. The holdout evaluation module does not generate plots."
**Recommendation**: Remove function and export (36 lines)

#### `save_pickle` and `load_pickle` (utils/serialization.py:90-103)
**Status**: Defined but never exported or used
**Evidence**: Not in `utils/__init__.py` exports, no callers found
**Recommendation**: Remove both functions (14 lines)

### 1.2 Unused Dependencies

#### plotly + kaleido (pyproject.toml:44-45)
**Status**: Listed as required dependencies but never imported
**Evidence**: No `import plotly` or `from plotly` found in codebase
**Impact**: ~50MB install footprint
**Recommendation**: Remove from dependencies

#### statsmodels (pyproject.toml:38)
**Status**: Listed as required dependency but never imported
**Evidence**: No `import statsmodels` found in codebase
**Recommendation**: Remove from dependencies

#### tqdm (pyproject.toml:43)
**Status**: Listed as required dependency but never imported
**Evidence**: No `import tqdm` found in codebase
**Recommendation**: Remove from dependencies

### 1.3 Unused Function Wrappers

#### `run_eval_holdout_with_config` (cli/eval_holdout.py:8-63)
**Status**: Never called in codebase
**Evidence**: Only `run_eval_holdout` is imported in `cli/main.py:314`
**Duplication**: 95% code overlap with `run_eval_holdout` (only difference is config loading)
**Recommendation**: Remove wrapper function (56 lines)

---

## 2. High Complexity Functions (C901 > 10)

### 2.1 Critical Mega-Functions

#### `run_aggregate_splits` (cli/aggregate_splits.py:1752)
**Complexity**: 108 (10.8x threshold)
**Lines**: ~813 lines in function
**File size**: 2,565 lines total
**Issue**: Single function orchestrates entire aggregation pipeline
**Recommendation**: Split into 8-10 smaller orchestration functions

#### `run_train` (cli/train.py:359)
**Complexity**: 78 (7.8x threshold)
**Lines**: ~1,300 lines in function
**File size**: 1,676 lines total
**Issue**: Single function handles entire training workflow
**Recommendation**: Extract into pipeline stages (data prep, feature selection, training, evaluation)

#### `plot_risk_distribution` (plotting/risk_dist.py:83)
**Complexity**: 62 (6.2x threshold)
**Lines**: ~550 lines in function
**Issue**: Single function creates 6-panel composite plot with extensive conditional logic
**Recommendation**: Split into panel-specific functions

#### `generate_aggregated_plots` (cli/aggregate_splits.py:1348)
**Complexity**: 36 (3.6x threshold)
**Issue**: Sequential plot generation with nested try-except blocks
**Recommendation**: Extract plot types into separate functions with error handling wrapper

#### `run_train_ensemble` (cli/train_ensemble.py:176)
**Complexity**: 30 (3.0x threshold)
**Recommendation**: Extract validation, training, and evaluation stages

### 2.2 Plotting Complexity

All plotting functions have high complexity due to multi-panel layouts and conditional styling:

| Function | File | Complexity | Recommendation |
|----------|------|------------|----------------|
| `_plot_logit_calibration_panel` | plotting/calibration.py:426 | 22 | Extract axis setup from plotting logic |
| `build_plot_metadata` | utils/metadata.py:18 | 23 | Simplify conditional metadata builder |
| `plot_roc_curve` | plotting/roc_pr.py:33 | 17 | Extract confidence interval plotting |
| `plot_dca_curve` | plotting/dca.py:120 | 15 | Extract annotation logic |
| `plot_model_comparison` | plotting/ensemble.py:116 | 15 | Separate data prep from plotting |

### 2.3 Data Processing Complexity

| Function | File | Complexity | Issue |
|----------|------|------------|-------|
| `prune_correlated_proteins` | features/panels.py:95 | 17 | Nested loops for correlation matrix |
| `_coerce_min_samples_leaf_list` | models/registry.py:169 | 16 | Type coercion with multiple branches |
| `collect_best_hyperparams` | cli/aggregate_splits.py:406 | 16 | Nested dict merging with validation |
| `compute_summary_stats` | cli/aggregate_splits.py:327 | 15 | Bootstrap CI computation across metrics |
| `validate_split_indices` | data/persistence.py:33 | 15 | Multi-condition validation logic |

---

## 3. Code Duplication

### 3.1 CLI Wrapper Pattern Duplication

**Pattern**: Every CLI command has two functions:
- `run_<command>(**kwargs)` - Direct function call
- `run_<command>_with_config(config_file, overrides, **kwargs)` - Config file wrapper

**Evidence**:
- `run_eval_holdout` vs `run_eval_holdout_with_config` (cli/eval_holdout.py)
- `run_aggregate_splits` vs `run_aggregate_splits_with_config` (cli/aggregate_splits.py)

**Issue**: 95% code duplication - only difference is config loading step

**Recommendation**: Create single decorator/helper to unify pattern:
```python
@with_config_loader(load_holdout_config)
def run_eval_holdout(**kwargs):
    # Main logic only
```

### 3.2 Logging Duplication

**Pattern**: Repetitive logging blocks in CLI functions:
```python
logger.info("Starting <operation>")
try:
    result = do_work()
    logger.info("<operation> complete")
    logger.info(f"Metric: {result['metric']:.4f}")
except Exception as e:
    logger.error(f"<operation> failed: {e}")
    raise
```

**Files affected**: All CLI modules (8 files)

**Recommendation**: Create context manager for operation logging

### 3.3 Scenario Validation Duplication

**Pattern**: Same validation logic repeated across modules:
```python
if scenario not in ["Incident", "IncidentPlusPrevalent"]:
    raise ValueError(...)
```

**Evidence**: Found in multiple modules

**Recommendation**: Centralize in `data/schema.py` as `validate_scenario(scenario)`

---

## 4. Unused Parameters

### 4.1 Unused Method Arguments (34 total)

#### High-Priority (Breaking API if removed)

**`scenario` parameter unused in 12 report methods** (evaluation/reports.py:366-660):
- All methods in report builder accept `scenario` but never use it
- Likely legacy parameter from earlier design
- **Recommendation**: Remove if not part of public API, otherwise deprecate

**CLI function unused params**:
- `cli/eval_holdout.py:82` - `kwargs` in `run_eval_holdout`
- `cli/main.py:469` - `ctx` in config commands
- `cli/train.py:293` - `logger` parameter
- `cli/aggregate_splits.py:1221` - `logger` parameter

#### Medium-Priority (Internal functions)

**Training complexity**:
- `models/training.py:512` - `random_state` in `_extract_selected_proteins`
- `models/optuna_search.py:308` - `fit_params` in `OptunaBayesSearch.fit`
- `models/stacking.py:636` - `scenario` in ensemble training

**Plotting unused params**:
- `plotting/oof.py:23-26` - 4 unused params (`scenario`, `seed`, `cv_folds`, `train_prev`)
- `plotting/calibration.py:875,878` - `bin_strategy`, `four_panel`
- `plotting/optuna_plots.py:33` - `plot_format`

**Data handling**:
- `data/columns.py:136` - `nrows` in column detection
- `data/persistence.py:116` - `n_splits` in split saver

#### Low-Priority (Intentional API design)

**Lambda/callback signatures**:
- `metrics/scorers.py:60` - `kwargs` in scorer lambda (matches sklearn API)
- `models/prevalence.py:163` - `X`, `y` in transform (matches sklearn API)

---

## 5. Unnecessary Complexity Patterns

### 5.1 Mega-Files

**Files exceeding 1000 lines**:

| File | Lines | Functions | Recommendation |
|------|-------|-----------|----------------|
| cli/aggregate_splits.py | 2,565 | 22 | Split into: metrics, plots, features, reports |
| cli/train.py | 1,676 | 11 | Split into: pipeline, training, evaluation |
| plotting/calibration.py | 1,054 | 14 | Split panels into separate files |

**Recommendation**: Target file size of 400-600 lines as per coding-style.md

### 5.2 Long Parameter Lists

**Functions with 10+ parameters**:
- `evaluate_holdout` (14 params) - evaluation/holdout.py:316
- `run_eval_holdout` (12 params) - cli/eval_holdout.py:65
- `run_aggregate_splits` (11 params) - cli/aggregate_splits.py:1752

**Recommendation**: Group related parameters into config objects (already exists for some)

### 5.3 Deep Nesting

**Functions with 5+ indentation levels**:
- `run_aggregate_splits` - 7 levels of nesting
- `run_train` - 6 levels of nesting
- `generate_aggregated_plots` - 5 levels of nesting

**Recommendation**: Use early returns and extract nested blocks into helper functions

---

## 6. Refactor Plan

### Phase 1: Quick Wins (Low Risk)

**Estimated Impact**: Remove ~150 lines, eliminate 3 dependencies

1. Remove unused functions:
   - `build_holdout_metadata` (36 lines)
   - `save_pickle`, `load_pickle` (14 lines)
   - `run_eval_holdout_with_config` (56 lines)

2. Remove unused dependencies from pyproject.toml:
   - plotly
   - kaleido
   - statsmodels
   - tqdm

3. Clean up unused parameters (34 occurrences):
   - Priority: `scenario` in reports.py methods (12 occurrences)
   - Priority: CLI wrapper `kwargs` parameters

**Tests Required**:
- Run full test suite (921 tests)
- Verify no import errors
- Check CLI help text unchanged

### Phase 2: Complexity Reduction (Medium Risk)

**Estimated Impact**: Improve maintainability of 2,565-line and 1,676-line files

1. Split `cli/aggregate_splits.py` (2,565 lines → 4 files @ ~640 lines each):
   - `aggregate_metrics.py` - metrics collection and summary stats
   - `aggregate_plots.py` - plot generation
   - `aggregate_features.py` - feature stability and consensus panels
   - `aggregate_main.py` - orchestration

2. Split `cli/train.py` (1,676 lines → 3 files @ ~560 lines each):
   - `train_pipeline.py` - data prep and feature selection
   - `train_model.py` - model training and calibration
   - `train_eval.py` - evaluation and reporting

3. Reduce complexity of mega-functions:
   - `run_aggregate_splits` (C901=108) - extract into 8-10 stage functions
   - `run_train` (C901=78) - extract into 6-8 stage functions
   - `plot_risk_distribution` (C901=62) - extract 6 panel functions

**Tests Required**:
- Full test suite must pass
- Add integration tests for new module boundaries
- No behavior changes allowed

### Phase 3: Eliminate Duplication (Medium Risk)

**Estimated Impact**: Remove ~200 lines of duplicated code

1. Create unified CLI config wrapper pattern:
   ```python
   # New: utils/cli_helpers.py
   def with_config_loader(config_loader_func):
       """Decorator to add config file loading to CLI function."""
   ```

2. Centralize scenario validation:
   ```python
   # Add to: data/schema.py
   def validate_scenario(scenario: str) -> None:
       """Validate scenario parameter."""
   ```

3. Create logging context manager:
   ```python
   # Add to: utils/logging.py
   @contextmanager
   def operation_logger(operation_name: str, logger=None):
       """Context manager for operation logging."""
   ```

**Tests Required**:
- All CLI commands must work identically
- Log output format unchanged
- Error messages preserved

---

## 7. Validation Tests Required

### Behavior Preservation Tests

**Before any changes**:
1. Run full test suite: `pytest tests/ -v` (921 tests must pass)
2. Generate baseline outputs:
   ```bash
   # Save current outputs for comparison
   ./run_local.sh --model LR_EN --split-seed 0
   cp -r results/LR_EN results/baseline_LR_EN
   ```

**After each refactor phase**:
1. Run full test suite (must maintain 921 passing tests, 65% coverage)
2. Run integration test with same seed:
   ```bash
   ./run_local.sh --model LR_EN --split-seed 0
   ```
3. Compare outputs (must be identical):
   ```bash
   # Metrics must match exactly
   diff results/LR_EN/metrics.json results/baseline_LR_EN/metrics.json

   # Predictions must match (allowing floating point tolerance)
   python scripts/compare_predictions.py \
       results/LR_EN/predictions.csv \
       results/baseline_LR_EN/predictions.csv \
       --tolerance 1e-10
   ```

### Regression Test Suite

**New tests needed** (add to `tests/test_refactor_validation.py`):

1. **Dead code removal validation**:
   - Test that removed functions are truly unreferenced
   - Test that removed dependencies don't break imports

2. **Complexity reduction validation**:
   - Integration tests for split modules (ensure same behavior)
   - Test that extracted functions preserve original logic

3. **Duplication removal validation**:
   - Test new CLI wrapper decorator with all commands
   - Test centralized scenario validation
   - Test logging context manager

---

## 8. Unresolved Questions

1. **`scenario` parameter in reports.py**: Is this part of a public API that external code depends on? If yes, deprecation warnings needed rather than removal.

2. **CLI wrapper functions**: Are the `_with_config` variants used by external scripts or only internally? If external, need deprecation path.

3. **Coverage target**: Current coverage is 65%. Should refactor aim to maintain or improve this? Some removed code (dead functions) will increase coverage automatically.

4. **plotly/statsmodels/tqdm dependencies**: Were these used previously and removed, or never used? Check git history for context on intended use.

5. **File splitting strategy**: Should split files maintain backward compatibility via `__init__.py` re-exports? This would preserve existing import paths.

6. **Complexity thresholds**: Current C901 threshold is 10. Should we increase to 15 for plotting/CLI orchestration functions, or enforce strict 10 everywhere?

---

## 9. Risk Assessment

### Low Risk (Phase 1)
- **Likelihood of breaking changes**: <5%
- **Impact if broken**: Low (test suite will catch)
- **Rollback difficulty**: Easy (single commit revert)

### Medium Risk (Phases 2-3)
- **Likelihood of breaking changes**: 10-15%
- **Impact if broken**: Medium (behavior changes in complex functions)
- **Rollback difficulty**: Moderate (multiple commits)
- **Mitigation**: Comprehensive integration tests with frozen outputs

### High Risk Areas
**Do not refactor without extensive validation**:
1. `oof_predictions_with_nested_cv` (models/training.py:73) - Core training logic
2. `prune_correlated_proteins` (features/panels.py:95) - Feature selection algorithm
3. Calibration functions - Statistical correctness critical

---

## 10. Deliverables Summary

### Code to Remove
- **Functions**: 4 functions, ~106 lines
- **Dependencies**: 4 packages (plotly, kaleido, statsmodels, tqdm)
- **Parameters**: 34 unused arguments

### Code to Simplify
- **Mega-files**: 3 files (split into 10 smaller modules)
- **Mega-functions**: 5 functions (extract into 30+ helpers)
- **Duplication**: ~200 lines of duplicate patterns

### Expected Outcomes
- **Line reduction**: ~500 lines removed
- **Complexity reduction**: 44 → <20 functions over C901 threshold
- **Dependency reduction**: 4 packages removed (~80MB install size)
- **Maintainability**: Average file size 320 lines (currently 420)
- **Test coverage**: Maintain 65% or improve to 70%+

---

## 11. Next Steps

1. **Review this analysis** with team/maintainer
2. **Answer unresolved questions** (Section 8)
3. **Create baseline** for regression testing (Section 7)
4. **Execute Phase 1** (low risk, high value)
5. **Re-assess** after Phase 1 before proceeding to Phases 2-3

---

**Analysis Complete**: 2026-01-23
**Agent ID**: a1352fc (for resuming)
