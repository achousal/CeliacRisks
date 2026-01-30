# Refactoring Plan: ced_ml Codebase

**Created**: 2026-01-28
**Updated**: 2026-01-29 (audit: refactor-cleaner agent + Codex repo scan)
**Status**: Planned
**Scope**: Exhaustive complexity reduction and modularity improvements

---

## Current State

**Stats**: ~33,900 lines source code (73 files), ~31,000 lines tests (51 files), 1,271 tests, 14% coverage.

**38 source files exceed 400 lines; 20 exceed 600.** The top offenders (updated 2026-01-29):

| File | Lines | Key Issue |
|------|-------|-----------|
| `cli/train.py` | 2,100 | `run_train()` God Function (1,608 lines) |
| `models/training.py` | 1,150 | `oof_predictions_with_nested_cv()` 376 lines |
| `cli/main.py` | 1,119 | CLI orchestration |
| `plotting/calibration.py` | 1,073 | 7 functions avg 133 lines; `_plot_logit_calibration_panel` 418 lines |
| `models/stacking.py` | 1,019 | Mixed class + module functions |
| `cli/optimize_panel.py` | 963 | Duplicated single/aggregated logic; `run_optimize_panel_aggregated` 393 lines |
| `cli/train_ensemble.py` | 963 | `run_train_ensemble` 712 lines |
| `models/optuna_search.py` | 849 | `OptunaSearchCV` 20 methods, 793 lines |
| `config/schema.py` | 850+ | 9 potentially unused dataclasses |
| `features/rfe.py` | 824 | 287-line core function |
| `evaluation/reports.py` | 755 | God class (`ResultsWriter`, 23 save_* methods) |
| `plotting/risk_dist.py` | ~600 | `plot_risk_distribution` 554 lines |
| `cli/aggregate_splits.py` | ~550 | `run_aggregate_splits` 516 lines |

**Cross-cutting issues:**
- 123 `if config.*` checks (missing Strategy pattern) -- 42 in `cli/train.py` alone
- Path discovery logic duplicated in 4+ modules
- 129 scattered `pd.read_csv`/`to_csv` calls without centralized validation
- CLI args (`--infile`, `--split-dir`, `--split-seed`) repeated across 5+ commands
- Test suite: 49 files in flat directory, ~1,020 lines of duplicated fixtures (98 fixtures, only 2 in conftest)
- `cli/train.py` imports 24 internal modules (highest coupling in codebase)
- 51 potentially dead symbols across 18 files
- 14 distinct boilerplate patterns repeated across CLI commands (~394 lines duplicated)

---

## Phase 0: Dead Code and Script Hygiene (Very Low Risk)

**Goal**: Remove unused code and archive superseded scripts before structural refactoring.

**Dead code removals:**
- `utils/paths.py`: `get_run_dir`, `get_core_dir`, `get_preds_dir`, `get_diagnostics_dir`, `get_reports_dir` have zero callers (only defined + re-exported via `__init__.py`). Remove functions and drop from `__all__`.
- `utils/metadata.py`: Docstring references `build_holdout_metadata` which does not exist. Remove stale reference. Also remove `format_split_info` (zero callers).
- `config/loader.py`: 6 dead functions -- `load_holdout_config`, `load_panel_optimize_config`, `convert_paths`, `print_config_summary`, `format_dict`, `resolve_value`. Remove all.
- `utils/logging.py`: 3 dead symbols -- `cleanup_live_logs`, `LoggerContext`, `log_dict`. Remove all.
- `config/schema.py`: 9 dataclasses potentially dead (`PanelConfig`, `SVMConfig`, `RFConfig`, `XGBoostConfig`, `EvaluationConfig`, `DCAConfig`, `OutputConfig`, `StrictnessConfig`, `ComputeConfig`, `RootConfig`). **Verify YAML deserialization usage before removing** -- these may be instantiated dynamically. Also: `rfe_min_auroc_frac` field unused -- implement or mark deprecated.

**Script archival:**
- `scripts/setup_optimize_panel.py`: Superseded by `ced optimize-panel --run-id` auto-detection. Move to `scripts/_legacy/`.
- `scripts/hpc_optimize_panel.sh`: Duplicates run-id/model discovery already in `post_training_pipeline.sh`. Move to `scripts/_legacy/`.
- `scripts/validate_rfecv_fix.py`: One-off validation. Convert assertions into tests under `tests/features/`, then delete script.

**Archive hygiene:**
- `docs/investigations/_archive/`: Add a `README.md` marker or rename to `docs/archive/investigations/` to prevent confusion with active investigation workflows.

**Logging consolidation (extend `utils/logging.py`):**

Two new helpers to eliminate divergent logger setup across all 6 CLI commands:

`verbose_to_log_level(verbose: int) -> int`:
- Standardize two divergent strategies: arithmetic (`20 - verbose*10` at aggregate_splits:344, train_ensemble:291) vs explicit mapping (optimize_panel:184-188, consensus_panel:280-285).

`setup_command_logger(command, verbose, outdir, run_id, model, split_seed, logger_name) -> Logger`:
- Combines `verbose_to_log_level` + `auto_log_path` + `setup_logger` + "Logging to file" info message.
- Replaces 6-10 lines of boilerplate in every CLI command (train:592-600, run_pipeline:164-171, aggregate_splits:356-362, train_ensemble:295-301, optimize_panel:635-641, consensus_panel:288-293).

**CLI validation helper:**
- Extract common mutually-exclusive flag validation and auto-detect config checks into `cli/_validators.py` to reduce boilerplate in `cli/main.py`.

**Verify**: `pytest tests/ -v` passes. `ruff check src/` clean. No broken imports.

---

## Phased Plan

Phases are ordered by risk/impact ratio (high impact, low risk first). Each phase is independently deployable -- tests must pass after each.

### Phase 1: Test Infrastructure (Low Risk)

**Goal**: Safe refactoring foundation.

**1a. Consolidate fixtures in `conftest.py`**
- Move `minimal_proteomics_data`, `temporal_proteomics_data` (duplicated in 6+ files)
- Move `toy_binary_classification` (duplicated 3x)
- Replace 170 inline `rng = np.random.default_rng(42)` with existing `rng` fixture
- Add `make_fast_training_config()`, `make_fast_splits_config()` helpers
- Savings: ~1,020 lines

**1b. Reorganize test directory**
```
tests/
  conftest.py (expanded)
  data/          <- test_io, test_splits, test_persistence, test_filters, test_columns
  models/        <- test_registry, test_stacking, test_calibration, test_training, test_optuna
  features/      <- (expand existing)
  metrics/       <- test_discrimination, test_thresholds, test_dca, test_bootstrap
  evaluation/    <- test_predict, test_reports, test_holdout
  plotting/      <- test_roc_pr, test_calibration, test_risk_dist, test_dca, test_ensemble
  cli/           <- (expand existing)
  e2e/           <- split test_e2e_runner.py (1,956 lines) into workflow-specific files
```

**1c. Split mega test files (>1,000 lines)**
- `test_e2e_runner.py` (1,956) -> 4 workflow files in `e2e/`
- `test_e2e_run_id_workflows.py` (1,288) -> discovery + CLI integration
- `test_models_stacking.py` (1,173) -> meta_features + training + prediction

**Verify**: `pytest tests/ -v` -- 1,271 tests pass, count unchanged.

---

### Phase 2: CLI Boilerplate Extraction (Low Risk)

**Goal**: Eliminate ~394 lines of duplicated boilerplate across 6 CLI commands using the `auto_log_path()` pattern -- one helper function, all commands delegate.

**Design principle**: Prefer standalone functions over classes. Each helper is a single function with explicit parameters, no shared mutable state.

#### 2a. `utils/paths.py` -- 4 helpers

| Function | Signature | Extracted from |
|----------|-----------|---------------|
| `find_results_root` | `() -> Path` | aggregate_splits:60-81, train_ensemble:199-207, optimize_panel:72-77, consensus_panel:301-311 |
| `extract_run_id_from_path` | `(path: Path) -> str \| None` | aggregate_splits:346-353, optimize_panel:627-633 |
| `resolve_split_files` | `(split_dir, split_seed, scenario) -> tuple[Path, Path]` | optimize_panel:276-287, 756-765 |
| `find_stability_file` | `(aggregated_dir: Path) -> Path` | optimize_panel:651-663, consensus_panel:136-149 |

`find_results_root`: Check `CED_RESULTS_DIR` env -> project root / results -> cwd / results.
`resolve_split_files`: Try scenario-specific name, fall back to old format without scenario.
`find_stability_file`: Check panels/ -> panels/features/ -> reports/feature_reports/ (legacy).

#### 2b. `utils/discovery.py` -- 2 helpers

| Function | Signature | Extracted from |
|----------|-----------|---------------|
| `discover_models` | `(run_dir: Path, skip_ensemble=True) -> list[str]` | train_ensemble:224-241, optimize_panel:100-108, consensus_panel:78-86 |
| `detect_latest_run_id` | `(results_dir: Path) -> str` | aggregate_splits:109-120, train_ensemble:210-220, optimize_panel:82-91 |

`discover_models`: Scan for model subdirs, filter `.`, `investigations`, `consensus`, optionally `ENSEMBLE`.
`detect_latest_run_id`: Glob `run_*`, sort descending, return latest.

#### 2c. `utils/metadata.py` -- 2 helpers (extend existing module)

| Function | Signature | Extracted from |
|----------|-----------|---------------|
| `load_run_metadata` | `(run_dir: Path) -> dict \| None` | run_pipeline:246-253, consensus_panel:213-239 |
| `extract_paths_from_metadata` | `(metadata: dict) -> tuple[str\|None, str\|None]` | consensus_panel:213-239 |

`extract_paths_from_metadata`: Handle nested (`models.{first}.infile`) and flat (`infile`) formats.

#### 2d. `utils/model_loading.py` -- 1 helper

| Function | Signature | Extracted from |
|----------|-----------|---------------|
| `load_representative_bundle` | `(model_dir: Path, model_name: str) -> tuple[dict, int]` | optimize_panel:697-720, consensus_panel:380-404 |

Find first `split_seed*` dir, load `core/{model}__final_model.joblib`, validate dict. Returns (bundle, seed).

#### 2e. CLI-to-helper adoption matrix

Each command replaces inline boilerplate with helper calls. No behavior changes.

| Command | Helpers to adopt |
|---------|-----------------|
| `train.py` | `setup_command_logger` |
| `run_pipeline.py` | `setup_command_logger`, `load_run_metadata` |
| `aggregate_splits.py` | `setup_command_logger`, `find_results_root`, `detect_latest_run_id`, `extract_run_id_from_path` |
| `train_ensemble.py` | `setup_command_logger`, `find_results_root`, `detect_latest_run_id`, `discover_models` |
| `optimize_panel.py` | `setup_command_logger`, `find_results_root`, `detect_latest_run_id`, `discover_models`, `extract_run_id_from_path`, `resolve_split_files`, `find_stability_file`, `load_representative_bundle` |
| `consensus_panel.py` | `setup_command_logger`, `find_results_root`, `discover_models`, `find_stability_file`, `load_representative_bundle`, `load_run_metadata`, `extract_paths_from_metadata` |

Also extract from `models/stacking.py` lines 454-536 (`_find_model_split_dir`) and `cli/aggregation/discovery.py`.

**Tests**: Unit tests for each helper in `tests/utils/test_paths.py`, `test_discovery.py`, `test_metadata.py`, `test_model_loading.py`.

**Verify**: `pytest tests/ -v` passes. `grep` for old inline patterns -> 0 remaining.

---

### Phase 3: Break Up cli/train.py (Medium Risk, Highest Impact)

**Goal**: Reduce `run_train()` from 1,551 lines to <200.

**Create `cli/orchestration/` package**
```
cli/orchestration/
  __init__.py
  context.py           <- TrainingContext dataclass (shared state across stages)
  data_stage.py        <- load data, resolve columns, apply filters (~150 lines)
  split_stage.py       <- load/validate split indices (~100 lines)
  feature_stage.py     <- screening, feature selection strategy dispatch (~200 lines)
  training_stage.py    <- nested CV, OOF calibration (~250 lines)
  evaluation_stage.py  <- evaluate on train/val/test, bootstrap CIs (~200 lines)
  plotting_stage.py    <- all plot generation calls (~150 lines)
  persistence_stage.py <- all file writes, metadata JSON (~200 lines)
```

**Strategy pattern for feature selection** (replaces ~31 config checks):
```python
class FeatureSelectionStrategy(Protocol):
    def select(self, X, y, config) -> list[str]: ...

class HybridStabilityStrategy: ...
class RFECVStrategy: ...
class NoneStrategy: ...
```

**Refactored cli/train.py (~250 lines)**:
```python
def run_train(...):
    ctx = TrainingContext.from_config(config, cli_args)
    ctx = load_data(ctx)
    ctx = load_splits(ctx)
    ctx = select_features(ctx)
    ctx = train_models(ctx)
    ctx = evaluate(ctx)
    save_artifacts(ctx)
    generate_plots(ctx)
```

**Verify**: `pytest tests/ -v` passes. `ced train --help` unchanged. Local smoke test.

---

### Phase 4: ResultsWriter Refactoring (Medium Risk)

**Goal**: Split 24-method God class into focused writers.

```
evaluation/
  reports.py            <- ResultsWriter facade (~150 lines, delegates to below)
  writers/
    __init__.py
    metrics_writer.py   <- save_val_metrics, save_test_metrics, save_cv_repeat_metrics, save_bootstrap_ci_metrics
    predictions_writer.py <- save_test/val/train_oof/controls predictions
    feature_writer.py   <- save_feature_report, save_stable_panel_report, save_panel_manifest
    artifacts_writer.py <- save_model_artifact, save_run_settings, save_best_params
    diagnostics_writer.py <- save_calibration_curve, save_learning_curve, summarize_outputs
```

`ResultsWriter` stays as a facade -- zero caller changes.

**Verify**: `pytest tests/test_reports.py -v` passes. `grep ResultsWriter` -> all usages work.

---

### Phase 5: I/O Consolidation (Low Risk)

**Goal**: Centralize 111+ scattered `pd.read_csv`/`to_csv` calls.

**Create `data/io_helpers.py` (~200 lines)**:
```python
def read_predictions(path, required_cols=None) -> pd.DataFrame: ...
def save_predictions(df, path, index=False) -> None: ...
def read_feature_report(path) -> pd.DataFrame: ...
def save_feature_report(df, path) -> None: ...
def read_metrics(path) -> pd.DataFrame: ...
```

**Files modified**: cli/train.py, cli/aggregate_splits.py, cli/aggregation/collection.py, models/stacking.py, evaluation/reports.py

---

### Phase 6: Split Oversized Source Modules (Low Risk)

**Goal**: No source file >600 lines.

| Module | Lines | Split Into |
|--------|-------|-----------|
| `models/training.py` | 1,150 | + `models/nested_cv.py` (fold execution, OOF aggregation) |
| `models/stacking.py` | 1,019 | + `models/stacking_utils.py` (module-level functions) |
| `features/rfe.py` | 824 | + `features/rfe_engine.py` (core elimination loop) |
| `cli/optimize_panel.py` | 963 | Deduplicate single vs aggregated via shared helpers (-150 lines) |
| `plotting/calibration.py` | 1,073 | + `plotting/calibration_reliability.py` |
| `cli/train_ensemble.py` | 963 | Split `run_train_ensemble` (712 lines) into staged helpers |
| `plotting/risk_dist.py` | ~600 | Extract subplot helpers from `plot_risk_distribution` (554 lines) |
| `cli/aggregate_splits.py` | ~550 | Split `run_aggregate_splits` (516 lines) into collect + report |
| `models/optuna_search.py` | 849 | Split `OptunaSearchCV` (20 methods, 793 lines) into search + callbacks |

---

### Phase 7: CLI Argument Groups (Very Low Risk)

**Goal**: DRY up repeated CLI args across 5+ commands.

**Create `cli/common_args.py`** with reusable option groups:
- `data_options`: --infile, --split-dir, --split-seed
- `output_options`: --verbose, --outdir
- `model_options`: --model, --run-id, --config

**Verify**: All `ced <cmd> --help` outputs unchanged.

---

### Phase 8: Strategy Pattern for Config-Driven Code (Medium Risk)

**Goal**: Replace remaining ~80 `if config.*` checks with polymorphism.

**Calibration strategy**:
```python
class CalibrationStrategy(Protocol):
    def calibrate(self, y_true, y_prob, cv_splits) -> CalibratedModel: ...
# Implementations: IsotonicCalibration, SigmoidCalibration, OOFPosthocCalibration, NoCalibration
```

**Threshold strategy**:
```python
class ThresholdStrategy(Protocol):
    def find_threshold(self, y_true, y_prob) -> float: ...
# Implementations: FixedSpecificityThreshold, YoudensJThreshold
```

**Verify**: `pytest tests/ -v` passes. Config YAML interface unchanged.

---

### Phase 9: Schema and Hyperparams Module Split (Low Risk)

**Goal**: Break up `config/schema.py` (850+ lines) and `models/hyperparams.py` into themed modules.

**`config/schema.py`** -> themed schema files:
- `config/data_schema.py` -- data loading, splits, columns
- `config/features_schema.py` -- feature selection, screening, stability
- `config/calibration_schema.py` -- calibration and prevalence
- `config/thresholds_schema.py` -- threshold strategies
- `config/ensemble_schema.py` -- ensemble and stacking

Re-export from `config/schema.py` for backward compatibility (facade).

**`models/hyperparams.py`** -> per-model modules with registry:
- `models/hyperparams_lr.py`, `hyperparams_rf.py`, `hyperparams_xgb.py`, `hyperparams_svm.py`
- Small registry in `hyperparams.py` dispatching to per-model modules

**Verify**: `pytest tests/ -v` passes. Config YAML loading unchanged. `from ced_ml.config.schema import *` still works.

---

## Execution Dependencies

```
Phase 0 (dead code) --> Phase 1 (tests) --> Phase 2 (paths) --> Phase 3 (train.py) --> Phase 4 (reports)
                                                                                    |-> Phase 5 (I/O)
                                                                                    |-> Phase 6 (splits)
Phase 7 (CLI args) -- independent, anytime after Phase 3
Phase 8 (strategies) -- after Phase 3
Phase 9 (schema split) -- independent, anytime after Phase 0
```

## Expected Outcomes

| Metric | Before | After |
|--------|--------|-------|
| Largest file | 2,100 lines | ~600 lines |
| God Functions (>200 lines) | 7 | 0 |
| God Classes (>15 methods) | 2 (`ResultsWriter` 23, `OptunaSearchCV` 20) | 0 |
| Dead symbols | 51 | 0 |
| Duplicated path discovery | 4 implementations | 1 |
| CLI boilerplate (14 patterns) | ~394 lines | ~120 lines shared helpers |
| Config if-checks | 123 | ~30 |
| Scattered CSV I/O | 129 | centralized |
| Test fixture duplication | ~1,020 lines | ~0 |
| Test organization | Flat (49 files) | Hierarchical (7 dirs) |

## End-to-End Verification

After all phases:
1. `pytest tests/ -v` -- all 1,271 tests pass
2. `ced --help` -- CLI interface unchanged
3. `ced train --config configs/training_config.yaml --model LR_EN --split-seed 0 --infile ../data/input.parquet` -- smoke test
4. `ruff check src/ tests/` -- no lint errors
5. `black --check src/ tests/` -- formatting clean
