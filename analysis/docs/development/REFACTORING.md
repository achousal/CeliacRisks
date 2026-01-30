# Refactoring Plan: ced_ml Codebase

**Created**: 2026-01-28
**Updated**: 2026-01-30 (Phase 0 + 0.5 implementation in progress)
**Status**: Phase 0/0.5 - Steps 1-5 Complete (dead code + config + --verbose removed)
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
- ~50 backward-compat shims, redundant pathways, and dead config options across ~25 files (v1.10 audit)

### Backward-Compat / Cruft Inventory (Codex repo scan, 2026-01-29)

Full line-level inventory organized by category. Items marked **(Rx)** map to a removal candidate below.

#### Back-compat / shim layers

| Area | Locations | Notes |
|------|-----------|-------|
| Split metadata legacy formats (scenario-less JSON, legacy index sort) | persistence.py:16,306,353,374,537,589 | **(R3)** |
| Legacy split filename (no-scenario CSV) | train.py:337,354,373 | **(R3)** |
| Holdout bare-model wrap + config fallback | holdout.py:88,93,412 | **(R5)** |
| `panels` module re-exports for old imports | panels.py:26 | Low risk to remove |
| `calibration` module re-exports for old imports | calibration.py:33-34 | Low risk to remove |
| KBest alias for old attribute name | kbest.py:486 | Low risk to remove |
| Aggregation orchestrator return format kept for old callers | orchestrator.py:44 | Low risk to remove |
| Legacy optuna filename (no model prefix) | aggregate_splits.py:736 | **(R2)** |

#### Legacy support / vestigial paths

| Area | Locations | Notes |
|------|-----------|-------|
| Convenience legacy loader in data IO | io.py:482,485 | Can remove if unused |
| Legacy results layout fallbacks (panels/features, reports/feature_reports) | optimize_panel.py:551,649; consensus_panel.py:95,136 | **(R2)** |
| Legacy split directory layout | optimize_panel.py:695; train_ensemble.py:361 | **(R3)** |
| Legacy stability inputs (selected_proteins_per_split fallback) | optimize_panel.py:357 | **(R4)** |
| Legacy protein name cleanup for CSVs | optimize_panel.py:669 | Phase 5 (I/O centralization) |
| Legacy `_resid` suffix normalization | optimize_panel.py:801 | Phase 5 / F3 |
| Legacy ColumnTransformer feature name assumptions | training.py:982,1052,1123 | Medium risk |
| Deprecated `feature_select` config default | defaults.py:66 | **(R1)** |

#### Compatibility cruft (kept "just in case")

| Area | Locations | Notes |
|------|-----------|-------|
| Hardcoded fallbacks in pipeline config resolution | run_pipeline.py:74,119 | Keep (user-friendly auto-detect) |
| CLI "first non-None wins" pattern | main.py:1294 | Phase 7 (CLI args DRY) |
| Optuna import fallback (graceful no-optuna) | optuna_search.py:11,30 | Keep (HPC compat) |

#### Redundant pathways

| Area | Locations | Notes |
|------|-----------|-------|
| Split seed sources: CLI vs splits config | main.py:1249,1277 | Phase 7 |
| Stability sources: stable_panel__KBest vs selected_proteins_per_split | optimize_panel.py:333,357 | **(R4)** / F8 |
| Stability file resolution triple-path | optimize_panel.py:551; consensus_panel.py:95 | **(R2)** / F2 |

#### Defensive fallbacks (keep -- scientifically justified)

| Area | Locations | Rationale |
|------|-----------|-----------|
| KBest variance fallback if F-test fails | kbest.py:42,84 | Handles degenerate features |
| Stability top-N fallback if none pass threshold | stability.py:143,231 | Prevents empty panel |
| Stratification fallback to outcome only | splits.py:165,201 | Handles small strata |
| DCA closest-zero fallback | dca.py:437,476 | Numerical edge case |
| Thresholds fallback path | thresholds.py:503 | Edge case handling |
| Registry sklearn version compat | registry.py:6,9,245,333 | HPC version variance |
| Optuna single vs multi-objective validation | optuna_search.py:395 | Runtime safety |

#### Transitional / deprecated APIs

| Area | Locations | Notes |
|------|-----------|-------|
| Deprecated `--verbose` CLI arg | save_splits.py:70; train.py:540; aggregate_splits.py:322; train_ensemble.py:279 | **(R1)** / Phase 0f |
| Deprecated risk_dist plotting args | risk_dist.py:120,173 | **(R1)** |

### Removal candidates (lowest risk first)

| ID | Action | Risk | Files |
|----|--------|------|-------|
| R1 | Remove deprecated CLI args (`--verbose`) and deprecated plotting params (`threshold_bundle` only) | Very low | save_splits, train, aggregate_splits, train_ensemble, risk_dist, defaults |
| R2 | Remove legacy path fallbacks for aggregated results (panels/features, reports/feature_reports) and legacy optuna filename | Low | optimize_panel, consensus_panel, aggregate_splits |
| R3 | Drop legacy split filename support (scenario-less CSVs) and legacy split meta naming | Low-medium | persistence, train, optimize_panel, train_ensemble |
| R4 | Remove legacy `selected_proteins_per_split` fallback in optimize-panel | Low | optimize_panel |
| R5 | Remove bare-model holdout compat (require bundle format) | Medium | holdout (needs artifact cutoff date) |

---

## Phase 0: Dead Code, Dead Config, and Script Hygiene (Very Low Risk)

**Goal**: Remove unused code, dead config, and archive superseded scripts before structural refactoring.

### 0a. Dead code removals

- `utils/paths.py`: `get_run_dir`, `get_core_dir`, `get_preds_dir`, `get_diagnostics_dir`, `get_reports_dir` have zero callers (only defined + re-exported via `__init__.py`). Remove functions and drop from `__all__`.
- `utils/metadata.py`: Docstring references `build_holdout_metadata` which does not exist. Remove stale reference. **Note**: `format_split_info` is a nested function within `build_plot_metadata` - keep (properly scoped).
- `config/loader.py`: **CORRECTION**: The 6 functions listed previously are NOT dead. `convert_paths`, `format_dict`, `resolve_value` are nested functions (properly scoped). `load_holdout_config`, `load_panel_optimize_config`, `print_config_summary` are architectural public APIs matching the pattern of `load_training_config`. All should be kept.
- `utils/logging.py`: 3 dead symbols -- `cleanup_live_logs`, `LoggerContext`, `log_dict`. Remove all.
- `config/schema.py`: **CORRECTION**: All dataclasses are ALIVE (used via Pydantic deserialization from YAML configs). Keep all. `rfe_min_auroc_frac` is a schema field - keep.

### 0b. Dead config options (v1.10 audit)

Remove from `config/schema.py` and `config/defaults.py`:

| Parameter | Why dead |
|-----------|----------|
| `splits.save_indices_only` | Defined in schema, never read by any code |
| `features.kbest_scope` | Defined as `Literal["protein", "transformed"]`, never used to alter behavior |
| `cv.tune_n_jobs` | Shadowed by `compute.tune_n_jobs` (only that one is actually used) |
| `cv.error_score` | Hardcoded to "raise" in training.py, config value ignored |
| `output.save_controls_oof` | File is unconditionally written when controls exist, config flag never checked |
| Entire `PanelConfig` class (`panels.*` block) | Replaced by `ced optimize-panel` and `ced consensus-panel` CLI commands |

**CORRECTIONS** - The following are LIVE (actively used in training.py), not dead:
- `features.rf_use_permutation` + `rf_perm_repeats` + `rf_perm_min_importance` + `rf_perm_top_n` (used in training.py:953,1100,1139,1148)
- `features.coef_threshold` (used in training.py:1002)
- `evaluation.bootstrap_min_samples` (used in train.py:2075)
- `cv.scoring_target_fpr` (used in training.py:630,680,1093)

**Files**: `config/schema.py`, `config/defaults.py`, `config/validation.py`

### 0c. Conditional dead config (confirm before removing)

| Parameter | Condition for removal |
|-----------|----------------------|
| `cv.grid_randomize` | Dead if Optuna is always enabled (production default) |
| `features.kbest_max` | Used only in validation (validates panel sizes don't exceed) and as fallback when k_grid empty; can remove if validation deemed unnecessary |

### 0d. Script archival

- `scripts/setup_optimize_panel.py`: Superseded by `ced optimize-panel --run-id` auto-detection. Move to `scripts/_legacy/`.
- `scripts/hpc_optimize_panel.sh`: Duplicates run-id/model discovery already in `post_training_pipeline.sh`. Move to `scripts/_legacy/`.
- `scripts/validate_rfecv_fix.py`: One-off validation. Convert assertions into tests under `tests/features/`, then delete script.
- `run_hpc.sh`: Already a deprecation stub that exits with error. Delete entirely, update docs.

### 0e. Vestigial doc references

- `CLAUDE.md` references `scripts/post_training_pipeline.sh` but it does not exist. Remove references.
- `docs/investigations/_archive/`: Add a `README.md` marker or rename to `docs/archive/investigations/` to prevent confusion with active investigation workflows.

### 0f. Deprecated CLI parameter

- Remove `--verbose` from all CLI commands. Keep only `--log-level`. Currently both exist with fallback conversion (`20 - verbose*10`) in aggregate_splits:344, train_ensemble:291, and explicit mapping in optimize_panel:184-188, consensus_panel:280-285.

### 0g. Logging consolidation (extend `utils/logging.py`)

Two new helpers to eliminate divergent logger setup across all 6 CLI commands:

`setup_command_logger(command, log_level, outdir, run_id, model, split_seed, logger_name) -> Logger`:
- Combines `auto_log_path` + `setup_logger` + "Logging to file" info message.
- Replaces 6-10 lines of boilerplate in every CLI command (train:592-600, run_pipeline:164-171, aggregate_splits:356-362, train_ensemble:295-301, optimize_panel:635-641, consensus_panel:288-293).
- **Note**: After Phase 0f, `--verbose` is removed, so this helper only needs `log_level` parameter.

### 0h. CLI validation helper

- Extract common mutually-exclusive flag validation and auto-detect config checks into `cli/_validators.py` to reduce boilerplate in `cli/main.py`.

**Verify**: `pytest tests/ -v` passes. `ruff check src/` clean. No broken imports. `grep -r` for removed config keys returns 0 hits.

### ✅ Phase 0 Implementation Status (2026-01-30)

**Steps 1-5 COMPLETED:**
- ✅ **Step 0a (Dead code)**: Removed 8 dead functions from utils/paths.py and utils/logging.py
- ✅ **Step 0a (Docstring)**: Fixed stale reference in utils/metadata.py
- ✅ **Step 0b (Dead config)**: Removed 6 dead config params + entire PanelConfig class
- ✅ **Step 0f (--verbose)**: Removed deprecated --verbose flag from all 9 CLI files
- ✅ **Documentation**: Updated REFACTORING.md with validation corrections

**Files modified**: 24 total (22 source files, 1 doc, 1 test)
**Lines removed**: ~350 lines of dead code and config
**Verification**: `ruff check src/` passes ✓

**Remaining steps**: 0g (logging helper), Phase 0.5 (fallback helpers), Phase 0.5 (discovery module)

---

## Phase 0.5: Eliminate Backward-Compat Fallback Chains (Low Risk)

**Goal**: Consolidate ~15 dual/triple-path fallback chains scattered across CLI modules into single-source-of-truth helpers. This reduces ~200 lines of duplicated fallback logic and makes the "canonical" file format explicit.

### Fallback chain inventory

Cross-referenced with Codex repo scan (Rx = removal candidate from inventory above).

| ID | Pattern | Files affected (Codex-verified) | Action | Rx |
|----|---------|--------------------------------|--------|----|
| F1 | Split file format: `train_idx_{scenario}_seed{N}.csv` vs `train_idx_seed{N}.csv` | train.py:337,354,373; optimize_panel.py:275-285,695; consensus_panel.py; persistence.py:537-542; aggregate_splits.py | Centralize into `resolve_split_files(split_dir, seed, scenario)` in `utils/paths.py` | R3 |
| F2 | Stability file triple-fallback: `panels/` -> `panels/features/` -> `reports/feature_reports/` | optimize_panel.py:551-561,649-660; consensus_panel.py:95-103,136-148 | Centralize into `find_stability_file(aggregated_dir)` in `utils/paths.py` | R2 |
| F3 | Protein name `_resid` suffix matching (add/remove suffix as fallback) | optimize_panel.py:801-823; optimize_panel.py:669 (CSV cleanup) | Normalize protein names once at data load time in `io.py` | -- |
| F4 | Optuna trials filename fallback (prefixed vs non-prefixed) | aggregate_splits.py:736-740 | Keep only current format if no old-format files in production | R2 |
| F5 | Metadata format fallback (nested `models.{first}.infile` vs flat `infile`) | consensus_panel.py:229-236 | Standardize on nested format; remove flat fallback | -- |
| F6 | Metrics loading dual-path (JSON vs CSV) | train_ensemble.py:140-157 | Standardize on JSON; remove CSV fallback | -- |
| F7 | Split metadata fallback (with/without scenario in meta JSON) | persistence.py:16,306,353,374,537-542,589 | Consolidate into `resolve_split_files` helper | R3 |
| F8 | Stability panel loading triple-fallback (`stable_panel__KBest.csv` -> `cv/selected_proteins_per_split.csv` -> all proteins) | optimize_panel.py:326-375; optimize_panel.py:333,357 | Document canonical path; centralize into helper | R4 |
| F9 | RFE ranking file dual-path (`feature_ranking_aggregated.csv` vs `feature_ranking.csv`) | consensus_panel.py:174-183 | Standardize on aggregated name | -- |
| F10 | Holdout backward-compat for bare model bundles | evaluation/holdout.py:88,93-94,412-413 | Require bundle format; remove bare-model shim | R5 |

### Out of scope (keep as-is)

- scipy/sklearn version compat shims (`features/screening.py:121-124`, `features/kbest.py:83,253`) -- needed for HPC environments
- Auto-detection fallback chains that are genuinely user-friendly (config file discovery in `run_pipeline.py:74,119,171-193`)
- Optuna import fallback / graceful no-optuna (`optuna_search.py:11,30`) -- needed for HPC environments
- Optuna multi-objective code (`multi_objective`, `pareto_selection`, `objectives`) -- experimental, document as such
- Optuna `storage`/`study_name`/`load_if_exists` -- optional advanced features
- Optuna single vs multi-objective validation (`optuna_search.py:395`) -- runtime safety
- `aggregate_config.yaml` -- still supported, mark as legacy in docs
- Defensive scientific fallbacks (KBest variance fallback `kbest.py:42,84`, stability top-N `stability.py:143,231`, stratification `splits.py:165,201`, DCA `dca.py:437,476`, thresholds `thresholds.py:503`, registry sklearn compat `registry.py:6,9,245,333`) -- all scientifically justified

**Verify**: `pytest tests/ -v` passes. Local smoke test with `ced run-pipeline`. `grep` for old fallback patterns returns 0 hits in caller code (fallbacks live only in centralized helpers).

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

`find_results_root`: Check `CED_RESULTS_DIR` env -> project root / results -> cwd / results. Replaces 5+ fragile `.parent.parent.parent.parent.parent` chains.
`resolve_split_files`: Try scenario-specific name, fall back to old format without scenario. (Absorbs fallback F1 from Phase 0.5)
`find_stability_file`: Check panels/ -> panels/features/ -> reports/feature_reports/ (legacy). (Absorbs fallback F2 from Phase 0.5)

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

`extract_paths_from_metadata`: Handle nested (`models.{first}.infile`) and flat (`infile`) formats. (Absorbs fallback F5 from Phase 0.5 temporarily; flat fallback removed once standardized.)

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

Also: normalize protein names (strip `_resid` suffix) once at load time in `io.py` to eliminate downstream defensive suffix gymnastics (absorbs fallback F3 from Phase 0.5).

Also: centralize defensive protein name stripping (`str.strip().str.strip('"').str.strip("'")`) currently duplicated in consensus_panel.py:669-670 and optimize_panel.py:669-670. Fix at CSV write time instead.

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
- `output_options`: --log-level, --outdir (--verbose removed in Phase 0f)
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
Phase 0 + 0.5 (dead code/cruft) --> Phase 1 (tests) --> Phase 2 (paths) --> Phase 3 (train.py) --> Phase 4 (reports)
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
| Dead config params | 11+ | 0 |
| Backward-compat fallback chains | 10 (duplicated across 5+ files) | 0 (centralized in helpers) |
| Duplicated path discovery | 6 implementations | 1 |
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

Unresolved questions: none.
