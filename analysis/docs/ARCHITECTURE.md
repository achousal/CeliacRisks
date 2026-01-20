# CeliacRisks Architecture

**Version:** 1.0
**Date:** 2026-01-20
**Status:** Current-state documentation

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Repo Layout & Module Responsibilities](#2-repo-layout--module-responsibilities)
3. [Primary Execution Flows](#3-primary-execution-flows)
4. [Data Flow Diagram](#4-data-flow-diagram)
5. [Key Data Contracts](#5-key-data-contracts)
6. [Configuration System](#6-configuration-system)
7. [Splitting & Leakage Prevention](#7-splitting--leakage-prevention)
8. [Feature Selection Pipeline](#8-feature-selection-pipeline)
9. [Model Training](#9-model-training)
10. [Calibration & Prevalence Adjustment](#10-calibration--prevalence-adjustment)
11. [Threshold Selection](#11-threshold-selection)
12. [Evaluation & Metrics](#12-evaluation--metrics)
13. [Output Artifacts & Reports](#13-output-artifacts--reports)
14. [Reproducibility & Determinism](#14-reproducibility--determinism)
15. [HPC Orchestration](#15-hpc-orchestration)
16. [Extension Points](#16-extension-points)
17. [Non-Negotiable Behaviors](#17-non-negotiable-behaviors)

---

## 1. System Overview

### Purpose

CeliacRisks is a modular ML pipeline for predicting **incident Celiac Disease (CeD)** risk from proteomics biomarkers measured before clinical diagnosis. The system generates calibrated risk scores suitable for clinical screening decisions.

**Key Characteristics:**
- Proteomics-based biomarker discovery
- Nested cross-validation optimized for discrimination (AUROC)
- Leakage-proof split strategy with VAL set for threshold tuning
- Calibrated probability estimates with prevalence adjustment
- Reproducible outputs with full metadata tracking

### Dataset

- **Size:** 43,960 samples (43,662 controls, 148 incident, 150 prevalent)
- **Features:** 2,920 proteins (columns ending with `_resid` suffix)
- **Metadata:** age, BMI, sex, ethnicity (flexible via `ColumnsConfig`)
- **Target:** Binary incident CeD diagnosis
- **Prevalence:** ~1:300 (0.33% incident cases)

**Where in code:**
- [data/schema.py:1-50](../src/ced_ml/data/schema.py#L1-L50) - Column definitions
- [data/io.py](../src/ced_ml/data/io.py) - Data loading functions

### Supported Models

Four classifiers with nested hyperparameter tuning:
1. **RF** (Random Forest)
2. **XGBoost**
3. **LinSVM_cal** (Linear SVM with CalibratedClassifierCV)
4. **LR_EN** (Logistic Regression with ElasticNet)

**Where in code:**
- [models/registry.py](../src/ced_ml/models/registry.py) - Model factory functions

### Architecture Decision Records

Key design decisions are documented in separate ADR files:
- [ADR-001: Split Strategy](adr/ADR-001-split-strategy.md) - Why 50/25/25 three-way split
- [ADR-002: Prevalent→TRAIN](adr/ADR-002-prevalent-train-only.md) - Why prevalent cases in TRAIN only
- [ADR-003: Control Downsampling](adr/ADR-003-control-downsampling.md) - Why 1:5 ratio
- [ADR-004: AUROC Optimization](adr/ADR-004-auroc-optimization.md) - AUROC as primary optimization metric
- [ADR-005: Prevalence Adjustment](adr/ADR-005-prevalence-adjustment.md) - Logit shift method
- [ADR-015: Flexible Metadata Columns](adr/ADR-015-flexible-metadata-columns.md) - Column configuration system

See [docs/adr/](adr/) for complete list.

---

## 2. Repo Layout & Module Responsibilities

### High-Level Structure

```
analysis/
  src/ced_ml/           # Python package (15,109 lines)
    cli/                # Command-line interface
    config/             # Configuration system
    data/               # Data I/O, splits, persistence
    features/           # Feature selection pipeline
    models/             # Model training, calibration
    metrics/            # Performance metrics
    evaluation/         # Prediction & reporting
    plotting/           # Visualization
    utils/              # Shared utilities
  tests/                # 753 tests, 82% coverage
  docs/                 # Documentation
    ARCHITECTURE.md     # This file
    adr/                # Architecture Decision Records
  Legacy/               # Archived monolithic scripts
splits_production/      # Persisted split indices
results_production/     # Training outputs
```

**Where in code:**
- [src/ced_ml/](../src/ced_ml/) - Package root
- [tests/](../tests/) - Test suite

### Module Responsibilities

#### `cli/` - Command-Line Interface
- `main.py` - Entry point (`ced` command via Click)
- `save_splits.py` - Split generation CLI
- `train.py` - Model training CLI
- `postprocess.py` - Multi-model aggregation CLI
- `eval_holdout.py` - Holdout evaluation CLI (run ONCE)
- `config_tools.py` - Config migration/validation utilities

**Where in code:**
- [cli/main.py](../src/ced_ml/cli/main.py) - CLI entry point
- [cli/train.py:204-474](../src/ced_ml/cli/train.py#L204-L474) - Main training orchestration

#### `config/` - Configuration System
- `schema.py` - Pydantic models (~200 parameters)
- `loader.py` - YAML loading + CLI overrides
- `validation.py` - Cross-field validation, leakage detection

**Key Config Classes:**
- `TrainingConfig` - Top-level training config
- `SplitsConfig` - Split generation parameters
- `CVConfig` - Cross-validation structure
- `FeatureConfig` - Feature selection methods
- `ThresholdConfig` - Threshold selection objectives
- `CalibrationConfig` - Calibration settings
- `ColumnsConfig` - Flexible metadata column configuration
- Model-specific configs: `LRConfig`, `SVMConfig`, `RFConfig`, `XGBoostConfig`

**Where in code:**
- [config/schema.py:292-343](../src/ced_ml/config/schema.py#L292-L343) - `TrainingConfig`
- [config/loader.py](../src/ced_ml/config/loader.py) - YAML loader
- [config/validation.py](../src/ced_ml/config/validation.py) - Validators

See [ADR-012: Pydantic Config Schema](adr/ADR-012-pydantic-config.md) for rationale.

#### `data/` - Data Layer (386 lines, 111 tests)
- `io.py` - Data loading with flexible column selection
- `splits.py` - Stratified splitting, downsampling, prevalent handling
- `persistence.py` - Split index CSV I/O
- `schema.py` - Column name constants, scenario definitions
- `filters.py` - Row filtering logic
- `columns.py` - Metadata column resolution (auto/explicit modes)

**Where in code:**
- [data/io.py](../src/ced_ml/data/io.py) - `load_data`, `usecols_for_proteomics`
- [data/splits.py:374-438](../src/ced_ml/data/splits.py#L374-L438) - `stratified_train_val_test_split`
- [data/columns.py](../src/ced_ml/data/columns.py) - `resolve_columns`, `ResolvedColumns`

See [ADR-015: Flexible Metadata Columns](adr/ADR-015-flexible-metadata-columns.md) for column configuration design.

#### `features/` - Feature Engineering (433 lines, 128 tests)
- `screening.py` - Mann-Whitney U / F-statistic screening
- `kbest.py` - SelectKBest wrapper
- `stability.py` - Stability panel extraction
- `corr_prune.py` - Correlation-based pruning
- `panels.py` - Biomarker panel building

**Where in code:**
- [features/screening.py](../src/ced_ml/features/screening.py) - `mann_whitney_screen`
- [features/stability.py:124-216](../src/ced_ml/features/stability.py#L124-L216) - `extract_stable_panel`

See [ADR-006: Hybrid Feature Selection](adr/ADR-006-hybrid-feature-selection.md).

#### `models/` - Model Training (549 lines, 103 tests)
- `registry.py` - Model factory (RF, XGBoost, LinSVM, LR)
- `hyperparams.py` - Hyperparameter grids for RandomizedSearchCV
- `training.py` - Nested CV orchestration, OOF predictions
- `calibration.py` - Calibration wrappers, prevalence adjustment
- `prevalence.py` - Prevalence adjustment utilities

**Where in code:**
- [models/registry.py](../src/ced_ml/models/registry.py) - `build_<model_name>` functions
- [models/training.py:29-192](../src/ced_ml/models/training.py#L29-L192) - `oof_predictions_with_nested_cv`
- [models/calibration.py:152-188](../src/ced_ml/models/calibration.py#L152-L188) - `PrevalenceAdjustedModel`

See [ADR-008: Nested CV Structure](adr/ADR-008-nested-cv.md), [ADR-011: PrevalenceAdjustedModel](adr/ADR-011-prevalence-wrapper.md).

#### `metrics/` - Performance Metrics (421 lines, 163 tests)
- `discrimination.py` - AUROC, PR-AUC, Brier score
- `thresholds.py` - Threshold selection (Youden, max_f1, fixed_spec, etc.)
- `dca.py` - Decision Curve Analysis (net benefit)
- `bootstrap.py` - Stratified bootstrap confidence intervals

**Where in code:**
- [metrics/discrimination.py](../src/ced_ml/metrics/discrimination.py) - `auroc`, `prauc`, `compute_brier_score`
- [metrics/thresholds.py:326-377](../src/ced_ml/metrics/thresholds.py#L326-L377) - `choose_threshold_objective`

See [ADR-009: Threshold on VAL](adr/ADR-009-threshold-on-val.md), [ADR-010: Fixed Spec 95%](adr/ADR-010-fixed-spec-95.md).

#### `evaluation/` - Prediction & Reporting (381 lines, 81 tests)
- `predict.py` - Generate predictions for TRAIN/VAL/TEST
- `reports.py` - `ResultsWriter` class (metrics, plots, metadata)
- `holdout.py` - Holdout set evaluation (run ONCE)

**Where in code:**
- [evaluation/reports.py](../src/ced_ml/evaluation/reports.py) - `ResultsWriter`, `OutputDirectories`
- [evaluation/holdout.py](../src/ced_ml/evaluation/holdout.py) - Holdout CLI logic

#### `plotting/` - Visualization (1,082 lines, 122 tests)
- `roc_pr.py` - ROC + PR curves
- `calibration.py` - Calibration plots
- `risk_dist.py` - Risk distribution histograms
- `dca.py` - Decision curve analysis plots
- `learning_curve.py` - Learning curve analysis

**Where in code:**
- [plotting/](../src/ced_ml/plotting/) - All plotting modules

#### `utils/` - Shared Utilities
- `logging.py` - Logging configuration
- `paths.py` - Path handling helpers
- `random.py` - RNG utilities
- `serialization.py` - Pickle save/load wrappers

---

## 3. Primary Execution Flows

### 3.1 `save-splits` - Split Generation

**Purpose:** Generate reproducible train/val/test split indices and save to CSV files.

**Inputs:**
- Raw data CSV (`--infile`)
- Splits configuration (YAML or CLI args)

**Outputs:**
- Split index CSV files: `{scenario}_{split}_idx_seed{N}.csv`
- Resolved config: `splits_config.yaml`

**Flow:**
1. Load splits configuration
2. Load raw data (minimal columns: ID, target, scenario columns)
3. For each scenario:
   a. Filter rows by scenario definition
   b. Downsample controls (1:5 ratio)
   c. Stratified 3-way split (50/25/25)
   d. Add prevalent cases to TRAIN (50% sample)
   e. Save indices as CSV
4. Save resolved config

**Where in code:**
- [cli/save_splits.py](../src/ced_ml/cli/save_splits.py) - CLI entry point
- [data/splits.py:374-438](../src/ced_ml/data/splits.py#L374-L438) - `stratified_train_val_test_split`
- [data/splits.py:326-366](../src/ced_ml/data/splits.py#L326-L366) - `add_prevalent_to_train`
- [data/splits.py:193-250](../src/ced_ml/data/splits.py#L193-L250) - `downsample_controls`
- [data/persistence.py](../src/ced_ml/data/persistence.py) - `save_split_indices`

**See ADRs:**
- [ADR-001: Split Strategy](adr/ADR-001-split-strategy.md)
- [ADR-002: Prevalent→TRAIN](adr/ADR-002-prevalent-train-only.md)
- [ADR-003: Control Downsampling](adr/ADR-003-control-downsampling.md)
- [ADR-013: Split Persistence Format](adr/ADR-013-split-persistence.md)

### 3.2 `train` - Model Training

**Purpose:** Train a single model with nested CV, generate OOF predictions, select thresholds, evaluate on VAL/TEST.

**Inputs:**
- Raw data CSV
- Split index CSVs (if `--split-dir` provided)
- Training configuration (YAML + CLI overrides)

**Outputs:**
- Final trained model: `final_model.pkl` (wrapped in `PrevalenceAdjustedModel`)
- OOF predictions: `oof_predictions.csv`, `val_predictions.csv`, `test_predictions.csv`
- Metrics: `*_metrics.json`
- Plots: ROC, PR, calibration, risk distribution, DCA
- Metadata: `run_settings.json`

**Flow:**
1. Load training configuration
2. Resolve metadata columns (auto or explicit mode)
3. Load raw data with resolved columns
4. Load or generate splits
5. Apply row filters
6. Screening: Reduce to top N proteins (e.g., 1000)
7. Nested CV loop (5 outer folds × 10 repeats):
   a. For each outer fold:
      - Split TRAIN into inner train/val
      - Inner CV: RandomizedSearchCV (5 inner folds × 200 iterations)
      - Select best hyperparameters (AUROC optimization)
      - Train model on inner train with best params
      - Generate OOF predictions for held-out outer fold samples
   b. Aggregate OOF predictions across all outer folds
8. Stability panel extraction: Keep proteins selected in ≥75% of CV folds
9. Train final model on full TRAIN set with stable panel
10. Calibrate model (if configured)
11. Wrap in `PrevalenceAdjustedModel` with target prevalence
12. Generate predictions on TRAIN/VAL/TEST
13. Select threshold on VAL set (e.g., Youden, max_f1, fixed_spec=0.95)
14. Compute metrics on TRAIN/VAL/TEST with selected threshold
15. Generate plots
16. Save model, predictions, metrics, plots, run_settings

**Where in code:**
- [cli/train.py:204-474](../src/ced_ml/cli/train.py#L204-L474) - Main training orchestration
- [models/training.py:29-192](../src/ced_ml/models/training.py#L29-L192) - `oof_predictions_with_nested_cv`
- [features/stability.py:124-216](../src/ced_ml/features/stability.py#L124-L216) - `extract_stable_panel`
- [models/calibration.py:152-188](../src/ced_ml/models/calibration.py#L152-L188) - `PrevalenceAdjustedModel`
- [metrics/thresholds.py:326-377](../src/ced_ml/metrics/thresholds.py#L326-L377) - `choose_threshold_objective`
- [evaluation/reports.py](../src/ced_ml/evaluation/reports.py) - `ResultsWriter`

**See ADRs:**
- [ADR-004: AUROC Optimization](adr/ADR-004-auroc-optimization.md)
- [ADR-005: Prevalence Adjustment](adr/ADR-005-prevalence-adjustment.md)
- [ADR-006: Hybrid Feature Selection](adr/ADR-006-hybrid-feature-selection.md)
- [ADR-007: Stability Panel](adr/ADR-007-stability-panel.md)
- [ADR-008: Nested CV Structure](adr/ADR-008-nested-cv.md)
- [ADR-009: Threshold on VAL](adr/ADR-009-threshold-on-val.md)
- [ADR-011: PrevalenceAdjustedModel](adr/ADR-011-prevalence-wrapper.md)
- [ADR-015: Flexible Metadata Columns](adr/ADR-015-flexible-metadata-columns.md)

### 3.3 `postprocess` - Multi-Model Aggregation

**Purpose:** Aggregate results across multiple trained models, generate comparison plots.

**Inputs:**
- Multiple model result directories (e.g., `results_RF/`, `results_XGBoost/`)

**Outputs:**
- Comparison tables: `model_comparison.csv`
- Comparison plots: Multi-model ROC/PR curves

**Flow:**
1. Load predictions and metrics from each model directory
2. Aggregate metrics into comparison table
3. Generate multi-model comparison plots
4. Save aggregated results

**Where in code:**
- [cli/postprocess.py](../src/ced_ml/cli/postprocess.py) - Postprocessing CLI

### 3.4 `eval-holdout` - Holdout Evaluation

**Purpose:** Evaluate trained model on a completely held-out dataset (run ONCE).

**Inputs:**
- Trained model pickle file
- Holdout dataset CSV

**Outputs:**
- Holdout predictions: `holdout_predictions.csv`
- Holdout metrics: `holdout_metrics.json`
- Holdout plots

**Flow:**
1. Load trained model
2. Load holdout data
3. Generate predictions
4. Compute metrics with previously selected threshold
5. Generate plots
6. Save results

**Where in code:**
- [evaluation/holdout.py](../src/ced_ml/evaluation/holdout.py) - Holdout evaluation logic

---

## 4. Data Flow Diagram

```
Raw Data CSV (43,960 samples, 2,920 proteins)
           |
           v
    [save-splits]
           |
           +---> {scenario}_train_idx_seed{N}.csv (50%)
           +---> {scenario}_val_idx_seed{N}.csv   (25%)
           +---> {scenario}_test_idx_seed{N}.csv  (25%)
           |
           v
    [train] with config.yaml
           |
           +---> Screening (Mann-Whitney U / F-stat → 1000 proteins)
           |
           +---> Nested CV Loop (5 outer × 10 repeats)
           |       |
           |       +---> Inner CV: RandomizedSearchCV (5 inner × 200 iter)
           |       +---> OOF predictions for each outer fold
           |       +---> Track selected features per CV repeat
           |
           +---> Stability Panel Extraction (≥75% selection rate)
           |
           +---> Final Model Training (full TRAIN, stable panel)
           |
           +---> Calibration (if enabled)
           |
           +---> Prevalence Adjustment Wrapper
           |
           +---> Predictions: TRAIN / VAL / TEST
           |
           +---> Threshold Selection (on VAL)
           |
           +---> Metrics Computation (with threshold)
           |
           +---> Outputs:
                   - final_model.pkl (PrevalenceAdjustedModel)
                   - oof_predictions.csv
                   - val_predictions.csv
                   - test_predictions.csv
                   - train_metrics.json, val_metrics.json, test_metrics.json
                   - plots/ (ROC, PR, calibration, risk_dist, DCA)
                   - run_settings.json
```

**Key Leakage Prevention:**
- VAL and TEST splits never used for hyperparameter tuning (only TRAIN)
- Threshold selected on VAL, never on TEST
- Prevalent cases only in TRAIN, never in VAL/TEST

**See ADRs:**
- [ADR-001: Split Strategy](adr/ADR-001-split-strategy.md)
- [ADR-002: Prevalent→TRAIN](adr/ADR-002-prevalent-train-only.md)
- [ADR-009: Threshold on VAL](adr/ADR-009-threshold-on-val.md)

---

## 5. Key Data Contracts

### 5.1 Column Schema

**Protein Columns:** Must end with `_resid` suffix (e.g., `APOE_resid`, `SERPINA1_resid`).

**Metadata Columns:** Configurable via `ColumnsConfig`:
- **Auto mode** (default): Auto-detect from default list (age, BMI, sex, ethnicity)
- **Explicit mode**: User-specified custom columns

**Reserved Columns:**
- `ID_COL = 'eid'` - Sample identifier
- `TARGET_COL = 'incident_CeD'` - Binary target
- `INCIDENT_COL = 'incident_CeD'` - Incident flag
- `PREVALENT_COL = 'prevalent_CeD'` - Prevalent flag

**Where in code:**
- [data/schema.py:1-50](../src/ced_ml/data/schema.py#L1-L50) - Column constants
- [data/columns.py](../src/ced_ml/data/columns.py) - `ColumnsConfig`, `resolve_columns`
- [config/schema.py](../src/ced_ml/config/schema.py) - `ColumnsConfig` class

**See ADR:**
- [ADR-015: Flexible Metadata Columns](adr/ADR-015-flexible-metadata-columns.md)

### 5.2 Metadata Column Resolution

**Purpose:** Handle datasets with varying metadata availability (e.g., protein-only datasets, custom metadata columns).

**Modes:**
- `auto` (default): Auto-detect from default metadata columns
- `explicit`: Use user-specified lists

**Resolution Process:**
1. If mode is `auto`:
   - Check data file for presence of default metadata columns
   - Use only those present
   - If none present, use 0 metadata columns (protein-only)
2. If mode is `explicit`:
   - Use user-specified lists (features, filters, stratify)
   - Validate that all specified columns exist

**Where in code:**
- [data/columns.py](../src/ced_ml/data/columns.py) - `resolve_columns`, `get_available_columns_from_file`
- [config/schema.py](../src/ced_ml/config/schema.py) - `ColumnsConfig`
- [data/io.py](../src/ced_ml/data/io.py) - `usecols_for_proteomics` (uses resolved columns)
- [data/filters.py](../src/ced_ml/data/filters.py) - `apply_row_filters` (uses resolved columns)
- [cli/train.py](../src/ced_ml/cli/train.py) - Column resolution before data loading

**See ADR:**
- [ADR-015: Flexible Metadata Columns](adr/ADR-015-flexible-metadata-columns.md)

### 5.3 Split Index CSVs

**Format:** Single-column CSV files with row indices (int64).

**Filename Pattern:** `{scenario}_{split}_idx_seed{N}.csv`

**Example:**
```
IncidentPlusPrevalent_train_idx_seed42.csv
IncidentPlusPrevalent_val_idx_seed42.csv
IncidentPlusPrevalent_test_idx_seed42.csv
```

**Contents:**
```csv
row_idx
0
5
12
...
```

**Where in code:**
- [data/persistence.py](../src/ced_ml/data/persistence.py) - `save_split_indices`, `load_split_indices`

**See ADR:**
- [ADR-013: Split Persistence Format](adr/ADR-013-split-persistence.md)

### 5.4 Output Artifacts

**Directory Structure:**
```
results_production/
  {model}_{scenario}_seed{N}/
    final_model.pkl            # PrevalenceAdjustedModel wrapper
    oof_predictions.csv        # OOF predictions (TRAIN set)
    val_predictions.csv        # VAL predictions
    test_predictions.csv       # TEST predictions
    train_metrics.json         # TRAIN metrics
    val_metrics.json           # VAL metrics
    test_metrics.json          # TEST metrics
    run_settings.json          # Full config + metadata
    plots/
      roc_pr.png               # ROC + PR curves
      calibration.png          # Calibration plot
      risk_dist.png            # Risk distribution histogram
      dca.png                  # Decision curve analysis
```

**Prediction CSV Format:**
```csv
eid,y_true,y_pred_proba,y_pred,fold,repeat
1001,0,0.012,0,0,0
1002,1,0.842,1,0,0
...
```

**Metrics JSON Format:**
```json
{
  "auroc": 0.85,
  "prauc": 0.42,
  "brier": 0.08,
  "threshold": 0.35,
  "sensitivity": 0.78,
  "specificity": 0.82,
  ...
}
```

**Where in code:**
- [evaluation/reports.py](../src/ced_ml/evaluation/reports.py) - `ResultsWriter.save_*` methods
- [data/schema.py:100-150](../src/ced_ml/data/schema.py#L100-L150) - Output filename constants

---

## 6. Configuration System

### 6.1 Schema (Pydantic)

**Top-Level Config:** `TrainingConfig`

**Sub-Configs:**
- `CVConfig` - Cross-validation structure
- `FeatureConfig` - Feature selection methods
- `ThresholdConfig` - Threshold selection objectives
- `CalibrationConfig` - Calibration settings
- `EvaluationConfig` - Metrics and reporting
- `DCAConfig` - Decision curve analysis
- `OutputConfig` - File generation control
- `ColumnsConfig` - Metadata column configuration
- Model-specific: `LRConfig`, `SVMConfig`, `RFConfig`, `XGBoostConfig`

**Where in code:**
- [config/schema.py:292-343](../src/ced_ml/config/schema.py#L292-L343) - `TrainingConfig`
- [config/schema.py](../src/ced_ml/config/schema.py) - All config classes

**See ADR:**
- [ADR-012: Pydantic Config Schema](adr/ADR-012-pydantic-config.md)

### 6.2 Sub-Configs

#### CVConfig (Cross-Validation)
```python
folds: int = 5                # Outer CV folds
repeats: int = 10             # CV repeats
inner_folds: int = 5          # Inner CV folds for hyperparameter tuning
scoring: str = "roc_auc"      # Optimization metric (AUROC for discrimination)
n_iter: int = 200             # RandomizedSearchCV iterations
random_state: int = 42        # RNG seed
n_jobs: int = -1              # Parallel jobs
verbose: int = 1              # Verbosity level
```

**Where in code:**
- [config/schema.py](../src/ced_ml/config/schema.py) - `CVConfig`

**See ADR:**
- [ADR-004: AUROC Optimization](adr/ADR-004-auroc-optimization.md)
- [ADR-008: Nested CV Structure](adr/ADR-008-nested-cv.md)

#### FeatureConfig (Feature Selection)
```python
feature_select: str = "hybrid"       # hybrid | kbest | stability | none
screen_method: str = "mann_whitney"  # mann_whitney | f_statistic
screen_top_n: int = 1000             # Screening output size
kbest_k: int | None = None           # KBest k (tuned if None)
stability_thresh: float = 0.75       # Stability selection threshold
corr_thresh: float = 0.9             # Correlation pruning threshold
hybrid_kbest_first: bool = True      # Hybrid mode: KBest before stability
...
```

**Where in code:**
- [config/schema.py:83-105](../src/ced_ml/config/schema.py#L83-L105) - `FeatureConfig`

**See ADRs:**
- [ADR-006: Hybrid Feature Selection](adr/ADR-006-hybrid-feature-selection.md)
- [ADR-007: Stability Panel](adr/ADR-007-stability-panel.md)

#### ThresholdConfig (Threshold Selection)
```python
objective: str = "youden"               # youden | max_f1 | fixed_spec | fixed_ppv
threshold_source: str = "val"           # val | train
fixed_spec: float | None = None         # For fixed_spec objective
fixed_ppv: float | None = None          # For fixed_ppv objective
target_prevalence_source: str = "test"  # test | val | train | fixed
target_prevalence_fixed: float | None = None  # For fixed source
...
```

**Where in code:**
- [config/schema.py:198-208](../src/ced_ml/config/schema.py#L198-L208) - `ThresholdConfig`

**See ADRs:**
- [ADR-009: Threshold on VAL](adr/ADR-009-threshold-on-val.md)
- [ADR-010: Fixed Spec 95%](adr/ADR-010-fixed-spec-95.md)

#### CalibrationConfig (Calibration)
```python
calibrate: bool = False         # Enable calibration
method: str = "sigmoid"         # sigmoid | isotonic
cv_folds: int = 5               # Calibration CV folds
ensemble: bool = True           # Ensemble calibration
```

**Where in code:**
- [config/schema.py](../src/ced_ml/config/schema.py) - `CalibrationConfig`

#### ColumnsConfig (Metadata Columns)
```python
mode: str = "auto"                      # auto | explicit
metadata_features: list[str] = []       # Explicit mode: feature columns
metadata_filters: list[str] = []        # Explicit mode: filter columns
metadata_stratify: list[str] = []       # Explicit mode: stratify columns
default_metadata: list[str] = ["age", "BMI", "sex", "ethnicity"]  # Auto mode defaults
```

**Where in code:**
- [config/schema.py](../src/ced_ml/config/schema.py) - `ColumnsConfig`
- [data/columns.py](../src/ced_ml/data/columns.py) - `resolve_columns`

**See ADR:**
- [ADR-015: Flexible Metadata Columns](adr/ADR-015-flexible-metadata-columns.md)

### 6.3 YAML Loading

**YAML Structure:**
```yaml
model: LR_EN
scenario: IncidentPlusPrevalent
split_seed: 42

cv:
  folds: 5
  repeats: 10
  scoring: neg_brier_score

features:
  feature_select: hybrid
  screen_top_n: 1000
  stability_thresh: 0.75

thresholds:
  objective: max_f1
  threshold_source: val

columns:
  mode: auto  # or explicit with custom lists
```

**CLI Overrides:**
```bash
ced train --config config.yaml \
    --override cv.folds=10 \
    --override features.screen_top_n=2000
```

**Where in code:**
- [config/loader.py](../src/ced_ml/config/loader.py) - `load_config_with_overrides`

### 6.4 Validation

**Cross-Field Validation:**
- VAL + TEST fractions must sum to < 1.0
- If `threshold_source='val'`, requires `cv.folds >= 2`
- If `target_prevalence_source='fixed'`, requires `target_prevalence_fixed` is not None

**Where in code:**
- [config/validation.py](../src/ced_ml/config/validation.py) - Validator functions
- [config/schema.py](../src/ced_ml/config/schema.py) - `@model_validator` decorators

---

## 7. Splitting & Leakage Prevention

### 7.1 Stratified 3-Way Split

**Strategy:** 50% TRAIN / 25% VAL / 25% TEST

**Stratification:** By target (incident_CeD) to preserve class balance across splits.

**Flow:**
1. Filter data by scenario (e.g., `IncidentPlusPrevalent`)
2. Downsample controls to 1:5 ratio
3. Stratified split into TRAIN (50%), VAL (25%), TEST (25%)
4. Save split indices as CSV files

**Why 3-Way Split?** See [ADR-001: Split Strategy](adr/ADR-001-split-strategy.md).

**Where in code:**
- [data/splits.py:374-438](../src/ced_ml/data/splits.py#L374-L438) - `stratified_train_val_test_split`
- [config/schema.py](../src/ced_ml/config/schema.py) - `SplitsConfig.validate_split_sizes`

**Tests:**
- `tests/test_data_splits.py` - Validates stratified split logic

### 7.2 Prevalent→TRAIN Only

**Rule:** Prevalent cases (diagnosed before plasma sample) are added to TRAIN set only, never to VAL or TEST.

**Rationale:**
- Prevalent cases provide signal enrichment for training
- VAL/TEST remain prospective (incident-only) for clinically relevant evaluation
- 50% sampling of prevalent cases balances signal vs. distribution shift

**Where in code:**
- [data/splits.py:326-366](../src/ced_ml/data/splits.py#L326-L366) - `add_prevalent_to_train`
- [data/schema.py:49](../src/ced_ml/data/schema.py#L49) - `SCENARIO_DEFINITIONS`

**Tests:**
- `tests/test_data_splits.py` - Validates prevalent handling

**See ADR:**
- [ADR-002: Prevalent→TRAIN](adr/ADR-002-prevalent-train-only.md)

### 7.3 Control Downsampling

**Ratio:** 1:5 case:control (down from ~1:300 original ratio)

**Method:** Random sampling of controls without replacement, stratified by split.

**Rationale:**
- Reduces computational cost (300x fewer controls)
- Preserves adequate negative signal for model training
- Requires prevalence adjustment for deployment (see Section 10.2)

**Where in code:**
- [data/splits.py:193-250](../src/ced_ml/data/splits.py#L193-L250) - `downsample_controls`

**Tests:**
- `tests/test_data_splits.py` - Validates downsampling logic

**See ADR:**
- [ADR-003: Control Downsampling](adr/ADR-003-control-downsampling.md)

### 7.4 Split Persistence

**Format:** CSV index files with single column of row indices (int64).

**Filename Pattern:** `{scenario}_{split}_idx_seed{N}.csv`

**Benefits:**
- Human-readable and version-controllable
- Language-agnostic (Python, R, etc.)
- Enables exact reproducibility across runs

**Where in code:**
- [data/persistence.py](../src/ced_ml/data/persistence.py) - `save_split_indices`, `load_split_indices`

**Tests:**
- `tests/test_data_persistence.py` - Validates CSV I/O

**See ADR:**
- [ADR-013: Split Persistence Format](adr/ADR-013-split-persistence.md)

---

## 8. Feature Selection Pipeline

### 8.1 Screening

**Purpose:** Reduce feature space from ~2,920 proteins to top N (default: 1,000) based on univariate association with target.

**Methods:**
- `mann_whitney` - Mann-Whitney U test (default for non-normal distributions)
- `f_statistic` - F-statistic (ANOVA F-test)

**Where in code:**
- [features/screening.py](../src/ced_ml/features/screening.py) - `mann_whitney_screen`, `f_statistic_screen`

**Tests:**
- `tests/test_features_screening.py` - Validates screening methods

### 8.2 KBest (SelectKBest)

**Purpose:** Select top K features using sklearn's `SelectKBest` with `f_classif` scoring.

**Hyperparameter Tuning:** K is tuned via RandomizedSearchCV if not specified.

**Where in code:**
- [features/kbest.py](../src/ced_ml/features/kbest.py) - `SelectKBest` wrapper

**Tests:**
- `tests/test_features_kbest.py` - Validates KBest selection

### 8.3 Stability Selection

**Purpose:** Extract proteins selected in ≥75% of CV folds for robust panel.

**Method:**
1. Track selected features across all CV repeats
2. Compute selection frequency for each feature
3. Keep features with frequency ≥ `stability_thresh` (default: 0.75)
4. Fallback: If no features meet threshold, keep top 20 by frequency

**Where in code:**
- [features/stability.py:124-216](../src/ced_ml/features/stability.py#L124-L216) - `extract_stable_panel`

**Tests:**
- `tests/test_features_stability.py` - Validates stability logic

**See ADR:**
- [ADR-007: Stability Panel](adr/ADR-007-stability-panel.md)

### 8.4 Correlation Pruning

**Purpose:** Remove highly correlated features (|r| > threshold) to reduce redundancy.

**Method:**
1. Compute pairwise Pearson correlation matrix
2. For each pair with |r| > `corr_thresh` (default: 0.9), keep feature with higher univariate association
3. Remove redundant features

**Where in code:**
- [features/corr_prune.py](../src/ced_ml/features/corr_prune.py) - `prune_correlated_features`

**Tests:**
- `tests/test_features_corr_prune.py` - Validates pruning logic

### 8.5 Hybrid Mode

**Hybrid Feature Selection:** Combines KBest + Stability.

**Order (if `hybrid_kbest_first=True`):**
1. Screening → 1,000 proteins
2. KBest → tune K via inner CV
3. Stability → extract stable panel from KBest selections

**Alternative Order (if `hybrid_kbest_first=False`):**
1. Screening → 1,000 proteins
2. Stability → extract stable panel from screening
3. KBest → refine stable panel

**Where in code:**
- [config/schema.py:83-105](../src/ced_ml/config/schema.py#L83-L105) - `FeatureConfig.hybrid_kbest_first`

**See ADR:**
- [ADR-006: Hybrid Feature Selection](adr/ADR-006-hybrid-feature-selection.md)

---

## 9. Model Training

### 9.1 Registry

**Model Factory:** `registry.py` provides builder functions for each model:
- `build_logistic_regression()` - LR with ElasticNet penalty
- `build_random_forest()` - RF with balanced class weights
- `build_xgboost()` - XGBoost with scale_pos_weight
- `build_linear_svm_calibrated()` - LinearSVC + CalibratedClassifierCV

**Where in code:**
- [models/registry.py](../src/ced_ml/models/registry.py) - `build_*` functions

**Tests:**
- `tests/test_models_registry.py` - Validates model builders

### 9.2 Hyperparameters

**RandomizedSearchCV:** 200 iterations per inner CV fold.

**Hyperparameter Grids:**
- **LR:** C (regularization strength), l1_ratio (ElasticNet mix)
- **RF:** n_estimators, max_depth, min_samples_split, min_samples_leaf
- **XGBoost:** max_depth, learning_rate, n_estimators, subsample, colsample_bytree
- **LinSVM:** C (regularization strength)

**Where in code:**
- [models/hyperparams.py](../src/ced_ml/models/hyperparams.py) - Hyperparameter distributions

**Tests:**
- `tests/test_models_hyperparams.py` - Validates hyperparameter grids

### 9.3 Nested CV

**Structure:** 5 outer folds × 10 repeats × 5 inner folds = 50,000 model fits per model type.

**Outer CV (5 folds × 10 repeats):**
- Generates out-of-fold (OOF) predictions for TRAIN set
- Repeated 10 times for robust estimates

**Inner CV (5 folds × 200 iterations):**
- RandomizedSearchCV for hyperparameter tuning
- Optimizes AUROC (discrimination-focused)
- Selects best hyperparameters per outer fold

**Where in code:**
- [models/training.py:29-192](../src/ced_ml/models/training.py#L29-L192) - `oof_predictions_with_nested_cv`
- [config/schema.py](../src/ced_ml/config/schema.py) - `CVConfig` (folds, repeats, inner_folds)

**Tests:**
- `tests/test_training.py` - Validates nested CV logic

**See ADR:**
- [ADR-008: Nested CV Structure](adr/ADR-008-nested-cv.md)

---

## 10. Calibration & Prevalence Adjustment

### 10.1 Calibration Methods

**Purpose:** Improve probability calibration (align predicted probabilities with true frequencies).

**Methods:**
- `sigmoid` - Platt scaling (logistic regression on classifier scores)
- `isotonic` - Isotonic regression (non-parametric, monotonic)

**Wrapper:** sklearn's `CalibratedClassifierCV` with 5-fold CV.

**Where in code:**
- [models/calibration.py](../src/ced_ml/models/calibration.py) - `maybe_calibrate_estimator`

**Tests:**
- `tests/test_models_calibration.py` - Validates calibration logic

### 10.2 Prevalence Shift (Logit Adjustment)

**Problem:** Training prevalence (1:5 after downsampling) ≠ Deployment prevalence (1:300 real-world).

**Solution:** Logit shift formula (Steyerberg, 2019):

```
P(Y=1|X, prev_new) = sigmoid(logit(p) + logit(prev_new) - logit(prev_old))
```

**Where:**
- `p` = Model's predicted probability
- `prev_old` = Training prevalence (e.g., 1/6 ≈ 0.167)
- `prev_new` = Deployment prevalence (e.g., 1/300 ≈ 0.0033)

**Where in code:**
- [models/calibration.py:117-149](../src/ced_ml/models/calibration.py#L117-L149) - `adjust_probabilities_for_prevalence`

**Tests:**
- `tests/test_models_calibration.py`, `tests/test_prevalence.py` - Validates adjustment logic

**References:**
- Steyerberg (2019). *Clinical Prediction Models (2nd ed.)*, Chapter 13.

**See ADR:**
- [ADR-005: Prevalence Adjustment](adr/ADR-005-prevalence-adjustment.md)

### 10.3 Wrapper Model (PrevalenceAdjustedModel)

**Purpose:** Wrap trained model to automatically apply prevalence adjustment at prediction time.

**Benefits:**
- Serialized model artifact includes adjustment
- Prevents deployment errors (forgetting to adjust)
- Sklearn-compatible (`BaseEstimator`, `ClassifierMixin`)

**Where in code:**
- [models/calibration.py:152-188](../src/ced_ml/models/calibration.py#L152-L188) - `PrevalenceAdjustedModel`
- [evaluation/reports.py](../src/ced_ml/evaluation/reports.py) - `ResultsWriter.save_model_artifact` (uses wrapper)

**Tests:**
- `tests/test_models_calibration.py` - Validates wrapper behavior

**See ADR:**
- [ADR-011: PrevalenceAdjustedModel](adr/ADR-011-prevalence-wrapper.md)

---

## 11. Threshold Selection

### 11.1 Objectives

**Objectives:** Choose decision threshold to optimize a specific criterion.

**Available Objectives:**
- `youden` - Youden's J statistic (sensitivity + specificity - 1)
- `max_f1` - Maximize F1 score
- `fixed_spec` - Achieve fixed specificity (e.g., 0.95)
- `fixed_ppv` - Achieve fixed positive predictive value
- `fixed_sens` - Achieve fixed sensitivity

**Default:** `youden` (balances sensitivity and specificity).

**Where in code:**
- [metrics/thresholds.py:326-377](../src/ced_ml/metrics/thresholds.py#L326-L377) - `choose_threshold_objective`

**Tests:**
- `tests/test_metrics_thresholds.py` - Validates threshold selection

### 11.2 Configuration

**ThresholdConfig:**
```python
objective: str = "youden"               # Objective function
threshold_source: str = "val"           # val | train (never test!)
fixed_spec: float | None = None         # For fixed_spec objective
target_prevalence_source: str = "test"  # test | val | train | fixed
```

**Where in code:**
- [config/schema.py:198-208](../src/ced_ml/config/schema.py#L198-L208) - `ThresholdConfig`

**See ADRs:**
- [ADR-009: Threshold on VAL](adr/ADR-009-threshold-on-val.md)
- [ADR-010: Fixed Spec 95%](adr/ADR-010-fixed-spec-95.md)

---

## 12. Evaluation & Metrics

### 12.1 Discrimination Metrics

**Metrics:**
- `AUROC` - Area under ROC curve (discrimination)
- `PR-AUC` - Area under Precision-Recall curve
- `Brier Score` - Mean squared error of predicted probabilities
- `Sensitivity` - True positive rate
- `Specificity` - True negative rate
- `PPV` - Positive predictive value
- `NPV` - Negative predictive value

**Where in code:**
- [metrics/discrimination.py](../src/ced_ml/metrics/discrimination.py) - `auroc`, `prauc`, `compute_brier_score`

**Tests:**
- `tests/test_metrics_discrimination.py` - Validates metric computations

### 12.2 Decision Curve Analysis (DCA)

**Purpose:** Evaluate net benefit of using the model for clinical decisions across a range of threshold probabilities.

**Metric:** Net benefit = (TP / N) - (FP / N) × (p_t / (1 - p_t))

**Where:**
- `p_t` = Threshold probability
- `TP` = True positives
- `FP` = False positives
- `N` = Total samples

**Where in code:**
- [metrics/dca.py](../src/ced_ml/metrics/dca.py) - `compute_net_benefit`

**Tests:**
- `tests/test_metrics_dca.py` - Validates DCA computations

### 12.3 Bootstrap Confidence Intervals

**Purpose:** Estimate 95% CIs for metrics via stratified bootstrap resampling.

**Method:**
1. Resample TEST set with replacement (stratified by target)
2. Compute metrics on each bootstrap sample
3. Compute 2.5th and 97.5th percentiles

**Default:** 500 bootstrap iterations.

**Where in code:**
- [metrics/bootstrap.py](../src/ced_ml/metrics/bootstrap.py) - `bootstrap_ci`

**Tests:**
- `tests/test_metrics_bootstrap.py` - Validates bootstrap logic

---

## 13. Output Artifacts & Reports

### 13.1 Directory Structure

**Output Directory Pattern:** `results_production/{model}_{scenario}_seed{N}/`

**Contents:**
```
results_production/
  RF_IncidentPlusPrevalent_seed42/
    final_model.pkl               # PrevalenceAdjustedModel wrapper
    oof_predictions.csv           # OOF predictions (TRAIN set)
    val_predictions.csv           # VAL predictions
    test_predictions.csv          # TEST predictions
    train_metrics.json            # TRAIN metrics
    val_metrics.json              # VAL metrics
    test_metrics.json             # TEST metrics
    run_settings.json             # Full config + metadata
    stable_features.txt           # Stable panel proteins
    plots/
      roc_pr.png                  # ROC + PR curves
      calibration.png             # Calibration plot
      risk_dist.png               # Risk distribution
      dca.png                     # Decision curve analysis
```

**Where in code:**
- [evaluation/reports.py](../src/ced_ml/evaluation/reports.py) - `ResultsWriter`, `OutputDirectories`

**Tests:**
- `tests/test_evaluation_reports.py` - Validates output generation

### 13.2 Plots

**Generated Plots:**
1. **ROC + PR Curves** - Discrimination metrics visualization
2. **Calibration Plot** - Predicted vs. observed probabilities
3. **Risk Distribution** - Histogram of predicted probabilities
4. **Decision Curve Analysis** - Net benefit vs. threshold probability

**Where in code:**
- [plotting/roc_pr.py](../src/ced_ml/plotting/roc_pr.py) - ROC/PR plots
- [plotting/calibration.py](../src/ced_ml/plotting/calibration.py) - Calibration plots
- [plotting/risk_dist.py](../src/ced_ml/plotting/risk_dist.py) - Risk distribution plots
- [plotting/dca.py](../src/ced_ml/plotting/dca.py) - DCA plots

**Tests:**
- `tests/test_plotting_*.py` - Validates plotting functions

---

## 14. Reproducibility & Determinism

### 14.1 Seeds

**Random Seeds:**
- `split_seed` - Data split generation
- `random_state` (CVConfig) - Cross-validation fold assignment, RandomizedSearchCV

**Where in code:**
- [config/schema.py](../src/ced_ml/config/schema.py) - `TrainingConfig.split_seed`, `CVConfig.random_state`
- [utils/random.py](../src/ced_ml/utils/random.py) - RNG utilities

### 14.2 Split Persistence

**Mechanism:** Save split indices as CSV files for exact reproducibility.

**Benefits:**
- Same splits across multiple runs
- Comparable results across different models
- Auditable and version-controllable

**Where in code:**
- [data/persistence.py](../src/ced_ml/data/persistence.py) - `save_split_indices`, `load_split_indices`
- [cli/train.py](../src/ced_ml/cli/train.py) - `--split-dir` flag

### 14.3 Metadata Logging

**`run_settings.json`** - Contains:
- Full resolved configuration (all parameters)
- Resolved metadata columns (auto-detected or explicit)
- Split seed, random state
- Model hyperparameters
- Feature selection parameters
- Threshold selection settings
- Software versions (Python, numpy, sklearn, etc.)
- Timestamp, runtime

**Where in code:**
- [evaluation/reports.py](../src/ced_ml/evaluation/reports.py) - `ResultsWriter.save_run_settings`

---

## 15. HPC Orchestration

### 15.1 LSF Template

**Job Array:** 4 models (RF, XGBoost, LinSVM_cal, LR_EN) run in parallel.

**Resource Allocation:**
- 16 cores per job
- 8 GB/core = 128 GB total per job
- 12-hour wall time

**Where in code:**
- `CeD_production.lsf.template` - LSF batch script template

**See ADR:**
- [ADR-014: HPC Job Array](adr/ADR-014-hpc-job-array.md)

### 15.2 Setup Script

**Purpose:** Initialize HPC environment (modules, virtual environment, dependencies).

**Where in code:**
- `scripts/hpc_setup.sh` - Environment setup script

### 15.3 Workflow

**WORKFLOW.md:** High-level pipeline execution steps.

**Steps:**
1. Generate splits: `ced save-splits`
2. Submit job array: `bsub < CeD_production.lsf`
3. Monitor jobs: `bjobs`
4. Postprocess: `ced postprocess`
5. Holdout evaluation: `ced eval-holdout` (run ONCE)

**Where in code:**
- `WORKFLOW.md` - Pipeline documentation

---

## 16. Extension Points

### 16.1 New Models

**Add New Model:**
1. Create builder function in `models/registry.py`
2. Add hyperparameter grid in `models/hyperparams.py`
3. Add config class in `config/schema.py` (if model-specific params needed)
4. Update model registry mapping
5. Add tests in `tests/test_models_*.py`

**Where in code:**
- [models/registry.py](../src/ced_ml/models/registry.py) - Model factory

### 16.2 New Features

**Add New Feature Selection Method:**
1. Implement in `features/` module
2. Add config option in `FeatureConfig`
3. Integrate in training pipeline (`cli/train.py`)
4. Add tests in `tests/test_features_*.py`

**Where in code:**
- [features/](../src/ced_ml/features/) - Feature engineering modules

### 16.3 New Metrics

**Add New Metric:**
1. Implement in `metrics/` module
2. Add config option in `EvaluationConfig` (if needed)
3. Integrate in evaluation pipeline (`evaluation/reports.py`)
4. Add tests in `tests/test_metrics_*.py`

**Where in code:**
- [metrics/](../src/ced_ml/metrics/) - Metrics modules

---

## 17. Non-Negotiable Behaviors

These behaviors are enforced by the codebase and must not be violated:

1. **Leakage Prevention**
   - Prevalent cases never in VAL/TEST (enforced in `add_prevalent_to_train`)
   - Threshold selected on VAL, never on TEST (enforced in `ThresholdConfig`)
   - Hyperparameter tuning only on TRAIN (enforced in nested CV logic)

2. **Split Validation**
   - VAL + TEST fractions must sum to < 1.0 (enforced in `SplitsConfig.validate_split_sizes`)

3. **Threshold Source Validation**
   - If `threshold_source='val'`, requires `cv.folds >= 2` (enforced in `TrainingConfig.validate_config`)

4. **Prevalence Source Validation**
   - If `target_prevalence_source='fixed'`, requires `target_prevalence_fixed` is not None (enforced in `TrainingConfig.validate_config`)

5. **Protein Column Suffix**
   - Protein columns must end with `_resid` suffix (defined in `data/schema.py`)

6. **Metadata Column Flexibility**
   - Metadata columns are flexible via `ColumnsConfig` (auto-detect or explicit mode)
   - Protein-only datasets supported (0 metadata columns)

7. **Split Persistence Format**
   - Indices saved as int64 CSV files (enforced in `data/persistence.py`)

8. **Random State**
   - All random operations seeded for reproducibility (enforced throughout)

9. **PrevalenceAdjustedModel Wrapper**
   - Final model must be wrapped in `PrevalenceAdjustedModel` for deployment (enforced in `ResultsWriter.save_model_artifact`)

**Where in code:**
- [data/splits.py:326-366](../src/ced_ml/data/splits.py#L326-L366) - Prevalent→TRAIN enforcement
- [config/schema.py](../src/ced_ml/config/schema.py) - Config validation
- [config/validation.py](../src/ced_ml/config/validation.py) - Cross-field validation
- [data/schema.py](../src/ced_ml/data/schema.py) - Column constants
- [data/columns.py](../src/ced_ml/data/columns.py) - Flexible column resolution
- [data/persistence.py](../src/ced_ml/data/persistence.py) - Split persistence
- [models/calibration.py:152-188](../src/ced_ml/models/calibration.py#L152-L188) - PrevalenceAdjustedModel wrapper

---

**End of ARCHITECTURE.md**
