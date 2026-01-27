# CLI Reference - CeD-ML Pipeline

High-level command-line interface reference for the Celiac Disease risk prediction pipeline.

## Table of Contents
- [Available Commands](#available-commands)
- [Typical Workflow Order](#typical-workflow-order)
- [Command Descriptions](#command-descriptions)
- [Configuration System](#configuration-system)
- [Environment Notes](#environment-notes)

---

## Available Commands

The CLI provides seven main commands accessed via `ced <command>`:

1. `save-splits` - Generate stratified train/val/test splits
2. `train` - Train single model on one split (supports fixed-panel validation)
3. `train-ensemble` - Train stacking meta-learner on base model predictions
4. `optimize-panel` - Post-hoc RFE panel size optimization (deployment trade-offs)
5. `aggregate-splits` - Aggregate results across multiple splits with bootstrap CIs
6. `eval-holdout` - Evaluate trained model on external holdout data
7. `config` - Validate and compare configuration files

Run `ced --help` or `ced <command> --help` for detailed usage.

---

## Typical Workflow Order

### Standard Pipeline (Single Split)
1. **Generate splits** - Create reproducible train/val/test splits from input data
2. **Train base models** - Train individual models (LR_EN, RF, XGBoost, LinSVM_cal)
3. **Train ensemble** (optional) - Train stacking meta-learner on base model OOF predictions
4. **Evaluate** - Review calibration, discrimination, clinical utility metrics

### Multi-Split Pipeline (Production)
1. **Generate splits** - Create multiple random splits (typically 10)
2. **Train models** - Train on each split independently (parallelizable on HPC)
3. **Train ensembles** (optional) - Train stacking ensemble per split
4. **Aggregate** - Pool results across splits, compute bootstrap confidence intervals
5. **Compare** - Visualize multi-model performance comparisons

### Holdout Validation
1. **Train on primary dataset** - Complete standard pipeline
2. **Evaluate holdout** - Apply trained model to external/temporal validation set

---

## Command Descriptions

### `ced save-splits`

**Purpose:** Generate stratified train/val/test splits with configurable downsampling and optional temporal ordering.

**Key Capabilities:**
- Stratified splitting by outcome (incident/prevalent cases)
- Control downsampling in training set (e.g., 5:1 control:case ratio)
- Multiple random splits for variance estimation
- Temporal splits for chronological validation
- Reproducible via seed control

**Required Inputs:**
- Input data file (CSV/Parquet with proteomics + demographics)
- Split configuration (sizes, downsampling ratio, number of splits)

**Outputs:**
- Pickled split indices (train/val/test row indices per split)

### `ced train`

**Purpose:** Train and evaluate a single model on one data split using nested cross-validation.

**Key Capabilities:**
- Nested CV for unbiased hyperparameter tuning (outer x inner folds)
- Optuna or RandomizedSearchCV for hyperparameter optimization
- Four feature selection strategies (choose via config or CLI flag):
  - **Strategy 1**: `hybrid_stability` (default) - Fast, tuned k-best + stability
  - **Strategy 2**: `rfecv` - Automatic sizing, consensus panels (slow)
  - **Strategy 3**: Post-hoc RFE via `ced optimize-panel` (run after training)
  - **Strategy 4**: Fixed panel via `--fixed-panel` flag (validation mode)
  - See [FEATURE_SELECTION.md](FEATURE_SELECTION.md) for detailed comparison and [ADR-013](../adr/ADR-013-four-strategy-feature-selection.md) for rationale
- Isotonic or Platt scaling calibration
- OOF (out-of-fold) or per-fold calibration strategies
- Prevalence adjustment for deployment scenarios
- Threshold optimization on validation set (fixed specificity, Youden, etc.)
- Comprehensive evaluation (AUROC, PR-AUC, calibration, DCA, risk distributions)

**Required Inputs:**
- Training configuration (model type, CV settings, feature selection, calibration)
- Input data file
- Split seed (which split to use)

**Outputs:**
- Trained model artifacts (pickled estimators, scalers, feature selectors)
- OOF predictions (train/val/test)
- Metrics (discrimination, calibration, thresholds, DCA)
- Plots (ROC, PR, calibration, risk distributions, learning curves)
- Feature selection records (selected proteins per fold)

### `ced train-ensemble`

**Purpose:** Train stacking ensemble meta-learner on base model out-of-fold predictions.

**Key Capabilities:**
- L2 logistic regression meta-learner (default)
- Learns optimal weighting of base model predictions
- Trains on OOF predictions to avoid overfitting
- Expected +2-5% AUROC improvement over best single model
- Supports same calibration and threshold strategies as base models

**Required Inputs:**
- Base model names (comma-separated)
- Split seed
- Base model outputs (OOF predictions must exist)

**Outputs:**
- Ensemble model artifacts
- Ensemble OOF predictions
- Ensemble metrics and plots

### `ced optimize-panel`

**Purpose:** Post-training panel size optimization via Recursive Feature Elimination.

**Key Capabilities:**
- Iterative feature elimination with validation set AUROC tracking
- Pareto frontier analysis (panel size vs. performance)
- Stakeholder-friendly cost-benefit recommendations
- Adaptive, linear, or geometric elimination strategies
- Very fast (~5 minutes for LR_EN, ~15-25 minutes for RF/XGBoost)

**Required Inputs:**
- Trained model bundle (.joblib file from `ced train`)
- Input data file (same as training)
- Split directory and seed

**Outputs:**
- Panel curve plot (Pareto frontier: size vs. AUROC)
- Feature ranking CSV (elimination order)
- Recommendations JSON (knee points, minimum viable panels)
- RFE curve CSV (AUROC at each panel size)

**When to use:**
- Clinical deployment: "What's the smallest panel maintaining 0.90 AUROC?"
- Stakeholder decisions: Cost per protein vs. performance trade-offs
- Rapid iteration: Test 10/20/50 protein panels in minutes
- Complements both hybrid_stability and rfecv strategies (Strategy 3 in four-strategy framework)

**Related documentation:**
- Detailed guide: [FEATURE_SELECTION.md](FEATURE_SELECTION.md) (see Strategy 3: Post-hoc RFE)
- Architecture decision: [ADR-013](../adr/ADR-013-four-strategy-feature-selection.md)

### `ced aggregate-splits`

**Purpose:** Pool results across multiple splits and compute bootstrap confidence intervals.

**Key Capabilities:**
- Bootstrap resampling for 95% CIs on all metrics
- Aggregated calibration (Brier score, slope, intercept)
- Aggregated discrimination (AUROC, PR-AUC, sensitivity, specificity)
- Clinical utility summaries (DCA net benefit curves)
- Feature stability analysis (selection frequency across splits)
- Consensus feature panels (proteins selected in ≥75% of splits)
- Multi-model comparison plots

**Required Inputs:**
- Configuration specifying models, number of splits, bootstrap iterations
- Completed model outputs for each split

**Outputs:**
- Aggregated metrics with bootstrap CIs (JSON + CSV)
- Consensus feature panels
- Comparison plots (ROC, calibration, risk distributions)

### `ced eval-holdout`

**Purpose:** Apply trained model to external holdout dataset for validation.

**Key Capabilities:**
- Load pre-trained model artifacts
- Apply same preprocessing and feature selection
- Generate predictions on new data
- Compute metrics (if outcome labels available)
- Produce evaluation reports

**Required Inputs:**
- Trained model directory (from `train` or `train-ensemble`)
- Holdout data file

**Outputs:**
- Holdout predictions
- Holdout metrics (if labels available)
- Holdout evaluation plots

### `ced config`

**Purpose:** Validate configuration files and compare config versions.

**Key Capabilities:**
- Schema validation (checks required fields, types, ranges)
- Config diffing (compare two YAML configs)
- Override simulation (test CLI overrides without running pipeline)

**Required Inputs:**
- Configuration file path(s)

**Outputs:**
- Validation status or diff output

---

## Configuration System

### Config Hierarchy (Lower Overrides Higher)
1. Base YAML files (`configs/*.yaml`)
2. Environment variables (e.g., `RUN_MODELS`, `DRY_RUN`)
3. CLI flags (e.g., `--model`, `--split-seed`)
4. CLI overrides (`--override path.to.param=value`)

### Core Config Files
- **`training_config.yaml`** - Model settings, CV parameters, feature selection, calibration, thresholds
- **`splits_config.yaml`** - Split generation settings (sizes, downsampling, temporal)
- **`pipeline_local.yaml`** - Local execution settings (paths, models, bootstrap iterations)
- **`pipeline_hpc.yaml`** - HPC execution settings (job resources, parallelization)

### CLI Override Syntax
```bash
--override path.to.param=value
```

Override paths follow YAML nesting (e.g., `cv.folds=10`, `optuna.enabled=true`).

### Common Override Use Cases
- Quick development runs (reduce CV folds, trials)
- Change scoring metrics (AUROC vs PR-AUC vs Brier)
- Toggle Optuna on/off
- Adjust feature selection thresholds
- Enable/disable bootstrap CIs
- Change threshold optimization strategy

---

## Environment Notes

### Working Directory
Always run commands from the `analysis/` directory:
```bash
cd analysis/
ced <command> [options]
```

### Installation
```bash
cd analysis/
pip install -e .          # Standard install
pip install -e ".[dev]"   # With development tools
```

### HPC Setup
HPC environments require one-time setup:
```bash
bash scripts/hpc_setup.sh
source venv/bin/activate
```

### Execution Scripts
- **Local:** `./run_local.sh` (single model, single split, quick validation)
- **HPC:** `./run_hpc.sh` (job array, multiple splits, production runs)
- **Post-training:** `bash scripts/post_training_pipeline.sh --run-id <ID>` (aggregation after HPC jobs)

### Job Monitoring (HPC)
```bash
bjobs -w | grep CeD_                              # Check job status
tail -f logs/CeD_<MODEL>_seed<N>.log             # Monitor logs
ls results/<MODEL>/run_*/split_seed*/            # Check outputs
```

---

## Quick Reference

### Minimal Example (Local Testing)
```bash
cd analysis/
ced save-splits --config configs/splits_config.yaml --infile ../data/input.parquet
ced train --config configs/training_config.yaml --model LR_EN --split-seed 0
```

### Production Example (HPC Multi-Split)
```bash
cd analysis/
bash scripts/hpc_setup.sh
./run_hpc.sh  # Submits job array for all models x all splits
# Wait for jobs to complete
bash scripts/post_training_pipeline.sh --run-id <RUN_ID>
```

### Ensemble Example
```bash
ced train --model LR_EN --split-seed 0
ced train --model RF --split-seed 0
ced train --model XGBoost --split-seed 0
ced train-ensemble --base-models LR_EN,RF,XGBoost --split-seed 0
```

---

**Last Updated:** 2026-01-25
**Pipeline Version:** ced_ml v1.1.0
