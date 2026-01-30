# CLI Reference - CeD-ML Pipeline

High-level command-line interface reference for the Celiac Disease risk prediction pipeline.

## Table of Contents
- [Available Commands](#available-commands)
- [Typical Workflow Order](#typical-workflow-order)
- [Command Descriptions](#command-descriptions)
- [Configuration System](#configuration-system)
- [Automatic Logging](#automatic-logging)
- [Environment Notes](#environment-notes)

---

## Available Commands

The CLI provides ten main commands accessed via `ced <command>`:

1. `run-pipeline` - **Run complete end-to-end workflow (RECOMMENDED)**
2. `save-splits` - Generate stratified train/val/test splits
3. `train` - Train single model on one split (supports fixed-panel validation)
4. `train-ensemble` - Train stacking meta-learner on base model predictions
5. `optimize-panel` - Post-hoc RFE panel size optimization for single models
6. `consensus-panel` - Cross-model consensus panel via Robust Rank Aggregation
7. `aggregate-splits` - Aggregate results across multiple splits with bootstrap CIs
8. `eval-holdout` - Evaluate trained model on external holdout data
9. `config` - Validate and compare configuration files
10. `convert-to-parquet` - Convert proteomics CSV to Parquet format

Run `ced --help` or `ced <command> --help` for detailed usage.

---

## Typical Workflow Order

### Recommended: One-Command Pipeline
**NEW: Run the entire workflow with a single command:**
```bash
ced run-pipeline --infile data/celiac.parquet
```
This automatically orchestrates all steps: splits → train → aggregate → ensemble → optimize panel → consensus.

See [`ced run-pipeline`](#ced-run-pipeline) for details and customization options.

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
5. **Optimize panels** - Determine minimum viable panel sizes per model
6. **Generate consensus** - Create cross-model consensus panel via RRA
7. **Compare** - Visualize multi-model performance comparisons

### Holdout Validation
1. **Train on primary dataset** - Complete standard pipeline
2. **Evaluate holdout** - Apply trained model to external/temporal validation set

---

## Command Descriptions

### `ced run-pipeline`

**Purpose:** Run the complete ML pipeline end-to-end with a single command (RECOMMENDED).

**Key Capabilities:**
- **Auto-discovers input data file** from config or common locations
- Orchestrates the full workflow from data splitting to panel optimization
- Auto-generates splits if not provided
- Trains all specified models across all split seeds
- Automatically aggregates results per model
- Optionally trains ensemble meta-learner
- Optionally optimizes panel sizes for each model
- Optionally generates cross-model consensus panel
- Maintains consistent run_id across all models for easy tracking

**Workflow Steps:**
1. Generate splits (if needed or `--overwrite-splits`)
2. Train base models (all models × all seeds)
3. Aggregate results per model
4. Train ensemble meta-learner (if `--ensemble`, default: enabled)
5. Aggregate ensemble results
6. Optimize panel sizes (if `--optimize-panel`, default: enabled)
7. Generate consensus panel (if `--consensus`, default: enabled)

**Required Inputs:**
- None (fully auto-discovered!)

**Optional Inputs:**
- Input data file (auto-discovered from config or common locations if not provided)
- Configuration file (training settings, feature selection, calibration)
- Models list (default: LR_EN,RF,XGBoost)
- Split seeds list (default: 0,1,2)
- Custom run ID (default: auto-generated timestamp)

**Outputs:**
```
results/
  run_<RUN_ID>/
    LR_EN/
      splits/split_seed0/, split_seed1/, ...
      aggregated/
        optimize_panel/
    RF/
      splits/...
      aggregated/
        optimize_panel/
    XGBoost/
      splits/...
      aggregated/
        optimize_panel/
    ENSEMBLE/  (if --ensemble)
      splits/...
      aggregated/
    consensus/  (if --consensus)
      final_panel.txt
      consensus_ranking.csv
```

**Examples:**

**Quick start - auto-discover everything** (SIMPLEST):
```bash
ced run-pipeline
```

**Or specify data file explicitly:**
```bash
ced run-pipeline --infile data/celiac.parquet
```

**Custom models and seeds:**
```bash
ced run-pipeline \
  --models LR_EN,RF,LinSVM_cal \
  --split-seeds 0,1,2,3,4
```

**Disable optional features** (faster for testing):
```bash
ced run-pipeline \
  --no-ensemble \
  --no-consensus \
  --no-optimize-panel
```

**Custom run ID and output:**
```bash
ced run-pipeline \
  --run-id production_v1 \
  --outdir ../results_production
```

**With config file:**
```bash
ced run-pipeline \
  --config configs/training_config.yaml
```

**Save logs to file:**
```bash
ced run-pipeline \
  --log-file logs/run_$(date +%Y%m%d_%H%M%S).log
```

**Options:**
- `--config, -c PATH`: Path to YAML configuration file
- `--infile PATH`: Input data file (Parquet/CSV, auto-discovered if not provided)
- `--split-dir PATH`: Directory for split indices (auto-generated if not provided)
- `--models TEXT`: Comma-separated list of models (default: LR_EN,RF,XGBoost)
- `--split-seeds TEXT`: Comma-separated list of split seeds (default: 0,1,2)
- `--run-id TEXT`: Shared run ID for all models (default: auto-generated)
- `--outdir PATH`: Output directory (default: results/)
- `--ensemble/--no-ensemble`: Train ensemble meta-learner (default: enabled)
- `--consensus/--no-consensus`: Generate consensus panel (default: enabled)
- `--optimize-panel/--no-optimize-panel`: Optimize panel sizes (default: enabled)
- `--overwrite-splits`: Regenerate splits even if they exist
- `--log-file PATH`: Save pipeline logs to file (in addition to console output)
- `--override KEY=VALUE`: Override config values (repeatable)
- `--pipeline-config PATH`: Pipeline config file (auto-detected: pipeline_local.yaml or pipeline_hpc.yaml)
- `--hpc`: Submit LSF jobs to HPC cluster instead of running locally
- `--hpc-config PATH`: HPC pipeline config (default: configs/pipeline_hpc.yaml)
- `--dry-run`: Preview HPC job submission without executing (--hpc mode only)

**Data File Auto-Discovery:**
When `--infile` is not provided, the command searches for data files in this order:
1. `configs/pipeline_local.yaml` or `configs/pipeline_hpc.yaml` → `paths.infile`
2. Training config file → `infile` key
3. Common locations: `../data/Celiac_dataset_proteomics_w_demo.parquet`, `data/celiac.parquet`, etc.

**HPC mode** (submit LSF jobs with dependency chains):
```bash
# Submit to HPC cluster
ced run-pipeline --hpc

# Preview without submitting
ced run-pipeline --hpc --dry-run

# With custom HPC config
ced run-pipeline --hpc --hpc-config configs/pipeline_custom.yaml
```

**Notes:**
- This command is equivalent to running all individual commands in sequence
- **Zero configuration required** - just run `ced run-pipeline` from the project root
- Progress is logged to stdout with timestamps (optionally save to file with `--log-file`)
- Failed steps will halt the pipeline with an error message
- Use `--no-ensemble --no-consensus --no-optimize-panel` for faster testing
- All models share the same run_id for easy aggregation and comparison

---

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
- CSV split indices (train/val/test row indices per split per scenario per seed)

### `ced train`

**Purpose:** Train and evaluate a single model on one data split using nested cross-validation.

**Key Capabilities:**
- Nested CV for unbiased hyperparameter tuning (outer x inner folds)
- Optuna or RandomizedSearchCV for hyperparameter optimization
- Five feature selection strategies (choose via config or CLI flag):
  - **Strategy 1**: `hybrid_stability` (default) - Fast, tuned k-best + stability
  - **Strategy 2**: `rfecv` - Automatic sizing, consensus panels (slow)
  - **Strategy 3**: Post-hoc RFE via `ced optimize-panel` (single-model deployment)
  - **Strategy 4**: Consensus panel via `ced consensus-panel` (cross-model deployment)
  - **Strategy 5**: Fixed panel via `--fixed-panel` flag (validation mode)
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
- **Auto-detects** results directory and base models from `--run-id`

**Required Inputs:**
- Base model names (auto-detected from run-id) OR explicit list
- Split seed
- Base model outputs (OOF predictions must exist)

**Outputs:**
- Ensemble model artifacts
- Ensemble OOF predictions
- Ensemble metrics and plots

**Usage Examples:**
```bash
# Auto-detection (RECOMMENDED) - discovers all base models automatically
ced train-ensemble --run-id 20260127_115115 --split-seed 0

# Manual specification (legacy)
ced train-ensemble \
  --results-dir results/ \
  --base-models LR_EN,RF,XGBoost \
  --split-seed 0
```

### `ced optimize-panel`

**Purpose:** Post-aggregation panel size optimization via RFE on consensus stable proteins.

**Key Capabilities:**
- Uses consensus stable proteins from ALL splits (eliminates split variability)
- Pools train/val data for maximum robustness
- Iterative feature elimination with cross-validation AUROC tracking
- Pareto frontier analysis (panel size vs. performance)
- Stakeholder-friendly cost-benefit recommendations
- Adaptive, linear, or geometric elimination strategies
- Fast (~10 minutes on pooled data)

**Required Inputs:**
- Aggregated results directory (must run `ced aggregate-splits` first)
- Input data file (same as training)
- Split directory

**Outputs (saved in `results_dir/aggregated/optimize_panel/`):**
- `panel_curve_aggregated.csv` - Full RFE curve with all metrics
- `feature_ranking_aggregated.csv` - Protein elimination order with importance scores
- `recommended_panels_aggregated.json` - Minimum viable panel sizes at 95%/90%/85% thresholds
- `panel_curve_aggregated.png` - Visualization of size vs. AUROC trade-off

**When to use:**
- Clinical deployment: "What's the smallest panel maintaining 0.90 AUROC?"
- Stakeholder decisions: Cost per protein vs. performance trade-offs
- Authoritative panel sizing (uses consensus across all splits)
- Post-aggregation analysis (complements hybrid_stability and rfecv strategies)

**Examples:**
```bash
# Optimize ALL models under a run-id (RECOMMENDED - auto-detects paths)
ced optimize-panel --run-id 20260127_115115

# Optimize specific model by run-id
ced optimize-panel --run-id 20260127_115115 --model LR_EN

# Optimize single model with explicit path (legacy)
ced optimize-panel \
  --results-dir results/LR_EN/run_20260127_115115 \
  --infile data/Celiac_dataset_proteomics_w_demo.parquet \
  --split-dir splits/
```

**Related documentation:**
- Detailed guide: [FEATURE_SELECTION.md](FEATURE_SELECTION.md) (see Aggregated RFE section)
- Architecture decision: [ADR-013](../adr/ADR-013-four-strategy-feature-selection.md)

### `ced consensus-panel`

**Purpose:** Generate cross-model consensus protein panel via Robust Rank Aggregation (RRA).

**Key Capabilities:**
- Aggregates rankings from multiple base models (LR_EN, RF, XGBoost, etc.)
- Combines stability frequency + RFE importance into per-model composite scores
- Cross-model aggregation via geometric mean of reciprocal ranks
- Correlation clustering to remove redundant proteins
- Produces final panel compatible with `--fixed-panel` validation

**Required Inputs:**
- Aggregated results from multiple models (must run `ced aggregate-splits` first)
- Run ID to auto-discover all models with stability results

**Outputs (saved in `results/run_<RUN_ID>/consensus/`):**
- `final_panel.txt` - One protein per line (for `--fixed-panel` training)
- `final_panel.csv` - Panel with consensus scores
- `consensus_ranking.csv` - All proteins with RRA scores
- `per_model_rankings.csv` - Per-model composite rankings
- `correlation_clusters.csv` - Cluster assignments and pruning info
- `consensus_metadata.json` - Run parameters and statistics

**When to use:**
- Final panel selection using consensus across multiple model types
- More robust than single-model panel (reduces model-specific bias)
- Before fixed-panel validation training

**Examples:**
```bash
# Generate consensus panel from all models
ced consensus-panel --run-id 20260127_115115

# Custom parameters
ced consensus-panel --run-id 20260127_115115 \
  --target-size 30 \
  --corr-threshold 0.90 \
  --rfe-weight 0.3

# Validate the resulting panel (use NEW split seed)
ced train --model LR_EN \
  --infile data/input.parquet \
  --fixed-panel results/run_20260127_115115/consensus/final_panel.txt \
  --split-seed 10
```

**Related documentation:**
- Configuration: `configs/consensus_panel.yaml`

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
- **Auto-detects** results directory from `--run-id`

**Required Inputs:**
- Configuration specifying models, number of splits, bootstrap iterations (config-based mode)
- OR run-id + model name for auto-detection mode
- Completed model outputs for each split

**Outputs:**
- Aggregated metrics with bootstrap CIs (JSON + CSV)
- Consensus feature panels
- Comparison plots (ROC, calibration, risk distributions)

**Usage Examples:**
```bash
# Auto-detection with run-id (RECOMMENDED) - specific model
ced aggregate-splits --run-id 20260127_115115 --model LR_EN

# Auto-detection with run-id - all models for a run
ced aggregate-splits --run-id 20260127_115115

# Explicit path (alternative)
ced aggregate-splits --results-dir results/run_20260127_115115/LR_EN/
```

**Notes:**
- `--results-dir` and `--run-id` are mutually exclusive
- If multiple models exist for a run-id, use `--model` to specify which one
- Run this AFTER completing training on all desired splits

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

### `ced convert-to-parquet`

**Purpose:** Convert proteomics CSV files to Parquet format with compression.

**Key Capabilities:**
- Efficient binary storage with compression (snappy/gzip/brotli/zstd)
- Faster I/O for large datasets
- Type-safe column storage
- Includes only modeling-relevant columns (matches training pipeline filter)

**Required Inputs:**
- CSV file path

**Outputs:**
- Parquet file (default: same name with .parquet extension)

**Usage Examples:**
```bash
# Basic conversion (uses snappy compression by default)
ced convert-to-parquet data/celiac_proteomics.csv

# Custom output path and compression
ced convert-to-parquet data/celiac_proteomics.csv \
  -o data/celiac.parquet \
  --compression gzip

# Available compression algorithms: snappy, gzip, brotli, zstd, none
ced convert-to-parquet data/celiac.csv --compression zstd
```

**When to use:**
- Large CSV files (>100 MB) for faster loading
- Repeated training runs (I/O speedup)
- Storage optimization (compression reduces file size)

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

## Automatic Logging

All CLI commands automatically write log files to structured subdirectories under `logs/`, alongside `results/`. No extra flags are needed -- logs are created on every invocation.

### Log directory structure

```
logs/
  training/
    run_{ID}/
      LR_EN_seed0.log
      RF_seed0.log
      XGBoost_seed1.log
  ensemble/
    run_{ID}/
      ENSEMBLE_seed0.log
  aggregation/
    run_{ID}/
      LR_EN.log
      RF.log
  optimization/
    run_{ID}/
      LR_EN.log
  consensus/
    run_{ID}/
      consensus.log
  pipeline/
    run_{ID}.log
```

### Command-to-log mapping

| Command | Log path |
|---------|----------|
| `ced train` | `logs/training/run_{ID}/{model}_seed{N}.log` |
| `ced train-ensemble` | `logs/ensemble/run_{ID}/ENSEMBLE_seed{N}.log` |
| `ced aggregate-splits` | `logs/aggregation/run_{ID}/{model}.log` |
| `ced optimize-panel` | `logs/optimization/run_{ID}/{model}.log` |
| `ced consensus-panel` | `logs/consensus/run_{ID}/consensus.log` |
| `ced run-pipeline` | `logs/pipeline/run_{ID}.log` |

### Notes

- Logs are written in addition to console output (both always active).
- The `--log-file` flag on `ced run-pipeline` overrides the auto-generated path.
- Log files use the same format: `[YYYY-MM-DD HH:MM:SS] LEVEL - message`.
- The `logs/` directory is created as a sibling of the `results/` output directory.

---

## Environment Notes

### Working Directory
Always run commands from the project root (`CeliacRisks/`):
```bash
cd CeliacRisks/
ced <command> [options]
```

### Installation
```bash
cd CeliacRisks/
pip install -e analysis/          # Standard install
pip install -e "analysis/[dev]"   # With development tools
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
cd CeliacRisks/
ced save-splits --config configs/splits_config.yaml --infile data/input.parquet
ced train --config configs/training_config.yaml --model LR_EN --infile data/input.parquet --split-seed 0
```

### Production Example (HPC Multi-Split)
```bash
cd CeliacRisks/
bash analysis/scripts/hpc_setup.sh

# Submit HPC pipeline (auto-manages job dependencies)
ced run-pipeline --hpc

# Monitor jobs
bjobs -w | grep CeD_
```

### Ensemble Example
```bash
# Train base models
ced train --model LR_EN --infile data/input.parquet --split-seed 0
ced train --model RF --infile data/input.parquet --split-seed 0
ced train --model XGBoost --infile data/input.parquet --split-seed 0

# Train ensemble (auto-detects base models from run-id)
ced train-ensemble --run-id 20260127_115115 --split-seed 0

# Aggregate results
ced aggregate-splits --run-id 20260127_115115 --model LR_EN
ced aggregate-splits --run-id 20260127_115115 --model ENSEMBLE

# Optimize panels
ced optimize-panel --run-id 20260127_115115 --model LR_EN
ced consensus-panel --run-id 20260127_115115
```

---

**Last Updated:** 2026-01-30
**Pipeline Version:** ced_ml v1.3.0 (complete `--run-id` auto-detection, `--hpc` mode)
