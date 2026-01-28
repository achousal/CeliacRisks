# CeliacRisks Project Documentation

**Project**: Machine Learning Pipeline for Incident Celiac Disease Risk Prediction
**Version**: 1.3.0
**Updated**: 2026-01-28
**Primary Package**: ced-ml
**Python**: 3.10+
**Project Owner**: Andres Chousal (Chowell Lab)
**Status**: Production-ready with stacking ensemble, OOF-posthoc calibration, temporal validation, panel optimization, and cross-model consensus

---

## Project Mission

Build calibrated ML models to predict **incident Celiac Disease (CeD) risk** from proteomics biomarkers measured **before clinical diagnosis**. Generate continuous risk scores for apparently healthy individuals to inform follow-up testing decisions.

### Clinical Workflow
```
Blood proteomics panel → ML risk score → [High risk?] → Anti-tTG antibody test → Endoscopy
```

---

## Quick Start

### Local testing
```bash
cd analysis/
pip install -e .
./run_local.sh
```

### HPC production
```bash
cd analysis/
bash scripts/hpc_setup.sh
./run_hpc.sh
```

### CLI reference
```bash
ced --help
ced save-splits --config configs/splits_config.yaml --infile ../data/input.parquet
ced train --config configs/training_config.yaml --model LR_EN --infile ../data/input.parquet --split-seed 0
ced train-ensemble --base-models LR_EN,RF,XGBoost --split-seed 0
ced aggregate-splits --config configs/aggregate_config.yaml
ced optimize-panel --run-id 20260127_115115 --model LR_EN
ced consensus-panel --run-id 20260127_115115
```

---

## Dataset

| Attribute | Value |
|-----------|-------|
| **Total samples** | 43,960 |
| **Controls** | 43,662 |
| **Incident CeD** | 148 (0.34%) - biomarkers BEFORE diagnosis |
| **Prevalent CeD** | 150 - used in TRAIN only (50% sampling) |
| **Proteins** | 2,920 (`*_resid` columns) |
| **Demographics** | age, BMI, sex, Genetic ethnic grouping (configurable via `ColumnsConfig`) |
| **Missing proteins** | Zero |
| **Missing ethnicity** | 17% (handled as "Missing" category) |

## Package Architecture

**Stats**: ~56,800 lines of code (including tests), 1,271 tests (14% coverage after refactoring expansion).

For detailed architecture with code pointers, see [docs/ARCHITECTURE.md](analysis/docs/ARCHITECTURE.md).

### Library Modules
| Layer | Modules | Purpose |
|-------|---------|---------|
| Data | `io`, `splits`, `persistence`, `filters`, `schema`, `columns` | Data loading, split generation, column resolution |
| Features | `screening`, `kbest`, `stability`, `corr_prune`, `panels`, `rfe`, `nested_rfe`, `consensus` | Feature selection and cross-model consensus |
| Models | `registry`, `hyperparams`, `optuna_search`, `training`, `calibration`, `prevalence`, `stacking` | Model training, hyperparameter optimization, and ensemble learning |
| Metrics | `discrimination`, `thresholds`, `dca`, `bootstrap` | Performance metrics |
| Evaluation | `predict`, `reports`, `holdout` | Prediction and reporting |
| Plotting | `roc_pr`, `calibration`, `risk_dist`, `dca`, `learning_curve`, `oof`, `optuna_plots`, `panel_curve`, `ensemble` | Visualization |

For output structure details, see [docs/reference/ARTIFACTS.md](analysis/docs/reference/ARTIFACTS.md).

---

## Key Architecture Decisions

The [docs/adr/](analysis/docs/adr/) directory contains 15 Architecture Decision Records documenting critical statistical and methodological design choices, organized by pipeline stage:

**Stage 1: Data Preparation**
- [ADR-001](analysis/docs/adr/ADR-001-split-strategy.md): 50/25/25 train/val/test split strategy
- [ADR-002](analysis/docs/adr/ADR-002-prevalent-train-only.md): Prevalent cases in training only
- [ADR-003](analysis/docs/adr/ADR-003-control-downsampling.md): Control downsampling ratio

**Stage 2: Feature Selection**
- [ADR-013](analysis/docs/adr/ADR-013-four-strategy-feature-selection.md): Four-strategy feature selection framework (rationale, use cases, trade-offs)
- [ADR-004](analysis/docs/adr/ADR-004-hybrid-feature-selection.md): Strategy 1 - Hybrid Stability (production default, tuned k-best)
- [ADR-005](analysis/docs/adr/ADR-005-stability-panel.md): Stability panel extraction (0.75 threshold, used by all strategies)

**Stage 3: Model Training & Ensembling**
- [ADR-006](analysis/docs/adr/ADR-006-nested-cv.md): Nested cross-validation structure
- [ADR-007](analysis/docs/adr/ADR-007-auroc-optimization.md): AUROC as optimization metric
- [ADR-008](analysis/docs/adr/ADR-008-optuna-hyperparameter-optimization.md): Optuna Bayesian hyperparameter optimization
- [ADR-009](analysis/docs/adr/ADR-009-oof-stacking-ensemble.md): Out-of-fold stacking ensemble (implemented 2026-01-22)

**Stage 4: Calibration**
- [ADR-010](analysis/docs/adr/ADR-010-prevalence-adjustment.md): Prevalence adjustment (speculative deployment concern)
- [ADR-014](analysis/docs/adr/ADR-014-oof-posthoc-calibration.md): OOF-posthoc calibration strategy

**Stage 5: Evaluation & Thresholds**
- [ADR-011](analysis/docs/adr/ADR-011-threshold-on-val.md): Threshold optimization on validation set
- [ADR-012](analysis/docs/adr/ADR-012-fixed-spec-95.md): Fixed specificity 0.95 for high-specificity screening

---

---

## Core Workflows

### 1. Generate Splits
```bash
ced save-splits \
  --config configs/splits_config.yaml \
  --infile ../data/Celiac_dataset_proteomics_w_demo.parquet
```
Creates stratified train/val/test splits with configurable:
- `val_size`, `test_size` (default: 0.25 each)
- `n_splits` (number of random splits, default: 10)
- `train_control_per_case` (downsampling ratio, default: 5.0)
- `seed_start` (reproducibility)

Output: `../splits/splits_*.pkl`

### 2. Train Models
```bash
ced train \
  --config configs/training_config.yaml \
  --model LR_EN \
  --infile ../data/Celiac_dataset_proteomics_w_demo.parquet \
  --split-seed 0
```
Trains single model on one split with:
- Nested CV (5-fold outer x 3 repeats x 3-fold inner by default)
- Optuna hyperparameter tuning (100 trials, TPE sampler, median pruner)
- **Feature selection** (choose one strategy):
  - **hybrid_stability** (default): screen → kbest (tuned) → stability → model
  - **rfecv**: screen → RFECV (within CV) → model
- Calibration (isotonic regression + prevalence adjustment)
- Threshold optimization (fixed specificity 0.95 on validation set)

Output: `../results/{model}/split_seed{N}/`

### 3. Feature Selection

The pipeline provides **four** distinct feature selection methods, each optimized for different use cases:

| Method | Type | Use Case | Speed |
|--------|------|----------|-------|
| **Hybrid Stability** | During training | Production models (default) | Fast (~30 min) |
| **Nested RFECV** | During training | Scientific discovery | Slow (~22 hours) |
| **Aggregated RFE** | After aggregation | **Single-model deployment optimization** | Fast (~10 min) |
| **Consensus Panel** | After aggregation | **Cross-model deployment optimization** | Fast (~15 min) |
| **Fixed Panel** | During training | Panel validation | Fast (~30 min) |

**Quick Start:**
- Default: `feature_selection_strategy: hybrid_stability` (recommended)
- For feature stability analysis: `feature_selection_strategy: rfecv`
- **For single-model panel sizing: `ced optimize-panel --run-id <RUN_ID> --model LR_EN`**
- **For cross-model consensus panel: `ced consensus-panel --run-id <RUN_ID>`**
- For panel validation: `ced train --fixed-panel panel.csv --split-seed 10`

**IMPORTANT:** Methods 1-2 are mutually exclusive (choose during training). Methods 3-5 are post-training tools.

**For detailed documentation, see [docs/reference/FEATURE_SELECTION.md](analysis/docs/reference/FEATURE_SELECTION.md)**

### 4. Train Ensemble (Optional)
```bash
# Train base models first
ced train --model LR_EN --split-seed 0
ced train --model RF --split-seed 0
ced train --model XGBoost --split-seed 0

# Train stacking ensemble (auto-detects base models from run-id)
ced train-ensemble --run-id 20260127_115115 --split-seed 0

# Alternative: Explicit base models (manual)
ced train-ensemble --results-dir results/ --base-models LR_EN,RF,XGBoost --split-seed 0
```
Trains L2 logistic regression meta-learner on OOF predictions from base models.
Expected improvement: +2-5% AUROC over best single model.

**Auto-detection** (recommended): Use `--run-id` to automatically discover results directory and base models.

Output: `../results/ENSEMBLE/run_{RUN_ID}/split_seed{N}/`

### 5. Post-Training Pipeline (HPC only)
After HPC jobs complete, run the comprehensive post-processing script:
```bash
# Check job status
bjobs -w | grep CeD_

# When all jobs done, run post-processing (auto-detects everything from run-id)
bash scripts/post_training_pipeline.sh --run-id <RUN_ID>

# With ensemble training
bash scripts/post_training_pipeline.sh --run-id <RUN_ID> --train-ensemble

# Manual overrides (disables auto-detection)
bash scripts/post_training_pipeline.sh --run-id <RUN_ID> \
  --base-models LR_EN,RF,XGBoost \
  --results-dir ../results
```

This automated pipeline **auto-detects** models and splits from the run-id:
1. **Auto-detects** base models by scanning for `run_{RUN_ID}` directories
2. **Validates** base model outputs (checks required files per split)
3. **Trains ensemble** meta-learner on OOF predictions (if `--train-ensemble`, auto-detects base models)
4. **Aggregates** results across splits (per model, with bootstrap CIs)
5. **Generates** validation reports and summary JSON

**No config coordination needed** - just provide `--run-id` and the script handles the rest. Ensemble training now uses `ced train-ensemble --run-id <RUN_ID>` for full auto-detection (no need to specify base models).

Aggregation includes:
- Calibration: Brier score, slope, intercept
- Discrimination: AUROC, PR-AUC, sensitivity/specificity
- Clinical utility: DCA net benefit curves
- Risk distributions and calibration plots
- Feature stability and consensus panels

Output: `../results/{MODEL}/run_{RUN_ID}/aggregated/`

**Local alternative** (manual aggregation):
```bash
ced aggregate-splits --config configs/aggregate_config.yaml
```

### 6. Visualize (Optional)
```bash
Rscript scripts/compare_models.R --results_root ../results
```
Multi-model comparison plots (requires R + ggplot2)

---

## Configuration System

**Config hierarchy** (lower overrides higher):
1. `configs/training_config.yaml` (model settings, feature selection)
2. `configs/splits_config.yaml` (CV split settings)
3. `configs/pipeline_local.yaml` or `pipeline_hpc.yaml` (execution settings)
4. Environment variables (e.g., `RUN_MODELS`, `DRY_RUN`)
5. CLI flags (e.g., `--model`, `--split-seed`)

**Key config files**:

### pipeline_local.yaml
```yaml
environment: local
paths:
  infile: ../data/Celiac_dataset_proteomics_w_demo.parquet
  splits_dir: ../splits
  results_dir: ../results
execution:
  models: [LR_EN, RF, LinSVM_cal, XGBoost]
  n_boot: 100
  overwrite_splits: false
```

### pipeline_hpc.yaml
```yaml
environment: hpc
hpc:
  project: YOUR_ALLOCATION  # Update this!
  queue: medium
  cores: 4
  memory: 16G
  walltime: "12:00"
execution:
  models: [LR_EN, RF, LinSVM_cal, XGBoost]
  n_boot: 500
```

### training_config.yaml
```yaml
cv:
  n_outer: 5
  n_repeats: 3
  n_inner: 3

optuna:
  enabled: true
  n_trials: 100
  sampler: tpe
  pruner: median

features:
  feature_selection_strategy: hybrid_stability  # Options: hybrid_stability, rfecv, none
  screen_method: mannwhitney
  screen_top_n: 1000

  # Hybrid Stability parameters
  k_grid: [25, 50, 100, 150, 200, 300, 400]
  stability_thresh: 0.70
  stable_corr_thresh: 0.85

  # Nested RFECV parameters (used if strategy=rfecv)
  rfe_target_size: 50        # Minimum features (stops at 50//2 = 25)
  rfe_step_strategy: adaptive
  rfe_consensus_thresh: 0.80

calibration:
  enabled: true
  strategy: per_fold    # or oof_posthoc (unbiased), none
  method: isotonic      # or sigmoid for Platt scaling

ensemble:
  enabled: false        # Opt-in
  method: stacking
  base_models: [LR_EN, RF, XGBoost, LinSVM_cal]
  meta_model:
    type: logistic_regression
    penalty: l2
    C: 1.0

splits:
  temporal_split: false           # Enable for chronological validation
  temporal_column: "sample_date"  # Column with dates/timestamps

thresholds:
  objective: fixed_spec
  fixed_spec: 0.95
  target_prevalence_source: test
```

---

## CLI Reference

For complete CLI documentation, see [analysis/docs/reference/CLI_REFERENCE.md](analysis/docs/reference/CLI_REFERENCE.md)

### CLI Commands
| Command | Module | Purpose |
|---------|--------|---------|
| `ced save-splits` | `cli/save_splits.py` | Split generation |
| `ced train` | `cli/train.py` | Model training |
| `ced train-ensemble` | `cli/train_ensemble.py` | Ensemble meta-learner training |
| `ced optimize-panel` | `cli/optimize_panel.py` | **Panel optimization (aggregated RFE)** |
| `ced consensus-panel` | `cli/consensus_panel.py` | **Cross-model consensus panel (RRA)** |
| `ced aggregate-splits` | `cli/aggregate_splits.py` | Results aggregation |
| `ced eval-holdout` | `cli/eval_holdout.py` | Holdout evaluation |
| `ced config` | `cli/config_tools.py` | Config validation and diff |

## Testing

```bash
# Run all tests
pytest tests/ -v
```

**Test suite:** 1,271 tests covering:
- Data I/O (CSV/Parquet), column resolution, and split generation
- Feature screening, k-best, stability, correlation pruning, panels, RFE, consensus
- Model registry, hyperparameters, training, calibration, prevalence, stacking
- Discrimination, thresholds, DCA, bootstrap
- Prediction, reports, holdout evaluation
- ROC/PR, calibration, risk distribution, DCA, learning curve, OOF, panel optimization plots
- CLI integration (including optimize-panel, consensus-panel, ensemble)
- Config validation and comparison

**Coverage:** 14% overall (after major refactoring expansion)

**Test markers:**
- `slow`: Integration tests that train real models (10-20s each). Skip with `pytest -m "not slow"` for faster development.

---

### Pipeline Configs
- **`configs/pipeline_local.yaml`**: Local development pipeline settings
- **`configs/pipeline_hpc.yaml`**: HPC production pipeline settings

These files control:
- Input/output paths
- Models to run
- Number of bootstrap iterations
- Execution flags (dry_run, overwrite_splits)
- HPC resources (cores, memory, walltime, queue, project allocation)

### Core Configs
- **`configs/splits_config.yaml`**: CV split generation settings
- **`configs/training_config.yaml`**: Model training and evaluation settings

---

## Documentation

| Document | Purpose |
|----------|---------|
| [analysis/docs/ARCHITECTURE.md](analysis/docs/ARCHITECTURE.md) | Technical architecture with code pointers |
| [analysis/docs/adr/](analysis/docs/adr/) | Architecture Decision Records (19 decisions) |
| [analysis/docs/reference/CLI_REFERENCE.md](analysis/docs/reference/CLI_REFERENCE.md) | Complete CLI command reference |
| [analysis/docs/reference/FEATURE_SELECTION.md](analysis/docs/reference/FEATURE_SELECTION.md) | Feature selection guide (all 4 strategies, workflows, troubleshooting) |
| [analysis/SETUP_README.md](analysis/SETUP_README.md) | Environment setup and getting started |
| [README.md](README.md) | Root project overview |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guidelines |
---

## Development Workflow

### Local development
```bash
# 1. Setup
cd analysis/
conda create -n ced_ml python=3.10
conda activate ced_ml
pip install -e ".[dev]"

# 2. Test changes
pytest tests/ -v
pytest tests/ --cov=ced_ml

# 3. Lint/format
ruff check src/ tests/
black src/ tests/

# 4. Quick validation
./run_local.sh

# 5. Commit
git add .
git commit -m "feat(models): add Optuna pruning support"
```

### HPC deployment
```bash
# 1. Test locally first
./run_local.sh

# 2. Setup HPC environment
bash scripts/hpc_setup.sh
source venv/bin/activate

# 3. Edit HPC config
nano configs/pipeline_hpc.yaml

# 4. Dry run
DRY_RUN=1 ./run_hpc.sh

# 5. Submit production jobs
./run_hpc.sh

# 6. Monitor
bjobs -w | grep CeD_
```

---

## Enhanced Testing Info

**Test suite**: 1,271 tests covering data I/O, feature selection, models, metrics, evaluation, plotting, and CLI integration.

```bash
# With coverage
pytest tests/ --cov=ced_ml --cov-report=term-missing

# Fast tests only
pytest tests/ -m "not slow"
```

---

## Reproducibility and Provenance

**Reproducibility guarantees**:
1. Fixed RNG seeds (all randomness seeded)
2. Config logging (full YAML configs saved)
3. Versioning (Git commit hash logged)
4. Environment tracking (package versions recorded)
5. Deterministic splits (persisted and reused)

---

## Common Tasks

### Add a new model
1. Edit [analysis/src/ced_ml/models/registry.py](analysis/src/ced_ml/models/registry.py)
2. Add param grid and model factory
3. Add tests
4. Update configs
5. Run `./run_local.sh`

### Train stacking ensemble
```bash
ced train --model LR_EN --split-seed 0
ced train --model RF --split-seed 0
ced train --model XGBoost --split-seed 0
ced train-ensemble --base-models LR_EN,RF,XGBoost --split-seed 0
```

### Optimize panel size for clinical deployment

**Aggregated RFE** (run AFTER aggregation, pools all splits):
```bash
# Method 1: Optimize ALL models under a run-id (RECOMMENDED - auto-detects paths)
ced optimize-panel --run-id 20260127_115115

# Method 2: Optimize specific model(s) by run-id
ced optimize-panel --run-id 20260127_115115 --model LR_EN

# Method 3: Optimize single model with explicit path (legacy)
ced optimize-panel \
  --results-dir results/LR_EN/run_20260127_115115 \
  --infile ../data/Celiac_dataset_proteomics_w_demo.parquet \
  --split-dir ../splits/
```
**Advantages:**
- Uses consensus stable proteins from ALL splits (eliminates variability)
- Pools train/val data for maximum robustness
- Generates single authoritative panel size recommendation
- Matches aggregated analysis philosophy

See [docs/reference/FEATURE_SELECTION.md](analysis/docs/reference/FEATURE_SELECTION.md) for detailed comparison.

### Validate a deployment panel (fixed-panel training)

Validate a specific panel with unbiased AUROC estimate:

```bash
# Step 1: Extract consensus panel from previous training
awk -F',' 'NR==1 || $2 >= 0.70 {print $1}' \
  results/LR_EN/aggregated/feature_stability.csv \
  > deployment_panel.csv

# Step 2: Validate with NEW split seed (critical for unbiased estimate)
ced train \
  --model LR_EN \
  --fixed-panel deployment_panel.csv \
  --split-seed 10 \
  --config configs/training_config.yaml
```

**Key points:**
- Always use a **new split seed** (prevents peeking at discovery splits)
- Feature selection is completely bypassed (trains on exact panel)
- Provides regulatory-grade performance estimate
- Use for FDA submission, clinical deployment, literature comparison

See [docs/reference/FEATURE_SELECTION.md](analysis/docs/reference/FEATURE_SELECTION.md) for detailed workflows.

### Use OOF-posthoc calibration
Set `calibration.strategy: oof_posthoc` in `configs/training_config.yaml`

### Enable temporal validation
Set `splits.temporal_split: true` and specify `temporal_column` in `configs/training_config.yaml`

---

## Environment Variables

**Pipeline runners**:
- `PIPELINE_CONFIG`: Override config file path
- `RUN_MODELS`: Comma-separated list (e.g., "LR_EN,RF")
- `DRY_RUN`: Preview without execution
- `OVERWRITE_SPLITS`: Regenerate splits

**HPC-specific**:
- `PROJECT`: HPC allocation (required)
- `QUEUE`: Queue name (default: medium)

---

## Troubleshooting

### "ced: command not found"
```bash
pip install -e .
```

### "XGBoost Library could not be loaded" (macOS)
```bash
brew install libomp
```

### Tests failing
```bash
rm -rf .pytest_cache __pycache__
pip install -e . --force-reinstall --no-deps
pytest tests/ -vv
```

### HPC jobs complete but no aggregated results
```bash
# Check job status
bjobs -w | grep CeD_

# When done, run post-processing
bash scripts/post_training_pipeline.sh --run-id <RUN_ID>
```

### Ensemble training fails
Check base model outputs exist:
```bash
ls results/{MODEL}/run_{RUN_ID}/split_seed*/preds/train_oof/
```

---

## Key Files to Know

**Core modules**:
- [analysis/src/ced_ml/cli/main.py](analysis/src/ced_ml/cli/main.py) - CLI entrypoint
- [analysis/src/ced_ml/cli/train_ensemble.py](analysis/src/ced_ml/cli/train_ensemble.py) - Ensemble training
- [analysis/src/ced_ml/cli/optimize_panel.py](analysis/src/ced_ml/cli/optimize_panel.py) - Panel optimization
- [analysis/src/ced_ml/cli/consensus_panel.py](analysis/src/ced_ml/cli/consensus_panel.py) - Cross-model consensus panel
- [analysis/src/ced_ml/data/splits.py](analysis/src/ced_ml/data/splits.py) - Splitting with temporal support
- [analysis/src/ced_ml/features/rfe.py](analysis/src/ced_ml/features/rfe.py) - RFE algorithm
- [analysis/src/ced_ml/features/consensus.py](analysis/src/ced_ml/features/consensus.py) - RRA consensus aggregation
- [analysis/src/ced_ml/models/stacking.py](analysis/src/ced_ml/models/stacking.py) - Stacking meta-learner
- [analysis/src/ced_ml/models/calibration.py](analysis/src/ced_ml/models/calibration.py) - OOF calibration

**Helper scripts**:
- [analysis/scripts/post_training_pipeline.sh](analysis/scripts/post_training_pipeline.sh) - HPC post-processing
- [analysis/scripts/hpc_setup.sh](analysis/scripts/hpc_setup.sh) - HPC environment setup

---

## Recent Major Changes

**2026-01-28:**
1. **Ensemble Training Auto-Detection** - `ced train-ensemble` now supports `--run-id` for automatic base model discovery (no config coordination needed)
2. **Post-Training Pipeline Auto-Detection** - `post_training_pipeline.sh` now auto-detects models and splits from run-id (no config coordination needed)
3. **Conda Environment Support** - HPC post-training pipeline now supports both venv and conda environments

**2026-01-27:**
1. **Cross-Model Consensus Panel** - `ced consensus-panel` generates consensus protein panel via Robust Rank Aggregation across multiple models
2. **Multi-Model Batch Processing** - `ced optimize-panel --run-id` now processes all base models automatically
3. **Investigation Workflows Consolidation** - Factorial experiment framework for prevalent case and control ratio optimization
4. **Test Suite Modernization** - Expanded to 1,271 tests with improved E2E coverage and ensemble testing
5. **Modular Aggregation Refactor** - Extracted orchestration helpers and metrics aggregation for maintainability

**2026-01-26:**
1. **Aggregated Panel Optimization** - Panel sizing via RFE on consensus stable proteins across all splits
2. **Model Stacking Ensemble** - L2 meta-learner, +2-5% AUROC expected
3. **OOF-Posthoc Calibration** - Eliminates ~0.5-1% optimistic bias
4. **Expanded Optuna Ranges** - Wider hyperparameter search space
5. **Temporal Validation** - Chronological train/val/test splits
6. **DCA Auto-Range** - Prevalence-based threshold configuration

---

**Last Updated**: 2026-01-28
**Status**: Fully integrated with panel optimization, cross-model consensus, and investigation framework for clinical deployment
