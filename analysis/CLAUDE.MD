# CLAUDE.MD - Celiac Disease ML Pipeline (ced-ml)

**Project Owner:** Andres Chousal
**Lab:** Elahi Lab
**Last Updated:** 2026-01-21
**Status:** Production-ready with Optuna hyperparameter optimization

---

## Project Mission

Build calibrated ML models to predict **incident Celiac Disease (CeD) risk** from proteomics biomarkers measured **before clinical diagnosis**. Generate continuous risk scores for apparently healthy individuals to inform follow-up testing decisions.

### Clinical Workflow
```
Blood proteomics panel → ML risk score → [High risk?] → Anti-tTG antibody test → Endoscopy
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

### Data Files
```
../data/Celiac_dataset_proteomics_w_demo.parquet (recommended)
../data/Celiac_dataset_proteomics.csv (~2.5 GB, also supported)
```

**Supported formats:** Parquet (recommended for speed) and CSV (auto-detected from file extension)

---

## Key Design Decisions

### 1. IncidentPlusPrevalent with Prevalent in TRAIN Only
Use IncidentPlusPrevalent scenario but restrict prevalent cases to TRAIN set only (50% sampling). VAL and TEST contain incident cases exclusively for clean prospective evaluation.

**Rationale:**
- Training signal enrichment: Prevalent cases provide additional signal for biomarker discovery
- Prospective validation: VAL/TEST remain incident-only for valid risk prediction evaluation
- Controlled sampling: 50% prevalent sampling + 1:5 case:control ratio balances classes

### 2. Three-Way Split: TRAIN/VAL/TEST
50% TRAIN / 25% VAL / 25% TEST

**Rationale:**
- **VAL (25%)**: Threshold selection and calibration tuning without test leakage
- **TEST (25%)**: Final performance reporting with prevalence-adjusted calibration
- **Threshold source = VAL**: All decision thresholds selected on validation set
- **Target prevalence = TEST**: Calibration uses test-set prevalence for deployment realism

### 3. Brier Score Optimization
Primary metric is `neg_brier_score` (not AUROC/PR-AUC)

**Rationale:** For clinical screening, calibrated probabilities matter more than ranking. A patient with 2% predicted risk should truly have ~2% incidence. Brier score directly optimizes calibration quality.

### 4. Control Downsampling (1:5 ratio)
TRAIN controls are downsampled to 5 controls per case (incident + prevalent)

**Rationale:**
- Addresses extreme class imbalance (148 incident vs 43,662 controls)
- Reduces computational burden while preserving discrimination
- Combined with prevalent sampling: ~74 incident + ~75 prevalent + ~745 controls in TRAIN

### 5. Missing as Category
17% of subjects have missing ethnicity, treated as explicit "Missing" category (not imputed or dropped)

**Rationale:** Preserves 7,674 subjects and allows model to learn if missingness is informative without introducing imputation bias.

### 6. Flexible Metadata Columns
Metadata columns are configurable via `ColumnsConfig` with two modes:
- **Auto mode** (default): Auto-detect standard columns (age, BMI, sex, ethnicity) from data
- **Explicit mode**: Specify exact metadata columns to use

**Rationale:** Enables pipeline to work with different datasets or metadata schemas without code changes. See [ADR-015](docs/adr/ADR-015-flexible-metadata-columns.md) for details.

---

## Package Architecture

**Stats:** ~19,657 lines of code, 762 tests (82% coverage), zero duplication.

For detailed architecture with code pointers, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## CLI Reference

The `ced` command provides 5 core subcommands:

### 1. Split Generation
```bash
ced save-splits --config configs/splits_config.yaml \
    --infile ../data/Celiac_dataset_proteomics_w_demo.parquet
```

**What it does:** Creates stratified TRAIN/VAL/TEST splits with prevalent enrichment and control downsampling.

**Output:** `../splits/train_idx_seed0.csv`, etc.

### 2. Model Training
```bash
ced train --config configs/training_config.yaml \
    --model LR_EN \
    --infile ../data/Celiac_dataset_proteomics_w_demo.parquet \
    --split-dir ../splits \
    --outdir ../results
```

**What it does:** Trains models with nested CV, hyperparameter tuning, feature selection, calibration, and evaluation.

**Output:** `../results/LR_EN__5x10__val0.25__test0.25__hybrid/`

### 3. Aggregate Results
```bash
ced aggregate-splits --results-dir results/ --n-boot 500
```

**What it does:** Aggregates metrics across splits, computes bootstrap CIs, creates combined plots.

**Output:** `results/aggregated/aggregated_metrics.csv`, combined plots

### 4. Holdout Evaluation
```bash
ced eval-holdout --config configs/holdout_config.yaml \
    --infile ../data/Celiac_dataset_proteomics_w_demo.parquet \
    --holdout-idx ../splits/HOLDOUT_idx.csv \
    --model-artifact ../results/.../core/final_model.joblib \
    --outdir ../results/HOLDOUT_FINAL
```

**What it does:** Applies trained model to held-out data (run ONCE for final validation).

**Output:** `../results/HOLDOUT_FINAL/holdout_metrics.csv`, predictions

---

## Configuration System

All pipeline parameters are managed via YAML configuration files:

### Example: Training Config
```yaml
# configs/training_config.yaml
model: LR_EN
scenario: IncidentPlusPrevalent

# Column configuration (NEW in v1.1)
columns:
  mode: auto  # or "explicit" for custom metadata
  # numeric_metadata: [age, bmi]  # only if mode=explicit
  # categorical_metadata: [sex, ethnicity]  # only if mode=explicit
  warn_missing_defaults: true

cv:
  folds: 5
  repeats: 10
  scoring: roc_auc  # Options: roc_auc, average_precision, neg_brier_score
  n_iter: 5  # RandomizedSearchCV iterations
  inner_folds: 200
  grid_randomize: true  # Randomize numeric grid values

# Optuna hyperparameter optimization (optional)
optuna:
  enabled: false  # Set to true to use Optuna instead of RandomizedSearchCV
  n_trials: 100
  sampler: tpe  # Options: tpe, random, cmaes, grid
  pruner: median  # Options: median, percentile, hyperband, none
  save_study: true
  save_trials_csv: true

features:
  feature_select: hybrid  # Options: none, kbest, l1_stability, hybrid
  kbest_scope: protein
  kbest_max: 800
  k_grid: [25, 50, 100, 200, 300, 400, 600, 800]
  screen_method: mannwhitney  # Options: mannwhitney, f_classif
  screen_top_n: 1000  # Pre-filter before k-best (0 = no screening)
  stability_thresh: 0.75  # Minimum frequency across folds
  stable_corr_thresh: 0.85  # Remove correlated features (r > 0.85)

thresholds:
  objective: fixed_spec  # Options: max_f1, max_fbeta, youden, fixed_spec, fixed_ppv
  fixed_spec: 0.95  # Target 95% specificity
  fbeta: 1.0  # Beta parameter for max_fbeta
  threshold_source: val  # Select threshold on VAL set
  target_prevalence_source: test  # Use TEST prevalence for calibration
  risk_prob_source: test  # Options: test, val

evaluation:
  test_ci_bootstrap: true
  n_boot: 500
  boot_random_state: 0
  learning_curve: true
  lc_train_sizes: [0.1, 0.25, 0.5, 0.75, 1.0]
  feature_reports: true
  feature_report_max: 200
  control_spec_targets: [0.90, 0.95, 0.99]
  toprisk_fracs: [0.01, 0.05, 0.10]

# Decision curve analysis
dca:
  compute_dca: true
  dca_threshold_min: 0.0005
  dca_threshold_max: 1.0
  dca_threshold_step: 0.001
  dca_report_points: [0.01, 0.05, 0.10, 0.20]

# Outputs
output:
  save_train_oof: true
  save_val_preds: true
  save_test_preds: true
  save_calibration: true
  calib_bins: 6
  save_feature_importance: true
  plot_format: png  # Options: png, pdf, svg
  plot_dpi: 300
```

### Example: Custom Metadata Columns
```yaml
# For datasets with different metadata schemas
columns:
  mode: explicit
  numeric_metadata: [age_at_sample, body_mass_index, creatinine]
  categorical_metadata: [gender, ancestry, study_site]
```

**Config tools:**
```bash
# Validate config
ced config validate config.yaml --strict

# Compare configs
ced config diff config1.yaml config2.yaml
```

---

## Quick Start

### Installation
```bash
cd analysis
pip install -e .
ced --help
```

### Basic Workflow

#### Option A: Orchestrated (Recommended)

**Local:**
```bash
# Config-driven pipeline: edit configs/pipeline_local.yaml first
./run_local.sh
```

**HPC:**
```bash
# Config-driven pipeline: edit configs/pipeline_hpc.yaml first
./run_hpc.sh
```

#### Option B: Manual (Step-by-step)

**Step 1: Generate splits**
```bash
# Edit configs/splits_config.yaml to configure split parameters
ced save-splits \
    --config configs/splits_config.yaml \
    --infile ../data/Celiac_dataset_proteomics_w_demo.parquet
```

**Step 2: Train models**
```bash
# Local (sequential)
ced train --config configs/training_config.yaml \
    --model LR_EN \
    --infile ../data/Celiac_dataset_proteomics_w_demo.parquet \
    --split-dir ../splits \
    --outdir ../results \
    --split-seed 0

# HPC (parallel via pipeline)
# Edit configs/pipeline_hpc.yaml, then:
./run_hpc.sh
```

**Step 3: Aggregate results**
```bash
ced aggregate-splits --results-dir ../results --n-boot 500
```

---

## Models

| Model | Algorithm | Notes |
|-------|-----------|-------|
| **RF** | Random Forest | 500 trees, class_weight tuning |
| **XGBoost** | Gradient Boosting | Auto scale_pos_weight, GPU support |
| **LinSVM_cal** | Linear SVM | Sigmoid calibration |
| **LR_EN** | Logistic Regression | ElasticNet penalty |

All models use:
- Nested CV: 5 outer folds × 10 repeats × 200 inner folds
- Hyperparameter optimization: RandomizedSearchCV (default) or Optuna (optional)
- Scoring: ROC-AUC, PR-AUC, or Brier score (configurable)
- Prevalence adjustment (train 16.7% → deployment 0.34%)

---

## Evaluation Metrics

### Primary (Model Selection)
- **Brier Score**: Calibration quality (lower is better)

### Discrimination
- **AUROC**: Ranking ability
- **PR-AUC**: Precision-recall for imbalanced data

### Clinical Utility
- **DCA net benefit**: Clinical utility at decision thresholds
- **Calibration slope/intercept**: Should be ~1.0/~0.0
- **Sensitivity at 95%/99% specificity**: High-specificity performance

---

## Testing

```bash
cd analysis

# Run all tests
pytest tests/ -v

# Run fast tests only (skip slow integration tests)
pytest tests/ -m "not slow"

# Run only slow tests
pytest tests/ -m slow

# With coverage
pytest tests/ --cov=ced_ml --cov-report=term-missing
```

**Test suite:** 762 tests covering:
- Data I/O (CSV/Parquet), column resolution, and split generation
- Feature screening, k-best, stability, correlation pruning, panels
- Model registry, hyperparameters, training, calibration, prevalence
- Discrimination, thresholds, DCA, bootstrap
- Prediction, reports, holdout evaluation
- ROC/PR, calibration, risk distribution, DCA, learning curve, OOF plots
- CLI integration (zero duplication verified)
- Config validation and comparison

**Coverage:** 82% overall (90-100% for most modules)

**Test markers:**
- `slow`: Integration tests that train real models (10-20s each). Skip with `pytest -m "not slow"` for faster development.

---

## File Reference

### CLI Commands
| Command | Module | Purpose |
|---------|--------|---------|
| `ced save-splits` | `cli/save_splits.py` | Split generation |
| `ced train` | `cli/train.py` | Model training |
| `ced aggregate-splits` | `cli/aggregate_splits.py` | Results aggregation |
| `ced eval-holdout` | `cli/eval_holdout.py` | Holdout evaluation |
| `ced config` | `cli/config_tools.py` | Config validation and diff |

### Library Modules
| Layer | Modules | Purpose |
|-------|---------|---------|
| Data | `io`, `splits`, `persistence`, `filters`, `schema`, `columns` | Data loading, split generation, column resolution |
| Features | `screening`, `kbest`, `stability`, `corr_prune`, `panels` | Feature selection |
| Models | `registry`, `hyperparams`, `optuna_search`, `training`, `calibration`, `prevalence` | Model training and hyperparameter optimization |
| Metrics | `discrimination`, `thresholds`, `dca`, `bootstrap` | Performance metrics |
| Evaluation | `predict`, `reports`, `holdout` | Prediction and reporting |
| Plotting | `roc_pr`, `calibration`, `risk_dist`, `dca`, `learning_curve`, `oof`, `optuna_plots` | Visualization |

For output structure details, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#13-output-artifacts--reports).

---

## Pipeline Configuration

All pipeline orchestration is managed via YAML configuration files:

### Pipeline Configs
- **`configs/pipeline_local.yaml`**: Local development pipeline settings
- **`configs/pipeline_hpc.yaml`**: HPC production pipeline settings

These files control:
- Input/output paths
- Models to run
- Number of bootstrap iterations
- Execution flags (dry_run, overwrite_splits, postprocess_only)
- HPC resources (cores, memory, walltime, queue, project allocation)

### Core Configs
- **`configs/splits_config.yaml`**: CV split generation settings
- **`configs/training_config.yaml`**: Model training and evaluation settings

### Running Pipelines

**Local:**
```bash
# Edit configs/pipeline_local.yaml, then:
./run_local.sh

# Override settings:
RUN_MODELS="LR_EN,RF" ./run_local.sh
DRY_RUN=1 ./run_local.sh
```

**HPC:**
```bash
# Edit configs/pipeline_hpc.yaml (especially hpc.project), then:
./run_hpc.sh

# Monitor jobs:
bjobs -w | grep CeD_
```

For detailed setup instructions, see:
- [SETUP_README.md](SETUP_README.md) - Environment setup and getting started
- [HPC_README.md](HPC_README.md) - HPC quick reference
- [WORKFLOW.md](WORKFLOW.md) - Visual workflow guide

---

## Biological Validation

Top proteins include established CeD biomarkers:

| Protein | Cohen's d | Clinical Relevance |
|---------|-----------|-------------------|
| **TGM2** | 1.73 | Primary CeD autoantigen (gold standard) |
| **CXCL9** | 1.53 | Inflammatory chemokine |
| **ITGB7** | 1.50 | Gut-homing integrin |
| **MUC2** | 0.96 | Intestinal mucin |

**Conclusion**: Models capture genuine biological signal, not overfitting.

---

## Troubleshooting

### Issue: Command not found
**Symptom:** `ced: command not found`

**Solution:**
```bash
cd analysis
pip install -e .
```

### Issue: Config validation errors
**Symptom:** `ValidationError: cv.folds must be positive integer`

**Solution:**
```bash
# Validate config before running
ced config validate config.yaml --strict

# Fix errors reported in output
```

### Issue: Tests failing
**Symptom:** Some tests fail with matplotlib/XGBoost errors

**Solutions:**
- Matplotlib backend: Set `MPLBACKEND=Agg` for headless rendering
- XGBoost: `pip install xgboost` (optional dependency)

### Issue: Parquet file read errors on HPC
**Symptom:** `Failed to read Parquet file` or `ArrowInvalid` errors on HPC

**Solutions:**
1. Verify complete file transfer: `ls -lh path/to/file.parquet`
2. Clear filesystem cache: `cat path/to/file.parquet > /dev/null`
3. Check file integrity: `python -c "import pyarrow.parquet as pq; pq.read_metadata('file.parquet')"`
4. If corrupted, regenerate or use CSV format instead

### Issue: Column not found errors
**Symptom:** `ValueError: Specified metadata columns not found in data`

**Solutions:**
1. Check your data's column names: `ced save-splits --infile data.csv --config config.yaml` will auto-detect
2. Use auto mode (default) to auto-detect standard columns
3. For custom columns, use explicit mode in config:
   ```yaml
   columns:
     mode: explicit
     numeric_metadata: [age, bmi]
     categorical_metadata: [sex]
   ```

For more troubleshooting, see [HPC_README.md](HPC_README.md) or [WORKFLOW.md](WORKFLOW.md).

---

## References

- **TRIPOD**: Collins et al. (2015). Transparent Reporting of Prediction Models. BMJ.
- **Calibration**: Van Calster et al. (2019). Calibration: the Achilles heel of predictive analytics. BMC Medicine.
- **DCA**: Vickers & Elkin (2006). Decision curve analysis. Medical Decision Making.
- **CeD Biology**: Sollid & Jabri (2013). Triggers and drivers of autoimmunity. Nature Reviews Immunology.

## Next Steps

**Current Status:** Production-ready ✅ (updated 2026-01-21)

### Recent Improvements (Jan 21, 2026)
- ✅ Optuna hyperparameter optimization with trial visualization ([ADR-018](docs/adr/ADR-018-optuna-hyperparameter-optimization.md))
- ✅ Split-specific output directories ([ADR-019](docs/adr/ADR-019-split-specific-output-directories.md))
- ✅ Combined OOF plotting across splits ([ADR-016](docs/adr/ADR-016-oof-combined-plotting.md))
- ✅ Config-driven pipeline orchestration (`pipeline_local.yaml`, `pipeline_hpc.yaml`)
- ✅ Unified CLI with `aggregate-splits` command
- ✅ Parquet file format support ([ADR-017](docs/adr/ADR-017-parquet-format-support.md))
- ✅ Flexible metadata column configuration (`ColumnsConfig`)
- ✅ Out-of-fold (OOF) prediction plotting module
- ✅ Improved error messages for file I/O and column resolution
- ✅ Standardized path structure (`../splits`, `../results`, `../logs`)
- ✅ Test suite: **762 tests**, Coverage: **82%**, Lines: **~19,657**

### Remaining Tasks
- [ ] Run validation study on HPC
- [ ] Performance benchmarking
- [ ] Docker container for reproducibility
- [ ] Publication artifacts (Zenodo/Figshare)

---

## More Documentation

| Document | Purpose |
|----------|---------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Technical architecture with code pointers |
| [docs/adr/](docs/adr/) | Architecture Decision Records (20 decisions) |
| [docs/reference/CLI_REFERENCE.md](docs/reference/CLI_REFERENCE.md) | Complete CLI command reference |
| [SETUP_README.md](SETUP_README.md) | Environment setup and getting started |
| [../README.md](../README.md) | Root project overview |
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | Contribution guidelines |
