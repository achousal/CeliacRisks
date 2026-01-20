# CLAUDE.MD - Celiac Disease ML Pipeline (ced-ml)

**Project Owner:** Andres Chousal
**Lab:** Elahi Lab
**Last Updated:** 2026-01-19
**Status:** Production-ready (refactored package)

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
| **Demographics** | age, BMI, sex, Genetic ethnic grouping |
| **Missing proteins** | Zero |
| **Missing ethnicity** | 17% (handled as "Missing" category) |

### Data File
```
../data/Celiac_dataset_proteomics.csv (~2.5 GB)
```

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

---

## Package Architecture

**Stats:** 15,109 lines, 753 tests (82% coverage), zero duplication.

For detailed architecture with code pointers, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## CLI Reference

The `ced` command provides 4 core subcommands:

### 1. Split Generation
```bash
ced save-splits --config configs/splits_config.yaml \
    --infile ../data/Celiac_dataset_proteomics.csv
```

**What it does:** Creates stratified TRAIN/VAL/TEST splits with prevalent enrichment and control downsampling.

**Output:** `splits_hpc/IncidentPlusPrevalent_train_idx_seed0.csv`, etc.

### 2. Model Training
```bash
ced train --config configs/training_config.yaml \
    --model LR_EN \
    --infile ../data/Celiac_dataset_proteomics.csv \
    --split-dir splits_hpc
```

**What it does:** Trains models with nested CV, hyperparameter tuning, feature selection, calibration, and evaluation.

**Output:** `results/IncidentPlusPrevalent__LR_EN__5x10__val0.25__test0.25__hybrid/`

### 3. Post-Processing
```bash
ced postprocess --results-dir results/ --n-boot 500
```

**What it does:** Aggregates metrics across splits, computes DCA curves, ranks models.

**Output:** `results/COMBINED/aggregated_metrics.csv`, DCA curves

### 4. Holdout Evaluation
```bash
ced eval-holdout --config configs/holdout_config.yaml \
    --infile ../data/Celiac_dataset_proteomics.csv \
    --holdout-idx splits/IncidentPlusPrevalent_HOLDOUT_idx.csv \
    --model-artifact results/.../core/final_model.joblib \
    --outdir results/HOLDOUT_FINAL
```

**What it does:** Applies trained model to held-out data (run ONCE for final validation).

**Output:** `results/HOLDOUT_FINAL/holdout_metrics.csv`, predictions

---

## Configuration System

All pipeline parameters are managed via YAML configuration files:

### Example: Training Config
```yaml
# configs/training_config.yaml
model: LR_EN
scenario: IncidentPlusPrevalent

cv:
  folds: 5
  repeats: 10
  scoring: neg_brier_score
  n_iter: 200
  inner_folds: 5

features:
  feature_select: hybrid
  screen_method: mannwhitney
  screen_top_n: 1000
  stability_thresh: 0.75

thresholds:
  objective: fixed_spec
  fixed_spec: 0.95
  threshold_source: val
  target_prevalence_source: test

evaluation:
  test_ci_bootstrap: true
  n_boot: 500
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

#### Option A: Orchestrated (Recommended for HPC)
```bash
# Handles all steps: generate splits, submit jobs, postprocess
./run_hpc.sh
```

#### Option B: Manual (Step-by-step)

**Step 1: Generate splits**
```bash
ced save-splits \
    --infile ../data/Celiac_dataset_proteomics.csv \
    --outdir splits_hpc \
    --mode development \
    --scenarios IncidentPlusPrevalent \
    --n-splits 10 \
    --val-size 0.25 \
    --test-size 0.25 \
    --prevalent-train-only \
    --prevalent-train-frac 0.5 \
    --train-control-per-case 5
```

**Step 2: Train models**
```bash
# Local (single model)
ced train --config configs/training_config.yaml \
    --model LR_EN \
    --infile ../data/Celiac_dataset_proteomics.csv \
    --split-dir splits_hpc

# HPC (4 models in parallel via LSF)
# IMPORTANT: Requires Step 1 to be completed first
bsub < CeD_hpc.lsf
```

**Step 3: Post-process**
```bash
ced postprocess --results-dir results_hpc --n-boot 500
```

**Step 4: Visualize (R)**
```bash
Rscript compare_models_faith.R --results_root results_hpc
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
- Nested CV: 5 outer folds × 10 repeats × 5 inner folds
- RandomizedSearchCV: 200 iterations
- Brier score optimization
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

**Test suite:** 768 tests covering:
- Data I/O and split generation
- Feature screening, k-best, stability, correlation pruning, panels
- Model registry, hyperparameters, training, calibration, prevalence
- Discrimination, thresholds, DCA, bootstrap
- Prediction, reports, holdout evaluation
- ROC/PR, calibration, risk distribution, DCA, learning curve plots
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
| `ced postprocess` | `cli/postprocess.py` | Results aggregation |
| `ced eval-holdout` | `cli/eval_holdout.py` | Holdout evaluation |
| `ced config` | `cli/config_tools.py` | Config validation and diff |

### Library Modules
| Layer | Modules | Purpose |
|-------|---------|---------|
| Data | `io`, `splits`, `persistence`, `filters`, `schema` | Data loading, split generation |
| Features | `screening`, `kbest`, `stability`, `corr_prune`, `panels` | Feature selection |
| Models | `registry`, `hyperparams`, `training`, `calibration`, `prevalence` | Model training |
| Metrics | `discrimination`, `thresholds`, `dca`, `bootstrap` | Performance metrics |
| Evaluation | `predict`, `reports`, `holdout` | Prediction and reporting |
| Plotting | `roc_pr`, `calibration`, `risk_dist`, `dca`, `learning_curve` | Visualization |

For output structure details, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#13-output-artifacts--reports).

---

## HPC Deployment

See dedicated guides:
- [HPC_README.md](HPC_README.md) - Quick reference
- [WORKFLOW.md](WORKFLOW.md) - Visual step-by-step guide

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

For more troubleshooting, see [HPC_README.md](HPC_README.md) or [WORKFLOW.md](WORKFLOW.md).

---

## References

- **TRIPOD**: Collins et al. (2015). Transparent Reporting of Prediction Models. BMJ.
- **Calibration**: Van Calster et al. (2019). Calibration: the Achilles heel of predictive analytics. BMC Medicine.
- **DCA**: Vickers & Elkin (2006). Decision curve analysis. Medical Decision Making.
- **CeD Biology**: Sollid & Jabri (2013). Triggers and drivers of autoimmunity. Nature Reviews Immunology.

## Next Steps

**Current Status:** Production-ready ✅ (updated 2026-01-19)

### Completed Today
- [x] Updated documentation and test status
  - Test suite: **768 tests**
  - Coverage: **82%**
  - All core functionality passing
  - Removed legacy tests
  - Added test marker documentation for slow tests

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
| [docs/adr/](docs/adr/) | Architecture Decision Records (15 decisions) |
| [HPC_README.md](HPC_README.md) | HPC quick reference |
| [WORKFLOW.md](WORKFLOW.md) | Visual HPC workflow guide |
| [../README.md](../README.md) | Root project overview |
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | Contribution guidelines |
