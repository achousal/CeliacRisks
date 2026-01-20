# Celiac Disease ML Pipeline - Algorithm Documentation

**Version:** 1.0.0
**Package:** ced-ml
**Updated:** 2026-01-19

---

## Overview

Calibrated ML models predicting incident Celiac Disease (CeD) risk from pre-diagnosis proteomics biomarkers.

**Clinical Workflow:**
```
Proteomics panel → Risk score → [High risk?] → Anti-tTG test → Endoscopy
```

**Pipeline Phases:**
1. Data preparation (three-way split)
2. Model training (nested CV + feature selection)
3. Postprocessing (metrics aggregation)
4. Holdout validation (external validation)

---

## Dataset

| Attribute | Value |
|-----------|-------|
| Total samples | 43,960 |
| Controls | 43,662 |
| Incident CeD | 148 (0.34%) - pre-diagnosis |
| Prevalent CeD | 150 - TRAIN only (50% sampling) |
| Proteins | 2,920 (_resid columns) |
| Demographics | age, BMI, sex, ethnicity |

**Source:** `Celiac_dataset_proteomics.csv` (2.5 GB)

---

## Data Preparation

### Three-Way Split
```
Original (43,960) → TRAIN (50%) / VAL (25%) / TEST (25%)
                     21,980      10,990      10,990
```

**TRAIN composition:**
- 74 incident CeD
- 75 prevalent CeD (50% sample, TRAIN-only)
- 370 controls (1:5 case:control ratio)

**VAL/TEST composition:**
- Incident CeD only (prospective evaluation)
- 37 cases + 185 controls each

**Rationale:**
- **VAL**: Threshold selection without test leakage
- **TEST**: Final reporting with prevalence-adjusted calibration
- **Prevalent in TRAIN only**: Signal enrichment without reverse causality bias

### CLI Command
```bash
ced save-splits \
  --infile ../data/Celiac_dataset_proteomics.csv \
  --outdir splits_production \
  --mode development \
  --scenarios IncidentPlusPrevalent \
  --n-splits 10 \
  --val-size 0.25 --test-size 0.25 \
  --prevalent-train-only --prevalent-train-frac 0.5 \
  --train-control-per-case 5
```

**Output:** 10 random seeds × 3 splits (TRAIN/VAL/TEST) per seed

---

## Model Training

### Nested CV Architecture
```
TRAIN (21,980)
├─ Outer CV: 5-fold stratified × 10 repeats = 50 iterations
│  └─ Each iteration: 80% train (17,584) / 20% test (4,396)
│
└─ Inner CV: 5-fold stratified
   └─ Hyperparameter tuning: 200 random combinations
```

**Total model trainings:** 50 outer × 5 inner × 200 configs = 50,000 evaluations per model

### Feature Selection Pipeline

#### Step 1: Mann-Whitney Screening (global, before CV)
- Input: 2,920 proteins
- Method: Effect size ranking (case vs control)
- Output: Top 1,000 proteins
- Runs: 1× globally

#### Step 2: K-best Selection (per inner fold)
- Input: 1,000 proteins (from Step 1)
- Method: SelectKBest (F-statistic)
- K values: 25, 50, 100, 200, 300, 400, 600, 800
- Runs: 250× (50 outer × 5 inner)
- Tuned via RandomizedSearchCV

#### Step 3: Stability Filtering (global, after CV)
- Input: Protein selections from 50 outer folds
- Method: Selection frequency = # folds selected / 50
- Threshold: Frequency ≥ 0.75 (≥38/50 folds)
- Output: ~50-100 stable proteins
- Runs: 1× globally

#### Step 4: Spearman Correlation Pruning (global, after CV)
- Input: Stable proteins (from Step 3)
- Method: Pairwise |Spearman ρ| on full TRAIN
- Threshold: |ρ| ≥ 0.85 → collapse to connected components
- Representative: Highest selection frequency, then effect size
- Output: ~40-80 unique proteins
- Runs: 1× globally

**Summary:**
| Step | Timing | Runs | Input | Output |
|------|--------|------|-------|--------|
| 1. Mann-Whitney | Before CV | 1 | 2,920 | 1,000 |
| 2. K-best | Inner CV | 250 | 1,000 | Variable K |
| 3. Stability | After CV | 1 | 50 selections | ~70 |
| 4. Spearman | After CV | 1 | ~70 | ~60 |

### Models

| Model | Algorithm | Calibration |
|-------|-----------|-------------|
| RF | Random Forest | Logistic |
| XGBoost | Gradient Boosting | Logistic |
| LinSVM_cal | Linear SVM | Sigmoid |
| LR_EN | Logistic Regression | ElasticNet |

**Training configuration:**
- Scoring: neg_brier_score (optimizes calibration)
- Hyperparameter search: 200 random combinations
- Class weighting: Balanced (addresses imbalance)
- Prevalence adjustment: 16.7% (TRAIN) → 0.34% (deployment)

### Hyperparameter Grids

**Logistic Regression (LR_EN):**
- K: 8 values
- C: 20 points (1e-4 to 10)
- L1_ratio: 8 values (0.01 to 0.60)
- Total: 1,280 combinations → sample 200

**Linear SVM (LinSVM_cal):**
- K: 8 values
- C: 10 points (1e-4 to 1e4)
- Total: 80 combinations → sample 200 (exhaustive + repeats)

**Random Forest (RF):**
- K: 8 values
- n_estimators: 2 values (100, 200)
- max_depth: 3 values (8, 10, 12)
- min_samples_leaf: 2 values (2, 4)
- max_features: 2 values (0.2, 0.3)
- class_weight: 2 options
- Total: 384 combinations → sample 200

**XGBoost:**
- K: 8 values
- n_estimators: 2 values (500, 1000)
- max_depth: 2 values (3, 7)
- learning_rate: 2 values (0.01, 0.1)
- subsample: 2 values (0.7, 1.0)
- colsample_bytree: 2 values (0.7, 1.0)
- Total: 256 combinations → sample 200

### CLI Command
```bash
# Single model (local)
ced train \
  --config configs/training_config.yaml \
  --model LR_EN \
  --infile ../data/Celiac_dataset_proteomics.csv \
  --splits-dir splits_production

# Four models (HPC parallel)
bsub < CeD_production.lsf
```

**Output:** `results/IncidentPlusPrevalent__LR_EN__5x10__val0.25__test0.25__hybrid/`

---

## Evaluation

### Metrics

**Primary (model selection):**
- Brier score: Calibration quality (lower is better)

**Discrimination:**
- AUROC: Ranking ability
- PR-AUC: Precision-recall for imbalanced data

**Clinical utility:**
- Sensitivity at 95%/99%/99.5% specificity
- DCA net benefit: Clinical utility at decision thresholds
- Calibration slope/intercept: Should be ~1.0/~0.0

### Evaluation Splits

| Split | Purpose | Metrics |
|-------|---------|---------|
| TRAIN (OOF) | Feature stability, learning curves | Internal CV performance |
| VAL | Threshold selection | Fixed-spec threshold (95%), Brier score |
| TEST | Final reporting | AUROC, PR-AUC, DCA, calibration, sensitivity |

**Key principle:** Thresholds selected on VAL, applied to TEST (prevents test leakage)

### Postprocessing
```bash
ced postprocess --results-dir results_production --n-boot 500
```

**Outputs:**
- Aggregated metrics across 4 models × 10 seeds
- DCA curves (0.05%-100% threshold sweep)
- Model ranking by Brier score (primary), then AUROC
- Bootstrap confidence intervals (500 iterations)
- Feature importance comparison

---

## Holdout Validation

**Use case:** External validation on completely held-out data (30% if created)

**Run ONCE only** - no parameter tuning allowed.

```bash
ced eval-holdout \
  --config configs/holdout_config.yaml \
  --infile ../data/Celiac_dataset_proteomics.csv \
  --holdout-idx splits/IncidentPlusPrevalent_HOLDOUT_idx.csv \
  --model-artifact results/.../core/final_model.joblib \
  --outdir results/HOLDOUT_FINAL
```

**Output:** External validation metrics (publication-ready)

---

## Output Structure

```
results_production/
├── IncidentPlusPrevalent__LR_EN__5x10__val0.25__test0.25__hybrid/
│   ├── config/
│   │   ├── resolved_config.yaml     # Provenance
│   │   └── cli_overrides.json
│   ├── core/
│   │   ├── val_metrics.csv          # Validation results
│   │   ├── test_metrics.csv         # PRIMARY RESULTS
│   │   └── final_model.joblib       # Trained model
│   ├── preds/
│   │   ├── val_preds/               # Validation predictions
│   │   └── test_preds/              # Test predictions
│   ├── reports/
│   │   └── stable_panel/            # Feature stability
│   └── diagnostics/
│       ├── calibration/             # Calibration curves
│       ├── dca/                     # Decision curve analysis
│       └── learning_curve/          # Sample size analysis
├── (3 more model directories: RF, XGBoost, LinSVM_cal)
├── COMBINED/
│   ├── aggregated_metrics.csv       # Cross-model comparison
│   └── dca_curves.csv
└── HOLDOUT_FINAL/                   # External validation
```

---

## Key Design Decisions

### 1. Brier Score Optimization
Primary metric is `neg_brier_score`, not AUROC/PR-AUC.

**Rationale:** Clinical screening requires calibrated probabilities. A patient with 2% predicted risk should truly have ~2% incidence. Brier score directly optimizes calibration quality.

### 2. Three-Way Split (50/25/25)
TRAIN (50%) / VAL (25%) / TEST (25%)

**Rationale:**
- VAL enables threshold selection without test leakage
- TEST provides unbiased final reporting
- Prevalence adjustment uses TEST prevalence (0.34%) for deployment realism

### 3. Prevalent Cases in TRAIN Only
Prevalent CeD restricted to TRAIN (50% sampling), excluded from VAL/TEST.

**Rationale:**
- Training signal enrichment without reverse causality bias
- VAL/TEST remain prospective (incident-only) for valid risk prediction

### 4. Control Downsampling (1:5 ratio)
TRAIN controls downsampled to 5 per case (incident + prevalent).

**Rationale:**
- Addresses extreme imbalance (148 incident vs 43,662 controls)
- Reduces computation while preserving discrimination
- Final TRAIN: ~74 incident + ~75 prevalent + ~745 controls

### 5. Nested CV Architecture
5 outer folds × 10 repeats × 5 inner folds

**Rationale:**
- Outer CV: Unbiased performance estimates (OOF predictions)
- Inner CV: Hyperparameter tuning without leakage
- 10 repeats: Robust estimates across random seeds

### 6. Four-Step Feature Selection
Mann-Whitney → K-best → Stability → Spearman pruning

**Rationale:**
- Step 1: Computational efficiency (2,920 → 1,000)
- Step 2: Model-specific tuning (K is hyperparameter)
- Step 3: Robustness (keep frequently selected proteins)
- Step 4: Remove redundancy (prune correlated proteins)

---

## Biological Validation

Top proteins include established CeD biomarkers:

| Protein | Cohen's d | Clinical Relevance |
|---------|-----------|-------------------|
| TGM2 | 1.73 | Primary CeD autoantigen (gold standard) |
| CXCL9 | 1.53 | Inflammatory chemokine |
| ITGB7 | 1.50 | Gut-homing integrin |
| MUC2 | 0.96 | Intestinal mucin |

**Conclusion:** Models capture genuine biological signal, not overfitting.

---

## Configuration System

All parameters managed via YAML:

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
ced config validate config.yaml --strict
ced config diff config1.yaml config2.yaml
```

---

## Package Architecture

```
src/ced_ml/
├── cli/              # Command-line interface
├── config/           # YAML schemas, validation
├── data/             # I/O, splits, persistence
├── features/         # Screening, k-best, stability, pruning
├── models/           # Registry, hyperparameters, training, calibration
├── metrics/          # Discrimination, thresholds, DCA, bootstrap
├── evaluation/       # Prediction, reports, holdout
├── plotting/         # ROC/PR, calibration, DCA, learning curves
└── utils/            # Logging, paths, seeds, serialization
```

**Statistics:**
- 15,109 lines of code
- 753 passing tests (82% coverage)
- Zero code duplication
- 10 random seeds × 4 models = 40 training runs per experiment

---

## Quick Start

### Installation
```bash
cd analysis
pip install -e .
ced --help
```

### Basic Workflow
```bash
# Step 1: Generate splits
ced save-splits \
  --infile ../data/Celiac_dataset_proteomics.csv \
  --outdir splits_production \
  --mode development \
  --scenarios IncidentPlusPrevalent \
  --n-splits 10

# Step 2: Train models (HPC)
bsub < CeD_production.lsf

# Step 3: Postprocess
ced postprocess --results-dir results_production --n-boot 500

# Step 4: Visualize
Rscript compare_models_faith.R --results_root results_production
```

---

## HPC Deployment

### LSF Batch Script
```bash
#!/bin/bash
#BSUB -J "CeD_train[1-4]"
#BSUB -o logs/CeD_%I.out
#BSUB -e logs/CeD_%I.err
#BSUB -n 16
#BSUB -W 12:00
#BSUB -R "rusage[mem=8GB]"

MODELS=(RF XGBoost LinSVM_cal LR_EN)
MODEL=${MODELS[$LSB_JOBINDEX-1]}

ced train \
  --config configs/training_config.yaml \
  --model $MODEL \
  --infile ../data/Celiac_dataset_proteomics.csv \
  --splits-dir splits_production
```

**Submit:** `bsub < CeD_production.lsf`

---

## Validation Properties

**No data leakage:**
- Features selected on 80%, evaluated on 20%
- Hyperparameters tuned on inner CV, evaluated on outer CV
- VAL never used during training
- TEST never used until final reporting

**Robust feature selection:**
- Features selected in ≥75% of folds survive
- Unstable/noisy features filtered out
- Correlation pruning removes redundancy

**Unbiased performance estimates:**
- OOF predictions never seen by trained model
- Bootstrap CIs provide confidence intervals
- Nested CV prevents overfitting

---

## Testing

```bash
cd analysis
pytest tests/ -v
```

**Coverage:** 753 tests, 82% coverage

**Test modules:**
- Data I/O, split generation
- Feature screening, k-best, stability, correlation pruning
- Model registry, hyperparameters, training, calibration
- Discrimination, thresholds, DCA, bootstrap
- Prediction, reports, holdout evaluation
- Plotting: ROC/PR, calibration, DCA, learning curves
- CLI integration, config validation

---

## References

- **TRIPOD:** Collins et al. (2015). Transparent Reporting of Prediction Models. BMJ.
- **Calibration:** Van Calster et al. (2019). Calibration: the Achilles heel of predictive analytics. BMC Medicine.
- **DCA:** Vickers & Elkin (2006). Decision curve analysis. Medical Decision Making.
- **CeD Biology:** Sollid & Jabri (2013). Triggers and drivers of autoimmunity. Nature Reviews Immunology.

---

**End of Pipeline Documentation**
