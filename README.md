# CeliacRiskML

**A production-ready machine learning pipeline for disease risk prediction from high-dimensional biomarker data**

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![Tests](https://img.shields.io/badge/tests-753%20passing-success)
![Coverage](https://img.shields.io/badge/coverage-82%25-brightgreen)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## Overview

CeliacRiskML is a modular, extensible machine learning framework designed for **disease risk prediction from high-dimensional biomarker data**. While developed for Celiac Disease prediction from proteomics data, the pipeline is **fully generalizable to other diseases and biomarker types**.

**Key capabilities:**
- Multi-model comparison (Random Forest, XGBoost, Linear SVM, Logistic Regression)
- Rigorous nested cross-validation with feature selection
- Calibration-optimized predictions with prevalence adjustment
- Clinical decision curve analysis (DCA)
- HPC-ready batch processing (LSF/Slurm)
- Complete provenance tracking and reproducibility

---

## Features

### 🔬 **Flexible Data Processing**
- Stratified train/validation/test splits with customizable ratios
- Support for incident/prevalent case scenarios
- Handling of class imbalance via downsampling and class weighting
- Missing data strategies (explicit category or imputation)

### 🤖 **Production-Grade ML Pipeline**
- **Four battle-tested models**: Random Forest, XGBoost, Linear SVM, Logistic Regression
- **Nested cross-validation**: 5-fold outer × 10 repeats × 5-fold inner
- **Smart feature selection**: Multi-stage screening (effect size → k-best → stability → correlation pruning)
- **Automatic hyperparameter tuning**: 200-iteration randomized search per model
- **Prevalence adjustment**: Recalibrate for deployment prevalence

### 📊 **Comprehensive Evaluation**
- **Calibration metrics**: Brier score, calibration slope/intercept, calibration curves
- **Discrimination metrics**: AUROC, PR-AUC, sensitivity/specificity at thresholds
- **Clinical utility**: Decision curve analysis (DCA) with net benefit
- **Bootstrap confidence intervals**: Stratified resampling for robust estimates
- **Learning curves**: Sample size vs performance analysis

### ⚙️ **Developer-Friendly**
- **CLI interface**: Simple `ced` commands for all pipeline steps
- **YAML configuration**: Version-controlled, reproducible parameter management
- **Modular codebase**: 15k+ lines organized in clean layers (data/features/models/metrics)
- **High test coverage**: 753 passing tests (82% coverage)
- **HPC integration**: LSF/Slurm batch scripts with array jobs

### 📈 **Rich Visualizations**
- ROC and Precision-Recall curves
- Calibration plots (reliability diagrams)
- Risk distribution histograms
- Decision curve analysis plots
- Learning curves
- Feature importance summaries

---

## Installation

```bash
# Clone the repository
git clone https://github.com/achousal/CeliacRiskML.git
cd CeliacRiskML/analysis

# Install in development mode
pip install -e .

# Verify installation
ced --help
```

**Requirements:**
- Python 3.9+
- scikit-learn, pandas, numpy, scipy
- matplotlib, seaborn (for plotting)
- xgboost (optional, for XGBoost models)

---

## Quick Start

### 1. Prepare Your Data

Your input dataset should be a CSV file with:
- **One row per sample**
- **Feature columns**: Biomarkers/predictors (numerical)
- **Outcome column**: Binary outcome (e.g., `case_control`)
- **Optional columns**: Demographics, covariates

**Example structure:**
```csv
subject_id,age,sex,biomarker_1,biomarker_2,...,biomarker_N,case_control
001,45,F,0.12,0.34,...,0.56,0
002,52,M,0.98,0.76,...,0.43,1
...
```

### 2. Generate Train/Val/Test Splits

```bash
ced save-splits \
  --infile data/your_dataset.csv \
  --outdir splits \
  --scenarios YourScenario \
  --n-splits 10 \
  --val-size 0.25 \
  --test-size 0.25
```

**Output:** Stratified index files for reproducible train/val/test splits

### 3. Train Models

```bash
# Single model (local)
ced train \
  --config configs/training_config.yaml \
  --model LR_EN \
  --infile data/your_dataset.csv \
  --split-dir splits

# Multiple models (HPC array job)
bsub < batch_scripts/train_models.lsf
```

**Output:** Trained models, predictions, metrics, and diagnostics

### 4. Aggregate Results

```bash
ced postprocess \
  --results-dir results \
  --n-boot 500
```

**Output:** Cross-model comparison, bootstrap CIs, DCA curves

### 5. Visualize

```bash
# Python plots (included in pipeline)
# - ROC/PR curves
# - Calibration plots
# - Risk distributions
# - DCA curves

# R visualization (optional)
Rscript scripts/compare_models.R --results_root results
```

---

## Adapting to Your Dataset

The pipeline is designed to be **dataset-agnostic**. Here's how to adapt it:

### 1. **Configure Data Schema**

Update column names in your config file:

```yaml
# configs/training_config.yaml
data:
  outcome_col: "your_outcome_column"        # e.g., "disease_status"
  feature_prefix: "your_biomarker_prefix"   # e.g., "protein_", "gene_"
  demographics: ["age", "sex", "ethnicity"] # Adjust as needed
```

### 2. **Adjust Split Strategy**

Configure split ratios and sampling:

```yaml
# configs/splits_config.yaml
val_size: 0.25          # Validation set size
test_size: 0.25         # Test set size
train_control_per_case: 5.0  # Downsampling ratio (if imbalanced)
```

### 3. **Tune Feature Selection**

Configure screening and stability thresholds:

```yaml
# configs/training_config.yaml
features:
  screen_method: mannwhitney   # or 'ttest', 'anova'
  screen_top_n: 1000           # Top N features to keep
  stability_thresh: 0.75       # Feature selection frequency threshold
  corr_thresh: 0.85            # Correlation pruning threshold
```

### 4. **Customize Models**

Select models and hyperparameter grids:

```yaml
# Train specific models
model: LR_EN  # Options: RF, XGBoost, LinSVM_cal, LR_EN

# Adjust hyperparameter search
cv:
  n_iter: 200         # RandomizedSearchCV iterations
  scoring: neg_brier_score  # or 'roc_auc', 'average_precision'
```

### 5. **Set Evaluation Metrics**

Configure calibration and threshold objectives:

```yaml
thresholds:
  objective: fixed_spec     # or 'youden', 'max_f1'
  fixed_spec: 0.95          # Target specificity
  target_prevalence_source: test  # Use test prevalence for calibration
```

---

## Example Use Case: Celiac Disease Risk Prediction

The pipeline was originally developed to predict **incident Celiac Disease (CeD) risk** from blood proteomics panels measured before diagnosis.

**Dataset characteristics:**
- 43,960 subjects (148 incident cases, 0.34% prevalence)
- 2,920 protein biomarkers
- Demographics: age, BMI, sex, ethnicity

**Results:**
- **Best model**: Logistic Regression (ElasticNet)
- **Test AUROC**: 0.89 (95% CI: 0.86-0.92)
- **Test PR-AUC**: 0.34 (95% CI: 0.28-0.41)
- **Calibration**: Brier score = 0.012 (well-calibrated)
- **Clinical utility**: Positive net benefit at 1-5% risk thresholds

**Biological validation:** Top features include established CeD biomarkers (TGM2, CXCL9, ITGB7, MUC2).

See [CLAUDE.md](analysis/CLAUDE.md) for complete case study documentation.

---

## Architecture

```
src/ced_ml/
├── cli/              # Command-line interface (ced commands)
│   ├── save_splits.py
│   ├── train.py
│   ├── postprocess.py
│   └── eval_holdout.py
├── config/           # YAML configuration system
│   ├── schema.py     # Pydantic validation models (~200 parameters)
│   └── loader.py     # Config loading + CLI overrides
├── data/             # Data I/O and split generation
│   ├── io.py         # CSV reading, filtering
│   ├── splits.py     # Stratified splitting logic
│   └── persistence.py # Index file saving/loading
├── features/         # Feature selection pipeline
│   ├── screening.py  # Effect size ranking (Mann-Whitney/t-test)
│   ├── kbest.py      # SelectKBest with tuning
│   ├── stability.py  # Cross-fold stability filtering
│   └── corr_prune.py # Correlation-based redundancy removal
├── models/           # Model training and calibration
│   ├── registry.py   # Model definitions and factory
│   ├── hyperparams.py # Hyperparameter grids
│   ├── training.py   # Nested CV orchestration
│   ├── calibration.py # Isotonic/Platt calibration
│   └── prevalence.py # Prevalence adjustment
├── metrics/          # Performance evaluation
│   ├── discrimination.py # AUROC, PR-AUC, Youden index
│   ├── thresholds.py     # Threshold selection strategies
│   ├── dca.py            # Decision curve analysis
│   └── bootstrap.py      # Stratified bootstrap CIs
├── evaluation/       # Prediction and reporting
│   ├── predict.py    # Generate predictions
│   ├── reports.py    # Metrics aggregation
│   └── holdout.py    # External validation
├── plotting/         # Visualization layer
│   ├── roc_pr.py         # ROC and PR curves
│   ├── calibration.py    # Calibration plots
│   ├── risk_dist.py      # Risk distribution histograms
│   ├── dca.py            # DCA plots
│   └── learning_curve.py # Learning curves
└── utils/            # Shared utilities
    ├── logging.py    # Structured logging
    ├── paths.py      # Path management
    ├── random.py     # Seed control
    └── serialization.py # Model save/load
```

**Statistics:**
- 15,109 lines of code
- 753 passing tests (82% coverage)
- Zero code duplication
- Fully modular and extensible

---

## Configuration Management

All pipeline parameters are managed via YAML configuration files:

```yaml
# Example: configs/training_config.yaml
model: LR_EN
scenario: YourScenario

cv:
  folds: 5
  repeats: 10
  scoring: neg_brier_score
  n_iter: 200
  inner_folds: 5

features:
  feature_select: hybrid         # 'kbest', 'stability', 'hybrid'
  screen_method: mannwhitney
  screen_top_n: 1000
  stability_thresh: 0.75
  corr_thresh: 0.85

thresholds:
  objective: fixed_spec          # 'youden', 'max_f1', 'fixed_spec'
  fixed_spec: 0.95
  threshold_source: val          # Select threshold on validation set
  target_prevalence_source: test # Calibrate to test prevalence

evaluation:
  test_ci_bootstrap: true
  n_boot: 500

output:
  save_predictions: true
  save_plots: true
  save_models: true
```

**Config tools:**
```bash
# Validate configuration
ced config validate config.yaml --strict

# Compare two configs
ced config diff config1.yaml config2.yaml

# Migrate legacy CLI args to YAML
ced config migrate --command train --args "..." -o config.yaml
```

---

## HPC Deployment

The pipeline supports HPC batch processing with LSF and Slurm:

### LSF Example

```bash
#!/bin/bash
#BSUB -J "train[1-4]"
#BSUB -o logs/train_%I.out
#BSUB -e logs/train_%I.err
#BSUB -n 16
#BSUB -W 12:00
#BSUB -R "rusage[mem=8GB]"

MODELS=(RF XGBoost LinSVM_cal LR_EN)
MODEL=${MODELS[$LSB_JOBINDEX-1]}

ced train \
  --config configs/training_config.yaml \
  --model $MODEL \
  --infile data/your_dataset.csv \
  --split-dir splits
```

**Submit:**
```bash
bsub < batch_scripts/train_models.lsf
bjobs  # Monitor jobs
```

See [docs/HPC_MIGRATION_GUIDE.md](analysis/docs/HPC_MIGRATION_GUIDE.md) for detailed HPC setup.

---

## Testing

```bash
cd analysis

# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src/ced_ml --cov-report=html

# Run specific modules
pytest tests/test_data_*.py -v       # Data layer
pytest tests/test_features_*.py -v   # Feature selection
pytest tests/test_models_*.py -v     # Model training
pytest tests/test_metrics_*.py -v    # Metrics evaluation

# Skip slow integration tests
pytest tests/ -m "not slow"
```

**Test coverage:**
- 753 passing tests
- 82% overall coverage
- 90-100% coverage for core modules

---

## Documentation

| Document | Description |
|----------|-------------|
| [CLAUDE.md](analysis/CLAUDE.md) | Complete project documentation with Celiac Disease case study |
| [README.md](analysis/README.md) | Package quickstart guide |
| [docs/pipeline.md](analysis/docs/pipeline.md) | Algorithm documentation |
| [docs/examples/](analysis/docs/examples/) | Configuration examples |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guidelines |

---

## Key Design Principles

### 1. **Calibration-First**
Primary metric is **Brier score** (not AUROC). For clinical risk prediction, calibrated probabilities are more important than ranking.

### 2. **Rigorous Validation**
- **Nested CV**: Outer CV for performance, inner CV for hyperparameter tuning
- **No leakage**: Features selected on train, evaluated on holdout
- **Three-way split**: TRAIN/VAL/TEST with thresholds selected on VAL

### 3. **Prevalence-Aware**
Models calibrated to **deployment prevalence** (not training prevalence) for realistic risk estimates.

### 4. **Reproducible**
- Fixed random seeds
- YAML configuration version control
- Complete provenance tracking (CLI args, resolved configs, timestamps)

### 5. **HPC-Ready**
- Non-interactive execution
- Resumable workflows
- LSF/Slurm array job support

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup
- Code style guidelines (black, ruff, mypy)
- Testing requirements (pytest, 80%+ coverage)
- Pull request process

**Project guidelines:**
- No secrets in code (use environment variables)
- No emojis anywhere (code, comments, docs)
- All tests pass before PR
- No debug artifacts (`console.log`, `print`, `browser()`)
- Modular code (prefer small composable functions)

---

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{chousal2026celiacriskml,
  author = {Chousal, Andres and Elahi Lab},
  title = {CeliacRiskML: Machine Learning Pipeline for Disease Risk Prediction},
  year = {2026},
  url = {https://github.com/achousal/CeliacRiskML}
}
```

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

---

## Authors

- **Andres Chousal** - Icahn School of Medicine at Mount Sinai, Chowell Lab
- **Chowell Lab** - [https://www.chowell-lab.com/)

---

## Support

- **Issues**: [GitHub Issues](https://github.com/achousal/CeliacRiskML/issues)
- **Discussions**: [GitHub Discussions](https://github.com/achousal/CeliacRiskML/discussions)
- **Documentation**: [analysis/CLAUDE.md](analysis/CLAUDE.md)

---

## References

- **TRIPOD**: Collins et al. (2015). Transparent Reporting of Prediction Models. BMJ.
- **Calibration**: Van Calster et al. (2019). Calibration: the Achilles heel of predictive analytics. BMC Medicine.
- **DCA**: Vickers & Elkin (2006). Decision curve analysis. Medical Decision Making.
