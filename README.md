# CeliacRiskML

**A research-ready machine learning pipeline for disease risk prediction from high-dimensional biomarker data**

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![Tests](https://img.shields.io/badge/tests-1081%20passing-success)
![Coverage](https://img.shields.io/badge/coverage-65%25-yellowgreen)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## Overview

CeliacRiskML is a modular, extensible machine learning framework for **disease risk prediction from high-dimensional biomarker data**. While developed for Celiac Disease prediction from proteomics data, the pipeline is **fully generalizable to other diseases and biomarker types**.

**Key capabilities:**
- Multi-model comparison (Random Forest, XGBoost, Linear SVM, Logistic Regression)
- **Model stacking ensemble** with OOF meta-learner for improved predictions
- Rigorous nested cross-validation with feature selection
- **OOF-posthoc calibration** for unbiased probability estimates
- Probability calibration at training prevalence (16.7%)
- **Temporal validation** support for chronological splits
- Clinical decision curve analysis (DCA) with auto-range
- HPC-ready batch processing (LSF/Slurm)
- Complete provenance tracking and reproducibility

---

## Features

### Flexible Data Processing
- Support for CSV and Parquet file formats (auto-detected)
- Configurable metadata columns (auto-detect or explicit specification)
- Stratified train/validation/test splits with customizable ratios
- Support for incident/prevalent case scenarios
- Handling of class imbalance via downsampling and class weighting

### Research-Grade ML Pipeline
- **Four models**: Random Forest, XGBoost, Linear SVM, Logistic Regression
- **Stacking ensemble**: L2 meta-learner combining base model OOF predictions (+2-5% AUROC)
- **Nested cross-validation**: N-fold outer x N repeats x N-fold inner
- **Smart feature selection**: Multi-stage screening (effect size, k-best, stability, correlation pruning)
- **Optuna hyperparameter optimization**: Bayesian TPE with expanded search ranges
- **Temporal validation**: Chronological train/val/test splits for time-series data

### Comprehensive Evaluation
- **Calibration metrics**: Brier score, calibration slope/intercept, calibration curves
- **OOF-posthoc calibration**: Unbiased calibration strategy eliminating ~0.5-1% optimistic bias
- **Discrimination metrics**: AUROC, PR-AUC, sensitivity/specificity at thresholds
- **Clinical utility**: Decision curve analysis (DCA) with auto-configured threshold ranges
- **Bootstrap confidence intervals**: Stratified resampling for robust estimates

### Rich Visualizations
- ROC and Precision-Recall curves
- Calibration plots (reliability diagrams)
- Risk distribution histograms
- Decision curve analysis plots
- Out-of-fold (OOF) prediction plots with confidence bands
- Feature importance summaries

---

## Installation

```bash
git clone https://github.com/achousal/CeliacRiskML.git
cd CeliacRiskML/analysis
pip install -e .
ced --help
```

**Requirements:** Python 3.10+, scikit-learn, pandas, numpy, scipy, matplotlib, seaborn, xgboost (optional)

---

## Quick Start

### Recommended: Run Complete Pipeline

**Local testing:**
```bash
cd analysis
./run_local.sh
```

**HPC deployment:**
```bash
cd analysis
PROJECT=your_allocation ./run_hpc.sh

# After jobs complete, run post-training pipeline
bash scripts/post_training_pipeline.sh --run-id <RUN_ID>
```

For detailed workflows, see "Post-Training Pipeline (HPC)" section below.

---

## Manual Pipeline Execution

For granular control over individual pipeline steps, use the `ced` CLI directly:

### 1. Generate Train/Val/Test Splits
```bash
ced save-splits --config configs/splits_config.yaml --infile data/your_dataset.csv
```

### 2. Train Models
```bash
ced train --config configs/training_config.yaml --model LR_EN
```

### 3. Train Ensemble (Optional)
```bash
# After training base models on same split
ced train-ensemble --base-models LR_EN,RF,XGBoost --split-seed 0
```

### 4. Aggregate Results
```bash
ced aggregate-splits --config configs/aggregate_config.yaml
```

### 5. Visualize
```bash
Rscript scripts/compare_models.R --results_root results
```

For detailed CLI usage, see [CLI Reference](analysis/docs/reference/CLI_REFERENCE.md).

---

## Post-Training Pipeline (HPC)

After submitting HPC jobs via `./run_hpc.sh`, a comprehensive post-processing pipeline automates ensemble training, result aggregation, and validation reporting.

### What It Does

The `post_training_pipeline.sh` script performs four automated steps:

1. **Validate Base Model Outputs** - Checks that all base models completed successfully for each split
2. **Train Ensemble Meta-Learner** - Combines base model predictions using stacking (if 2+ models available)
3. **Aggregate Results** - Computes bootstrap confidence intervals across all splits
4. **Generate Validation Report** - Creates summary JSON with pipeline status and metrics

### Usage

**Basic:**
```bash
cd analysis/
bash scripts/post_training_pipeline.sh --run-id 20260122_120000
```

**Skip ensemble training:**
```bash
bash scripts/post_training_pipeline.sh --run-id 20260122_120000 --skip-ensemble
```

**Custom configuration:**
```bash
bash scripts/post_training_pipeline.sh \
  --run-id 20260122_120000 \
  --results-dir ../results \
  --config configs/pipeline_hpc.yaml \
  --base-models LR_EN,RF,XGBoost \
  --min-splits 8
```

### Key Options

| Option | Description | Default |
|--------|-------------|---------|
| `--run-id` | HPC run identifier (required) | - |
| `--results-dir` | Results directory path | From config |
| `--config` | Pipeline config file | `configs/pipeline_hpc.yaml` |
| `--base-models` | Comma-separated model list | From config |
| `--skip-ensemble` | Skip ensemble training | false |
| `--min-splits` | Minimum required splits | 1 |

### Expected Outputs

**Per-model aggregated results** (under `results/{MODEL}/run_{RUN_ID}/aggregated/`):
- `aggregation_metadata.json` - Run provenance and configuration
- `test_metrics_agg.csv` - Bootstrap CI for AUROC, PR-AUC, Brier score
- `calibration_agg.csv` - Calibration slope/intercept across splits
- `feature_stability.csv` - Feature selection frequency and importance
- `dca_net_benefit.csv` - Decision curve analysis metrics
- Plots: ROC curves, calibration plots, risk distributions

**Ensemble outputs** (if trained, under `results/ENSEMBLE/run_{RUN_ID}/`):
- Same structure as base models
- Expected improvement: +2-5% AUROC over best single model

**Pipeline logs** (under `logs/post/run_{RUN_ID}/`):
- `post_training.log` - Detailed execution log with timestamps
- `pipeline_summary.json` - Validation report and file counts

### Configuration

The pipeline reads settings from `configs/pipeline_hpc.yaml`:

```yaml
execution:
  models: [LR_EN, RF, LinSVM_cal, XGBoost]  # Base models to aggregate
  n_boot: 1000  # Bootstrap iterations for CIs

ensemble:
  base_models: [LR_EN, RF, XGBoost]  # Models to combine in ensemble
  meta_model:
    type: logistic_regression
    penalty: l2
    C: 1.0
```

Ensemble training uses the `train-ensemble` CLI command internally. For manual ensemble training:

```bash
# After training base models for a single split
ced train-ensemble \
  --config configs/training_config.yaml \
  --base-models LR_EN,RF,XGBoost \
  --split-seed 0 \
  --results-dir ../results \
  --outdir ../results/ENSEMBLE/run_20260122/split_seed0
```

### Typical HPC Workflow

```bash
# 1. Submit training jobs
cd analysis/
./run_hpc.sh

# 2. Monitor job progress
bjobs -w | grep CeD_
# or for Slurm: squeue -u $USER | grep CeD_

# 3. When all jobs complete, run post-processing
bash scripts/post_training_pipeline.sh --run-id 20260122_120000

# 4. Review results
cat logs/post/run_20260122_120000/pipeline_summary.json
ls results/*/run_20260122_120000/aggregated/
```

### Troubleshooting

**Ensemble training skipped:**
- Check that at least 2 base models completed successfully
- Verify OOF prediction files exist: `results/{MODEL}/run_{RUN_ID}/split_seed*/preds/train_oof/*.csv`
- Use `--skip-ensemble` if ensemble is not needed

**Aggregation failed:**
- Check individual split outputs: `results/{MODEL}/run_{RUN_ID}/split_seed*/core/test_metrics.csv`
- Ensure minimum required splits completed (default: 1, configurable via `--min-splits`)
- Review log file: `logs/post/run_{RUN_ID}/post_training.log`

**Missing run ID:**
- Check HPC job logs for run timestamp: `logs/training/CeD_*.log`
- List results directories: `ls -d results/*/run_*`

---

## Adapting to Your Dataset

### 1. Configure Data Schema
```yaml
# configs/training_config.yaml
columns:
  mode: auto  # or "explicit" for custom metadata columns
  # numeric_metadata: [age, bmi]  # if mode=explicit
  # categorical_metadata: [sex, ethnicity]  # if mode=explicit
```

### 2. Adjust Split Strategy
```yaml
# configs/splits_config.yaml
val_size: 0.25
test_size: 0.25
train_control_per_case: 5.0  # Downsampling ratio
```

### 3. Tune Feature Selection
```yaml
features:
  screen_method: mannwhitney
  screen_top_n: 1000
  stability_thresh: 0.75
  corr_thresh: 0.85
```

### 4. Set Evaluation Metrics
```yaml
thresholds:
  objective: fixed_spec
  fixed_spec: 0.95
  target_prevalence_source: test
```

### 5. Configure Calibration Strategy
```yaml
calibration:
  enabled: true
  strategy: oof_posthoc  # Unbiased (or per_fold for legacy)
  method: isotonic
```

### 6. Enable Stacking Ensemble
```yaml
ensemble:
  enabled: true
  base_models: [LR_EN, RF, XGBoost]
  meta_model:
    type: logistic_regression
    penalty: l2
```

### 7. Enable Temporal Validation
```yaml
splits:
  temporal_split: true
  temporal_column: "sample_date"
```
---

## Example Use Case: Celiac Disease Risk Prediction

The pipeline was developed to predict **incident Celiac Disease (CeD) risk** from blood proteomics panels measured before diagnosis.

**Dataset:** 43,960 subjects (148 incident cases, 0.34% prevalence), 2,920 protein biomarkers

**Results:**
- **Best model**: Logistic Regression (ElasticNet)
- **Test AUROC**: 0.89 (95% CI: 0.86-0.92)
- **Test PR-AUC**: 0.34 (95% CI: 0.28-0.41)
- **Calibration**: Brier score = 0.012 (well-calibrated)

**Biological validation:** Top features include established CeD biomarkers (TGM2, CXCL9, ITGB7, MUC2).

---

## Key Design Principles

1. **Discrimination-First with Post-Hoc Calibration** - Optimize AUROC, then calibrate with isotonic regression
2. **Rigorous Validation** - Nested CV, no leakage, three-way split with thresholds on VAL
3. **Unbiased Calibration** - OOF-posthoc strategy eliminates calibration data leakage
4. **Ensemble Learning** - Stacking meta-learner combines diverse base models for improved predictions
5. **Temporal-Aware** - Optional chronological splits prevent future data leakage
6. **Reproducible** - Fixed seeds, YAML configs, complete provenance tracking
7. **HPC-Ready** - Non-interactive, resumable, LSF/Slurm array job support

---

## Documentation

| Document | Description |
|----------|-------------|
| [analysis/docs/ARCHITECTURE.md](analysis/docs/ARCHITECTURE.md) | Technical architecture with code pointers |
| [analysis/docs/adr/](analysis/docs/adr/) | Architecture Decision Records (15 decisions) |
| [analysis/docs/reference/CLI_REFERENCE.md](analysis/docs/reference/CLI_REFERENCE.md) | Complete CLI command reference |
| [analysis/docs/reference/HYPERPARAMETER_TUNING.md](analysis/docs/reference/HYPERPARAMETER_TUNING.md) | Hyperparameter tuning guide |
| [analysis/docs/reference/METRICS_REFERENCE.md](analysis/docs/reference/METRICS_REFERENCE.md) | Metrics behavior reference |
| [analysis/docs/reference/ARTIFACTS.md](analysis/docs/reference/ARTIFACTS.md) | Output structure and file formats |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guidelines |

---

## Testing

```bash
cd analysis
pytest tests/ -v                    # Run all tests
pytest tests/ --cov=src/ced_ml     # With coverage
```

**Stats:** 1,081 tests, 65% coverage

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and PR process.

**Non-negotiables:** No secrets in code, no emojis, tests pass before PR, no debug artifacts.

---

## Citation

```bibtex
@software{chousal2026celiacriskml,
  author = {Chousal, Andres and Chowell Lab},
  title = {CeliacRiskML: Machine Learning Pipeline for Disease Risk Prediction},
  year = {2026},
  url = {https://github.com/achousal/CeliacRiskML}
}
```

---

## License

MIT License - see [LICENSE](LICENSE)

---

## References

- **TRIPOD**: Collins et al. (2015). Transparent Reporting of Prediction Models. BMJ.
- **Calibration**: Van Calster et al. (2019). Calibration: the Achilles heel of predictive analytics. BMC Medicine.
- **DCA**: Vickers & Elkin (2006). Decision curve analysis. Medical Decision Making.
