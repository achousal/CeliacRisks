# CeliacRiskML

**Production-grade ML pipeline for disease risk prediction from high-dimensional biomarker data**

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![Tests](https://img.shields.io/badge/tests-1271%20passing-success)
![Coverage](https://img.shields.io/badge/coverage-14%25-red)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## What is this?

A modular ML framework for predicting disease risk from proteomics or other high-dimensional biomarker data. Built for Celiac Disease prediction, but generalizable to any disease/biomarker combination.

**Core features:**
- Multi-model ensemble with stacking (RF, XGBoost, SVM, Logistic Regression)
- Five feature selection strategies (hybrid stability, nested RFECV, post-hoc RFE, cross-model consensus, fixed panel)
- Nested cross-validation with Bayesian hyperparameter optimization
- Unbiased calibration (OOF-posthoc strategy)
- Clinical decision curve analysis
- Cross-model consensus panel generation via Robust Rank Aggregation
- HPC-ready (LSF/Slurm) with full provenance tracking

---

## Quick Start

### Local
```bash
git clone https://github.com/achousal/CeliacRiskML.git
cd CeliacRiskML/analysis
pip install -e .
./run_local.sh
```

### HPC
```bash
cd analysis
bash scripts/hpc_setup.sh
./run_hpc.sh

# After jobs finish
bash scripts/post_training_pipeline.sh --run-id <RUN_ID>
```

**Pipeline flow:** Data → Split → Feature Selection → Model Training → Calibration → Ensemble → Evaluation

---

## What Can This Software Do?

### 1. Train Models Locally
Run the full pipeline on your machine with default configs:

```bash
cd analysis/
./run_local.sh
```

**What happens:** Trains 4 models (LR, RF, SVM, XGBoost) on 10 splits with nested CV, hyperparameter tuning, and calibration. Results in `../results/`.

### 2. Train Models on HPC
Submit batch jobs with automated resource management:

```bash
cd analysis/
./run_hpc.sh

# Monitor jobs
bjobs -w | grep CeD_
```

**Configure resources** in `configs/pipeline_hpc.yaml` (cores, memory, walltime, queue).

### 3. Post-Training Pipeline (HPC)
After HPC jobs complete, run automated aggregation and ensemble training:

```bash
bash scripts/post_training_pipeline.sh --run-id 20260127_115115 --train-ensemble
```

**What happens:**
1. Validates all model outputs
2. Trains stacking ensemble meta-learner
3. Aggregates results across splits (metrics + bootstrap CIs)
4. Generates validation reports and summary JSON

**No manual steps needed** - auto-detects models and paths from run-id.

### 4. Optimize Panel Size (Single Model)
Find the minimal protein panel for clinical deployment:

```bash
ced optimize-panel --run-id 20260127_115115 --model LR_EN
```

**Output:** `panel_curve.png`, `recommended_panel_50.csv`, `rfe_results.csv` in `results/LR_EN/run_{RUN_ID}/aggregated/panel_optimization/`

### 5. Cross-Model Consensus Panel
Generate a robust panel via Robust Rank Aggregation across all models:

```bash
ced consensus-panel --run-id 20260127_115115
```

**Output:** `consensus_panel_50.csv`, `rra_scores.csv`, `model_comparison.png` in `results/consensus_panel/run_{RUN_ID}/`

**Use this for deployment** when you want features that work well across multiple models.

---

### Configuration-Based Workflow

All commands use YAML configs in `analysis/configs/`:

- `pipeline_local.yaml` / `pipeline_hpc.yaml` - Models, paths, execution settings
- `training_config.yaml` - Feature selection, calibration, ensemble
- `splits_config.yaml` - Train/val/test ratios

**Example:** Enable temporal validation and OOF-posthoc calibration:

```yaml
# training_config.yaml
splits:
  temporal_split: true
  temporal_column: "sample_date"

calibration:
  strategy: oof_posthoc  # Unbiased calibration

ensemble:
  enabled: true
  base_models: [LR_EN, RF, XGBoost]
```

Then just run `./run_local.sh` - configs are applied automatically.

---

## Key Capabilities

### ML Pipeline
- **Models**: Logistic Regression (ElasticNet), Random Forest, XGBoost, Linear SVM
- **Ensemble**: Out-of-fold stacking with L2 meta-learner
- **Feature Selection**: Five strategies optimized for different use cases
  - Hybrid stability (default, ~30 min)
  - Nested RFECV (scientific discovery, ~22 hrs)
  - Post-hoc RFE (single-model deployment, ~10 min)
  - Cross-model consensus via RRA (robust deployment, ~15 min)
  - Fixed panel validation (regulatory, ~30 min)
- **Hyperparameter Tuning**: Bayesian optimization (Optuna TPE, 100 trials)
- **Temporal Validation**: Chronological splits to prevent future data leakage

### Evaluation
- **Discrimination**: AUROC, PR-AUC with bootstrap CIs
- **Calibration**: Brier score, slope/intercept, calibration curves (OOF-posthoc)
- **Clinical Utility**: Decision curve analysis with auto-configured threshold ranges
- **Visualizations**: ROC/PR curves, calibration plots, risk distributions, DCA plots

### Production Features
- **Reproducibility**: Fixed seeds, YAML configs, Git commit tracking
- **HPC Support**: LSF/Slurm array jobs with automated post-processing (conda/venv compatible)
- **Provenance**: Complete metadata for every run (config, environment, timing)
- **Testing**: 1,271 tests with comprehensive E2E coverage
- **Investigation Framework**: Factorial experiment design for methodological optimization

---

## Configuration

All settings in YAML configs under `analysis/configs/`:

**Essential configs:**
- `pipeline_{local,hpc}.yaml` - Execution settings (models, paths, HPC resources)
- `training_config.yaml` - Model/feature/calibration settings
- `splits_config.yaml` - Train/val/test split ratios

**Example: Enable ensemble + temporal validation**
```yaml
# training_config.yaml
ensemble:
  enabled: true
  base_models: [LR_EN, RF, XGBoost]

splits:
  temporal_split: true
  temporal_column: "sample_date"

calibration:
  strategy: oof_posthoc  # Unbiased
```

---

## Example Results: Celiac Disease

**Dataset:** 43,960 subjects (148 incident cases, 150 prevalent cases), 2,920 proteins, plus demographics (age, BMI, sex, genetic ancestry)

**Top Features:** TGM2, CXCL9, ITGB7, MUC2 (known CeD biomarkers)

---

## Documentation

| Document | Description |
|----------|-------------|
| [CLAUDE.md](CLAUDE.md) | Project overview (START HERE) |
| [ARCHITECTURE.md](analysis/docs/ARCHITECTURE.md) | Technical architecture + code pointers |
| [ADRs](analysis/docs/adr/) | 15 architectural decisions (split strategy, calibration, ensembles, etc.) |
| [CLI_REFERENCE.md](analysis/docs/reference/CLI_REFERENCE.md) | Complete command reference |
| [FEATURE_SELECTION.md](analysis/docs/reference/FEATURE_SELECTION.md) | Five feature selection strategies guide |
| [HYPERPARAMETER_TUNING.md](analysis/docs/reference/HYPERPARAMETER_TUNING.md) | Tuning guide |
| [ARTIFACTS.md](analysis/docs/reference/ARTIFACTS.md) | Output structure |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and guidelines.

---

## Citation

```bibtex
@software{chousal2026celiacriskml,
  author = {Chousal, Andres and Chowell Lab},
  title = {CeliacRiskML: ML Pipeline for Disease Risk Prediction},
  year = {2026},
  url = {https://github.com/achousal/CeliacRiskML}
}
```

---

## License

MIT License - see [LICENSE](LICENSE)
