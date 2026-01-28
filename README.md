# CeliacRiskML

**Production-grade ML pipeline for disease risk prediction from high-dimensional biomarker data**

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![Tests](https://img.shields.io/badge/tests-1081%20passing-success)
![Coverage](https://img.shields.io/badge/coverage-65%25-yellowgreen)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## What is this?

A modular ML framework for predicting disease risk from proteomics or other high-dimensional biomarker data. Built for Celiac Disease prediction, but generalizable to any disease/biomarker combination.

**Core features:**
- Multi-model ensemble with stacking (RF, XGBoost, SVM, Logistic Regression)
- Nested cross-validation with rigorous feature selection
- Unbiased calibration (OOF-posthoc strategy)
- Clinical decision curve analysis
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

For manual control over individual steps, see [CLI Reference](analysis/docs/reference/CLI_REFERENCE.md).

---

## Key Capabilities

### ML Pipeline
- **Models**: Logistic Regression (ElasticNet), Random Forest, XGBoost, Linear SVM
- **Ensemble**: Out-of-fold stacking with L2 meta-learner
- **Feature Selection**: Effect size → k-best → stability selection → correlation pruning
- **Hyperparameter Tuning**: Bayesian optimization (Optuna TPE, 100 trials)
- **Temporal Validation**: Chronological splits to prevent future data leakage

### Evaluation
- **Discrimination**: AUROC, PR-AUC with bootstrap CIs
- **Calibration**: Brier score, slope/intercept, calibration curves (OOF-posthoc)
- **Clinical Utility**: Decision curve analysis with auto-configured threshold ranges
- **Visualizations**: ROC/PR curves, calibration plots, risk distributions, DCA plots

### Production Features
- **Reproducibility**: Fixed seeds, YAML configs, Git commit tracking
- **HPC Support**: LSF/Slurm array jobs with automated post-processing
- **Provenance**: Complete metadata for every run (config, environment, timing)
- **Testing**: 1,081 tests, 65% coverage

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
| [ARCHITECTURE.md](analysis/docs/ARCHITECTURE.md) | Technical architecture + code pointers |
| [ADRs](analysis/docs/adr/) | 15 architectural decisions (split strategy, calibration, ensembles, etc.) |
| [CLI_REFERENCE.md](analysis/docs/reference/CLI_REFERENCE.md) | Complete command reference |
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
