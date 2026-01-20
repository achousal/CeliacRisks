# CeliacRiskML

Machine learning pipeline for predicting incident Celiac Disease (CeD) risk from proteomics biomarkers.

[![Tests](https://github.com/achousal/CeliacRiskML/workflows/Tests/badge.svg)](https://github.com/achousal/CeliacRiskML/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Overview

This project develops calibrated ML models to predict incident Celiac Disease risk from blood proteomics panels measured before clinical diagnosis. The pipeline generates continuous risk scores for apparently healthy individuals to inform follow-up testing decisions.

**Clinical Workflow:**
```
Blood proteomics → ML risk score → [High risk?] → Anti-tTG test → Endoscopy
```

## Key Features

- **Production-ready package**: Modular architecture with 15k+ lines of tested code
- **Four ML models**: Random Forest, XGBoost, Linear SVM, Logistic Regression (ElasticNet)
- **Rigorous evaluation**: Nested CV (5x10), prevalence adjustment, calibration, DCA
- **HPC-optimized**: LSF/Slurm batch scripts, resumable workflows
- **High test coverage**: 832 passing tests (85% coverage)
- **CLI interface**: Simple commands for splits, training, evaluation
- **Reproducible**: YAML configs, fixed seeds, provenance tracking

## Quick Start

### Installation

```bash
cd analysis
pip install -e .
ced --help
```

### Basic Workflow

**1. Generate train/val/test splits:**
```bash
ced save-splits \
  --infile ../data/Celiac_dataset_proteomics.csv \
  --outdir splits_production \
  --scenarios IncidentPlusPrevalent \
  --n-splits 10
```

**2. Train models:**
```bash
# Local (single model)
ced train \
  --config configs/training_config.yaml \
  --model LR_EN \
  --infile ../data/Celiac_dataset_proteomics.csv \
  --split-dir splits_production

# HPC (4 models in parallel)
bsub < CeD_production.lsf
```

**3. Aggregate results:**
```bash
ced postprocess --results-dir results_production --n-boot 500
```

**4. Visualize (R):**
```bash
Rscript compare_models_faith.R --results_root results_production
```

## Dataset

| Attribute | Value |
|-----------|-------|
| Total samples | 43,960 |
| Incident CeD | 148 (0.34%) |
| Prevalent CeD | 150 (enrichment only) |
| Controls | 43,662 |
| Proteins | 2,920 |
| Demographics | Age, BMI, sex, ethnicity |

## Models

| Model | Algorithm | Notes |
|-------|-----------|-------|
| **RF** | Random Forest | 500 trees, class_weight tuning |
| **XGBoost** | Gradient Boosting | Auto scale_pos_weight, GPU support |
| **LinSVM_cal** | Linear SVM | Sigmoid calibration |
| **LR_EN** | Logistic Regression | ElasticNet penalty (recommended) |

All models use:
- Nested CV: 5 outer folds × 10 repeats × 5 inner folds
- Brier score optimization
- Prevalence adjustment (train 16.7% → deployment 0.34%)
- Feature selection: Mann-Whitney screening + k-best + stability

## Evaluation Metrics

- **Calibration**: Brier score, calibration slope/intercept
- **Discrimination**: AUROC, PR-AUC
- **Clinical utility**: DCA net benefit, sensitivity at 95%/99% specificity
- **Bootstrap CI**: 500 iterations on test set

## Documentation

- [CLAUDE.md](analysis/CLAUDE.md) - Complete project documentation
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [docs/](analysis/docs/) - Technical guides
  - [MIGRATION_GUIDE.md](analysis/docs/MIGRATION_GUIDE.md) - Legacy script migration
  - [HPC_MIGRATION_GUIDE.md](analysis/docs/HPC_MIGRATION_GUIDE.md) - HPC deployment

## Project Structure

```
CeliacRiskML/
├── analysis/                           # ML pipeline package (ced-ml)
│   ├── Celiac_dataset_proteomics.csv  # Main dataset (not in repo)
│   ├── src/ced_ml/                     # Source code
│   │   ├── cli/                        # Command-line interface
│   │   ├── data/                       # Data I/O, splits
│   │   ├── features/                   # Feature selection
│   │   ├── models/                     # Training, calibration
│   │   ├── metrics/                    # Performance metrics
│   │   ├── evaluation/                 # Prediction, reports
│   │   └── plotting/                   # Visualizations
│   ├── tests/                          # 832 unit/integration tests
│   ├── configs/                        # YAML configurations
│   ├── docs/                           # Documentation
│   ├── README.md                       # Package quickstart
│   ├── CLAUDE.md                       # Detailed project guide
│   └── pyproject.toml                  # Package configuration
├── Legacy/                             # Archived legacy scripts
├── .github/                            # GitHub workflows, templates
├── CONTRIBUTING.md                     # Contribution guide
├── LICENSE                             # MIT License
└── README.md                           # This file
```

## Testing

```bash
cd analysis

# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src/ced_ml --cov-report=term

# Skip slow tests
pytest tests/ -m "not slow"
```

**Test Coverage:** 85% overall (832 passing tests)

## HPC Deployment

### LSF Example

```bash
# Submit 4 models as array job
bsub < CeD_production.lsf

# Monitor
bjobs

# Check output
tail -f logs/CeD_*.out
```

See [HPC_MIGRATION_GUIDE.md](analysis/docs/HPC_MIGRATION_GUIDE.md) for detailed HPC setup.

## Configuration

All pipeline parameters are managed via YAML files in [configs/](analysis/configs/):

- `splits_config.yaml` - Data splitting parameters
- `training_config.yaml` - Model training settings
- `holdout_config.yaml` - External validation

**Example:**
```yaml
model: LR_EN
scenario: IncidentPlusPrevalent

cv:
  folds: 5
  repeats: 10
  scoring: neg_brier_score

features:
  feature_select: hybrid
  screen_top_n: 1000
  stability_thresh: 0.75

thresholds:
  objective: fixed_spec
  fixed_spec: 0.95
```

## Key Design Decisions

1. **IncidentPlusPrevalent scenario**: Prevalent cases enrichment in TRAIN only (50% sampling), VAL/TEST remain incident-only
2. **Three-way split**: 50% TRAIN / 25% VAL / 25% TEST
3. **Brier score optimization**: Prioritizes calibration over discrimination
4. **Control downsampling**: 1:5 case:control ratio to reduce imbalance
5. **Missing as category**: 17% missing ethnicity treated as explicit category

See [CLAUDE.md](analysis/CLAUDE.md) for detailed rationale.

## Biological Validation

Top proteins include established CeD biomarkers:

| Protein | Cohen's d | Clinical Relevance |
|---------|-----------|-------------------|
| **TGM2** | 1.73 | Primary CeD autoantigen |
| **CXCL9** | 1.53 | Inflammatory chemokine |
| **ITGB7** | 1.50 | Gut-homing integrin |
| **MUC2** | 0.96 | Intestinal mucin |

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup
- Code style guidelines
- Testing requirements
- Pull request process

**Non-negotiables:**
- No secrets in code
- No emojis anywhere
- All tests pass before PR
- No debug artifacts (console.log, print, browser)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{chousal2026celiacriskml,
  author = {Chousal, Andres and Elahi Lab},
  title = {CeliacRiskML: Machine Learning for Celiac Disease Risk Prediction},
  year = {2026},
  url = {https://github.com/achousal/CeliacRiskML}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## Authors

- **Andres Chousal** - [achousal](https://github.com/achousal)
- **Elahi Lab** - Stanford University

## References

- **TRIPOD**: Collins et al. (2015). Transparent Reporting of Prediction Models. BMJ.
- **Calibration**: Van Calster et al. (2019). Calibration: the Achilles heel of predictive analytics. BMC Medicine.
- **DCA**: Vickers & Elkin (2006). Decision curve analysis. Medical Decision Making.
- **CeD Biology**: Sollid & Jabri (2013). Triggers and drivers of autoimmunity. Nature Reviews Immunology.

## Support

- **Issues**: [GitHub Issues](https://github.com/achousal/CeliacRiskML/issues)
- **Questions**: Open a GitHub Discussion
- **Contact**: @achousal