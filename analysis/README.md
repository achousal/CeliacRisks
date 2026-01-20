# CeD-ML: Machine Learning Pipeline for Celiac Disease Risk Prediction

**Status:** Phase D Complete - Production Ready ✅

[![Tests](https://img.shields.io/badge/tests-832%20passing-success)]() [![Coverage](https://img.shields.io/badge/coverage-85%25-brightgreen)]()

A modular, reproducible ML pipeline for predicting incident Celiac Disease (CeD) risk from proteomics biomarkers.

---

## Quick Start

### Installation

```bash
# Development mode (editable install)
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

### Generate Splits

```bash
# Using config file
ced save-splits --config docs/examples/splits_config.yaml \
    --infile ../../Celiac_dataset_proteomics.csv

# Using CLI args only
ced save-splits \
    --infile ../../Celiac_dataset_proteomics.csv \
    --outdir splits_test \
    --scenarios IncidentPlusPrevalent \
    --n-splits 10 \
    --val-size 0.25 \
    --test-size 0.25 \
    --prevalent-train-only \
    --prevalent-train-frac 0.5 \
    --train-control-per-case 5
```

### Train Models

```bash
# Using config file
ced train --config docs/examples/training_config.yaml

# With CLI overrides
ced train --config docs/examples/training_config.yaml \
    --model RF \
    --override cv.folds=10 \
    --override features.screen_top_n=2000
```

---

### Configuration Schema (~200 Parameters)

All current pipeline parameters are organized into typed configuration classes:

- **SplitsConfig**: Data split generation (17 params)
- **CVConfig**: Cross-validation structure (8 params)
- **FeatureConfig**: Feature selection methods (14 params)
- **PanelConfig**: Biomarker panel building (9 params)
- **ThresholdConfig**: Threshold selection (8 params)
- **EvaluationConfig**: Metrics and reporting (12 params)
- **DCAConfig**: Decision curve analysis (5 params)
- **OutputConfig**: File generation control (8 params)
- **StrictnessConfig**: Validation levels (6 params)
- **Model-specific configs**: LR, SVM, RF, XGBoost hyperparameters (~60 params)

### CLI Commands

```bash
# Core pipeline commands
ced save-splits     # Generate train/val/test splits
ced train           # Train ML models
ced postprocess     # Aggregate results across models
ced eval-holdout    # Evaluate on holdout set (run ONCE only)

# Configuration management (Phase C)
ced config migrate  # Convert legacy CLI args to YAML
ced config validate # Validate config files
ced config diff     # Compare two configs
```

---

## Configuration Examples

### Splits Configuration

```yaml
# docs/examples/splits_config.yaml
mode: development
scenarios:
  - IncidentPlusPrevalent

n_splits: 10
val_size: 0.25
test_size: 0.25

prevalent_train_only: true
prevalent_train_frac: 0.5
train_control_per_case: 5.0

outdir: splits_production
```

### Training Configuration

```yaml
# docs/examples/training_config.yaml
model: LR_EN
scenario: IncidentPlusPrevalent

cv:
  folds: 5
  repeats: 10
  scoring: neg_brier_score
  n_iter: 200

features:
  feature_select: hybrid
  screen_top_n: 1000
  stability_thresh: 0.75

thresholds:
  objective: max_f1
  threshold_source: val
  target_prevalence_source: test

evaluation:
  test_ci_bootstrap: true
  n_boot: 500

outdir: results_production
```

### CLI Overrides

```bash
# Override nested config values
ced train --config training.yaml \
    --override cv.folds=10 \
    --override cv.repeats=20 \
    --override features.screen_top_n=2000 \
    --override thresholds.objective=youden
```

---

## Package Structure

```
src/ced_ml/                       # 15,109 lines of modular code
  cli/                            # Command-line interface
    main.py                       # Entry point (ced command)
    save_splits.py, train.py      # Pipeline commands
    postprocess.py, eval_holdout.py
    config_tools.py               # Config migration/validation
  config/                         # Configuration system
    schema.py                     # Pydantic models (~200 params)
    loader.py                     # YAML + CLI override loading
    validation.py                 # Leakage detection, strictness
  data/                           # Data layer (386 lines, 111 tests)
    io.py, splits.py              # I/O, split generation
    persistence.py, schema.py     # Index saving, constants
  features/                       # Feature engineering (433 lines, 128 tests)
    screening.py, kbest.py        # Mann-Whitney, K-best selection
    stability.py, corr_prune.py   # Stability tracking, correlation pruning
    panels.py                     # Biomarker panel building
  models/                         # Model training (549 lines, 103 tests)
    registry.py, hyperparams.py   # Model zoo, hyperparameter grids
    training.py                   # Nested CV orchestration
    calibration.py, prevalence.py # Calibration, prevalence adjustment
  metrics/                        # Performance metrics (421 lines, 163 tests)
    discrimination.py             # AUROC, PR-AUC, Youden
    thresholds.py, dca.py         # Threshold selection, decision curve analysis
    bootstrap.py                  # Stratified bootstrap CIs
  evaluation/                     # Prediction & reporting (381 lines, 81 tests)
    predict.py, reports.py        # Prediction generation, results writer
    holdout.py                    # Holdout set evaluation
  plotting/                       # Visualization (1,082 lines, 122 tests)
    roc_pr.py, calibration.py     # ROC/PR curves, calibration plots
    risk_dist.py, dca.py          # Risk distribution, DCA plots
    learning_curve.py             # Learning curve analysis
  utils/                          # Shared utilities
    logging.py, paths.py          # Logging, path handling
    random.py, serialization.py   # RNG, save/load helpers

Legacy/                           # Archived monolithic scripts
  celiacML_faith.py               # Original 4,000-line training script
  shared_utils.py                 # Original 3,000-line utilities
  save_splits.py                  # Original split generation
  postprocess_compare.py          # Original postprocessing

tests/                            # 832 tests, 85% coverage
  test_data_*.py                  # Data layer tests (111 tests)
  test_features_*.py              # Features layer tests (128 tests)
  test_models_*.py                # Models layer tests (103 tests)
  test_metrics_*.py               # Metrics layer tests (163 tests)
  test_evaluation_*.py            # Evaluation layer tests (81 tests)
  test_plotting_*.py              # Plotting layer tests (122 tests)
  test_cli_*.py                   # CLI integration tests (21 tests)
  test_config*.py                 # Config system tests (29 tests)
```

---

## Behavioral Equivalence Guarantee

**All defaults exactly match the current implementation** to ensure reproducibility:

- Default split sizes: `val_size=0.0`, `test_size=0.30`
- Default CV: `folds=5`, `repeats=3`, `scoring='average_precision'`
- Default feature selection: `screen_method='mannwhitney'`, `stability_thresh=0.70`
- Default model hyperparameters match `get_param_distributions()` in `celiacML_faith.py`

### Verification

```bash
# Test config roundtrip
ced save-splits --config docs/examples/splits_config.yaml --infile dummy.csv

# Resolved config is saved to: splits_production/splits_config.yaml
# Compare with original to verify all values were loaded correctly
```
### Next: Production Deployment

- [ ] Update HPC batch scripts to use `ced` CLI
- [ ] Performance benchmarking (training time, memory usage)
- [ ] Documentation for external users
- [ ] Publication-ready reproducibility artifacts

## Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run full test suite (832 tests)
pytest tests/ -v

# Run tests with coverage report
pytest tests/ --cov=src/ced_ml --cov-report=html

# Run specific test modules
pytest tests/test_data_*.py -v       # Data layer tests
pytest tests/test_features_*.py -v   # Features layer tests
pytest tests/test_models_*.py -v     # Models layer tests

# Check types
mypy src/ced_ml/

# Format code
black src/ced_ml/
ruff src/ced_ml/
```

**Test Summary:**
- 832 passing tests
- 85% code coverage
- All 6 layers tested (data, features, models, metrics, evaluation, plotting)
- CLI integration verified (zero duplication)
- Behavioral equivalence validated

---

## Contributing

See [CLAUDE.md](CLAUDE.md) for detailed project documentation and the refactoring plan.

## License

MIT

---