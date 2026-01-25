# CeD-ML Project Overview

**Updated**: 2026-01-21
**Version**: 1.0.0
**Package**: ced-ml
**Python**: 3.10+

Primary documentation: [README.md](../../README.md)

## Quick Facts

**Purpose**: Production ML pipeline for disease risk prediction from high-dimensional biomarker data

**Dataset**: 43,960 samples, 2,920 protein biomarkers, 0.34% prevalence (148 incident cases)

**Models**: Logistic Regression (ElasticNet/L1), Random Forest, XGBoost, Linear SVM (calibrated)

**Statistics**:
- Code: 20,821 lines
- Tests: 770 tests, 85% coverage
- ADRs: 19 Architecture Decision Records

## Core Components

**Data**: [src/ced_ml/data/](../../analysis/src/ced_ml/data/)
- Splits, preprocessing, loading (CSV/Parquet)

**Features**: [src/ced_ml/features/](../../analysis/src/ced_ml/features/)
- Multi-stage selection (effect size, k-best, stability, correlation)

**Models**: [src/ced_ml/models/](../../analysis/src/ced_ml/models/)
- Nested CV, Optuna optimization, registry

**Evaluation**: [src/ced_ml/evaluation/](../../analysis/src/ced_ml/evaluation/)
- Calibration (isotonic + prevalence adjustment), metrics, bootstrap

**Metrics**: [src/ced_ml/metrics/](../../analysis/src/ced_ml/metrics/)
- DCA, ROC/PR, threshold optimization

**Plotting**: [src/ced_ml/plotting/](../../analysis/src/ced_ml/plotting/)
- Calibration, ROC, DCA plots

## Recent Major Changes

**2026-01-20**: Optuna integration (Bayesian TPE + pruning for 2-5x speedup)
**2026-01-19**: Pipeline refactor (unified configs, environment variable overrides)
**2026-01-18**: Split-specific output directories for better parallel execution

## Key Design Principles

1. **Discrimination-first with post-hoc calibration**: Optimize AUROC, then calibrate
2. **Rigorous validation**: Nested CV, no leakage, three-way split
3. **Prevalence-aware**: Calibrated to deployment prevalence
4. **Reproducible**: Fixed seeds, config logging, git tracking
5. **HPC-ready**: Non-interactive, resumable, job array support

## Configuration System

**Hierarchy** (lower overrides higher):
1. `training_config.yaml` - Model settings, feature selection
2. `splits_config.yaml` - CV split settings
3. `pipeline_{local,hpc}.yaml` - Execution settings
4. Environment variables - Runtime overrides
5. CLI flags - Direct overrides

## Quick Commands

```bash
# Local testing
cd analysis/ && pip install -e . && ./run_local.sh

# HPC production
cd analysis/ && bash scripts/hpc_setup.sh && ./run_hpc.sh

# CLI
ced save-splits --config configs/splits_config.yaml --infile ../data/input.parquet
ced train --config configs/training_config.yaml --model LR_EN --split-index 0
ced postprocess --results-dir ../results --n-boot 500
```

## Documentation Map

- `README.md` - Primary project documentation (this is the source of truth)
- `SETUP_README.md` - Environment setup guide
- `docs/ARCHITECTURE.md` - Technical architecture
- `docs/adr/` - 19 Architecture Decision Records
- `CONTRIBUTING.md` - Development guidelines
