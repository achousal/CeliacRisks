# CeliacRisks Setup Guide

**Version**: 1.0.0
**Updated**: 2026-01-20
**Author**: Andres Chousal

This guide covers environment setup for both local development and HPC production runs.

---

## Prerequisites

- Python 3.10+
- Git (for version tracking)
- ~2 GB disk space for dependencies
- Input data file: `../data/Celiac_dataset_proteomics_w_demo.parquet`
- **macOS only**: OpenMP runtime (install via `brew install libomp`)

---

## Quick Start

### Local Development (Recommended)

If you already have a conda environment:

```bash
# 1. Ensure you're in the analysis directory
cd analysis/

# 2. Install the package in your conda environment
pip install -e .

# 3. Verify installation
ced --help

# 4. Run smoke test (1 split, 1 model, ~5 minutes)
./run_local.sh

# 5. Run full local test (multiple models)
# Edit configs/pipeline_local.yaml to customize models and settings
./run_local.sh
```

### HPC Production

For HPC job submission with LSF (`bsub`):

```bash
# 1. Run automated setup (creates venv, installs package)
bash scripts/hpc_setup.sh

# 2. Edit HPC config with your HPC project allocation
# Change project: YOUR_PROJECT_ALLOCATION in configs/pipeline_hpc.yaml
nano configs/pipeline_hpc.yaml

# 3. Submit production run
./run_hpc.sh
```

---

## Detailed Setup Instructions

### Option 1: Conda Environment (Local Development)

**Best for**: Local exploration, development, testing

```bash
# Create new conda environment
conda create -n ced_ml python=3.10
conda activate ced_ml

# Navigate to project
cd analysis/

# Install package
pip install -e .

# Optional: Install development tools
pip install -e ".[dev]"

# Verify installation
ced --help
ced --version

# Run local pipeline
./run_local.sh
```

**Advantages**:
- Easy to manage multiple environments
- Works seamlessly with Jupyter notebooks
- No need for separate venv setup

### Option 2: Virtual Environment (HPC and Local)

**Best for**: HPC production runs, reproducible deployments

**Automated setup**:

```bash
cd analysis/
bash scripts/hpc_setup.sh
```

This script will:
1. Check Python version (requires 3.10+)
2. Create virtual environment in `venv/`
3. Install package and dependencies
4. Run optional test suite
5. Create output directories
6. Record package versions and git state

**Virtual environment activation**:

The setup script **cannot** activate the venv in your shell (bash subprocess limitation).

- **For pipeline runs**: No action needed - `run_hpc.sh` and `run_local.sh` activate the venv automatically
- **For interactive CLI usage**: Manually activate after setup:
  ```bash
  source venv/bin/activate
  ced --help  # Verify installation
  ```

**Manual setup** (if needed):

```bash
cd analysis/

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install package
pip install -e .

# Record environment (for reproducibility)
pip freeze > requirements_frozen_$(date +%Y%m%d).txt
```

**Advantages**:
- Explicit dependency pinning
- Matches HPC production environment exactly
- Pipeline runners handle activation automatically

---

## Environment Comparison

| Feature | Conda | Venv (hpc_setup.sh) |
|---------|-------|---------------------|
| Setup time | Fast | ~2-3 minutes |
| Disk space | ~1.5 GB | ~800 MB |
| Jupyter integration | Excellent | Manual |
| HPC compatibility | Good | Excellent |
| Works with `run_local.sh` | ✓ | ✓ |
| Works with `run_hpc.sh` | Manual venv activation | ✓ |
| Reproducibility | Good | Excellent |

---

## Running the Pipeline

### Local Pipeline (`run_local.sh`)

**Features**:
- Auto-detects conda or venv
- Runs models sequentially (no job submission)
- Config-driven via `configs/pipeline_local.yaml`
- Quick smoke testing

**Configuration**: Edit [configs/pipeline_local.yaml](configs/pipeline_local.yaml) to set:
- Models to run
- Number of bootstrap iterations
- Paths and execution settings

**Examples**:

```bash
# Standard run (uses pipeline_local.yaml)
./run_local.sh

# Use custom pipeline config
PIPELINE_CONFIG=configs/custom_pipeline.yaml ./run_local.sh

# Override dry_run setting
DRY_RUN=1 ./run_local.sh

# Override models (comma-separated)
RUN_MODELS="LR_EN,RF,XGBoost" ./run_local.sh

# Regenerate splits
OVERWRITE_SPLITS=1 ./run_local.sh

# Re-aggregate existing results (manual aggregation per model)
ced aggregate-splits --results-dir results/LR_EN/run_20250120_143022 --n-boot 100
```

**Environment variables** (override config):
- `PIPELINE_CONFIG`: Path to pipeline config (default: configs/pipeline_local.yaml)
- `RUN_MODELS`: Comma-separated list (overrides config)
- `DRY_RUN`: Preview without execution (1 or 0)
- `OVERWRITE_SPLITS`: Regenerate splits (1 or 0)

### HPC Pipeline (`run_hpc.sh`)

**Features**:
- Submits jobs to LSF scheduler via `bsub`
- Parallel execution (configurable models and splits)
- Config-driven via `configs/pipeline_hpc.yaml`
- Production-grade logging

**Setup**:

1. Edit [configs/pipeline_hpc.yaml](configs/pipeline_hpc.yaml) to set your HPC project:
   ```yaml
   hpc:
     project: YOUR_PROJECT_ALLOCATION  # Update this!
   ```

2. Run setup (if not done already):
   ```bash
   bash scripts/hpc_setup.sh
   ```

3. Submit pipeline:
   ```bash
   ./run_hpc.sh
   ```

**Configuration**: Edit [configs/pipeline_hpc.yaml](configs/pipeline_hpc.yaml) to set:
- Models to run
- Number of bootstrap iterations
- HPC resources (cores, memory, walltime, queue)
- Paths and execution settings

**Examples**:

```bash
# Standard production run (uses pipeline_hpc.yaml)
./run_hpc.sh

# Use custom pipeline config
PIPELINE_CONFIG=configs/custom_hpc.yaml ./run_hpc.sh

# Override dry_run setting
DRY_RUN=1 ./run_hpc.sh

# Override models
RUN_MODELS="LR_EN,RF" ./run_hpc.sh

# Monitor jobs
bjobs -w | grep CeD_
```

---

## Verification and Testing

### Quick verification

```bash
# Check CLI is available
ced --help

# Check version
ced --version

# List available commands
ced --help
```

### Run test suite

```bash
# Full test suite
pytest tests/ -v

# With coverage
pytest tests/ --cov=ced_ml --cov-report=term-missing

# Fast tests only (skip slow integration tests)
pytest tests/ -v -m "not slow"
```

### Smoke test

```bash
# Generate splits, train models, postprocess
./run_local.sh

# Check outputs
ls -la ../results/
```

---

## Troubleshooting

### "ced: command not found"

**Solution**: Install package in current environment:
```bash
pip install -e .
```

### "Virtual environment not found at venv/bin/activate"

**Solution**: Either:
- Use conda: `conda activate ced_ml && pip install -e .`
- Create venv: `bash scripts/hpc_setup.sh`

### "bsub: command not found" (on local machine)

**Expected**: `bsub` is HPC-specific. Use `run_local.sh` instead:
```bash
./run_local.sh
```

### "XGBoost Library (libxgboost.dylib) could not be loaded" (macOS only)

**Cause**: XGBoost requires OpenMP runtime (`libomp`) for parallel processing.

**Solution**: Install OpenMP via Homebrew:
```bash
brew install libomp
```

**Verification**:
```bash
python -c "from xgboost import XGBClassifier; print('XGBoost OK')"
```

**Note**: This is macOS-specific. Linux and HPC environments typically have OpenMP pre-installed.

### Package import errors

**Solution**: Reinstall in development mode:
```bash
pip install -e . --force-reinstall --no-deps
```

### Tests fail after setup

**Common causes**:
- Missing data file: Check `../data/Celiac_dataset_proteomics_w_demo.parquet`
- Old cached files: `rm -rf .pytest_cache __pycache__`
- Version mismatch: `pip install -e . --upgrade`

---

## File Structure

After setup, you should have:

```
analysis/
├── venv/                  # Virtual environment (if using hpc_setup.sh)
├── src/ced_ml/            # Core package
│   ├── cli/               # CLI entrypoints
│   ├── data/              # Data loading, splits, preprocessing
│   ├── features/          # Feature selection
│   ├── models/            # Model training, Optuna integration
│   ├── evaluation/        # Calibration, metrics, bootstrapping
│   ├── metrics/           # DCA, ROC/PR, thresholds
│   ├── plotting/          # Visualizations
│   └── utils/             # Logging, config, reproducibility
├── tests/                 # Pytest tests
├── configs/               # YAML configuration files
│   ├── pipeline_local.yaml     # Local pipeline config
│   ├── pipeline_hpc.yaml       # HPC pipeline config
│   ├── splits_config.yaml      # CV split settings
│   └── training_config.yaml    # Model training settings
├── docs/                  # Documentation
│   ├── ARCHITECTURE.md    # Technical architecture
│   ├── adr/               # Architecture Decision Records
│   └── reference/         # API and CLI reference
├── scripts/               # Automation scripts
│   ├── hpc_setup.sh       # Automated setup script
│   └── aggregate_results.py    # Post-processing
├── splits/                # CV split indices (generated)
├── results/               # Model results and outputs (generated)
├── logs/                  # Run logs (generated)
├── run_local.sh           # Local pipeline runner
├── run_hpc.sh             # HPC pipeline runner
├── SETUP_README.md        # This file
└── pyproject.toml         # Package configuration
```

---

## Environment Recording (Reproducibility)

### Automatic (via hpc_setup.sh)

The setup script automatically creates:
- `requirements_frozen_YYYYMMDD.txt`: Exact package versions
- `git_version.txt`: Git commit at setup time

### Manual

```bash
# Record package versions
pip freeze > requirements_frozen_$(date +%Y%m%d).txt

# Record git state
git log -1 --oneline > git_version.txt

# Record Python version
python --version > python_version.txt
```

---

## Next Steps

### Local Development Workflow

1. **Explore data and configs**:
   ```bash
   ls ../data/
   ls configs/
   ```

2. **Quick test**:
   ```bash
   ./run_local.sh
   ```

3. **Iterate on configs**:
   ```bash
   # Edit pipeline config
   nano configs/pipeline_local.yaml

   # Edit training config
   nano configs/training_config.yaml

   # Rerun with new settings
   ./run_local.sh

   # Or force regenerate splits
   OVERWRITE_SPLITS=1 ./run_local.sh
   ```

4. **Full local test** (before HPC deployment):
   ```bash
   # Edit configs/pipeline_local.yaml to set multiple models
   # Then run:
   ./run_local.sh
   ```

### HPC Production Workflow

1. **Verify setup**:
   ```bash
   bash scripts/hpc_setup.sh
   source venv/bin/activate
   ced --help
   ```

2. **Dry run**:
   ```bash
   DRY_RUN=1 ./run_hpc.sh
   ```

3. **Single split test** (edit configs first):
   ```bash
   # Edit configs/pipeline_hpc.yaml to set 1 split, 1 model
   ./run_hpc.sh
   bjobs -w
   ```

4. **Full production**:
   ```bash
   # Ensure configs/pipeline_hpc.yaml has production settings
   ./run_hpc.sh
   ```

---

## Documentation

- **This guide**: [SETUP_README.md](SETUP_README.md) - Environment setup and getting started
- **Project overview**: [README.md](../README.md) - High-level project context
- **Architecture**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - Technical design and data flow
- **CLI Reference**: [docs/reference/CLI_REFERENCE.md](docs/reference/CLI_REFERENCE.md) - Command-line interface guide

---

## Support

If you encounter issues:

1. Check this guide's troubleshooting section
2. Verify Python version: `python --version` (requires 3.10+)
3. Check logs in `logs_local/` or `logs_a/`
4. Run tests: `pytest tests/ -v`
5. Contact: Andres Chousal

---

**Last Updated**: 2026-01-20
**Tested On**:
- macOS (local)
- HPC cluster with LSF scheduler
