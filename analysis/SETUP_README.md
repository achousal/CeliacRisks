# CeliacRisks Setup Guide

**Version**: 1.0.0
**Updated**: 2026-01-20
**Author**: Andres Chousal

This guide covers environment setup for both local development and HPC production runs.

---

## Prerequisites

- Python 3.8+
- Git (for version tracking)
- ~2 GB disk space for dependencies
- Input data file: `../data/Celiac_dataset_proteomics_w_demo.csv`

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
./my_run_local.sh

# 5. Run full local test (3 splits, 2 models)
N_SPLITS=3 RUN_MODELS="LR_EN,RF" ./my_run_local.sh
```

### HPC Production

For HPC job submission with LSF (`bsub`):

```bash
# 1. Run automated setup (creates venv, installs package)
bash scripts/hpc_setup.sh

# 2. Edit production script with your HPC project allocation
# Change PROJECT="acc_Chipuk_Laboratory" to your allocation
nano my_run_production.sh

# 3. Submit production run
./my_run_production.sh
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
./my_run_local.sh
```

**Advantages**:
- Easy to manage multiple environments
- Works seamlessly with Jupyter notebooks
- No need for separate venv setup

### Option 2: Virtual Environment (HPC and Local)

**Best for**: HPC production runs, reproducible deployments

This is what `scripts/hpc_setup.sh` does automatically:

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

# Verify installation
ced --help

# Record environment (for reproducibility)
pip freeze > requirements_frozen_$(date +%Y%m%d).txt
```

**Or run the automated script**:

```bash
bash scripts/hpc_setup.sh
```

This script will:
1. Check Python version (requires 3.8+)
2. Create virtual environment in `venv/`
3. Install package and dependencies
4. Run optional test suite
5. Create output directories
6. Record package versions and git state

**Advantages**:
- Explicit dependency pinning
- Matches HPC production environment exactly
- Works with `my_run_production.sh` out of the box

---

## Environment Comparison

| Feature | Conda | Venv (hpc_setup.sh) |
|---------|-------|---------------------|
| Setup time | Fast | ~2-3 minutes |
| Disk space | ~1.5 GB | ~800 MB |
| Jupyter integration | Excellent | Manual |
| HPC compatibility | Good | Excellent |
| Works with `my_run_local.sh` | ✓ | ✓ |
| Works with `my_run_production.sh` | ✗ | ✓ |
| Reproducibility | Good | Excellent |

---

## Running the Pipeline

### Local Pipeline (`my_run_local.sh`)

**Features**:
- Auto-detects conda or venv
- Runs models sequentially (no job submission)
- Local-friendly defaults (1 split, 100 bootstraps)
- Quick smoke testing

**Examples**:

```bash
# Quick smoke test (1 split, LR_EN only, ~5 min)
./my_run_local.sh

# Test multiple models
RUN_MODELS="LR_EN,RF,XGBoost" ./my_run_local.sh

# More splits for better estimates
N_SPLITS=5 ./my_run_local.sh

# Dry run to preview commands
DRY_RUN=1 ./my_run_local.sh

# Postprocess existing results
POSTPROCESS_ONLY=1 ./my_run_local.sh
```

**Environment variables**:
- `N_SPLITS`: Number of CV splits (default: 1)
- `RUN_MODELS`: Comma-separated list (default: LR_EN)
- `N_BOOT`: Bootstrap iterations (default: 100)
- `DRY_RUN`: Preview without execution (0 or 1)
- `OVERWRITE_SPLITS`: Regenerate splits (0 or 1)

### HPC Pipeline (`my_run_production.sh`)

**Features**:
- Submits jobs to LSF scheduler via `bsub`
- Parallel execution (4 models, 10 splits)
- Full bootstrap (500 iterations)
- Production-grade logging

**Setup**:

1. Edit `my_run_production.sh` to set your HPC project:
   ```bash
   PROJECT="your_project_allocation"  # Line 34
   ```

2. Run setup (if not done already):
   ```bash
   bash scripts/hpc_setup.sh
   ```

3. Submit pipeline:
   ```bash
   ./my_run_production.sh
   ```

**Examples**:

```bash
# Standard production run
./my_run_production.sh

# Single split test
N_SPLITS=1 ./my_run_production.sh

# Subset of models
RUN_MODELS="LR_EN,RF" ./my_run_production.sh

# Dry run
DRY_RUN=1 ./my_run_production.sh

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
# Generate 1 split, train 1 model, postprocess
./my_run_local.sh

# Check outputs
ls -la results_local/
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

**Expected**: `bsub` is HPC-specific. Use `my_run_local.sh` instead:
```bash
./my_run_local.sh
```

### Package import errors

**Solution**: Reinstall in development mode:
```bash
pip install -e . --force-reinstall --no-deps
```

### Tests fail after setup

**Common causes**:
- Missing data file: Check `../data/Celiac_dataset_proteomics_w_demo.csv`
- Old cached files: `rm -rf .pytest_cache __pycache__`
- Version mismatch: `pip install -e . --upgrade`

---

## File Structure

After setup, you should have:

```
analysis/
├── venv/                  # Virtual environment (if using hpc_setup.sh)
├── configs/               # YAML configuration files
├── logs_local/            # Local run logs
├── logs_a/                # HPC production logs
├── results_local/         # Local results
├── results_a/             # HPC production results
├── splits_local/          # Local CV splits
├── splits_1-10/           # Production CV splits
├── my_run_local.sh        # Local pipeline runner (NEW)
├── my_run_production.sh   # HPC pipeline runner
├── scripts/hpc_setup.sh   # Automated setup script
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
   ./my_run_local.sh
   ```

3. **Iterate on configs**:
   ```bash
   # Edit configs
   nano configs/my_training_config.yaml

   # Rerun
   OVERWRITE_SPLITS=1 ./my_run_local.sh
   ```

4. **Full local test** (before HPC deployment):
   ```bash
   N_SPLITS=5 RUN_MODELS="LR_EN,RF,XGBoost,LinSVM_cal" ./my_run_local.sh
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
   DRY_RUN=1 ./my_run_production.sh
   ```

3. **Single split test**:
   ```bash
   N_SPLITS=1 RUN_MODELS="LR_EN" ./my_run_production.sh
   bjobs -w
   ```

4. **Full production**:
   ```bash
   ./my_run_production.sh
   ```

---

## Documentation

- **Quick start**: [HPC_README.md](HPC_README.md)
- **Project overview**: [CLAUDE.MD](CLAUDE.MD)
- **Architecture**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Workflow guide**: [WORKFLOW.md](WORKFLOW.md)
- **This guide**: SETUP_README.md

---

## Support

If you encounter issues:

1. Check this guide's troubleshooting section
2. Verify Python version: `python --version` (requires 3.8+)
3. Check logs in `logs_local/` or `logs_a/`
4. Run tests: `pytest tests/ -v`
5. Contact: Andres Chousal

---

**Last Updated**: 2026-01-20
**Tested On**:
- macOS (local)
- HPC cluster with LSF scheduler
