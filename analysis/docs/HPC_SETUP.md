# HPC Setup Guide - CeliacRisks

**Project:** CeliacRisks v1.0.0
**Author:** Andres Chousal
**Date:** 2026-01-19

---

## Quick Start (TL;DR)

```bash
# 1. Clone repository on HPC
git clone git@github.com:achousal/CeliacRisks.git
cd CeliacRisks/analysis

# 2. Run automated setup
bash scripts/hpc_setup.sh

# 3. Activate environment and test
source venv/bin/activate
ced --help

# 4. Submit production job
bsub < CeD_production.lsf
```

---

## Prerequisites

### HPC Environment
- **Scheduler:** LSF (IBM Spectrum LSF)
- **Python:** 3.8+ (via module or system)
- **Git:** For repository cloning
- **Storage:** ~50 GB for data + results

### Required Data File
Place the dataset at:
```
CeliacRisks/Celiac_dataset_proteomics.csv
```
(~2.5 GB, 43,960 samples × 2,920 proteins)

---

## Step-by-Step Setup

### 1. Clone Repository

```bash
# SSH to HPC login node
ssh your_username@hpc.stanford.edu

# Navigate to your project directory
cd /labs/elahi/$USER/projects

# Clone the repository
git clone git@github.com:achousal/CeliacRisks.git
cd CeliacRisks/analysis
```

**Note:** If SSH keys are not configured, use HTTPS:
```bash
git clone https://github.com/achousal/CeliacRisks.git
```

### 2. Load Python Module

```bash
# Check available Python modules
module avail python

# Load Python 3.8+ (example - adjust for your HPC)
module load python/3.9.0

# Verify Python version
python3 --version  # Should be >= 3.8
```

### 3. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 4. Install CeliacRisks Package

```bash
# Install in editable mode with all dependencies
pip install -e .

# Verify installation
ced --help
```

**Expected output:**
```
Usage: ced [OPTIONS] COMMAND [ARGS]...

  CeliacRisks ML Pipeline CLI

Options:
  --version  Show version
  --help     Show this message and exit

Commands:
  config         Configuration management tools
  eval-holdout   Evaluate model on holdout set
  postprocess    Aggregate results across splits
  save-splits    Generate train/val/test splits
  train          Train ML models with nested CV
```

### 5. Verify Installation

```bash
# Run test suite (optional but recommended)
pytest tests/ -v --tb=short

# Expected: 846+ passing tests
```

### 6. Prepare Data

```bash
# Verify data file exists
ls -lh ../Celiac_dataset_proteomics.csv

# If missing, copy from shared storage
cp /labs/elahi/shared/data/Celiac_dataset_proteomics.csv ../

# Verify file size (~2.5 GB)
du -h ../Celiac_dataset_proteomics.csv
```

### 7. Create Output Directories

```bash
# Create directories for outputs
mkdir -p logs splits_production results_production

# Set permissions (if needed)
chmod 755 logs splits_production results_production
```

---

## Configuration

### Environment Variables

Create a `.env` file for custom paths (optional):

```bash
# analysis/.env (DO NOT COMMIT)
CED_DATA_DIR=/labs/elahi/$USER/data
CED_RESULTS_DIR=/labs/elahi/$USER/results
CED_SPLITS_DIR=/labs/elahi/$USER/splits
CED_LOG_LEVEL=INFO
```

**Important:** Add `.env` to `.gitignore` (already done in this project).

### LSF Configuration

The provided batch scripts use these defaults:

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Queue** | normal | Adjust per HPC policy |
| **Cores** | 16 | Good for RF/XGBoost |
| **Memory** | 8 GB/core | 128 GB total |
| **Time** | 12 hours | Conservative estimate |
| **Array** | 1-4 | 4 models in parallel |

Edit [CeD_production.lsf](../CeD_production.lsf) to customize.

---

## Running the Pipeline

### Method 1: Automated (Recommended)

```bash
# Run full production pipeline
bash run_production.sh
```

**What it does:**
1. Generates 10 train/val/test splits
2. Trains 4 models (RF, XGBoost, LinSVM_cal, LR_EN) × 10 splits
3. Aggregates results
4. Generates visualizations

**Monitor progress:**
```bash
# Check job status
bjobs

# Tail logs
tail -f logs/CeD_*.out
```

### Method 2: Manual (Step-by-Step)

**Step 1: Generate splits**
```bash
ced save-splits \
  --config configs/splits_config.yaml \
  --infile ../Celiac_dataset_proteomics.csv \
  --outdir splits_production \
  --n-splits 10
```

**Step 2: Submit training jobs**
```bash
# Submit LSF array job (4 models × 10 splits = 40 jobs)
bsub < CeD_production.lsf

# Monitor
bjobs -w
bpeek <job_id>  # View running job output
```

**Step 3: Post-process results**
```bash
# After all jobs complete
ced postprocess \
  --results-dir results_production \
  --n-boot 500
```

**Step 4: Generate visualizations**
```bash
# Requires R and ggplot2
module load r/4.2.0
Rscript compare_models_faith.R --results_root results_production
```

---

## Monitoring and Debugging

### Check Job Status

```bash
# List all jobs
bjobs

# List jobs for specific job array
bjobs -J "CeD_train*"

# Detailed job info
bjobs -l <job_id>

# Job history
bhist -l <job_id>
```

### View Logs

```bash
# Real-time monitoring
tail -f logs/CeD_1.out

# Check for errors
grep -i error logs/*.err
grep -i exception logs/*.err

# View completed job log
cat logs/CeD_1.out
```

### Common Issues

#### Issue: Module not found errors
**Symptom:** `ModuleNotFoundError: No module named 'ced_ml'`

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall package
pip install -e .
```

#### Issue: Data file not found
**Symptom:** `FileNotFoundError: ../Celiac_dataset_proteomics.csv`

**Solution:**
```bash
# Verify data path
ls -l ../Celiac_dataset_proteomics.csv

# Or specify absolute path in config
# configs/training_config.yaml:
# data:
#   infile: /labs/elahi/USER/data/Celiac_dataset_proteomics.csv
```

#### Issue: Out of memory
**Symptom:** Job killed with exit code 140 or memory errors in logs

**Solution:**
```bash
# Increase memory in LSF script
# Edit CeD_production.lsf:
#BSUB -R "rusage[mem=16GB]"  # Increase from 8GB

# Or reduce training sample size
# Edit configs/training_config.yaml:
# data:
#   train_control_per_case: 3  # Reduce from 5
```

#### Issue: Timeout
**Symptom:** Job killed after 12 hours

**Solution:**
```bash
# Increase wall time
# Edit CeD_production.lsf:
#BSUB -W 24:00  # Increase from 12:00
```

---

## Resource Optimization

### Choosing Resources

| Model | Cores | Memory | Time |
|-------|-------|--------|------|
| **RF** | 16 | 8 GB | 4-6 hours |
| **XGBoost** | 16 | 8 GB | 3-5 hours |
| **LinSVM_cal** | 8 | 4 GB | 1-2 hours |
| **LR_EN** | 8 | 4 GB | 1-2 hours |

**Recommendations:**
- Use 16 cores for ensemble models (RF, XGBoost)
- Use 8 cores for linear models (LinSVM_cal, LR_EN)
- Request 2× memory if using full control set (not recommended)

### Parallel Execution

```bash
# Run all 4 models in parallel (default)
bsub < CeD_production.lsf

# Run single model
bsub -J "CeD_RF" \
  -o logs/CeD_RF.out \
  -e logs/CeD_RF.err \
  -n 16 -W 12:00 -R "rusage[mem=8GB]" \
  "source venv/bin/activate && \
   ced train --config configs/training_config.yaml --model RF \
   --infile ../Celiac_dataset_proteomics.csv \
   --splits-dir splits_production"
```

---

## Output Structure

After successful execution:

```
CeliacRisks/
├── analysis/
│   ├── splits_production/                    # Train/val/test splits
│   │   ├── IncidentPlusPrevalent_train_idx_seed0.csv
│   │   ├── IncidentPlusPrevalent_val_idx_seed0.csv
│   │   └── IncidentPlusPrevalent_test_idx_seed0.csv
│   │   └── (27 more files for 9 more seeds)
│   ├── results_production/                   # Model outputs
│   │   ├── IncidentPlusPrevalent__RF__5x10__val0.25__test0.25__hybrid/
│   │   ├── IncidentPlusPrevalent__XGBoost__5x10__val0.25__test0.25__hybrid/
│   │   ├── IncidentPlusPrevalent__LinSVM_cal__5x10__val0.25__test0.25__hybrid/
│   │   ├── IncidentPlusPrevalent__LR_EN__5x10__val0.25__test0.25__hybrid/
│   │   └── COMBINED/                         # Aggregated results
│   │       ├── aggregated_metrics.csv
│   │       ├── dca_curves.csv
│   │       └── model_comparison_plots/
│   └── logs/                                 # Job logs
│       ├── CeD_1.out  # RF
│       ├── CeD_2.out  # XGBoost
│       ├── CeD_3.out  # LinSVM_cal
│       └── CeD_4.out  # LR_EN
└── Celiac_dataset_proteomics.csv            # Input data
```

---

## Validation

### Check Pipeline Success

```bash
# All jobs completed?
bjobs | grep CeD_train  # Should be empty

# All models finished?
ls results_production/IncidentPlusPrevalent__*/core/final_model.joblib | wc -l
# Expected: 4 (one per model)

# Metrics generated?
ls results_production/IncidentPlusPrevalent__*/core/test_metrics.csv | wc -l
# Expected: 4

# Aggregation completed?
ls results_production/COMBINED/aggregated_metrics.csv
# Should exist
```

### Quick Results Check

```bash
# View top model by Brier score
head -n 2 results_production/COMBINED/aggregated_metrics.csv | column -t -s,

# Expected output (example):
# Model      Brier_Score_mean  AUROC_mean  PR_AUC_mean
# LR_EN      0.0032           0.957       0.289
```

---

## Backup and Archiving

### Save Results

```bash
# Create timestamped archive
tar -czf CeliacRisks_v1.0.0_$(date +%Y%m%d).tar.gz \
  splits_production/ \
  results_production/ \
  configs/

# Move to archival storage
mv CeliacRisks_v1.0.0_*.tar.gz /labs/elahi/archives/
```

### Cleanup

```bash
# Remove intermediate files (optional)
rm -rf results_production/*/preds/val_preds/*.csv
rm -rf results_production/*/diagnostics/learning_curve/*.png

# Keep only final models and metrics
```

---

## Next Steps

After successful HPC run:

1. **Download results for local analysis**
   ```bash
   # From your local machine
   rsync -avz --progress \
     your_username@hpc.stanford.edu:/labs/elahi/$USER/projects/CeliacRisks/analysis/results_production/ \
     ~/Downloads/CeliacRisks_results/
   ```

2. **Generate publication figures**
   ```bash
   # Run locally with downloaded results
   Rscript compare_models_faith.R --results_root ~/Downloads/CeliacRisks_results
   ```

3. **Archive to Zenodo/Figshare**
   - Upload model artifacts
   - Upload aggregated metrics
   - Generate DOI for citation

4. **Prepare manuscript**
   - Use metrics from `results_production/COMBINED/aggregated_metrics.csv`
   - Include calibration curves and DCA plots
   - Report holdout validation (if performed)

---

## Support

**Documentation:**
- [CLAUDE.md](../CLAUDE.md) - Project overview
- [HPC_MIGRATION_GUIDE.md](HPC_MIGRATION_GUIDE.md) - Migration from legacy scripts
- [README.md](../README.md) - Package quickstart

**Troubleshooting:**
- Check logs in `logs/*.err` for error messages
- Review test suite: `pytest tests/ -v`
- Contact: Andres Chousal (achousal@stanford.edu)

**HPC Support:**
- Stanford Research Computing: srcc-support@stanford.edu
- LSF documentation: https://www.ibm.com/docs/en/spectrum-lsf

---

## Reproducibility Checklist

- [ ] Git commit hash recorded in results
- [ ] Random seeds documented (0-9 for 10 splits)
- [ ] Package versions logged (`pip freeze > requirements.txt`)
- [ ] Configuration files archived with results
- [ ] Data provenance documented (dataset version, filtering)
- [ ] Results backed up to archival storage

**Version tracking:**
```bash
# Record environment
pip freeze > results_production/requirements_$(date +%Y%m%d).txt

# Record git state
git log -1 --oneline > results_production/git_version.txt
git diff > results_production/git_diff.txt  # Any uncommitted changes
```

---

**Setup completed successfully? Proceed to run the pipeline!**
