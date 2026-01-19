# HPC Setup Checklist

**CeliacRisks v1.0.0**

Use this checklist for first-time HPC deployment. Print or copy to track progress.

---

## Pre-Setup (Do These First)

- [ ] SSH access to HPC confirmed (can login)
- [ ] Git installed on HPC (run `which git`)
- [ ] Python 3.8+ available (run `module avail python`)
- [ ] Dataset file `Celiac_dataset_proteomics.csv` (~2.5 GB) accessible
- [ ] Project storage allocation confirmed (~50 GB needed)
- [ ] LSF scheduler access confirmed (run `bjobs`)

---

## Initial Setup (One-Time)

### 1. Clone Repository
- [ ] Navigate to project directory on HPC
- [ ] Clone: `git clone git@github.com:achousal/CeliacRisks.git`
  - If SSH fails, use HTTPS: `git clone https://github.com/achousal/CeliacRisks.git`
- [ ] Change to analysis directory: `cd CeliacRisks/analysis`

### 2. Module Configuration
- [ ] Identify correct Python module name
  - Run: `module avail python`
  - Note the module name (e.g., `python/3.9.0`)
- [ ] Edit `scripts/load_modules.sh`
  - Update line: `module load python/YOUR_VERSION`
- [ ] Test module loading: `source scripts/load_modules.sh`
- [ ] Verify Python version: `python3 --version` (should be 3.8+)

### 3. Automated Setup
- [ ] Run setup script: `bash scripts/hpc_setup.sh`
- [ ] Wait for installation (~2-3 minutes)
- [ ] Note any errors or warnings
- [ ] Run test suite when prompted (recommended)
- [ ] Record environment: `pip freeze > requirements_snapshot.txt`

### 4. Verify Installation
- [ ] Activate environment: `source venv/bin/activate`
- [ ] Check CLI: `ced --help` (should show commands)
- [ ] Check version: `ced --version`
- [ ] Test import: `python -c "import ced_ml; print('OK')"`

---

## Configuration (Before First Run)

### 1. Data Preparation
- [ ] Copy dataset to parent directory
  - `cp /path/to/source/Celiac_dataset_proteomics.csv ../`
- [ ] Verify file size: `du -h ../Celiac_dataset_proteomics.csv` (~2.5 GB)
- [ ] Check file permissions: `ls -lh ../Celiac_dataset_proteomics.csv`

### 2. Batch Script Customization
- [ ] Copy template: `cp CeD_production.lsf.template CeD_production.lsf`
- [ ] Edit `CeD_production.lsf`:
  - [ ] Update `#BSUB -P YOUR_PROJECT_ALLOCATION` (line 22)
  - [ ] Update `#BSUB -q normal` to your queue name (line 23)
  - [ ] Update `BASE_DIR="/path/to/your/CeliacRisks/analysis"` (line 50)
- [ ] Save changes

### 3. Create Output Directories
- [ ] `mkdir -p logs splits_production results_production`
- [ ] Verify permissions: `ls -ld logs splits_production results_production`

### 4. Configuration Review
- [ ] Check splits config: `cat configs/splits_config.yaml`
- [ ] Check training config: `cat configs/training_config.yaml`
- [ ] Edit configs if customization needed (optional)

---

## First Test Run (Recommended)

### Single Split Test
- [ ] Activate environment: `source venv/bin/activate`
- [ ] Generate one split:
  ```bash
  ced save-splits \
    --config configs/splits_config.yaml \
    --infile ../Celiac_dataset_proteomics.csv \
    --n-splits 1
  ```
- [ ] Check output: `ls splits_production/`
- [ ] Expected files:
  - `IncidentPlusPrevalent_train_idx_seed0.csv`
  - `IncidentPlusPrevalent_val_idx_seed0.csv`
  - `IncidentPlusPrevalent_test_idx_seed0.csv`

### Single Model Test (Optional)
- [ ] Submit test job:
  ```bash
  MODEL=LR_EN bsub -J "CeD_test" -W 2:00 < CeD_production.lsf
  ```
- [ ] Monitor: `bjobs -w`
- [ ] Check logs: `tail -f logs/CeD_test_*.out`
- [ ] Wait for completion (~30-60 min)
- [ ] Verify output: `ls results_production/IncidentPlusPrevalent__LR_EN__*/core/`

---

## Production Run

### Submit Jobs
- [ ] Activate environment: `source venv/bin/activate`
- [ ] Review submission command
- [ ] Submit all models: `bsub < CeD_production.lsf`
- [ ] Note job IDs from output
- [ ] Verify submission: `bjobs -w | grep CeD`

### Monitor Progress
- [ ] Check job status regularly: `bjobs -w`
- [ ] Check logs: `ls -lth logs/` (most recent first)
- [ ] Tail active job: `tail -f logs/CeD_train_*.out`
- [ ] Watch for errors: `grep -i error logs/*.err`
- [ ] Track completion: `find results_production -name "test_metrics.csv" | wc -l`

### Expected Runtime
- [ ] LR_EN: 1-2 hours
- [ ] LinSVM_cal: 1-2 hours
- [ ] RF: 4-6 hours
- [ ] XGBoost: 3-5 hours

---

## Post-Processing

### Verify Completion
- [ ] All jobs done: `bjobs | grep CeD` (should be empty)
- [ ] Count outputs: `ls results_production/IncidentPlusPrevalent__*/core/final_model.joblib | wc -l`
- [ ] Expected: 4 model files (one per model)

### Aggregate Results
- [ ] Activate environment: `source venv/bin/activate`
- [ ] Run postprocessing:
  ```bash
  ced postprocess --results-dir results_production --n-boot 500
  ```
- [ ] Check combined results: `ls results_production/COMBINED/`
- [ ] View metrics:
  ```bash
  head -n 5 results_production/COMBINED/aggregated_metrics.csv | column -t -s,
  ```

---

## Validation

### Check Outputs
- [ ] Model artifacts exist:
  - `results_production/IncidentPlusPrevalent__RF__*/core/final_model.joblib`
  - `results_production/IncidentPlusPrevalent__XGBoost__*/core/final_model.joblib`
  - `results_production/IncidentPlusPrevalent__LinSVM_cal__*/core/final_model.joblib`
  - `results_production/IncidentPlusPrevalent__LR_EN__*/core/final_model.joblib`

- [ ] Metrics files exist:
  - `results_production/IncidentPlusPrevalent__*/core/test_metrics.csv` (4 files)
  - `results_production/COMBINED/aggregated_metrics.csv`

- [ ] Prediction files exist:
  - `results_production/IncidentPlusPrevalent__*/preds/test_preds/*.csv`

- [ ] Diagnostic plots exist:
  - `results_production/IncidentPlusPrevalent__*/diagnostics/calibration/*.png`

### Sanity Checks
- [ ] AUROC values reasonable (0.85-0.99)
- [ ] Brier scores reasonable (0.001-0.01)
- [ ] PR-AUC values reasonable (0.15-0.40)
- [ ] No suspicious warnings in logs

---

## Backup and Archive

### Save Results
- [ ] Create archive:
  ```bash
  tar -czf CeliacRisks_v1.0.0_$(date +%Y%m%d).tar.gz \
    splits_production/ results_production/ configs/ logs/
  ```
- [ ] Move to archival storage:
  ```bash
  mv CeliacRisks_v1.0.0_*.tar.gz /labs/elahi/archives/
  ```

### Record Metadata
- [ ] Save package versions: `pip freeze > results_production/requirements.txt`
- [ ] Save git state: `git log -1 > results_production/git_version.txt`
- [ ] Save run parameters: `cp configs/*.yaml results_production/`

---

## Download Results (Local)

### Transfer Files
- [ ] From local machine:
  ```bash
  rsync -avz --progress \
    your_username@hpc.stanford.edu:/path/to/CeliacRisks/analysis/results_production/ \
    ~/Downloads/CeliacRisks_results/
  ```

### Verify Transfer
- [ ] Check file count matches HPC
- [ ] Check file sizes reasonable
- [ ] Open metrics CSV in Excel/R

---

## Next Steps

- [ ] Generate publication figures (R script)
- [ ] Prepare manuscript tables
- [ ] Upload to Zenodo/Figshare (optional)
- [ ] Document any issues or deviations
- [ ] Update CLAUDE.md with any lessons learned

---

## Troubleshooting Reference

| Symptom | Check | Solution |
|---------|-------|----------|
| `ced: command not found` | Virtual env | `source venv/bin/activate` |
| `Data file not found` | Data path | Copy data to `../` |
| Job killed (exit 140) | Memory | Increase mem in batch script |
| Job timeout | Wall time | Increase `-W` in batch script |
| Module not found | Python path | Check `module load` command |
| Import errors | Package install | Reinstall: `pip install -e .` |

For detailed troubleshooting, see: [docs/HPC_SETUP.md](docs/HPC_SETUP.md)

---

## Completion

- [ ] All checkboxes completed
- [ ] Results validated
- [ ] Results backed up
- [ ] Documentation updated

**Setup Date:** _______________
**Completed By:** _______________
**HPC System:** _______________
**Notes:**

---

_Last updated: 2026-01-19_
