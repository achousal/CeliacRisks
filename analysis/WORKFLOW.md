# CeliacRisks Pipeline Workflow

**Visual guide to running the complete pipeline on HPC**

---

## Overview: Local → HPC → Results

```
┌─────────────────┐
│  Local Machine  │
└────────┬────────┘
         │ 1. git clone
         ↓
┌─────────────────┐
│   HPC Cluster   │
└────────┬────────┘
         │ 2. Setup
         │ 3. Submit Jobs
         ↓
┌─────────────────┐
│     Results     │
└────────┬────────┘
         │ 4. Download
         ↓
┌─────────────────┐
│  Local Analysis │
└─────────────────┘
```

---

## Phase 1: Initial Setup (One-Time)

```bash
# On HPC login node
ssh your_username@hpc.cluster.edu

# Clone repository
git clone git@github.com:achousal/CeliacRisks.git
cd CeliacRisks/analysis

# Automated setup
bash scripts/hpc_setup.sh
# ↓ Creates virtual environment
# ↓ Installs ced-ml package
# ↓ Runs test suite
# ↓ Records package versions

# Verify
source venv/bin/activate
ced --help
```

**Time required:** 5-10 minutes

---

## Phase 2: Data Preparation

```bash
# Copy dataset to project directory
cp /shared/storage/Celiac_dataset_proteomics.csv ../

# Verify file size
du -h ../Celiac_dataset_proteomics.csv
# Expected: ~2.5 GB

# Customize batch script
cp CeD_production.lsf.template CeD_production.lsf
vim CeD_production.lsf
# Edit: PROJECT, QUEUE, BASE_DIR
```

**Time required:** 5 minutes

---

## Phase 3: Test Run (Recommended)

```bash
# Generate single split
source venv/bin/activate
ced save-splits \
  --config configs/splits_config.yaml \
  --infile ../Celiac_dataset_proteomics.csv \
  --n-splits 1

# Check output
ls splits_production/
# ↓ IncidentPlusPrevalent_train_idx_seed0.csv
# ↓ IncidentPlusPrevalent_val_idx_seed0.csv
# ↓ IncidentPlusPrevalent_test_idx_seed0.csv

# Test single model
MODEL=LR_EN bsub -J "CeD_test" -W 2:00 < CeD_production.lsf

# Monitor
bjobs -w
tail -f logs/CeD_test_*.out
```

**Time required:** 30-60 minutes

---

## Phase 4: Production Run

### Step 4A: Generate Splits

```bash
source venv/bin/activate

ced save-splits \
  --config configs/splits_config.yaml \
  --infile ../Celiac_dataset_proteomics.csv \
  --n-splits 10

# Verify
ls splits_production/ | wc -l
# Expected: 30 files (3 files × 10 splits)
```

**Time required:** 2-3 minutes

### Step 4B: Submit Training Jobs

```bash
# Submit all 4 models as array job
bsub < CeD_production.lsf

# Or submit individually
MODEL=LR_EN bsub -J "CeD_LR" < CeD_production.lsf
MODEL=RF bsub -J "CeD_RF" < CeD_production.lsf
MODEL=XGBoost bsub -J "CeD_XGB" < CeD_production.lsf
MODEL=LinSVM_cal bsub -J "CeD_SVM" < CeD_production.lsf
```

**Time required:** 4-8 hours (model-dependent)

### Step 4C: Monitor Progress

```bash
# Check job status
bjobs -w | grep CeD

# View logs (real-time)
tail -f logs/CeD_train_*.out

# Count completed models
find results_production -name "final_model.joblib" | wc -l
# Target: 4 (one per model)

# Check for errors
grep -i error logs/*.err
```

### Step 4D: Wait for Completion

```
Job Timeline (Typical):
─────────────────────────────────────────
0h              LR_EN done ✓
│               LinSVM_cal done ✓
│
3h              XGBoost done ✓
│
6h              RF done ✓
│
─────────────────────────────────────────
All jobs complete
```

---

## Phase 5: Post-Processing

```bash
# Verify all models completed
ls results_production/IncidentPlusPrevalent__*/core/final_model.joblib | wc -l
# Expected: 4

# Aggregate results
source venv/bin/activate
ced postprocess \
  --results-dir results_production \
  --n-boot 500

# View top models
head -n 5 results_production/COMBINED/aggregated_metrics.csv | column -t -s,
```

**Output structure:**
```
results_production/
├── IncidentPlusPrevalent__RF__5x10__val0.25__test0.25__hybrid/
│   ├── core/
│   │   ├── final_model.joblib      ← Trained model
│   │   ├── test_metrics.csv        ← Primary results
│   │   └── val_metrics.csv
│   ├── preds/
│   │   ├── test_preds/             ← Predictions
│   │   └── val_preds/
│   └── diagnostics/
│       ├── calibration/            ← Calibration curves
│       ├── dca/                    ← Decision curve analysis
│       └── risk_dist/              ← Risk distribution plots
├── (3 more model directories)
└── COMBINED/
    ├── aggregated_metrics.csv      ← Cross-model comparison
    └── dca_curves.csv
```

**Time required:** 10-15 minutes

---

## Phase 6: Download Results

```bash
# From LOCAL machine
rsync -avz --progress \
  your_username@hpc.cluster.edu:/path/to/CeliacRisks/analysis/results_production/ \
  ~/Downloads/CeliacRisks_results/

# Verify transfer
ls ~/Downloads/CeliacRisks_results/COMBINED/
```

**Time required:** 5-10 minutes (network-dependent)

---

## Phase 7: Visualization (Local)

```bash
# In local analysis/ directory
cd ~/Projects/CeliacRisks/analysis

# Generate publication figures (R)
Rscript compare_models_faith.R \
  --results_root ~/Downloads/CeliacRisks_results

# Output
# ↓ plots/model_comparison_*.pdf
# ↓ plots/calibration_curves.pdf
# ↓ plots/dca_curves.pdf
```

**Time required:** 2-3 minutes

---

## Phase 8: Archival and Publication

```bash
# Create archive (on HPC)
tar -czf CeliacRisks_v1.0.0_$(date +%Y%m%d).tar.gz \
  splits_production/ \
  results_production/ \
  configs/ \
  logs/

# Move to archival storage
mv CeliacRisks_v1.0.0_*.tar.gz /labs/elahi/archives/

# Record metadata
pip freeze > results_production/requirements.txt
git log -1 > results_production/git_version.txt
cp configs/*.yaml results_production/
```

---

## Complete Timeline

```
Total Time: ~8-12 hours (mostly waiting for jobs)
Active Time: ~30 minutes (actual work)

Day 1
─────────────────────────────────────────
09:00  Setup (10 min)
09:10  Test run submission (5 min)
10:00  Test run complete ✓

10:30  Production splits (3 min)
10:33  Job submission (2 min)
10:35  ☕ Wait for jobs...

18:00  All jobs complete ✓
18:05  Post-process (15 min)
18:20  Download results (10 min)

Day 2
─────────────────────────────────────────
09:00  Generate figures (3 min)
09:03  Archive results (5 min)
09:08  Done ✓
```

---

## Troubleshooting Quick Reference

| Issue | Quick Fix |
|-------|-----------|
| `ced: command not found` | `source venv/bin/activate` |
| Job killed (OOM) | Increase mem: `-R "rusage[mem=16000]"` |
| Job timeout | Increase time: `-W 24:00` |
| Data not found | Copy to `../Celiac_dataset_proteomics.csv` |
| Import errors | Reinstall: `pip install -e .` |

**Full troubleshooting:** [docs/HPC_SETUP.md](docs/HPC_SETUP.md)

---

## Key Commands Reference

```bash
# Setup
bash scripts/hpc_setup.sh

# Activate
source venv/bin/activate

# Submit
bsub < CeD_production.lsf

# Monitor
bjobs -w

# Check logs
tail -f logs/CeD_train_*.out

# Postprocess
ced postprocess --results-dir results_production
```

---

## Pipeline Dependency Graph

```
Celiac_dataset_proteomics.csv
         ↓
    [save-splits]
         ↓
    ┌────┴────┬─────────┬──────────┐
    ↓         ↓         ↓          ↓
  train/    val/     test/    (× 10 splits)
    ↓         ↓         ↓
    └────┬────┴─────────┘
         ↓
    [ced train] ← configs/training_config.yaml
         ↓
    ┌────┴────┬─────────┬──────────┐
    ↓         ↓         ↓          ↓
   RF    XGBoost  LinSVM_cal   LR_EN
    ↓         ↓         ↓          ↓
    └────┬────┴─────────┴──────────┘
         ↓
  [ced postprocess]
         ↓
    COMBINED/aggregated_metrics.csv
         ↓
  [R visualization]
         ↓
   Publication figures
```

---

## Success Criteria

- [ ] 4 model artifacts created
- [ ] 4 test_metrics.csv files generated
- [ ] COMBINED/aggregated_metrics.csv exists
- [ ] AUROC values > 0.90 for all models
- [ ] Brier scores < 0.01 for all models
- [ ] No errors in logs

---

**Ready to begin? Start here:** [HPC_README.md](HPC_README.md)