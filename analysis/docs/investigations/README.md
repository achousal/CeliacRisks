# Prevalent vs Incident Score Difference Investigation

Diagnoses whether models systematically under-predict prevalent (confirmed) cases relative to incident (pre-diagnostic) cases, and whether this reflects **methodological artifacts** (class imbalance during training) or **biological differences**.

---

## Quick Start (2 min read)

### Run the Full Factorial Experiment

```bash
cd analysis/docs/investigations/
bash run_experiment.sh
```

This automatically:
1. **Generates splits** with different prevalent sampling (0.5, 1.0) and case:control ratios (1:1, 1:5)
2. **Retrains models** on each configuration
3. **Investigates** score differences for each configuration
4. **Outputs EXPERIMENT_COMPARISON.csv** with results

### Interpretation

Open `../../../results/investigations/EXPERIMENT_COMPARISON.csv`:

| Observation | Conclusion | Action |
|---|---|---|
| Score difference **shrinks 70%+** (1:5 → 1:1 ratio) | Artifact-driven (class imbalance) | Retrain with balanced sampling |
| Score difference **unchanged** (>80% persists) | Biology-driven | Accept, document, stratify reporting |
| Score difference **shrinks 30-70%** | Mixed causes | Retrain + case-aware calibration |

---

## Key Concepts

### What's Being Tested

**Phase 1 (Baseline):** Models trained on production config (downsampled controls)
- Score difference observed: ~20-30%

**Phase 2 (Artifact Test):** Models trained on balanced ratios (1:1 controls:cases)
- If score difference shrinks significantly → **imbalance is the problem**
- If score difference persists → **biological difference is the problem**

### Why This Matters

- **Artifact-driven**: Retraining with balanced sampling fixes the problem
- **Biology-driven**: Need case-specific thresholds or reporting

---

## Available Modes

### Option 1: Quick 2×2 Diagnostic (30 min)

```bash
bash run_experiment.sh
```

Tests:
- Prevalent sampling: 0.5, 1.0
- Case:control ratios: 1:1, 1:5

**Best for**: Initial assessment, laptop-friendly

### Option 2: Robust 2×3 Extended (60 min)

```bash
bash run_experiment.sh --case-control-ratios 1,5,10
```

Tests:
- Prevalent sampling: 0.5, 1.0
- Case:control ratios: 1:1, 1:5, 1:10

**Best for**: Dose-response validation (does artifact shrink proportionally?)

### Option 3: HPC Parallel (15 min wall-clock)

```bash
bash run_experiment.sh --mode hpc --case-control-ratios 1,5,10
```

**Best for**: Production runs, all models

### Option 4: Skip Retraining (Use Existing Models)

```bash
bash run_experiment.sh --skip-training
```

**Best for**: Re-running investigations on already-trained models

---

## Output Structure

```
results/investigations/
├── EXPERIMENT_COMPARISON.csv              # Main result: all configs compared
├── 0.5_1/                                 # Config: prevalent=0.5, case:control=1:1
│   ├── distributions_LR_EN_oof_seed0.png
│   ├── calibration_LR_EN_oof_seed0.png
│   ├── feature_bias_LR_EN_oof_seed0.png
│   └── scores_LR_EN_oof_seed0.csv
├── 0.5_5/                                 # Config: prevalent=0.5, case:control=1:5
├── 1.0_1/                                 # Config: prevalent=1.0, case:control=1:1
└── 1.0_5/                                 # Config: prevalent=1.0, case:control=1:5
```

---

## Core Metrics Reported

For each configuration and model:

| Metric | What It Shows |
|--------|---|
| **Median score difference** | Magnitude of score gap between incident and prevalent cases |
| **Mann-Whitney p-value** | Statistical significance of the difference |
| **Cohen's d** | Effect size (0.5 = medium, 0.8 = large) |
| **Calibration slope** | Are predictions overconfident (>1) or underconfident (<1)? |
| **Calibration intercept** | Systematic over/underprediction |
| **Feature bias %** | What fraction of selected features favor incident discrimination? |

---

## Recommended Workflow

### Day 1: Baseline (30 min)
```bash
bash run_experiment.sh
# → Review EXPERIMENT_COMPARISON.csv
# → Are Phase 1→2 changes substantial?
```

### Day 2: Decision (5 min)
- **If artifact-driven**: Plan retraining
- **If biology-driven**: Plan documentation and stratified reporting
- **If unclear**: Run extended 2×3 experiment

### Day 3 (Optional): Extended Analysis (60 min)
```bash
bash run_experiment.sh --case-control-ratios 1,5,10
```

---

## Advanced Options

### All Models (Default: LR_EN, RF)

```bash
bash run_experiment.sh --models LR_EN,RF,XGBoost,LinSVM_cal
```

### Specific Analyses Only

```bash
bash run_experiment.sh --analyses distributions,calibration
```

Options: `distributions`, `calibration`, `features`

### HPC with Custom Resources

```bash
bash run_experiment.sh --mode hpc --cores 8 --memory 32G --queue long
```

### Dry Run (Preview)

```bash
bash run_experiment.sh --dry-run
```

---

## HPC Monitoring

```bash
# Check job status
bjobs -w | grep experiment

# Watch logs
tail -f ../../../logs/investigation/investigate_*.log

# Count completed jobs
ls ../../../logs/investigation/investigate_*.log | wc -l
```

---

## Troubleshooting

### "ced: command not found"
```bash
cd analysis/
pip install -e .
```

### Models failing to train
```bash
# Check logs
tail ../logs/experiments/training_*.log

# Verify dataset exists
ls ../../data/*.parquet
```

### Investigations fail
```bash
# Ensure models trained for current config
ls ../../../results/{MODEL}/run_*/split_seed0/preds/
```

### No results after HPC completion
Check job logs:
```bash
cat ../../../logs/investigation/investigate_*.log | grep ERROR
```

---

## Technical Details

### Why This Experiment Design?

1. **Fixed test set**: Same held-out people across all training configurations → isolates training effects
2. **Vary training only**: Different case:control ratios during training → separates class imbalance from biology
3. **Multiple seeds**: Across 10 split seeds → rules out fold-specific luck
4. **Multiple metrics**: Discrimination + calibration + feature analysis → comprehensive diagnosis

### What It Can't Detect

- Hidden confounders (e.g., age bias in prevalent cases)
- Data quality differences between case types
- Measurement artifacts

**Recommendation**: Stratify Phase 2 results by age/sex to rule out confounding

---

## Architecture Overview

### Split Configuration Matrix

```
                    Case:Control Ratio
                      1:1        1:5
Prevalent=0.5     Config A   Config B   ← Current production default
Prevalent=1.0     Config C   Config D   ← Balanced
```

Each configuration:
- Generates new CV splits
- Retrains all specified models
- Runs full investigation (distributions, calibration, feature bias)
- Saves to separate output directory

### Automation Script Flow

```
run_experiment.sh
├── Parse arguments
├── PHASE 1: Generate split configs (1-2 min)
├── PHASE 2: Retrain models (30-60 min depending on config count)
├── PHASE 3: Run investigations (5-15 min)
└── PHASE 4: Generate EXPERIMENT_COMPARISON.csv (1 min)
```

---

## File Reference

| File | Purpose |
|------|---------|
| **README.md** | This file – entry point and execution guide |
| **METHODOLOGY.md** | Statistical design, interpretation guide, decision trees |
| **investigate.py** | Core investigation code (distributions, calibration, features) |
| **run_experiment.sh** | Main automation orchestrator |
| **consolidate_full.py** | Results aggregation |
| **splits_config_investigation.yaml** | Config template |

---

## Questions?

1. **How do I start?** → You're reading it
2. **How do I interpret results?** → See [METHODOLOGY.md](METHODOLOGY.md)
3. **Which execution option should I pick?** → See Available Modes above
4. **Need help troubleshooting?** → See Troubleshooting section

---

**Last Updated:** 2026-01-27
**Status:** Production-ready
**Tested:** All modes (local, HPC), all configurations
