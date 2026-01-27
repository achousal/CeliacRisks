# HPC Panel Optimization Guide

## Problem
After running the main training pipeline on HPC, you want to run panel optimization via RFE. The `ced optimize-panel` CLI command auto-detects paths relative to local `../results`, but your HPC results are on the HPC filesystem.

## Solution: Use HPC Script

The `hpc_optimize_panel.sh` script reuses [configs/pipeline_hpc.yaml](../configs/pipeline_hpc.yaml) for paths (just like `post_training_pipeline.sh`).

### Quick Start
```bash
# SSH to HPC
ssh <username>@minerva.hpc.mssm.edu

# Navigate to analysis directory
cd /sc/arion/projects/Chipuk_Laboratory/chousa01/CeliacRisks/analysis

# Run script (auto-detects first available model)
bash scripts/hpc_optimize_panel.sh --run-id 20260127_094454
```

### Custom Configuration
Override defaults via arguments:

```bash
# Optimize specific model/split
bash scripts/hpc_optimize_panel.sh \
  --run-id 20260127_094454 \
  --model XGBoost \
  --split-seed 5 \
  --start-size 150 \
  --min-size 10 \
  --min-auroc-frac 0.92
```

### Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--run-id` | (required) | Run timestamp from HPC training |
| `--model` | (auto-detect) | Model to optimize (LR_EN, RF, XGBoost, etc.) |
| `--split-seed` | `0` | Split seed to use |
| `--start-size` | `100` | Starting panel size |
| `--min-size` | `5` | Minimum panel size to test |
| `--min-auroc-frac` | `0.90` | Early stop threshold (fraction of max AUROC) |
| `--config` | `configs/pipeline_hpc.yaml` | Config file for paths |

### Expected Output
Results saved to:
```
/sc/arion/projects/Chipuk_Laboratory/chousa01/CeliacRisks/results/{MODEL}/run_{RUN_ID}/split_seed{SPLIT_SEED}/optimize_panel/
```

Files generated:
- `rfe_curve.csv` - AUROC vs panel size
- `feature_ranking.csv` - Features ranked by elimination order
- `pareto_curve.png` - Cost-performance trade-off plot
- `feature_ranking.png` - Feature importance waterfall
- `rfe_params.json` - Run parameters and metadata

### Monitor Progress
The script runs interactively and displays progress in real-time:
```bash
# If running in background, check logs
tail -f ../logs/features/optimize_panel_*.log
```

### After Completion
Review Pareto curve to select deployment panel size:

```bash
# View results on HPC
cd /sc/arion/projects/Chipuk_Laboratory/chousa01/CeliacRisks/results/LR_EN/run_20260127_094454/split_seed0/optimize_panel/

# Show top 20 features
head -20 feature_ranking.csv

# Show AUROC curve (key decision points)
awk -F',' 'NR==1 || $1 <= 50' rfe_curve.csv
```

### Extract Deployment Panel
Once you've selected a panel size (e.g., 25 features):

```bash
# Extract top N features
head -26 feature_ranking.csv > deployment_panel_25.csv  # +1 for header
```

### Validate Panel (NEW Split Seed)
Test the selected panel with unbiased cross-validation:

```bash
# On HPC: validate with NEW split seed (prevents peeking)
ced train \
    --model LR_EN \
    --fixed-panel deployment_panel_25.csv \
    --split-seed 10 \
    --config configs/training_config.yaml
```

**Critical**: Always use a **new split seed** (e.g., 10) when validating a panel discovered on split_seed 0-9. This prevents optimistic bias from data leakage.

## Troubleshooting

### Model not found
Check that your HPC training completed:
```bash
ls /sc/arion/projects/Chipuk_Laboratory/chousa01/CeliacRisks/results/LR_EN/run_20260127_094454/split_seed0/core/
```

Expected file: `LR_EN__final_model.joblib`

### No splits directory
Verify splits were generated during training:
```bash
ls /sc/arion/projects/Chipuk_Laboratory/chousa01/CeliacRisks/splits/
```

### Script fails immediately
Common issues:
- Virtual environment not activated (run `source venv/bin/activate` first)
- Missing input data file (check `configs/pipeline_hpc.yaml` paths)
- Incorrect model path (verify model trained for specified run_id)

## Alternative: Local Optimization

If you want to run locally (requires downloading HPC results):

```bash
# 1. Download from HPC
scp -r <username>@minerva.hpc.mssm.edu:/sc/arion/projects/Chipuk_Laboratory/chousa01/CeliacRisks/results/LR_EN/run_20260127_094454 \
    ../results/LR_EN/

# 2. Run locally (auto-detects paths)
ced optimize-panel --run-id 20260127_094454 --model LR_EN --split-seed 0
```

Note: Local optimization requires sufficient memory (~8GB+) and may be slower for large models.

## Next Steps

After panel optimization:

1. **Review Pareto curve** - Identify sweet spot (max AUROC, min cost)
2. **Select panel size** - Based on clinical/deployment constraints
3. **Extract features** - Create CSV with selected panel
4. **Validate panel** - Train on NEW split seed with `--fixed-panel`
5. **Update documentation** - Record panel selection rationale in ADR

See [FEATURE_SELECTION.md](reference/FEATURE_SELECTION.md) for detailed workflows.
