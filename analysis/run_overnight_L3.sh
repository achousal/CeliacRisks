#!/bin/bash
# Level 3 Overnight Production Run - Quick Launch Script
# Maximum resources: 10 splits, 100 trials, 1000 bootstrap, all metrics
# Expected runtime: 14-16 hours on premium queue

set -euo pipefail

echo "========================================"
echo "Level 3 Production Run - Overnight HPC"
echo "========================================"
echo ""

# Configuration paths
SPLITS_CONFIG="configs/splits_config.yaml"
TRAINING_CONFIG="configs/training_config_production.yaml"
PIPELINE_CONFIG="configs/pipeline_hpc.yaml"
DATA_FILE="../data/Celiac_dataset_proteomics_w_demo.parquet"

# Pre-flight checks
echo "[1/7] Pre-flight checks..."

if [[ ! -f "$DATA_FILE" ]]; then
    echo "ERROR: Data file not found: $DATA_FILE"
    exit 1
fi

if [[ ! -f "$SPLITS_CONFIG" ]]; then
    echo "ERROR: Splits config not found: $SPLITS_CONFIG"
    exit 1
fi

if [[ ! -f "$TRAINING_CONFIG" ]]; then
    echo "ERROR: Training config not found: $TRAINING_CONFIG"
    exit 1
fi

if [[ ! -f "$PIPELINE_CONFIG" ]]; then
    echo "ERROR: Pipeline config not found: $PIPELINE_CONFIG"
    exit 1
fi

echo "   ✓ All config files found"

# Check n_splits setting
CURRENT_SPLITS=$(grep "^n_splits:" "$SPLITS_CONFIG" | awk '{print $2}')
echo "[2/7] Current n_splits: $CURRENT_SPLITS"

if [[ "$CURRENT_SPLITS" != "10" ]]; then
    echo "   WARNING: n_splits is $CURRENT_SPLITS (expected 10 for L3)"
    read -p "   Update to 10 splits? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Create backup
        cp "$SPLITS_CONFIG" "${SPLITS_CONFIG}.bak_$(date +%Y%m%d_%H%M%S)"
        # Update n_splits
        sed -i.tmp 's/^n_splits:.*/n_splits: 10/' "$SPLITS_CONFIG"
        rm -f "${SPLITS_CONFIG}.tmp"
        echo "   ✓ Updated n_splits to 10 (backup created)"
    else
        echo "   Proceeding with current n_splits=$CURRENT_SPLITS"
    fi
fi

# Verify HPC allocation
echo "[3/7] Verifying HPC settings..."
HPC_PROJECT=$(grep "project:" "$PIPELINE_CONFIG" | awk '{print $2}')
HPC_QUEUE=$(grep "queue:" "$PIPELINE_CONFIG" | awk '{print $2}')
HPC_WALLTIME=$(grep "walltime:" "$PIPELINE_CONFIG" | awk -F'"' '{print $2}')
HPC_CORES=$(grep "cores:" "$PIPELINE_CONFIG" | tail -1 | awk '{print $2}')

echo "   Project: $HPC_PROJECT"
echo "   Queue: $HPC_QUEUE"
echo "   Walltime: $HPC_WALLTIME"
echo "   Cores per job: $HPC_CORES"

if [[ "$HPC_QUEUE" != "premium" ]]; then
    echo "   WARNING: Queue is '$HPC_QUEUE' (recommended: premium for overnight)"
fi

# Check Optuna settings
echo "[4/7] Verifying Optuna settings..."
OPTUNA_TRIALS=$(grep "n_trials:" "$TRAINING_CONFIG" | head -1 | awk '{print $2}')
CV_REPEATS=$(grep "repeats:" "$TRAINING_CONFIG" | head -1 | awk '{print $2}')
N_BOOT=$(grep "n_boot:" "$TRAINING_CONFIG" | tail -1 | awk '{print $2}')

echo "   Optuna trials: $OPTUNA_TRIALS"
echo "   CV repeats: $CV_REPEATS"
echo "   Bootstrap iterations: $N_BOOT"

if [[ "$OPTUNA_TRIALS" != "100" ]] || [[ "$CV_REPEATS" != "10" ]] || [[ "$N_BOOT" != "1000" ]]; then
    echo "   WARNING: Config doesn't match L3 specs (100 trials, 10 repeats, 1000 boot)"
fi

# Estimate compute requirements
FINAL_SPLITS=$(grep "^n_splits:" "$SPLITS_CONFIG" | awk '{print $2}')
N_MODELS=4  # LR_EN, LinSVM_cal, RF, XGBoost

echo "[5/7] Compute estimates..."
echo "   Total jobs: $FINAL_SPLITS splits × $N_MODELS models = $((FINAL_SPLITS * N_MODELS)) jobs"
echo "   Estimated runtime: 14-16 hours (bottleneck: Random Forest)"
echo "   Core-hours: ~4,500 (10 splits × 4 models × 8 cores × 14 hrs)"

# Check disk space
DATA_SIZE=$(du -h "$DATA_FILE" | awk '{print $1}')
AVAILABLE_SPACE=$(df -h ../results | tail -1 | awk '{print $4}')
echo "   Data file size: $DATA_SIZE"
echo "   Available space in results/: $AVAILABLE_SPACE"

# Dry run option
echo ""
echo "[6/7] Ready to launch..."
echo "   Run mode: ${DRY_RUN:-0}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo ""
    echo "DRY RUN MODE - Jobs will NOT be submitted"
    echo "Set DRY_RUN=0 or unset to submit for real"
    echo ""
fi

read -p "Proceed with launch? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted by user"
    exit 0
fi

# Launch pipeline
echo ""
echo "[7/7] Launching pipeline..."
echo "========================================"

OVERWRITE_SPLITS=1 ./run_hpc.sh

# Capture run ID from output
echo ""
echo "========================================"
echo "Jobs submitted successfully!"
echo ""
echo "Next steps:"
echo "1. Monitor jobs:"
echo "   bjobs -w | grep CeD_"
echo ""
echo "2. Check logs:"
echo "   tail -f logs/{RUN_ID}/CeD_*.out"
echo ""
echo "3. After all jobs complete (DONE status):"
echo "   bash scripts/post_training_pipeline.sh --run-id <RUN_ID>"
echo ""
echo "4. Optional: Train ensemble:"
echo "   bash scripts/post_training_pipeline.sh --run-id <RUN_ID> --train-ensemble"
echo ""
echo "Expected completion: ~14-16 hours from now"
echo "========================================"
