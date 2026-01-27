#!/usr/bin/env bash

##########################################################################################
# Full Factorial Experiment: Prevalent × Case:Control Ratios
##########################################################################################
#
# Automates the complete investigation across multiple split configurations to separate
# methodological artifacts (class imbalance, prevalent sampling) from biological differences.
#
# Usage:
#   bash run_experiment.sh [options]
#
# Options:
#   --prevalent-fracs FRAC1,FRAC2,...  Prevalent sampling fractions (default: 0.5,1.0)
#   --case-control-ratios RATIO1,...   Case:control ratios (default: 1,5)
#   --models MODEL1,MODEL2,...          Models to train/investigate (default: LR_EN,RF)
#   --mode {local|hpc}                  Execution mode (default: local)
#   --skip-training                     Skip retraining, use existing models
#   --skip-splits                       Skip split generation
#   --analyses LIST                     Analyses to run (default: distributions)
#   --cores N                           Cores per job (HPC only)
#   --memory MEM                        Memory per job (HPC only)
#   --queue QUEUE                       LSF queue (HPC only, default: medium)
#   --dry-run                           Preview without executing
#   --help                              Show this message
#
# Examples:
#   # 2x2 experiment (default)
#   bash run_experiment.sh
#
#   # 2x3 experiment with extended ratios
#   bash run_experiment.sh --case-control-ratios 1,5,10
#
#   # HPC execution with all models
#   bash run_fullexperiment.sh --mode hpc --models LR_EN,RF,XGBoost,LinSVM_cal
#
##########################################################################################

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/../../../results/investigations"
SPLITS_BASE_DIR="$ANALYSIS_DIR/../splits_experiments"
LOG_DIR="$ANALYSIS_DIR/../logs/experiments"

# Defaults
PREVALENT_FRACS=(0.5 1.0)
CASE_CONTROL_RATIOS=(1 5)
MODELS=("LR_EN" "RF")
MODE="local"
SKIP_TRAINING=false
SKIP_SPLITS=false
ANALYSES="distributions,calibration,features"
DRY_RUN=false
CORES=4
MEMORY="16G"
QUEUE="medium"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Helper functions
print_header() {
    echo ""
    echo "################################################################################################"
    echo "  $1"
    echo "################################################################################################"
    echo ""
}

print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

show_help() {
    head -60 "$0" | tail -n +4
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prevalent-fracs)
            IFS=',' read -ra PREVALENT_FRACS <<< "$2"
            shift 2
            ;;
        --case-control-ratios)
            IFS=',' read -ra CASE_CONTROL_RATIOS <<< "$2"
            shift 2
            ;;
        --models)
            IFS=',' read -ra MODELS <<< "$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --skip-splits)
            SKIP_SPLITS=true
            shift
            ;;
        --analyses)
            ANALYSES="$2"
            shift 2
            ;;
        --cores)
            CORES="$2"
            shift 2
            ;;
        --memory)
            MEMORY="$2"
            shift 2
            ;;
        --queue)
            QUEUE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Create directories
mkdir -p "$RESULTS_DIR" "$LOG_DIR" "$SPLITS_BASE_DIR"

# Summary tracking
TOTAL_CONFIGS=0
TRAINED_CONFIGS=0
FAILED_CONFIGS=0

print_header "FACTORIAL EXPERIMENT: Prevalent vs Case:Control Ratio"

print_status "Experiment Design:"
echo "  Prevalent fractions:  ${PREVALENT_FRACS[@]}"
echo "  Case:control ratios:  ${CASE_CONTROL_RATIOS[@]}"
echo "  Models:               ${MODELS[@]}"
echo "  Analyses:             $ANALYSES"
echo "  Mode:                 $MODE"
echo ""

# Calculate total configs
for pf in "${PREVALENT_FRACS[@]}"; do
    for ccr in "${CASE_CONTROL_RATIOS[@]}"; do
        TOTAL_CONFIGS=$((TOTAL_CONFIGS + 1))
    done
done

echo "  Total configurations: $TOTAL_CONFIGS"
echo ""

if [ "$DRY_RUN" = true ]; then
    print_warning "DRY RUN MODE - no actual training/investigation"
    echo ""
fi

##########################################################################################
# PHASE 1: Generate Splits for Each Configuration
##########################################################################################

if [ "$SKIP_SPLITS" = false ]; then
    print_header "PHASE 1: Generating Splits"

    CONFIG_ID=0
    for pf in "${PREVALENT_FRACS[@]}"; do
        for ccr in "${CASE_CONTROL_RATIOS[@]}"; do
            CONFIG_ID=$((CONFIG_ID + 1))
            CONFIG_NAME="config_${pf}_${ccr}"
            PROGRESS="[$CONFIG_ID/$TOTAL_CONFIGS]"

            print_status "$PROGRESS Generating splits: prevalent_frac=$pf, case_control=$ccr"

            # Create config-specific split directory
            CONFIG_SPLITS_DIR="$SPLITS_BASE_DIR/${pf}_${ccr}"
            mkdir -p "$CONFIG_SPLITS_DIR"

            # Create temporary config file
            TEMP_CONFIG="$ANALYSIS_DIR/configs/splits_config_experiment_${pf}_${ccr}.yaml"

            cat > "$TEMP_CONFIG" << EOF
mode: development
scenarios:
  - IncidentPlusPrevalent

val_size: 0.25
test_size: 0.25
holdout_size: 0.30

n_splits: 10
seed_start: 0

prevalent_train_only: false
prevalent_train_frac: $pf

train_control_per_case: $ccr
eval_control_per_case: $ccr

save_indices_only: false
EOF

            if [ "$DRY_RUN" = false ]; then
                if (cd "$ANALYSIS_DIR" && ced save-splits \
                    --config "configs/splits_config_experiment_${pf}_${ccr}.yaml" \
                    --infile ../data/Celiac_dataset_proteomics_w_demo.parquet \
                    --outdir "$CONFIG_SPLITS_DIR" \
                    --overwrite) >> "$LOG_DIR/splits_generation.log" 2>&1; then
                    print_success "$PROGRESS Splits generated"
                else
                    print_error "$PROGRESS Split generation FAILED"
                    FAILED_CONFIGS=$((FAILED_CONFIGS + 1))
                fi
            else
                print_status "$PROGRESS [DRY RUN] Would generate splits"
            fi
        done
    done

    echo ""
fi

##########################################################################################
# PHASE 2: Train Models for Each Configuration
##########################################################################################

if [ "$SKIP_TRAINING" = false ]; then
    print_header "PHASE 2: Training Models on Each Configuration"

    CONFIG_ID=0
    for pf in "${PREVALENT_FRACS[@]}"; do
        for ccr in "${CASE_CONTROL_RATIOS[@]}"; do
            CONFIG_ID=$((CONFIG_ID + 1))
            CONFIG_NAME="config_${pf}_${ccr}"
            PROGRESS="[$CONFIG_ID/$TOTAL_CONFIGS]"

            print_status "$PROGRESS Configuration: prevalent_frac=$pf, case_control=$ccr"

            # Get config-specific split directory
            CONFIG_SPLITS_DIR="$SPLITS_BASE_DIR/${pf}_${ccr}"

            MODEL_ID=0
            for MODEL in "${MODELS[@]}"; do
                MODEL_ID=$((MODEL_ID + 1))
                MODEL_PROGRESS="  [$MODEL_ID/${#MODELS[@]}]"

                print_status "$PROGRESS$MODEL_PROGRESS Training $MODEL (split_seed 0)..."

                if [ "$DRY_RUN" = false ]; then
                    if (cd "$ANALYSIS_DIR" && ced train \
                        --model "$MODEL" \
                        --split-seed 0 \
                        --split-dir "$CONFIG_SPLITS_DIR" \
                        --scenario IncidentPlusPrevalent \
                        --config configs/training_config.yaml) \
                        >> "$LOG_DIR/training_${pf}_${ccr}.log" 2>&1; then
                        print_success "$PROGRESS$MODEL_PROGRESS $MODEL trained"
                        TRAINED_CONFIGS=$((TRAINED_CONFIGS + 1))
                    else
                        print_error "$PROGRESS$MODEL_PROGRESS Training $MODEL FAILED"
                        FAILED_CONFIGS=$((FAILED_CONFIGS + 1))
                    fi
                else
                    print_status "$PROGRESS$MODEL_PROGRESS [DRY RUN] Would train $MODEL"
                fi
            done
        done
    done

    echo ""
fi

##########################################################################################
# PHASE 3: Run Investigations for Each Configuration
##########################################################################################

print_header "PHASE 3: Running Investigations"

CONFIG_ID=0
for pf in "${PREVALENT_FRACS[@]}"; do
    for ccr in "${CASE_CONTROL_RATIOS[@]}"; do
        CONFIG_ID=$((CONFIG_ID + 1))
        CONFIG_NAME="config_${pf}_${ccr}"
        PROGRESS="[$CONFIG_ID/$TOTAL_CONFIGS]"

        print_status "$PROGRESS Configuration: prevalent_frac=$pf, case_control=$ccr"

        # Create subdirectory for this configuration's results
        CONFIG_RESULTS_DIR="$RESULTS_DIR/${pf}_${ccr}"
        mkdir -p "$CONFIG_RESULTS_DIR"

        if [ "$MODE" == "local" ]; then
            # Local mode: run investigations sequentially

            MODEL_ID=0
            for MODEL in "${MODELS[@]}"; do
                MODEL_ID=$((MODEL_ID + 1))
                MODEL_PROGRESS="  [$MODEL_ID/${#MODELS[@]}]"

                print_status "$PROGRESS$MODEL_PROGRESS Investigating $MODEL..."

                if [ "$DRY_RUN" = false ]; then
                    if (cd "$ANALYSIS_DIR" && python docs/investigations/investigate.py \
                        --mode oof \
                        --model "$MODEL" \
                        --split-seed 0 \
                        --analyses "$ANALYSES") \
                        >> "$LOG_DIR/investigation_${pf}_${ccr}.log" 2>&1; then

                        # Move results to config-specific directory
                        mv "$RESULTS_DIR"/distributions_${MODEL}_oof_seed0.png "$CONFIG_RESULTS_DIR/" 2>/dev/null || true
                        mv "$RESULTS_DIR"/calibration_${MODEL}_oof_seed0.png "$CONFIG_RESULTS_DIR/" 2>/dev/null || true
                        mv "$RESULTS_DIR"/feature_bias_${MODEL}_oof_seed0.png "$CONFIG_RESULTS_DIR/" 2>/dev/null || true
                        mv "$RESULTS_DIR"/scores_${MODEL}_oof_seed0.csv "$CONFIG_RESULTS_DIR/" 2>/dev/null || true

                        print_success "$PROGRESS$MODEL_PROGRESS Investigation complete"
                    else
                        print_error "$PROGRESS$MODEL_PROGRESS Investigation FAILED"
                        FAILED_CONFIGS=$((FAILED_CONFIGS + 1))
                    fi
                else
                    print_status "$PROGRESS$MODEL_PROGRESS [DRY RUN] Would investigate $MODEL"
                fi
            done
        fi

        echo ""
    done
done

##########################################################################################
# PHASE 4: Generate Comparison Report
##########################################################################################

if [ "$DRY_RUN" = false ]; then
    print_header "PHASE 4: Generating Comparison Report"

    # Create Python script to compare results
    COMPARE_SCRIPT="$RESULTS_DIR/compare_configurations.py"

    cat > "$COMPARE_SCRIPT" << 'PYTHON_END'
#!/usr/bin/env python3
import pandas as pd
import json
from pathlib import Path
import glob

results_root = Path(__file__).parent

print("=" * 100)
print("FACTORIAL EXPERIMENT: RESULTS COMPARISON")
print("=" * 100)

# Collect all configuration results
configs_data = {}

for config_dir in sorted(results_root.glob("*_*")):
    if not config_dir.is_dir():
        continue

    config_name = config_dir.name
    parts = config_name.split("_")
    if len(parts) < 2:
        continue

    pf, ccr = parts[0], parts[1]

    # Look for summary CSVs in this config directory
    summary_files = sorted(glob.glob(str(config_dir / "summary_*.csv")))

    if summary_files:
        config_data = []
        for f in summary_files:
            try:
                df = pd.read_csv(f)
                config_data.append(df)
            except:
                pass

        if config_data:
            combined = pd.concat(config_data, ignore_index=True)
            configs_data[config_name] = combined

# Print comparison
print(f"\nTotal configurations found: {len(configs_data)}\n")

if configs_data:
    # Aggregate across models for each config
    summary = []

    for config_name, data in sorted(configs_data.items()):
        pf, ccr = config_name.split("_")[0], config_name.split("_")[1]

        if 'median_diff' in data.columns and len(data) > 0:
            row = {
                'Config': config_name,
                'Prevalent_Frac': pf,
                'Case_Control_Ratio': f"1:{ccr}",
                'Models': len(data['model'].unique()),
                'Runs': len(data),
                'Median_Diff_Mean': data['median_diff'].mean(),
                'Median_Diff_Std': data['median_diff'].std(),
                'Cohens_d_Mean': data['cohens_d'].mean(),
                'Cohens_d_Std': data['cohens_d'].std(),
                'Significant_Runs': sum(data['mw_pval'] < 0.05),
                'Total_Incident_Cases': data['n_incident'].sum(),
            }
            summary.append(row)

    if summary:
        summary_df = pd.DataFrame(summary)

        print("CONFIGURATION SUMMARY:")
        print("-" * 100)
        print(summary_df.to_string(index=False))
        print("-" * 100)

        # Save summary
        summary_path = results_root / "EXPERIMENT_COMPARISON.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nComparison saved to: {summary_path.name}\n")

        # Interpret results
        print("\nKEY FINDINGS:")
        print("-" * 100)

        for idx, row in summary_df.iterrows():
            print(f"\nConfig: {row['Config']}")
            print(f"  Case:Control = {row['Case_Control_Ratio']}, Prevalent = {row['Prevalent_Frac']}")
            print(f"  Median score difference: {row['Median_Diff_Mean']:+.4f} ± {row['Median_Diff_Std']:.4f}")
            print(f"  Effect size (Cohen's d): {row['Cohens_d_Mean']:.3f} ± {row['Cohens_d_Std']:.3f}")
            print(f"  Significant (p<0.05): {row['Significant_Runs']}/{row['Total_Incident_Cases']} runs")
else:
    print("WARNING: No results found. Check that investigations completed.")

print("\n" + "=" * 100)
PYTHON_END

    chmod +x "$COMPARE_SCRIPT"

    print_status "Running comparison script..."
    if python "$COMPARE_SCRIPT"; then
        print_success "Comparison report generated"
    else
        print_warning "Comparison had issues (check results manually)"
    fi
fi

##########################################################################################
# Summary
##########################################################################################

print_header "EXPERIMENT COMPLETE"

echo "Summary:"
echo "  Total configurations:  $TOTAL_CONFIGS"
echo "  Successfully trained:  $TRAINED_CONFIGS"
echo "  Failed:                $FAILED_CONFIGS"
echo ""
echo "Results location: $RESULTS_DIR"
echo "Logs location:    $LOG_DIR"
echo ""
echo "Key output files:"
echo "  - EXPERIMENT_COMPARISON.csv (configuration summary)"
echo "  - {prevalent_frac}_{case_control}/ (per-config results)"
echo ""

[ $FAILED_CONFIGS -eq 0 ] && exit 0 || exit 1
