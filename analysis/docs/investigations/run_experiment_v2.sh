#!/usr/bin/env bash

##########################################################################################
# Full Factorial Experiment v2: Prevalent × Case:Control Ratios (with Statistical Rigor)
##########################################################################################
#
# Enhanced factorial design with:
# - Multiple random seeds for variance estimation
# - Fixed 100-protein panel (eliminates FS variability)
# - Frozen hyperparameter/calibration config
# - Statistical testing (paired t-tests, Bonferroni, Cohen's d)
#
# Usage:
#   bash run_experiment_v2.sh [options]
#
# Options:
#   --prevalent-fracs FRAC1,FRAC2,...  Prevalent sampling fractions (default: 0.5,1.0)
#   --case-control-ratios RATIO1,...   Case:control ratios (default: 1,5)
#   --models MODEL1,MODEL2,...          Models to train (default: LR_EN,RF)
#   --split-seeds SEED1,SEED2,...       Random seeds (default: 0,1,2,3,4)
#   --skip-training                     Skip retraining
#   --skip-splits                       Skip split generation
#   --skip-panel                        Skip panel generation (use existing)
#   --dry-run                           Preview without executing
#   --help                              Show this message
#
# Examples:
#   # Full experiment (40 runs: 4 configs × 5 seeds × 2 models)
#   bash run_experiment_v2.sh --skip-splits
#
#   # Quick test (8 runs: 4 configs × 1 seed × 2 models)
#   bash run_experiment_v2.sh --skip-splits --split-seeds 0
#
##########################################################################################

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/../../../results"
INVEST_RESULTS_DIR="$SCRIPT_DIR/../../../results/investigations"
SPLITS_BASE_DIR="$ANALYSIS_DIR/../splits_experiments"
LOG_DIR="$ANALYSIS_DIR/../logs/experiments"
PANEL_FILE="$SCRIPT_DIR/top100_panel.csv"
FROZEN_CONFIG="$SCRIPT_DIR/training_config_frozen.yaml"

# Defaults
PREVALENT_FRACS=(0.5 1.0)
CASE_CONTROL_RATIOS=(1 5)
MODELS=("LR_EN" "RF")
SPLIT_SEEDS=(0 1 2 3 4)
SKIP_TRAINING=false
SKIP_SPLITS=false
SKIP_PANEL=false
DRY_RUN=false

# Experiment tracking
EXPERIMENT_ID=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_OUTPUT_DIR="$INVEST_RESULTS_DIR/experiment_${EXPERIMENT_ID}"

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
    head -50 "$0" | tail -n +4
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
        --split-seeds)
            IFS=',' read -ra SPLIT_SEEDS <<< "$2"
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
        --skip-panel)
            SKIP_PANEL=true
            shift
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
mkdir -p "$INVEST_RESULTS_DIR" "$LOG_DIR" "$SPLITS_BASE_DIR" "$EXPERIMENT_OUTPUT_DIR"

# Summary tracking
TOTAL_CONFIGS=0
TOTAL_RUNS=0
COMPLETED_RUNS=0
FAILED_RUNS=0

print_header "FACTORIAL EXPERIMENT v2 (Statistical Rigor)"

print_status "Experiment Design:"
echo "  Experiment ID:        $EXPERIMENT_ID"
echo "  Prevalent fractions:  ${PREVALENT_FRACS[@]}"
echo "  Case:control ratios:  ${CASE_CONTROL_RATIOS[@]}"
echo "  Models:               ${MODELS[@]}"
echo "  Random seeds:         ${SPLIT_SEEDS[@]}"
echo "  Fixed panel:          $PANEL_FILE"
echo "  Frozen config:        $FROZEN_CONFIG"
echo ""

# Calculate total configs and runs
for pf in "${PREVALENT_FRACS[@]}"; do
    for ccr in "${CASE_CONTROL_RATIOS[@]}"; do
        TOTAL_CONFIGS=$((TOTAL_CONFIGS + 1))
        for seed in "${SPLIT_SEEDS[@]}"; do
            for model in "${MODELS[@]}"; do
                TOTAL_RUNS=$((TOTAL_RUNS + 1))
            done
        done
    done
done

echo "  Total configurations: $TOTAL_CONFIGS"
echo "  Total runs:           $TOTAL_RUNS (configs × seeds × models)"
echo ""

if [ "$DRY_RUN" = true ]; then
    print_warning "DRY RUN MODE - no actual execution"
    echo ""
fi

##########################################################################################
# PHASE 0: Generate Fixed Panel
##########################################################################################

if [ "$SKIP_PANEL" = false ]; then
    print_header "PHASE 0: Generating Fixed 100-Protein Panel"

    if [ -f "$PANEL_FILE" ]; then
        print_warning "Panel file exists: $PANEL_FILE"
        read -p "Overwrite? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_status "Using existing panel"
            SKIP_PANEL=true
        fi
    fi

    if [ "$SKIP_PANEL" = false ]; then
        print_status "Generating panel (Mann-Whitney screening + k-best selection)..."

        if [ "$DRY_RUN" = false ]; then
            if (cd "$ANALYSIS_DIR" && python docs/investigations/generate_fixed_panel.py \
                --infile ../data/Celiac_dataset_proteomics_w_demo.parquet \
                --outfile docs/investigations/top100_panel.csv \
                --screen-method mannwhitney \
                --screen-top-n 1000 \
                --final-k 100) >> "$LOG_DIR/panel_generation.log" 2>&1; then
                print_success "Panel generated: $(wc -l < "$PANEL_FILE") proteins (including header)"
            else
                print_error "Panel generation FAILED"
                exit 1
            fi
        else
            print_status "[DRY RUN] Would generate panel"
        fi
    fi

    echo ""
fi

# Verify frozen config exists
if [ ! -f "$FROZEN_CONFIG" ]; then
    print_error "Frozen config not found: $FROZEN_CONFIG"
    echo "Create training_config_frozen.yaml first!"
    exit 1
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
            PROGRESS="[$CONFIG_ID/$TOTAL_CONFIGS]"

            print_status "$PROGRESS Generating splits: prevalent_frac=$pf, case_control=$ccr"

            CONFIG_SPLITS_DIR="$SPLITS_BASE_DIR/${pf}_${ccr}"
            mkdir -p "$CONFIG_SPLITS_DIR"

            TEMP_CONFIG="$SCRIPT_DIR/splits_config_experiment_${pf}_${ccr}.yaml"

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
                    --config "docs/investigations/splits_config_experiment_${pf}_${ccr}.yaml" \
                    --infile ../data/Celiac_dataset_proteomics_w_demo.parquet \
                    --outdir "$CONFIG_SPLITS_DIR" \
                    --overwrite) >> "$LOG_DIR/splits_generation.log" 2>&1; then
                    print_success "$PROGRESS Splits generated"
                else
                    print_error "$PROGRESS Split generation FAILED"
                    FAILED_RUNS=$((FAILED_RUNS + 1))
                fi
            else
                print_status "$PROGRESS [DRY RUN] Would generate splits"
            fi
        done
    done

    echo ""
fi

##########################################################################################
# PHASE 2: Train Models for Each Configuration × Seed
##########################################################################################

if [ "$SKIP_TRAINING" = false ]; then
    print_header "PHASE 2: Training Models (Frozen Config + Fixed Panel)"

    RUN_ID=0
    CONFIG_ID=0

    for pf in "${PREVALENT_FRACS[@]}"; do
        for ccr in "${CASE_CONTROL_RATIOS[@]}"; do
            CONFIG_ID=$((CONFIG_ID + 1))
            CONFIG_PROGRESS="[Config $CONFIG_ID/$TOTAL_CONFIGS]"

            print_status "$CONFIG_PROGRESS Configuration: prevalent_frac=$pf, case_control=$ccr"

            CONFIG_SPLITS_DIR="$SPLITS_BASE_DIR/${pf}_${ccr}"

            for SEED in "${SPLIT_SEEDS[@]}"; do
                SEED_PROGRESS="  [Seed $SEED]"

                for MODEL in "${MODELS[@]}"; do
                    RUN_ID=$((RUN_ID + 1))
                    RUN_PROGRESS="    [Run $RUN_ID/$TOTAL_RUNS]"

                    print_status "$CONFIG_PROGRESS$SEED_PROGRESS$RUN_PROGRESS Training $MODEL..."

                    if [ "$DRY_RUN" = false ]; then
                        if (cd "$ANALYSIS_DIR" && ced train \
                            --model "$MODEL" \
                            --split-seed "$SEED" \
                            --split-dir "$CONFIG_SPLITS_DIR" \
                            --scenario IncidentPlusPrevalent \
                            --infile ../data/Celiac_dataset_proteomics_w_demo.parquet \
                            --fixed-panel docs/investigations/top100_panel.csv \
                            --config docs/investigations/training_config_frozen.yaml) \
                            >> "$LOG_DIR/training_${pf}_${ccr}_seed${SEED}.log" 2>&1; then
                            print_success "$CONFIG_PROGRESS$SEED_PROGRESS$RUN_PROGRESS $MODEL complete"
                            COMPLETED_RUNS=$((COMPLETED_RUNS + 1))
                        else
                            print_error "$CONFIG_PROGRESS$SEED_PROGRESS$RUN_PROGRESS $MODEL FAILED"
                            FAILED_RUNS=$((FAILED_RUNS + 1))
                        fi
                    else
                        print_status "$CONFIG_PROGRESS$SEED_PROGRESS$RUN_PROGRESS [DRY RUN] Would train $MODEL"
                    fi
                done
            done
        done
    done

    echo ""
fi

##########################################################################################
# PHASE 3: Statistical Analysis
##########################################################################################

print_header "PHASE 3: Statistical Analysis"

if [ "$DRY_RUN" = false ]; then
    print_status "Running factorial analysis with statistical testing..."

    if (cd "$ANALYSIS_DIR" && python docs/investigations/investigate_factorial.py \
        --results-dir "$RESULTS_DIR" \
        --output-dir "$EXPERIMENT_OUTPUT_DIR" \
        --metric AUROC) >> "$LOG_DIR/analysis_${EXPERIMENT_ID}.log" 2>&1; then
        print_success "Analysis complete"

        echo ""
        print_status "Output files:"
        echo "  - metrics_all.csv (raw metrics, $TOTAL_RUNS rows)"
        echo "  - comparison_table.csv (config summary, 8 rows)"
        echo "  - statistical_tests.csv (paired comparisons, 8 tests)"
        echo "  - power_analysis.csv (post-hoc power)"
        echo "  - summary.md (human-readable findings)"
        echo ""
        echo "Location: $EXPERIMENT_OUTPUT_DIR"
    else
        print_error "Analysis FAILED"
        FAILED_RUNS=$((FAILED_RUNS + 1))
    fi
else
    print_status "[DRY RUN] Would run statistical analysis"
fi

echo ""

##########################################################################################
# Summary
##########################################################################################

print_header "EXPERIMENT COMPLETE"

echo "Summary:"
echo "  Experiment ID:         $EXPERIMENT_ID"
echo "  Total runs attempted:  $TOTAL_RUNS"
echo "  Successfully completed: $COMPLETED_RUNS"
echo "  Failed:                $FAILED_RUNS"
echo ""
echo "Experimental controls:"
echo "  Fixed panel:           $PANEL_FILE"
echo "  Frozen config:         $FROZEN_CONFIG"
echo "  Random seeds:          ${#SPLIT_SEEDS[@]} independent splits"
echo ""
echo "Results location:        $EXPERIMENT_OUTPUT_DIR"
echo "Training logs location:  $LOG_DIR"
echo ""

if [ "$DRY_RUN" = false ] && [ -f "$EXPERIMENT_OUTPUT_DIR/summary.md" ]; then
    print_status "Quick preview of findings:"
    head -n 50 "$EXPERIMENT_OUTPUT_DIR/summary.md"
fi

[ $FAILED_RUNS -eq 0 ] && exit 0 || exit 1
