#!/usr/bin/env bash

##########################################################################################
# Consolidated Factorial Experiment Runner
##########################################################################################
#
# Full-featured runner for factorial experiments testing prevalent fractions and
# case:control ratios with:
# - Multiple random seeds for robust statistics
# - Fixed 100-protein panel (eliminates FS variability)
# - Frozen hyperparameter/calibration config
# - Statistical analysis (paired t-tests, Bonferroni, Cohen's d)
# - Comprehensive logging and progress tracking
#
# Usage:
#   bash run_factorial_experiment.sh [options]
#
# Presets:
#   --quick                             Quick test (1 seed, 2 models)
#   --overnight                         Overnight run (5 seeds, 2 models, 6 configs)
#   --full                              Full experiment (10 seeds, 4 models, 6 configs)
#
# Options:
#   --prevalent-fracs FRAC1,FRAC2,...   Prevalent sampling fractions (default: 0.5,1.0)
#   --case-control-ratios RATIO1,...    Case:control ratios (default: 1,5,10)
#   --models MODEL1,MODEL2,...          Models to train (default: LR_EN,RF)
#   --split-seeds SEED1,SEED2,...       Random seeds (default: 0,1,2,3,4)
#   --skip-training                     Skip retraining (analyze existing)
#   --skip-splits                       Skip split generation (use existing)
#   --skip-panel                        Skip panel generation (use existing)
#   --force-panel                       Force panel regeneration
#   --dry-run                           Preview without executing
#   --help                              Show this message
#
# Examples:
#   # Quick test (12 runs: 6 configs × 1 seed × 2 models, ~30 min)
#   bash run_factorial_experiment.sh --quick
#
#   # Overnight run (60 runs: 6 configs × 5 seeds × 2 models, ~6-8 hours)
#   bash run_factorial_experiment.sh --overnight
#
#   # Full experiment (240 runs: 6 configs × 10 seeds × 4 models, ~24-30 hours)
#   bash run_factorial_experiment.sh --full
#
#   # Custom configuration
#   bash run_factorial_experiment.sh \
#     --prevalent-fracs 0.5,1.0 \
#     --case-control-ratios 1,5,10 \
#     --models LR_EN,RF,XGBoost \
#     --split-seeds 0,1,2,3,4,5,6,7,8,9 \
#     --skip-splits
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
CASE_CONTROL_RATIOS=(1 5 10)
MODELS=("LR_EN" "RF")
SPLIT_SEEDS=(0 1 2 3 4)
SKIP_TRAINING=false
SKIP_SPLITS=false
SKIP_PANEL=false
FORCE_PANEL=false
DRY_RUN=false
PRESET=""

# Experiment tracking
EXPERIMENT_ID=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_OUTPUT_DIR="$INVEST_RESULTS_DIR/experiment_${EXPERIMENT_ID}"
EXPERIMENT_LOG="$LOG_DIR/experiment_${EXPERIMENT_ID}.log"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
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

print_info() {
    echo -e "${CYAN}[i]${NC} $1"
}

show_help() {
    head -60 "$0" | tail -n +4
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            PRESET="quick"
            SPLIT_SEEDS=(0)
            MODELS=("LR_EN" "RF")
            shift
            ;;
        --overnight)
            PRESET="overnight"
            SPLIT_SEEDS=(0 1 2 3 4)
            MODELS=("LR_EN" "RF")
            shift
            ;;
        --full)
            PRESET="full"
            SPLIT_SEEDS=(0 1 2 3 4 5 6 7 8 9)
            MODELS=("LR_EN" "RF" "XGBoost" "LinSVM_cal")
            shift
            ;;
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
        --force-panel)
            FORCE_PANEL=true
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
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Create log directory
mkdir -p "$LOG_DIR"

# Redirect all output to log file AND terminal
exec > >(tee -a "$EXPERIMENT_LOG") 2>&1

##########################################################################################
# Print Header
##########################################################################################

print_header "FACTORIAL EXPERIMENT RUNNER"

print_status "Experiment ID: $EXPERIMENT_ID"
if [ -n "$PRESET" ]; then
    print_info "Running preset: $PRESET"
fi
print_status "Start time: $(date)"
print_status "Log file: $EXPERIMENT_LOG"
echo ""

##########################################################################################
# Pre-flight Checks
##########################################################################################

print_header "PRE-FLIGHT CHECKS"

# Check data file
DATA_FILE="$ANALYSIS_DIR/../data/Celiac_dataset_proteomics_w_demo.parquet"
if [ ! -f "$DATA_FILE" ]; then
    print_error "Data file not found: $DATA_FILE"
    exit 1
fi
print_success "Data file found ($(du -h "$DATA_FILE" | cut -f1))"

# Check environment
cd "$ANALYSIS_DIR"
if ! python -c "import ced_ml; print('OK')" > /dev/null 2>&1; then
    print_error "ced_ml package not installed"
    echo "Run: cd analysis && pip install -e ."
    exit 1
fi
print_success "ced_ml package available"

# Check frozen config
if [ ! -f "$FROZEN_CONFIG" ]; then
    print_error "Frozen config not found: $FROZEN_CONFIG"
    exit 1
fi
print_success "Frozen config found"

# Check panel
if [ -f "$PANEL_FILE" ]; then
    PANEL_SIZE=$(wc -l < "$PANEL_FILE" | tr -d ' ')
    print_success "Panel file found: $PANEL_SIZE proteins"
elif [ "$SKIP_PANEL" = true ]; then
    print_error "Panel file not found and --skip-panel specified"
    exit 1
fi

echo ""

##########################################################################################
# Configuration Summary
##########################################################################################

print_header "EXPERIMENT CONFIGURATION"

echo "Design:"
echo "  Prevalent fractions:    ${PREVALENT_FRACS[@]}"
echo "  Case:control ratios:    ${CASE_CONTROL_RATIOS[@]}"
echo "  Models:                 ${MODELS[@]}"
echo "  Random seeds:           ${SPLIT_SEEDS[@]}"
echo ""

echo "Files:"
echo "  Fixed panel:            $PANEL_FILE"
echo "  Frozen config:          $FROZEN_CONFIG"
echo "  Data file:              $DATA_FILE"
echo ""

# Calculate total configs and runs
TOTAL_CONFIGS=0
TOTAL_RUNS=0
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

echo "Execution:"
echo "  Total configurations:   $TOTAL_CONFIGS"
echo "  Seeds per config:       ${#SPLIT_SEEDS[@]}"
echo "  Models per seed:        ${#MODELS[@]}"
echo "  Total runs:             $TOTAL_RUNS"
echo ""

# Estimate duration
MINUTES_PER_RUN=5
TOTAL_MINUTES=$((TOTAL_RUNS * MINUTES_PER_RUN))
TOTAL_HOURS=$((TOTAL_MINUTES / 60))
echo "Time estimate:"
echo "  Minutes per run:        ~$MINUTES_PER_RUN"
echo "  Total duration:         ~$TOTAL_HOURS hours ($TOTAL_MINUTES minutes)"

# Use BSD date syntax for macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    COMPLETION_TIME=$(date -v +${TOTAL_HOURS}H '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "N/A")
else
    COMPLETION_TIME=$(date -d "+${TOTAL_HOURS} hours" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "N/A")
fi
echo "  Expected completion:    $COMPLETION_TIME"
echo ""

echo "Flags:"
echo "  Skip training:          $SKIP_TRAINING"
echo "  Skip splits:            $SKIP_SPLITS"
echo "  Skip panel:             $SKIP_PANEL"
echo "  Force panel:            $FORCE_PANEL"
echo "  Dry run:                $DRY_RUN"
echo ""

if [ "$DRY_RUN" = true ]; then
    print_warning "DRY RUN MODE - no actual execution"
    echo ""
fi

##########################################################################################
# PHASE 0: Generate Fixed Panel
##########################################################################################

if [ "$SKIP_PANEL" = false ] || [ "$FORCE_PANEL" = true ]; then
    print_header "PHASE 0: Fixed Panel Generation"

    if [ -f "$PANEL_FILE" ] && [ "$FORCE_PANEL" = false ]; then
        print_warning "Panel file exists: $PANEL_FILE ($(wc -l < "$PANEL_FILE" | tr -d ' ') proteins)"
        read -p "Overwrite? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_status "Using existing panel"
            SKIP_PANEL=true
        fi
    fi

    if [ "$SKIP_PANEL" = false ] || [ "$FORCE_PANEL" = true ]; then
        print_status "Generating panel (Mann-Whitney screening + k-best selection)..."

        if [ "$DRY_RUN" = false ]; then
            PANEL_LOG="$LOG_DIR/panel_generation_${EXPERIMENT_ID}.log"
            if (cd "$ANALYSIS_DIR" && python docs/investigations/generate_fixed_panel.py \
                --infile ../data/Celiac_dataset_proteomics_w_demo.parquet \
                --outfile docs/investigations/top100_panel.csv \
                --final-k 100) > "$PANEL_LOG" 2>&1; then
                PANEL_SIZE=$(wc -l < "$PANEL_FILE" | tr -d ' ')
                print_success "Panel generated: $PANEL_SIZE proteins"
                print_info "Panel log: $PANEL_LOG"
            else
                print_error "Panel generation FAILED"
                print_error "Check log: $PANEL_LOG"
                exit 1
            fi
        else
            print_status "[DRY RUN] Would generate panel"
        fi
    fi

    echo ""
fi

##########################################################################################
# PHASE 1: Generate Splits for Each Configuration
##########################################################################################

if [ "$SKIP_SPLITS" = false ]; then
    print_header "PHASE 1: Split Generation"

    CONFIG_ID=0
    SPLIT_FAILURES=0

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

n_splits: ${#SPLIT_SEEDS[@]}
seed_start: ${SPLIT_SEEDS[0]}

train_control_per_case: $ccr
prevalent_sampling_frac: $pf

split_metadata:
  experiment_id: $EXPERIMENT_ID
  prevalent_frac: $pf
  case_control_ratio: $ccr
EOF

            if [ "$DRY_RUN" = false ]; then
                SPLIT_LOG="$LOG_DIR/splits_${pf}_${ccr}_${EXPERIMENT_ID}.log"
                if (cd "$ANALYSIS_DIR" && ced save-splits \
                    --config "$TEMP_CONFIG" \
                    --infile "$DATA_FILE" \
                    --outdir "$CONFIG_SPLITS_DIR") > "$SPLIT_LOG" 2>&1; then
                    print_success "$PROGRESS Splits saved to: ${CONFIG_SPLITS_DIR##*/}"
                else
                    print_error "$PROGRESS Split generation FAILED"
                    SPLIT_FAILURES=$((SPLIT_FAILURES + 1))
                fi
            else
                print_status "[DRY RUN] Would generate splits to: ${CONFIG_SPLITS_DIR##*/}"
            fi

            # Clean up temp config
            rm -f "$TEMP_CONFIG"
        done
    done

    if [ $SPLIT_FAILURES -gt 0 ]; then
        print_error "Split generation failed for $SPLIT_FAILURES configurations"
        exit 1
    fi

    echo ""
fi

##########################################################################################
# PHASE 2: Train Models for Each Configuration × Seed
##########################################################################################

if [ "$SKIP_TRAINING" = false ]; then
    print_header "PHASE 2: Model Training"

    RUN_ID=0
    TRAINING_FAILURES=0
    TRAINING_SUCCESSES=0

    for pf in "${PREVALENT_FRACS[@]}"; do
        for ccr in "${CASE_CONTROL_RATIOS[@]}"; do
            CONFIG_SPLITS_DIR="$SPLITS_BASE_DIR/${pf}_${ccr}"

            for seed in "${SPLIT_SEEDS[@]}"; do
                # Check if split files exist (CSV format)
                TRAIN_IDX_FILE="$CONFIG_SPLITS_DIR/train_idx_IncidentPlusPrevalent_seed${seed}.csv"
                if [ ! -f "$TRAIN_IDX_FILE" ]; then
                    print_error "Split files not found for seed $seed in: $CONFIG_SPLITS_DIR"
                    TRAINING_FAILURES=$((TRAINING_FAILURES + 1))
                    continue
                fi

                for model in "${MODELS[@]}"; do
                    RUN_ID=$((RUN_ID + 1))
                    PROGRESS="[$RUN_ID/$TOTAL_RUNS]"

                    print_status "$PROGRESS Training: $model, prevalent=$pf, ccr=$ccr, seed=$seed"

                    RESULTS_SUBDIR="$INVEST_RESULTS_DIR/${pf}_${ccr}/${model}/split_seed${seed}"
                    mkdir -p "$RESULTS_SUBDIR"

                    if [ "$DRY_RUN" = false ]; then
                        TRAIN_LOG="$LOG_DIR/train_${model}_${pf}_${ccr}_seed${seed}_${EXPERIMENT_ID}.log"

                        if (cd "$ANALYSIS_DIR" && ced train \
                            --config "$FROZEN_CONFIG" \
                            --model "$model" \
                            --infile "$DATA_FILE" \
                            --split-dir "$CONFIG_SPLITS_DIR" \
                            --split-seed "$seed" \
                            --outdir "$RESULTS_SUBDIR" \
                            --fixed-panel docs/investigations/top100_panel.csv \
                            --metadata experiment_id="$EXPERIMENT_ID" \
                            --metadata prevalent_frac="$pf" \
                            --metadata case_control_ratio="$ccr" \
                            --metadata split_seed="$seed") > "$TRAIN_LOG" 2>&1; then
                            print_success "$PROGRESS Training complete"
                            TRAINING_SUCCESSES=$((TRAINING_SUCCESSES + 1))
                        else
                            print_error "$PROGRESS Training FAILED (log: ${TRAIN_LOG##*/})"
                            TRAINING_FAILURES=$((TRAINING_FAILURES + 1))
                        fi
                    else
                        print_status "[DRY RUN] Would train: $model (seed=$seed, pf=$pf, ccr=$ccr)"
                        TRAINING_SUCCESSES=$((TRAINING_SUCCESSES + 1))
                    fi
                done
            done
        done
    done

    echo ""
    print_status "Training summary:"
    echo "  Successful runs:  $TRAINING_SUCCESSES / $TOTAL_RUNS"
    echo "  Failed runs:      $TRAINING_FAILURES / $TOTAL_RUNS"
    echo ""

    if [ $TRAINING_FAILURES -gt 0 ]; then
        print_warning "Some training runs failed - check logs in $LOG_DIR"
    fi

    echo ""
fi

##########################################################################################
# PHASE 3: Statistical Analysis
##########################################################################################

print_header "PHASE 3: Statistical Analysis"

print_status "Analyzing results across configurations..."

if [ "$DRY_RUN" = false ]; then
    ANALYSIS_LOG="$LOG_DIR/analysis_${EXPERIMENT_ID}.log"

    mkdir -p "$EXPERIMENT_OUTPUT_DIR"

    if python "$SCRIPT_DIR/analyze_factorial_results.py" \
        --results-dir "$INVEST_RESULTS_DIR" \
        --output-dir "$EXPERIMENT_OUTPUT_DIR" \
        --experiment-id "$EXPERIMENT_ID" > "$ANALYSIS_LOG" 2>&1; then
        print_success "Statistical analysis complete"
        print_info "Results saved to: ${EXPERIMENT_OUTPUT_DIR##*/}"
    else
        print_error "Statistical analysis FAILED"
        print_error "Check log: $ANALYSIS_LOG"
    fi
else
    print_status "[DRY RUN] Would run statistical analysis"
fi

echo ""

##########################################################################################
# Summary
##########################################################################################

print_header "EXPERIMENT COMPLETE"

END_TIME=$(date)
print_status "End time: $END_TIME"

if [ "$DRY_RUN" = false ]; then
    echo ""
    print_info "Experiment ID: $EXPERIMENT_ID"
    print_info "Results directory: $EXPERIMENT_OUTPUT_DIR"
    print_info "Full log: $EXPERIMENT_LOG"

    if [ -f "$EXPERIMENT_OUTPUT_DIR/summary.md" ]; then
        echo ""
        print_status "Quick findings preview:"
        echo ""
        head -n 60 "$EXPERIMENT_OUTPUT_DIR/summary.md"
        echo ""
        print_info "Full summary: $EXPERIMENT_OUTPUT_DIR/summary.md"
    fi

    if [ -d "$EXPERIMENT_OUTPUT_DIR" ]; then
        echo ""
        print_status "Generated files:"
        ls -lh "$EXPERIMENT_OUTPUT_DIR"/*.{csv,md,json} 2>/dev/null || echo "  (checking...)"
    fi
else
    print_info "Dry run complete - no files generated"
fi

echo ""

if [ $TRAINING_FAILURES -gt 0 ]; then
    print_warning "Experiment completed with $TRAINING_FAILURES training failures"
    exit 1
else
    print_success "All operations completed successfully"
    exit 0
fi
