#!/bin/bash

#######################################################################################
# Unified Investigation Runner
# Supports three execution modes: single, full, hpc
#
# Usage:
#   bash run_investigation.sh --mode MODE [options]
#
# Modes:
#   single    Single split analysis (fast diagnostic, ~30 sec)
#   full      Full coverage across all splits (~10-15 min)
#   hpc       HPC distributed execution (~2-3 min wall-clock)
#
# Options:
#   --models MODEL1,MODEL2    Models to investigate (default: all)
#   --split-seed N            Split seed (single mode only, default: 0)
#   --oof-or-test MODE        oof or test (single mode only, default: oof)
#   --analyses LIST           Comma-separated analyses (default: distributions)
#   --cores N                 Cores per job (hpc mode, default: 2)
#   --memory MEM              Memory per job (hpc mode, default: 8G)
#   --walltime TIME           Wall clock time (hpc mode, default: 01:00)
#   --queue QUEUE             LSF queue (hpc mode, default: medium)
#   --dry-run                 Preview without executing (hpc mode)
#
# Examples:
#   # Quick diagnostic (single split, OOF)
#   bash run_investigation.sh --mode single
#
#   # Full coverage (all splits, OOF + test)
#   bash run_investigation.sh --mode full
#
#   # HPC execution
#   bash run_investigation.sh --mode hpc
#
#   # Specific models only
#   bash run_investigation.sh --mode full --models LR_EN,RF
#######################################################################################

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/../../../results/investigations"
LOG_DIR="$ANALYSIS_DIR/../logs/investigation"

# Defaults
MODE=""
MODELS=("LR_EN" "RF" "XGBoost" "LinSVM_cal")
SPLIT_SEEDS=(0 1 2 3 4 5 6 7 8 9)
SPLIT_SEED=0
OOF_OR_TEST="oof"
ANALYSES="distributions"
DRY_RUN=false
QUEUE="medium"
CORES=2
MEMORY="8G"
WALLTIME="01:00"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Helper functions
print_header() {
    echo ""
    echo "================================================================================================"
    echo "  $1"
    echo "================================================================================================"
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

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --models)
            IFS=',' read -ra MODELS <<< "$2"
            shift 2
            ;;
        --split-seed)
            SPLIT_SEED="$2"
            shift 2
            ;;
        --oof-or-test)
            OOF_OR_TEST="$2"
            shift 2
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
        --walltime)
            WALLTIME="$2"
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
        *)
            echo "Unknown option: $1"
            echo "Usage: bash run_investigation.sh --mode MODE [options]"
            echo "Modes: single, full, hpc"
            echo "Run with --help for details"
            exit 1
            ;;
    esac
done

# Validate mode
if [ -z "$MODE" ]; then
    print_error "Mode required: --mode {single|full|hpc}"
    exit 1
fi

# Create directories
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

#######################################################################################
# MODE: SINGLE (fast diagnostic)
#######################################################################################

if [ "$MODE" == "single" ]; then
    print_header "SINGLE SPLIT INVESTIGATION"

    LOG_FILE="$RESULTS_DIR/investigation_single_$(date +%Y%m%d_%H%M%S).log"

    print_status "Configuration:"
    echo "  Models:      ${MODELS[@]}"
    echo "  Split seed:  $SPLIT_SEED"
    echo "  Mode:        $OOF_OR_TEST"
    echo "  Analyses:    $ANALYSES"
    echo "  Output dir:  $RESULTS_DIR"
    echo ""

    COMPLETED=0
    FAILED=0
    TOTAL=${#MODELS[@]}

    for MODEL in "${MODELS[@]}"; do
        COMPLETED=$((COMPLETED + 1))
        PROGRESS="[$COMPLETED/$TOTAL]"

        print_status "$PROGRESS Investigating $MODEL..."

        if (cd "$ANALYSIS_DIR" && python docs/investigations/investigate.py \
            --mode "$OOF_OR_TEST" \
            --model "$MODEL" \
            --split-seed "$SPLIT_SEED" \
            --analyses "$ANALYSES") \
            >> "$LOG_FILE" 2>&1; then
            print_success "$PROGRESS $MODEL completed"
        else
            print_error "$PROGRESS $MODEL FAILED"
            FAILED=$((FAILED + 1))
        fi
    done

    print_header "INVESTIGATION COMPLETE"
    echo "  Total runs:  $TOTAL"
    echo "  Successful:  $((TOTAL - FAILED))"
    echo "  Failed:      $FAILED"
    echo "  Log file:    $LOG_FILE"
    echo ""
    echo "Results saved to: $RESULTS_DIR"

    [ $FAILED -eq 0 ] && exit 0 || exit 1

#######################################################################################
# MODE: FULL (comprehensive coverage)
#######################################################################################

elif [ "$MODE" == "full" ]; then
    print_header "FULL CASE COVERAGE INVESTIGATION"

    LOG_FILE="$RESULTS_DIR/investigation_full_$(date +%Y%m%d_%H%M%S).log"

    print_status "Strategy: Analyze OOF + Test across all 10 splits"
    echo "  - Each split has ~75 incident in OOF and ~15 in test"
    echo "  - Aggregating covers all ~148 incident cases"
    echo ""

    TOTAL_RUNS=$((${#MODELS[@]} * ${#SPLIT_SEEDS[@]} * 2))
    COMPLETED=0
    FAILED=0

    for MODEL in "${MODELS[@]}"; do
        for SEED in "${SPLIT_SEEDS[@]}"; do
            for RUN_MODE in "oof" "test"; do
                COMPLETED=$((COMPLETED + 1))
                PROGRESS="[$COMPLETED/$TOTAL_RUNS]"

                print_status "$PROGRESS $MODEL seed $SEED ($RUN_MODE)..."

                if (cd "$ANALYSIS_DIR" && python docs/investigations/investigate.py \
                    --mode "$RUN_MODE" \
                    --model "$MODEL" \
                    --split-seed "$SEED" \
                    --analyses "$ANALYSES") \
                    >> "$LOG_FILE" 2>&1; then
                    print_success "$PROGRESS Completed"
                else
                    print_error "$PROGRESS FAILED"
                    FAILED=$((FAILED + 1))
                fi
            done
        done
    done

    # Consolidate results
    print_header "CONSOLIDATING RESULTS"

    cat > "$RESULTS_DIR/consolidate_full.py" << 'PYTHON_SCRIPT'
import pandas as pd
from pathlib import Path
import glob

results_dir = Path(__file__).parent

# Load all summaries
oof_files = sorted(glob.glob(str(results_dir / "summary_oof_seed*.csv")))
test_files = sorted(glob.glob(str(results_dir / "summary_test_seed*.csv")))

print(f"Found {len(oof_files)} OOF + {len(test_files)} test summaries")

all_data = []
for f in oof_files + test_files:
    try:
        df = pd.read_csv(f)
        all_data.append(df)
    except Exception as e:
        print(f"WARNING: Could not load {f}: {e}")

if not all_data:
    print("ERROR: No data to aggregate")
    exit(1)

combined = pd.concat(all_data, ignore_index=True)

# Per-model summary
print("\n" + "="*80)
print("FULL CASE COVERAGE SUMMARY")
print("="*80)

for model in sorted(combined['model'].unique()):
    model_data = combined[combined['model'] == model]
    oof_data = model_data[model_data['mode'] == 'oof']
    test_data = model_data[model_data['mode'] == 'test']

    print(f"\n{model}:")
    print(f"  OOF runs:  {len(oof_data)} (~{oof_data['n_incident'].sum()} incident instances)")
    print(f"  Test runs: {len(test_data)} (~{test_data['n_incident'].sum()} incident instances)")

    if 'median_diff' in model_data.columns:
        avg_diff = model_data['median_diff'].mean()
        sig_count = sum(model_data['mw_pval'] < 0.05)
        print(f"  Avg median difference: {avg_diff:+.4f}")
        print(f"  Significant runs (p<0.05): {sig_count}/{len(model_data)}")
        print(f"  Avg Cohen's d: {model_data['cohens_d'].mean():.3f}")

# Save consolidated
output_path = results_dir / "FULL_COVERAGE_SUMMARY.csv"
combined.to_csv(output_path, index=False)
print(f"\nFull coverage summary: {output_path.name}")

# Per-model aggregate
per_model = combined.groupby(['model']).agg({
    'split_seed': 'count',
    'n_incident': 'sum',
    'n_prevalent': 'sum',
    'median_diff': ['mean', 'std'],
    'cohens_d': ['mean', 'std'],
    'mw_pval': lambda x: sum(x < 0.05)
}).round(4)

model_summary_path = results_dir / "FULL_COVERAGE_MODEL_SUMMARY.csv"
per_model.to_csv(model_summary_path)
print(f"Per-model summary: {model_summary_path.name}")
PYTHON_SCRIPT

    if python "$RESULTS_DIR/consolidate_full.py" >> "$LOG_FILE" 2>&1; then
        print_success "Results consolidated"
    else
        print_warning "Consolidation had issues (check $LOG_FILE)"
    fi

    print_header "INVESTIGATION COMPLETE"
    echo "  Total runs:  $TOTAL_RUNS"
    echo "  Successful:  $((TOTAL_RUNS - FAILED))"
    echo "  Failed:      $FAILED"
    echo "  Log file:    $LOG_FILE"
    echo ""
    echo "Key outputs:"
    echo "  - FULL_COVERAGE_SUMMARY.csv"
    echo "  - FULL_COVERAGE_MODEL_SUMMARY.csv"

    [ $FAILED -eq 0 ] && exit 0 || exit 1

#######################################################################################
# MODE: HPC (distributed execution)
#######################################################################################

elif [ "$MODE" == "hpc" ]; then
    print_header "HPC DISTRIBUTED INVESTIGATION"

    # Generate job array
    JOB_LIST_FILE="$LOG_DIR/job_list_$(date +%Y%m%d_%H%M%S).txt"
    > "$JOB_LIST_FILE"

    JOB_INDEX=0
    for MODEL in "${MODELS[@]}"; do
        for SEED in "${SPLIT_SEEDS[@]}"; do
            for RUN_MODE in "oof" "test"; do
                JOB_INDEX=$((JOB_INDEX + 1))
                echo "$JOB_INDEX $MODEL $SEED $RUN_MODE" >> "$JOB_LIST_FILE"
            done
        done
    done

    TOTAL_JOBS=$JOB_INDEX

    print_status "Job configuration:"
    echo "  Total jobs:  $TOTAL_JOBS"
    echo "  Models:      ${MODELS[@]}"
    echo "  Split seeds: ${SPLIT_SEEDS[@]}"
    echo "  Queue:       $QUEUE"
    echo "  Resources:   ${CORES} cores, ${MEMORY} memory, ${WALLTIME} walltime"
    echo ""

    if [ "$DRY_RUN" = true ]; then
        print_warning "DRY RUN - Jobs NOT submitted"
        echo "Job list saved to: $JOB_LIST_FILE"
        head -10 "$JOB_LIST_FILE"
        echo "..."
        echo ""
        echo "To submit for real, run without --dry-run"
        exit 0
    fi

    # Generate LSF submission script
    SUBMIT_SCRIPT="$LOG_DIR/submit_$(date +%Y%m%d_%H%M%S).sh"

    cat > "$SUBMIT_SCRIPT" << EOF
#!/bin/bash
#BSUB -J investigate[1-${TOTAL_JOBS}]
#BSUB -q ${QUEUE}
#BSUB -n ${CORES}
#BSUB -R "rusage[mem=${MEMORY}]"
#BSUB -W ${WALLTIME}
#BSUB -o ${LOG_DIR}/investigate_%I.log
#BSUB -e ${LOG_DIR}/investigate_%I.err

set -e

# Activate environment
cd $ANALYSIS_DIR
source venv/bin/activate 2>/dev/null || source ../venv/bin/activate 2>/dev/null || true

# Read job parameters
JOB_INFO=\$(sed -n "\${LSB_JOBINDEX}p" $JOB_LIST_FILE)
read -r JOB_ID MODEL SEED RUN_MODE <<< "\$JOB_INFO"

echo "Job \$LSB_JOBINDEX: \$MODEL seed \$SEED (\$RUN_MODE)"

# Run investigation
python docs/investigations/investigate.py \\
    --mode "\$RUN_MODE" \\
    --model "\$MODEL" \\
    --split-seed "\$SEED" \\
    --analyses "$ANALYSES"

echo "Job \$LSB_JOBINDEX completed successfully"
EOF

    chmod +x "$SUBMIT_SCRIPT"

    # Submit
    print_status "Submitting job array..."
    if JOB_ID=$(bsub < "$SUBMIT_SCRIPT" 2>&1 | grep -oP 'Job <\K[0-9]+'); then
        print_success "Job array submitted: $JOB_ID"
        echo ""
        echo "Monitor progress:"
        echo "  bjobs -w $JOB_ID"
        echo "  tail -f $LOG_DIR/investigate_*.log"
        echo ""
        echo "After completion, consolidate results:"
        echo "  bash run_investigation.sh --mode full  # (will reuse existing outputs)"
    else
        print_error "Job submission failed"
        exit 1
    fi

else
    print_error "Invalid mode: $MODE (choose: single, full, hpc)"
    exit 1
fi
