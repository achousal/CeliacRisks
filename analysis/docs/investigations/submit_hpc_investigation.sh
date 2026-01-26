#!/bin/bash

#######################################################################################
# HPC Investigation Job Submission
# Submits investigation runs as array jobs to HPC (LSF scheduler)
# Non-interactive, batch-safe, resumable
#
# Usage:
#   bash submit_hpc_investigation.sh [options]
#
# Options:
#   --models MODEL1,MODEL2    Models to investigate (default: all)
#   --seeds START-END         Seed range (e.g., 0-4) or list (e.g., 0,1,2)
#   --queue QUEUE_NAME        LSF queue (default: medium)
#   --cores N                 Cores per job (default: 2)
#   --memory MEM              Memory per job (default: 8G)
#   --walltime TIME           Wall clock time (default: 01:00)
#   --dry-run                 Preview jobs without submitting
#
# Examples:
#   bash submit_hpc_investigation.sh                           # All models, all seeds
#   bash submit_hpc_investigation.sh --models LR_EN,RF         # Specific models
#   bash submit_hpc_investigation.sh --seeds 0-4               # Specific seed range
#   bash submit_hpc_investigation.sh --dry-run                 # Preview without submitting
#######################################################################################

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECT_ROOT="$(cd "$ANALYSIS_DIR/.." && pwd)"

# Defaults
MODELS=("LR_EN" "RF" "XGBoost" "LinSVM_cal")
SEEDS=(0 1 2 3 4 5 6 7 8 9)
MODE="oof"
DRY_RUN=false
QUEUE="medium"
CORES=2
MEMORY="8G"
WALLTIME="01:00"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --models)
            IFS=',' read -ra MODELS <<< "$2"
            shift 2
            ;;
        --seeds)
            if [[ "$2" == *"-"* ]]; then
                IFS='-' read -r START END <<< "$2"
                SEEDS=($(seq $START $END))
            else
                IFS=',' read -ra SEEDS <<< "$2"
            fi
            shift 2
            ;;
        --queue)
            QUEUE="$2"
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
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "ERROR: Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_header() {
    echo ""
    echo "================================================================================================"
    echo "  $1"
    echo "================================================================================================"
}

print_status() {
    echo -e "${BLUE}[*]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Build job list
JOB_LIST=()
for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        JOB_LIST+=("$MODEL:$SEED")
    done
done

TOTAL_JOBS=${#JOB_LIST[@]}

print_header "HPC INVESTIGATION SUBMISSION"

print_status "Configuration:"
echo "  Models:       ${MODELS[@]}"
echo "  Seeds:        ${SEEDS[@]}"
echo "  Total jobs:   $TOTAL_JOBS"
echo "  Queue:        $QUEUE"
echo "  Resources:    $CORES cores, $MEMORY RAM, $WALLTIME walltime"
echo "  Dry run:      $DRY_RUN"

# Create logs directory
mkdir -p "$ANALYSIS_DIR/logs"

# Create job submission file
SUBMIT_FILE="$ANALYSIS_DIR/logs/submit_investigation_$(date +%Y%m%d_%H%M%S).sh"

cat > "$SUBMIT_FILE" << 'JOBSCRIPT'
#!/bin/bash

#BSUB -J "CeD_Investigate[1-ARRAY_SIZE]"
#BSUB -n CORES
#BSUB -M MEMORY
#BSUB -W WALLTIME
#BSUB -q QUEUE
#BSUB -o logs/investigate_%I.log
#BSUB -e logs/investigate_%I.err
#BSUB -R "rusage[mem=MEMORY]"

set -e

# Initialize environment
if command -v module &> /dev/null; then
    module load python || true
fi

# Activate conda environment if available
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -n "$CONDA_PREFIX" ]; then
    source "$CONDA_PREFIX/bin/activate" ced_ml 2>/dev/null || true
fi

# Get job array index and configuration
JOB_INDEX=$((LSB_JOBINDEX - 1))

# Job mapping array (populated below by script)
declare -a JOBS_ARRAY
JOBS_ARRAY+=(JOBS_PLACEHOLDER)

# Extract model and seed from job mapping
JOB_CONFIG="${JOBS_ARRAY[$JOB_INDEX]}"
MODEL=$(echo "$JOB_CONFIG" | cut -d: -f1)
SEED=$(echo "$JOB_CONFIG" | cut -d: -f2)

# Log header
echo "================================================================================================"
echo "Investigation Job: $((JOB_INDEX + 1))/TOTAL_JOBS"
echo "Model: $MODEL, Seed: $SEED"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Hostname: $(hostname)"
echo "================================================================================================"

cd "ANALYSIS_DIR"

# Run investigation
python docs/investigations/investigate.py \
    --mode oof \
    --model "$MODEL" \
    --split-seed "$SEED" \
    --analyses "distributions,calibration,features"

echo ""
echo "Investigation completed successfully"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
JOBSCRIPT

# Replace placeholders in job script
sed -i "s|ARRAY_SIZE|$TOTAL_JOBS|g" "$SUBMIT_FILE"
sed -i "s|CORES|$CORES|g" "$SUBMIT_FILE"
sed -i "s|MEMORY|$MEMORY|g" "$SUBMIT_FILE"
sed -i "s|WALLTIME|$WALLTIME|g" "$SUBMIT_FILE"
sed -i "s|QUEUE|$QUEUE|g" "$SUBMIT_FILE"
sed -i "s|ANALYSIS_DIR|$ANALYSIS_DIR|g" "$SUBMIT_FILE"
sed -i "s|TOTAL_JOBS|$TOTAL_JOBS|g" "$SUBMIT_FILE"

# Build and insert job mapping
JOB_MAPPING_LINES=""
for i in "${!JOB_LIST[@]}"; do
    JOB_MAPPING_LINES+="JOBS_ARRAY[$i]=\"${JOB_LIST[$i]}\""$'\n'
done

# Replace placeholder with actual mapping
sed -i "/JOBS_ARRAY+=(JOBS_PLACEHOLDER)/c\\
$JOB_MAPPING_LINES" "$SUBMIT_FILE"

chmod +x "$SUBMIT_FILE"

print_status "Job submission script created:"
echo "  $SUBMIT_FILE"

print_status "Job mapping ($TOTAL_JOBS jobs):"
for i in "${!JOB_LIST[@]}"; do
    printf "  %2d. %s\n" "$((i+1))" "${JOB_LIST[$i]}"
done

echo ""

if [ "$DRY_RUN" = true ]; then
    print_warning "DRY RUN MODE - no jobs submitted"
    print_status "To submit jobs, run:"
    echo "  bsub < $SUBMIT_FILE"
else
    print_status "Submitting job array..."
    JOB_ID=$(bsub < "$SUBMIT_FILE" | grep -oP 'Job <\K[0-9]+')

    if [ -z "$JOB_ID" ]; then
        print_warning "Could not extract job ID. Check submission manually:"
        bsub < "$SUBMIT_FILE"
    else
        print_success "Job array submitted with ID: $JOB_ID"
        echo ""
        print_status "Monitor progress:"
        echo "  bjobs -w $JOB_ID                     # Job status"
        echo "  tail -f logs/investigate_*.log       # Live logs"
        echo "  bjobs -a $JOB_ID | wc -l             # Count completed jobs"
        echo ""
        print_status "After all jobs complete (bjobs shows no active jobs):"
        echo "  python docs/investigations/consolidate_hpc_results.py"
        echo ""
        print_status "Job logs saved to: logs/investigate_*.log and logs/investigate_*.err"
    fi
fi

print_status "Next steps:"
echo "  1. Monitor job status: bjobs -w"
echo "  2. Check individual logs: logs/investigate_*.log"
echo "  3. When all jobs done, aggregate results (see above)"
