#!/bin/bash

#######################################################################################
# Full Investigation Runner
# Runs investigation on ALL models across ALL split seeds
# Comprehensive analysis with distributions, calibration, and feature bias
#######################################################################################

set -e

# Configuration
INVESTIGATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS=("LR_EN" "RF" "XGBoost" "LinSVM_cal")
SPLIT_SEEDS=(0 1 2 3 4 5 6 7 8 9)
ANALYSES="distributions,calibration,features"
MODE="oof"

# Output tracking
RESULTS_DIR="../../../results/investigations"
LOG_FILE="$RESULTS_DIR/investigation_run_$(date +%Y%m%d_%H%M%S).log"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Create results directory
mkdir -p "$RESULTS_DIR"

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

# Summary tracking
TOTAL_RUNS=$((${#MODELS[@]} * ${#SPLIT_SEEDS[@]}))
COMPLETED=0
FAILED=0
FAILED_RUNS=()

print_header "COMPREHENSIVE INVESTIGATION: ALL MODELS × ALL SEEDS"

print_status "Configuration:"
echo "  Models:      ${MODELS[@]}"
echo "  Split seeds: ${SPLIT_SEEDS[@]}"
echo "  Analyses:    $ANALYSES"
echo "  Mode:        $MODE"
echo "  Output dir:  $RESULTS_DIR"
echo "  Log file:    $LOG_FILE"
echo "  Total runs:  $TOTAL_RUNS"

print_status "Starting investigation runs..."
echo ""

# Run investigation for each model and seed combination
for MODEL in "${MODELS[@]}"; do
    for SEED in "${SPLIT_SEEDS[@]}"; do
        COMPLETED=$((COMPLETED + 1))
        PROGRESS="[$COMPLETED/$TOTAL_RUNS]"

        print_status "$PROGRESS Investigating $MODEL (seed $SEED)..."

        # Run investigation (from analysis directory to ensure ced_ml is available)
        ANALYSIS_DIR="$INVESTIGATION_DIR/../../"
        if (cd "$ANALYSIS_DIR" && python docs/investigations/investigate.py \
            --mode "$MODE" \
            --model "$MODEL" \
            --split-seed "$SEED" \
            --analyses "$ANALYSES") \
            >> "$LOG_FILE" 2>&1; then
            print_success "$PROGRESS $MODEL seed $SEED completed"
        else
            print_error "$PROGRESS $MODEL seed $SEED FAILED"
            FAILED=$((FAILED + 1))
            FAILED_RUNS+=("$MODEL seed $SEED")
        fi
    done
done

print_header "INVESTIGATION COMPLETE"

print_status "Results Summary:"
echo "  Total runs:     $TOTAL_RUNS"
echo "  Successful:     $((TOTAL_RUNS - FAILED))"
echo "  Failed:         $FAILED"

if [ $FAILED -gt 0 ]; then
    print_warning "Failed runs:"
    for run in "${FAILED_RUNS[@]}"; do
        echo "    - $run"
    done
fi

print_status "Generating consolidated summary..."

# Generate Python script to create cross-seed summary
cat > "$RESULTS_DIR/consolidate_results.py" << 'PYTHON_SCRIPT'
import pandas as pd
from pathlib import Path
import glob

results_dir = Path(__file__).parent

# Find all summary CSVs
summary_files = sorted(glob.glob(str(results_dir / "summary_*.csv")))

if not summary_files:
    print("WARNING: No summary files found")
    exit(1)

# Load and combine all summaries
all_summaries = []
for f in summary_files:
    try:
        df = pd.read_csv(f)
        all_summaries.append(df)
    except Exception as e:
        print(f"WARNING: Could not load {f}: {e}")

if not all_summaries:
    print("ERROR: No summaries could be loaded")
    exit(1)

combined = pd.concat(all_summaries, ignore_index=True)

# Group by model
print("\n" + "="*100)
print("SUMMARY BY MODEL")
print("="*100)

for model in sorted(combined['model'].unique()):
    model_data = combined[combined['model'] == model]
    print(f"\n{model}:")
    print(f"  Splits analyzed: {model_data['split_seed'].nunique()}")

    # Distribution stats
    if 'mw_pval' in model_data.columns:
        sig_count = sum(model_data['mw_pval'] < 0.05)
        print(f"  Significant differences (p<0.05): {sig_count}/{len(model_data)}")
        print(f"  Median effect size (Cohen's d): {model_data['cohens_d'].mean():.3f}")
        print(f"  Average median difference: {model_data['median_diff'].mean():+.3f}")
        print(f"  Direction consistency: {sum(model_data['median_diff'] > 0)}/{len(model_data)} incidents higher")

# Overall patterns
print("\n" + "="*100)
print("OVERALL PATTERNS")
print("="*100)

if 'cohens_d' in combined.columns:
    avg_effect = combined['cohens_d'].mean()
    print(f"\nAverage effect size across all runs: {avg_effect:.3f}")
    if avg_effect < 0.2:
        print("  → Negligible effect (both case types treated similarly)")
    elif avg_effect < 0.5:
        print("  → Small effect (minor differences)")
    elif avg_effect < 0.8:
        print("  → Medium effect (notable differences)")
    else:
        print("  → Large effect (substantial differences)")

if 'mw_pval' in combined.columns:
    total_sig = sum(combined['mw_pval'] < 0.05)
    print(f"\nTotal significant differences: {total_sig}/{len(combined)} runs (p<0.05)")

# Save consolidated summary
output_path = Path(__file__).parent / "CONSOLIDATED_SUMMARY.csv"
combined.to_csv(output_path, index=False)
print(f"\nConsolidated summary saved: {output_path.name}")

# Save per-model summary
per_model = combined.groupby('model').agg({
    'split_seed': 'count',
    'median_diff': ['mean', 'std'],
    'cohens_d': ['mean', 'std'],
    'mw_pval': lambda x: sum(x < 0.05)
}).round(3)

model_summary_path = Path(__file__).parent / "MODEL_SUMMARY.csv"
per_model.to_csv(model_summary_path)
print(f"Per-model summary saved: {model_summary_path.name}")

PYTHON_SCRIPT

# Run consolidation
print_status "Consolidating results across all seeds..."
if python "$RESULTS_DIR/consolidate_results.py" >> "$LOG_FILE" 2>&1; then
    print_success "Results consolidated successfully"
else
    print_warning "Consolidation script encountered issues (check $LOG_FILE)"
fi

# Print output paths
print_status "Output locations:"
echo "  Results directory: $RESULTS_DIR"
echo "  Log file:          $LOG_FILE"
echo ""
echo "  Key output files:"
echo "    - CONSOLIDATED_SUMMARY.csv      (all runs combined)"
echo "    - MODEL_SUMMARY.csv             (per-model aggregates)"
echo "    - summary_*.csv                 (per-seed summaries)"
echo "    - distributions_*.png           (score distribution plots)"
echo "    - calibration_*.png             (calibration analysis plots)"
echo "    - feature_bias_*.png            (feature bias plots)"
echo "    - feature_bias_details_*.csv    (per-protein AUROC data)"

# Final status
echo ""
if [ $FAILED -eq 0 ]; then
    print_success "All investigations completed successfully!"
    print_status "Next steps:"
    echo "  1. Review consolidated results: CONSOLIDATED_SUMMARY.csv"
    echo "  2. Check per-model patterns:    MODEL_SUMMARY.csv"
    echo "  3. Examine individual plots in $RESULTS_DIR"
    exit 0
else
    print_warning "Investigation completed with $FAILED failures"
    print_status "Check log file for details: $LOG_FILE"
    exit 1
fi
