#!/usr/bin/env python3

"""
Consolidate investigation results across HPC job array.

Run after all investigation jobs have completed:
  python consolidate_hpc_results.py

This script:
  1. Verifies all expected output files exist
  2. Combines per-seed results into summary tables
  3. Generates per-model aggregates
  4. Produces interpretation guide
"""

import pandas as pd
import json
from pathlib import Path
import sys
from datetime import datetime

def main():
    results_dir = Path(__file__).parent.parent.parent.parent / "results" / "investigations"

    print("\n" + "=" * 100)
    print("INVESTIGATION RESULTS CONSOLIDATION")
    print("=" * 100)
    print(f"\nResults directory: {results_dir}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Find all summary CSVs
    summary_files = sorted(results_dir.glob("summary_*.csv"))

    if not summary_files:
        print("\nERROR: No summary files found in results directory")
        print(f"Expected files: summary_*.csv in {results_dir}")
        print("\nPossible issues:")
        print("  1. Investigation jobs may not have completed")
        print("  2. Results directory path is incorrect")
        print("  3. Job failures prevented output generation")
        return 1

    print(f"\nFound {len(summary_files)} summary files")

    # Load and combine all summaries
    all_summaries = []
    failed_files = []

    for f in summary_files:
        try:
            df = pd.read_csv(f)
            all_summaries.append(df)
            print(f"  ✓ {f.name}")
        except Exception as e:
            print(f"  ✗ {f.name}: {e}")
            failed_files.append(f.name)

    if not all_summaries:
        print("\nERROR: Could not load any summary files")
        return 1

    if failed_files:
        print(f"\nWARNING: {len(failed_files)} files failed to load")

    # Combine all data
    combined = pd.concat(all_summaries, ignore_index=True)
    print(f"\nTotal records: {len(combined)}")

    # Save consolidated summary
    consolidated_path = results_dir / "CONSOLIDATED_SUMMARY.csv"
    combined.to_csv(consolidated_path, index=False)
    print(f"\nConsolidated summary saved: {consolidated_path.name}")

    # Generate per-model analysis
    print("\n" + "=" * 100)
    print("SUMMARY BY MODEL")
    print("=" * 100)

    model_summaries = []

    for model in sorted(combined['model'].unique()):
        model_data = combined[combined['model'] == model]
        n_seeds = model_data['split_seed'].nunique()

        print(f"\n{model}:")
        print(f"  Splits analyzed: {n_seeds}")

        # Distribution stats
        if 'mw_pval' in model_data.columns:
            sig_count = sum(model_data['mw_pval'] < 0.05)
            print(f"  Significant differences (p<0.05): {sig_count}/{len(model_data)}")

            avg_d = model_data['cohens_d'].mean()
            std_d = model_data['cohens_d'].std()
            print(f"  Cohen's d: {avg_d:.3f} ± {std_d:.3f}")

            avg_median_diff = model_data['median_diff'].mean()
            print(f"  Average median score difference: {avg_median_diff:+.3f}")

            n_higher = sum(model_data['median_diff'] > 0)
            print(f"  Incident > Prevalent: {n_higher}/{len(model_data)} seeds")

        # Power analysis
        if 'power' in model_data.columns:
            avg_power = model_data['power'].mean()
            adequate_power = sum(model_data['power'] > 0.80) / len(model_data) * 100
            print(f"  Average power: {avg_power:.1%} (adequate: {adequate_power:.0f}%)")

        model_summaries.append({
            'model': model,
            'n_seeds': n_seeds,
            'sig_count': sig_count,
            'cohens_d_mean': avg_d,
            'cohens_d_std': std_d,
            'median_diff_mean': avg_median_diff,
            'incident_higher_pct': n_higher / len(model_data) * 100,
        })

    # Save per-model summary
    if model_summaries:
        model_summary_df = pd.DataFrame(model_summaries)
        model_summary_path = results_dir / "MODEL_SUMMARY.csv"
        model_summary_df.to_csv(model_summary_path, index=False)
        print(f"\nPer-model summary saved: {model_summary_path.name}")

    # Overall interpretation
    print("\n" + "=" * 100)
    print("OVERALL PATTERNS")
    print("=" * 100)

    if 'cohens_d' in combined.columns:
        avg_effect = combined['cohens_d'].mean()
        print(f"\nAverage effect size across all runs: {avg_effect:.3f}")

        if avg_effect < 0.2:
            interpretation = "NEGLIGIBLE (both case types treated similarly)"
        elif avg_effect < 0.5:
            interpretation = "SMALL (minor differences observed)"
        elif avg_effect < 0.8:
            interpretation = "MEDIUM (notable differences)"
        else:
            interpretation = "LARGE (substantial differences)"

        print(f"  Interpretation: {interpretation}")

    if 'mw_pval' in combined.columns:
        total_sig = sum(combined['mw_pval'] < 0.05)
        sig_pct = total_sig / len(combined) * 100
        print(f"\nStatistical significance:")
        print(f"  {total_sig}/{len(combined)} runs with p<0.05 ({sig_pct:.1f}%)")

    # Output file summary
    print("\n" + "=" * 100)
    print("GENERATED FILES")
    print("=" * 100)

    output_files = {
        'Summary tables': [
            ('CONSOLIDATED_SUMMARY.csv', 'All runs combined'),
            ('MODEL_SUMMARY.csv', 'Per-model aggregates'),
            ('summary_*.csv', 'Individual seed results'),
        ],
        'Visualizations': [
            ('distributions_*.png', 'Score distribution plots'),
            ('calibration_*.png', 'Calibration analysis'),
            ('feature_bias_*.png', 'Feature bias analysis'),
        ],
        'Feature details': [
            ('feature_bias_details_*.csv', 'Per-protein AUROC scores'),
        ],
    }

    for category, files in output_files.items():
        print(f"\n{category}:")
        for pattern, description in files:
            count = len(list(results_dir.glob(pattern)))
            if count > 0:
                print(f"  {pattern:30s} ({count:2d} files) - {description}")

    # Recommendations
    print("\n" + "=" * 100)
    print("NEXT STEPS")
    print("=" * 100)

    print("""
1. Review consolidated results:
   - Open: CONSOLIDATED_SUMMARY.csv
   - Check: Average Cohen's d, significance rates, patterns

2. Interpret by model:
   - Open: MODEL_SUMMARY.csv
   - Compare models on effect sizes and consistency

3. Visual inspection:
   - Review: distributions_*.png (score distributions)
   - Review: calibration_*.png (prediction quality)
   - Review: feature_bias_*.png (feature selection bias)

4. Statistical interpretation guide:

   IF avg Cohen's d < 0.2 AND < 20% significant:
     → CONCLUSION: No meaningful difference
     → ACTION: Models treat incident/prevalent equally

   IF avg Cohen's d > 0.3 AND > 50% significant:
     → CONCLUSION: Systematic difference exists
     → ACTION: Investigate root cause:
       • Check calibration plots (methodology issue?)
       • Check feature bias (selection bias?)
       • Consider case-stratified reporting

   IF patterns consistent across all models:
     → Likely biological difference (genuine risk variation)

   IF patterns inconsistent across models:
     → Likely methodological artifact (training imbalance)

5. Document findings in project ADRs if needed
6. Update CLAUDE.md with investigation results
""")

    print("=" * 100)
    print("CONSOLIDATION COMPLETE")
    print("=" * 100 + "\n")

    return 0

if __name__ == "__main__":
    sys.exit(main())
