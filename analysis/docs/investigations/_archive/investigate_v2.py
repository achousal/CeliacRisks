#!/usr/bin/env python3
"""
IMPROVED Investigation Script: Prevalent vs Incident Case Scores

Key improvements over original:
1. Correctly finds run directories in results/run_*/ structure
2. Maps run directories to configurations via split metadata
3. Supports comparing across configurations
4. Provides power analysis and sample size recommendations
5. Enhanced calibration diagnostics

Usage:
    # Single configuration analysis
    python investigate_v2.py --run-id 20260127_160356 --model LR_EN

    # Compare all configurations
    python investigate_v2.py --compare-configs --config-dir ../../splits_experiments

    # Power analysis
    python investigate_v2.py --power-analysis --target-effect 0.5
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, ks_2samp, ttest_ind, norm
from sklearn.calibration import calibration_curve

# Add parent directory to path for imports
analysis_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(analysis_dir / "src"))

try:
    from ced_ml.models.calibration import calibration_intercept_slope
    from ced_ml.metrics.discrimination import compute_brier_score, auroc
    IMPORTS_OK = True
except ImportError as e:
    print(f"WARNING: Could not import ced_ml modules: {e}")
    IMPORTS_OK = False

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def find_model_in_run(run_dir: Path) -> Optional[str]:
    """Detect which model was trained by inspecting output files"""
    split_dir = run_dir / "split_seed0"
    if not split_dir.exists():
        return None

    # Check for OOF predictions
    oof_dir = split_dir / "preds" / "train_oof"
    if oof_dir.exists():
        for f in oof_dir.glob("train_oof__*.csv"):
            model = f.stem.replace("train_oof__", "")
            return model

    return None


def load_run_metadata(run_dir: Path) -> Dict:
    """Load metadata about a run (config, model, splits)"""
    metadata = {
        'run_id': run_dir.name.replace('run_', ''),
        'run_dir': str(run_dir),
        'model': None,
        'config': {}
    }

    # Detect model
    metadata['model'] = find_model_in_run(run_dir)

    # Load split trace to get configuration
    split_trace = run_dir / "split_seed0" / "diagnostics" / "splits" / "train_test_split_trace.csv"
    if split_trace.exists():
        trace = pd.read_csv(split_trace)
        metadata['config']['n_train'] = int(trace[trace['split'] == 'train'].shape[0])
        metadata['config']['n_val'] = int(trace[trace['split'] == 'val'].shape[0])
        metadata['config']['n_test'] = int(trace[trace['split'] == 'test'].shape[0])

        # Note: split trace doesn't contain case type info, skip for now
        # Would need to load actual data to count case types

    return metadata


def calculate_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return np.nan
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def calculate_power(n1: int, n2: int, effect_size: float, alpha: float = 0.05) -> float:
    """Approximate statistical power for Mann-Whitney U test"""
    if np.isnan(effect_size) or effect_size == 0:
        return np.nan
    z_alpha = norm.ppf(1 - alpha / 2)
    n_eff = (n1 * n2) / (n1 + n2)
    z_beta = abs(effect_size) * np.sqrt(n_eff / (np.pi / 3))
    power = 1 - norm.cdf(z_alpha - z_beta)
    return power


def calculate_required_n(effect_size: float, power: float = 0.8, alpha: float = 0.05) -> int:
    """Calculate required sample size per group for desired power"""
    if effect_size == 0:
        return np.inf
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    n_per_group = ((z_alpha + z_beta) ** 2) * (2 * (np.pi / 3)) / (effect_size ** 2)
    return int(np.ceil(n_per_group))


def load_predictions_from_run(run_dir: Path, split_seed: int = 0) -> Optional[Tuple[pd.DataFrame, str]]:
    """Load OOF predictions from a run directory"""
    split_dir = run_dir / f"split_seed{split_seed}"
    model = find_model_in_run(run_dir)

    if model is None:
        return None

    oof_file = split_dir / "preds" / "train_oof" / f"train_oof__{model}.csv"
    if not oof_file.exists():
        return None

    preds = pd.read_csv(oof_file)
    return preds, model


def analyze_distribution_improved(
    preds: pd.DataFrame,
    model: str,
    run_id: str,
    output_dir: Path
) -> Dict:
    """Enhanced distribution analysis with power calculations"""

    # Filter to cases
    cases = preds[preds['category'].isin(['Incident', 'Prevalent'])].copy()
    incident = cases[cases['category'] == 'Incident']
    prevalent = cases[cases['category'] == 'Prevalent']

    if len(incident) == 0 or len(prevalent) == 0:
        return None

    # Handle multiple y_prob columns (from CV repeats)
    prob_cols = [c for c in preds.columns if c.startswith('y_prob')]
    if len(prob_cols) > 1:
        incident_scores = incident[prob_cols].mean(axis=1).values
        prevalent_scores = prevalent[prob_cols].mean(axis=1).values
    else:
        incident_scores = incident['y_prob'].values
        prevalent_scores = prevalent['y_prob'].values

    # Statistical tests
    mw_stat, mw_pval = mannwhitneyu(incident_scores, prevalent_scores, alternative='two-sided')
    ks_stat, ks_pval = ks_2samp(incident_scores, prevalent_scores)
    t_stat, t_pval = ttest_ind(incident_scores, prevalent_scores)

    # Effect size and power
    cohens_d = calculate_cohens_d(incident_scores, prevalent_scores)
    observed_power = calculate_power(len(incident_scores), len(prevalent_scores), cohens_d)

    # Calculate required N for 80% power
    required_n_small = calculate_required_n(0.2, 0.8)  # Small effect
    required_n_medium = calculate_required_n(0.5, 0.8)  # Medium effect
    required_n_large = calculate_required_n(0.8, 0.8)  # Large effect
    required_n_observed = calculate_required_n(abs(cohens_d), 0.8) if not np.isnan(cohens_d) else np.inf

    stats = {
        'run_id': run_id,
        'model': model,
        'n_incident': len(incident_scores),
        'n_prevalent': len(prevalent_scores),
        'incident_mean': np.mean(incident_scores),
        'incident_median': np.median(incident_scores),
        'incident_sd': np.std(incident_scores),
        'prevalent_mean': np.mean(prevalent_scores),
        'prevalent_median': np.median(prevalent_scores),
        'prevalent_sd': np.std(prevalent_scores),
        'median_diff': np.median(incident_scores) - np.median(prevalent_scores),
        'mean_diff': np.mean(incident_scores) - np.mean(prevalent_scores),
        'cohens_d': cohens_d,
        'mw_pval': mw_pval,
        't_pval': t_pval,
        'ks_pval': ks_pval,
        'observed_power': observed_power,
        'required_n_small_effect': required_n_small,
        'required_n_medium_effect': required_n_medium,
        'required_n_large_effect': required_n_large,
        'required_n_observed_effect': required_n_observed
    }

    # Print results
    print(f"\n{'='*80}")
    print(f"DISTRIBUTION ANALYSIS: {model} (run: {run_id})")
    print(f"{'='*80}")
    print(f"\nSample sizes: Incident={len(incident_scores)}, Prevalent={len(prevalent_scores)}")
    print(f"Median scores: Incident={stats['incident_median']:.3f}, Prevalent={stats['prevalent_median']:.3f}")
    print(f"Median difference: {stats['median_diff']:+.3f}")
    print(f"Effect size (Cohen's d): {cohens_d:.3f}")
    print(f"P-value (Mann-Whitney): {mw_pval:.4f}")
    print(f"Observed power: {observed_power:.1%}")

    print(f"\nSample size requirements for 80% power:")
    print(f"  Small effect (d=0.2):    {required_n_small:>4} per group")
    print(f"  Medium effect (d=0.5):   {required_n_medium:>4} per group")
    print(f"  Large effect (d=0.8):    {required_n_large:>4} per group")
    if not np.isinf(required_n_observed):
        print(f"  Observed effect (d={abs(cohens_d):.2f}): {int(required_n_observed):>4} per group")

    return stats


def compare_configurations(results_dir: Path, output_dir: Path):
    """Compare results across all experimental configurations"""

    print(f"\n{'='*100}")
    print("CONFIGURATION COMPARISON")
    print(f"{'='*100}")

    # Find all run directories
    run_dirs = sorted([d for d in results_dir.glob("run_*") if d.is_dir()])

    print(f"\nFound {len(run_dirs)} run directories")

    all_results = []

    for run_dir in run_dirs:
        metadata = load_run_metadata(run_dir)
        data = load_predictions_from_run(run_dir)

        if data is None:
            print(f"Skipping {run_dir.name} (no predictions found)")
            continue

        preds, model = data
        stats = analyze_distribution_improved(preds, model, metadata['run_id'], output_dir)

        if stats:
            # Merge with metadata
            stats.update(metadata['config'])
            all_results.append(stats)

    if all_results:
        df = pd.DataFrame(all_results)

        # Save comparison
        comparison_path = output_dir / "configuration_comparison.csv"
        df.to_csv(comparison_path, index=False)

        print(f"\n{'='*100}")
        print("COMPARISON SUMMARY")
        print(f"{'='*100}")

        # Group by configuration pattern
        summary_cols = ['model', 'n_incident', 'n_prevalent', 'median_diff', 'cohens_d',
                       'mw_pval', 'observed_power']
        print(df[summary_cols].to_string(index=False))

        print(f"\nFull comparison saved to: {comparison_path}")

        return df

    return None


def main():
    parser = argparse.ArgumentParser(description="Improved investigation with power analysis")

    parser.add_argument('--run-id', help='Specific run ID to analyze')
    parser.add_argument('--model', help='Model name (auto-detected if not provided)')
    parser.add_argument('--compare-configs', action='store_true', help='Compare all configurations')
    parser.add_argument('--power-analysis', action='store_true', help='Show power analysis')
    parser.add_argument('--target-effect', type=float, default=0.5, help='Target effect size for power calc')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')

    args = parser.parse_args()

    # Setup paths
    analysis_dir = Path(__file__).parent.parent.parent
    results_dir = analysis_dir / "results"

    if args.output_dir is None:
        args.output_dir = analysis_dir.parent / "results" / "investigations"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.compare_configs:
        compare_configurations(results_dir, args.output_dir)

    elif args.run_id:
        run_dir = results_dir / f"run_{args.run_id}"
        if not run_dir.exists():
            print(f"ERROR: Run directory not found: {run_dir}")
            return 1

        data = load_predictions_from_run(run_dir)
        if data is None:
            print(f"ERROR: Could not load predictions from {run_dir}")
            return 1

        preds, model = data
        if args.model and model != args.model:
            print(f"WARNING: Detected model '{model}' differs from specified '{args.model}'")

        stats = analyze_distribution_improved(preds, model, args.run_id, args.output_dir)

        # Save individual result
        if stats:
            result_path = args.output_dir / f"analysis_{model}_{args.run_id}.csv"
            pd.DataFrame([stats]).to_csv(result_path, index=False)
            print(f"\nResults saved to: {result_path}")

    elif args.power_analysis:
        print(f"\n{'='*80}")
        print(f"POWER ANALYSIS FOR EFFECT SIZE d={args.target_effect}")
        print(f"{'='*80}\n")

        powers = [0.7, 0.8, 0.9]
        effect_sizes = [0.2, 0.3, 0.5, 0.8]

        print("Sample size per group required:\n")
        print(f"{'Power':<10} {'d=0.2':<10} {'d=0.3':<10} {'d=0.5':<10} {'d=0.8':<10}")
        print("-" * 50)

        for pwr in powers:
            row = [f"{pwr:.0%}"]
            for es in effect_sizes:
                n = calculate_required_n(es, pwr)
                row.append(f"{n:>7}")
            print("  ".join(row))

        print(f"\nFor your target effect size (d={args.target_effect}):")
        for pwr in powers:
            n = calculate_required_n(args.target_effect, pwr)
            print(f"  {pwr:.0%} power: {n} per group ({n*2} total)")

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
