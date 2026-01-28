#!/usr/bin/env python3
"""
Comprehensive Investigation: Prevalent vs Incident Case Scores

Analyzes whether prevalent and incident celiac cases receive different risk scores.
Supports both out-of-fold (OOF) and test set analysis across multiple models.

Three types of analysis:
1. distributions - Score distribution comparison (Mann-Whitney, Cohen's d, power)
2. calibration  - Calibration quality by case type (intercept, slope, Brier score)
3. features     - Feature bias analysis (per-protein discrimination by case type)

Usage:
    # Full analysis (all three)
    python investigate.py --mode oof --model LR_EN --analyses distributions,calibration,features

    # OOF analysis with distributions only (fastest)
    python investigate.py --mode oof --model LR_EN --analyses distributions

    # Test set analysis (requires investigation splits)
    python investigate.py --mode test --model LR_EN

    # All models
    python investigate.py --mode oof --all-models
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
    print("Calibration and feature bias analyses will be unavailable.")
    IMPORTS_OK = False
    calibration_intercept_slope = None
    compute_brier_score = None
    auroc = None

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Styling
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


class InvestigationConfig:
    """Configuration for investigation paths and settings"""

    def __init__(self, base_dir: Path = None):
        if base_dir is None:
            # base_dir is analysis/ directory
            base_dir = Path(__file__).parent.parent.parent

        self.base_dir = base_dir
        # Project root is parent of analysis/
        project_root = base_dir.parent
        self.results_dir = project_root / "results"
        self.data_dir = project_root / "data"
        self.splits_dir = project_root / "splits"
        self.output_dir = project_root / "results" / "investigations"

        # Available models
        self.all_models = ["LR_EN", "RF", "XGBoost", "LinSVM_cal"]

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)


def find_latest_run(results_dir: Path, model: str) -> Optional[str]:
    """Find the most recent run directory for a model"""
    model_dir = results_dir / model
    if not model_dir.exists():
        return None

    run_dirs = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not run_dirs:
        return None

    # Sort by directory name (timestamp-based)
    latest = sorted(run_dirs, key=lambda x: x.name)[-1]
    return latest.name


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


def load_predictions(
    config: InvestigationConfig,
    model: str,
    split_seed: int,
    mode: str,
    run_id: Optional[str] = None
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Load predictions and metadata

    Returns:
        Tuple of (predictions, metadata, full_data) or None if loading fails
    """
    # Find run directory
    if run_id is None:
        run_id = find_latest_run(config.results_dir, model)
        if run_id is None:
            print(f"ERROR: No run directory found for {model}")
            return None

    run_dir = config.results_dir / model / run_id / f"split_seed{split_seed}"

    # Load predictions based on mode
    if mode == "oof":
        preds_path = run_dir / "preds" / "train_oof" / f"train_oof__{model}.csv"
    else:  # test
        preds_path = run_dir / "preds" / "test_preds" / f"test_preds__{model}.csv"

    if not preds_path.exists():
        print(f"ERROR: Predictions not found at {preds_path}")
        return None

    print(f"Loading predictions from: {preds_path.name}")
    preds = pd.read_csv(preds_path)

    # Load metadata
    metadata_path = config.data_dir / "Celiac_dataset_proteomics_w_demo.parquet"
    if not metadata_path.exists():
        print(f"ERROR: Metadata not found at {metadata_path}")
        return None

    metadata = pd.read_parquet(metadata_path)

    # Load indices to map predictions to metadata
    if mode == "oof":
        idx_path = config.splits_dir / f"train_idx_IncidentPlusPrevalent_seed{split_seed}.csv"
    else:
        idx_path = config.splits_dir / f"test_idx_IncidentPlusPrevalent_seed{split_seed}.csv"

    if not idx_path.exists():
        print(f"ERROR: Index file not found at {idx_path}")
        return None

    indices = pd.read_csv(idx_path)['idx'].values
    subset_metadata = metadata.iloc[indices].reset_index(drop=True)

    # Add case type to predictions
    if 'category' in preds.columns:
        # OOF predictions already have category column
        preds['case_type'] = preds['category']
    elif 'idx' in preds.columns:
        # Merge from metadata using idx
        preds = preds.merge(
            metadata[['CeD_comparison']],
            left_on='idx',
            right_index=True,
            how='left'
        )
        preds['case_type'] = preds['CeD_comparison']
    else:
        # Test set - use order
        preds['CeD_comparison'] = subset_metadata['CeD_comparison'].values
        preds['case_type'] = preds['CeD_comparison']

    # Handle probability column (OOF may have repeats)
    if 'y_prob' not in preds.columns:
        prob_cols = [c for c in preds.columns if c.startswith('y_prob')]
        if prob_cols:
            preds['y_prob'] = preds[prob_cols].mean(axis=1)
        else:
            print("ERROR: No probability columns found")
            return None

    return preds, subset_metadata, metadata


def analyze_distributions(
    preds: pd.DataFrame,
    model: str,
    split_seed: int,
    mode: str,
    output_dir: Path
) -> Dict:
    """Analyze score distributions by case type"""

    print("\n" + "=" * 80)
    print(f"DISTRIBUTION ANALYSIS: {model} ({mode.upper()})")
    print("=" * 80)

    # Filter to cases only
    cases = preds[preds['case_type'].isin(['Incident', 'Prevalent'])].copy()

    if len(cases) == 0:
        print("ERROR: No cases found")
        return None

    incident = cases[cases['case_type'] == 'Incident']
    prevalent = cases[cases['case_type'] == 'Prevalent']

    if len(incident) == 0 or len(prevalent) == 0:
        print(f"WARNING: Missing case types (Incident: {len(incident)}, Prevalent: {len(prevalent)})")
        return None

    # Extract scores
    incident_scores = incident['y_prob'].values
    prevalent_scores = prevalent['y_prob'].values

    # Statistical tests
    mw_stat, mw_pval = mannwhitneyu(incident_scores, prevalent_scores, alternative='two-sided')
    ks_stat, ks_pval = ks_2samp(incident_scores, prevalent_scores)
    t_stat, t_pval = ttest_ind(incident_scores, prevalent_scores)

    # Effect size and power
    cohens_d = calculate_cohens_d(incident_scores, prevalent_scores)
    power = calculate_power(len(incident_scores), len(prevalent_scores), cohens_d)

    # Descriptive statistics
    stats = {
        'model': model,
        'mode': mode,
        'split_seed': split_seed,
        'n_incident': len(incident_scores),
        'n_prevalent': len(prevalent_scores),
        'incident_mean': np.mean(incident_scores),
        'incident_median': np.median(incident_scores),
        'incident_sd': np.std(incident_scores),
        'incident_q25': np.percentile(incident_scores, 25),
        'incident_q75': np.percentile(incident_scores, 75),
        'prevalent_mean': np.mean(prevalent_scores),
        'prevalent_median': np.median(prevalent_scores),
        'prevalent_sd': np.std(prevalent_scores),
        'prevalent_q25': np.percentile(prevalent_scores, 25),
        'prevalent_q75': np.percentile(prevalent_scores, 75),
        'median_diff': np.median(incident_scores) - np.median(prevalent_scores),
        'median_pct_diff': ((np.median(incident_scores) - np.median(prevalent_scores)) / np.median(prevalent_scores) * 100) if np.median(prevalent_scores) > 0 else np.nan,
        'mean_diff': np.mean(incident_scores) - np.mean(prevalent_scores),
        'mean_pct_diff': ((np.mean(incident_scores) - np.mean(prevalent_scores)) / np.mean(prevalent_scores) * 100) if np.mean(prevalent_scores) > 0 else np.nan,
        'cohens_d': cohens_d,
        'mw_stat': mw_stat,
        'mw_pval': mw_pval,
        't_stat': t_stat,
        't_pval': t_pval,
        'ks_stat': ks_stat,
        'ks_pval': ks_pval,
        'power': power
    }

    # Print summary
    print(f"\nSample sizes:")
    print(f"  Incident:  n={len(incident_scores)}")
    print(f"  Prevalent: n={len(prevalent_scores)}")

    print(f"\nDescriptive statistics:")
    print(f"  Incident  - Mean: {stats['incident_mean']:.3f}, Median: {stats['incident_median']:.3f}, SD: {stats['incident_sd']:.3f}")
    print(f"  Prevalent - Mean: {stats['prevalent_mean']:.3f}, Median: {stats['prevalent_median']:.3f}, SD: {stats['prevalent_sd']:.3f}")

    print(f"\nDifferences (Incident - Prevalent):")
    print(f"  Median: {stats['median_diff']:+.3f} ({stats['median_pct_diff']:+.1f}%)")
    print(f"  Mean:   {stats['mean_diff']:+.3f} ({stats['mean_pct_diff']:+.1f}%)")

    print(f"\nEffect size:")
    print(f"  Cohen's d: {cohens_d:.3f}")
    magnitude = "negligible" if abs(cohens_d) < 0.2 else "small" if abs(cohens_d) < 0.5 else "medium" if abs(cohens_d) < 0.8 else "large"
    print(f"  Magnitude: {magnitude}")

    print(f"\nStatistical tests:")
    print(f"  Mann-Whitney U: stat={mw_stat:.1f}, p={mw_pval:.4f}")
    print(f"  t-test:         t={t_stat:.3f}, p={t_pval:.4f}")
    print(f"  K-S test:       D={ks_stat:.3f}, p={ks_pval:.4f}")

    print(f"\nPower analysis:")
    print(f"  Observed power: {power:.1%}")
    print(f"  Adequate for medium effects (d≥0.5): {'YES' if power > 0.8 else 'NO'}")

    # Interpretation
    sig_level = "VERY STRONG" if mw_pval < 0.001 else "STRONG" if mw_pval < 0.01 else "MODERATE" if mw_pval < 0.05 else "NO"
    print(f"\nInterpretation:")
    print(f"  {sig_level} evidence of different distributions (p={mw_pval:.4f})")

    direction = "HIGHER" if stats['median_diff'] > 0 else "LOWER"
    print(f"  Incident cases score {direction} than prevalent")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Histogram
    axes[0, 0].hist(incident_scores, bins=30, alpha=0.6, label=f'Incident (n={len(incident_scores)})',
                    density=True, color='#2E86AB', edgecolor='black', linewidth=0.5)
    axes[0, 0].hist(prevalent_scores, bins=30, alpha=0.6, label=f'Prevalent (n={len(prevalent_scores)})',
                    density=True, color='#A23B72', edgecolor='black', linewidth=0.5)
    axes[0, 0].axvline(np.median(incident_scores), color='#2E86AB', linestyle='--', linewidth=2)
    axes[0, 0].axvline(np.median(prevalent_scores), color='#A23B72', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Risk Score', fontweight='bold')
    axes[0, 0].set_ylabel('Density', fontweight='bold')
    axes[0, 0].set_title('Score Distributions', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Box plot
    data_box = pd.DataFrame({
        'Risk Score': np.concatenate([incident_scores, prevalent_scores]),
        'Case Type': ['Incident'] * len(incident_scores) + ['Prevalent'] * len(prevalent_scores)
    })
    sns.boxplot(data=data_box, x='Case Type', y='Risk Score', ax=axes[0, 1],
                palette={'Incident': '#2E86AB', 'Prevalent': '#A23B72'})
    axes[0, 1].set_title(f'Box Plot\np={mw_pval:.4f}', fontweight='bold')
    axes[0, 1].grid(alpha=0.3, axis='y')

    # Violin plot
    sns.violinplot(data=data_box, x='Case Type', y='Risk Score', ax=axes[1, 0],
                   palette={'Incident': '#2E86AB', 'Prevalent': '#A23B72'})
    axes[1, 0].set_title(f'Violin Plot\nCohen\'s d={cohens_d:.2f}', fontweight='bold')
    axes[1, 0].grid(alpha=0.3, axis='y')

    # ECDF
    incident_sorted = np.sort(incident_scores)
    prevalent_sorted = np.sort(prevalent_scores)
    axes[1, 1].plot(incident_sorted, np.arange(1, len(incident_sorted) + 1) / len(incident_sorted),
                   label='Incident', color='#2E86AB', linewidth=2)
    axes[1, 1].plot(prevalent_sorted, np.arange(1, len(prevalent_sorted) + 1) / len(prevalent_sorted),
                   label='Prevalent', color='#A23B72', linewidth=2)
    axes[1, 1].set_xlabel('Risk Score', fontweight='bold')
    axes[1, 1].set_ylabel('Cumulative Probability', fontweight='bold')
    axes[1, 1].set_title(f'ECDF\nKS D={ks_stat:.3f}', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.suptitle(f'{model} - {mode.upper()} Predictions\nIncident vs Prevalent Case Scores',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    plot_path = output_dir / f"distributions_{model}_{mode}_seed{split_seed}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved: {plot_path.name}")
    plt.close()

    # Save scores for further analysis
    scores_df = pd.DataFrame({
        'case_type': cases['case_type'].values,
        'risk_score': cases['y_prob'].values,
        'y_true': cases['y_true'].values if 'y_true' in cases.columns else np.ones(len(cases))
    })
    scores_path = output_dir / f"scores_{model}_{mode}_seed{split_seed}.csv"
    scores_df.to_csv(scores_path, index=False)

    return stats


def analyze_calibration(
    preds: pd.DataFrame,
    model: str,
    split_seed: int,
    mode: str,
    output_dir: Path
) -> Dict:
    """Analyze calibration separately for incident vs prevalent cases"""

    print("\n" + "=" * 80)
    print(f"CALIBRATION ANALYSIS: {model} ({mode.upper()})")
    print("=" * 80)

    # Filter to cases and controls
    cases = preds[preds['case_type'].isin(['Incident', 'Prevalent'])].copy()
    controls = preds[preds['case_type'] == 'Controls'].copy()

    if len(cases) == 0:
        print("ERROR: No cases found")
        return None

    incident = cases[cases['case_type'] == 'Incident']
    prevalent = cases[cases['case_type'] == 'Prevalent']

    if len(incident) == 0 or len(prevalent) == 0:
        print(f"WARNING: Missing case types (Incident: {len(incident)}, Prevalent: {len(prevalent)})")
        return None

    # Check for required modules
    if not IMPORTS_OK or calibration_intercept_slope is None:
        print("ERROR: Required calibration functions not available")
        print("Please ensure ced_ml package is installed (pip install -e .)")
        return None

    # For calibration, we need both classes (cases=1, controls=0)
    # Combine incident/prevalent with controls separately
    incident_with_controls = pd.concat([incident, controls])
    prevalent_with_controls = pd.concat([prevalent, controls])

    incident_y = np.concatenate([np.ones(len(incident)), np.zeros(len(controls))])
    prevalent_y = np.concatenate([np.ones(len(prevalent)), np.zeros(len(controls))])

    # Calculate calibration metrics
    incident_scores = incident_with_controls['y_prob'].values
    prevalent_scores = prevalent_with_controls['y_prob'].values

    # Overall metrics
    try:
        inc_intercept, inc_slope = calibration_intercept_slope(incident_y, incident_scores)
        prev_intercept, prev_slope = calibration_intercept_slope(prevalent_y, prevalent_scores)

        # Compute Brier scores (cases only for comparison)
        inc_brier = compute_brier_score(incident_y, incident_scores)
        prev_brier = compute_brier_score(prevalent_y, prevalent_scores)
    except Exception as e:
        print(f"ERROR calculating calibration metrics: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Calibration curves (with controls)
    n_bins = 10
    try:
        inc_prob_true, inc_prob_pred = calibration_curve(
            incident_y, incident_scores, n_bins=n_bins, strategy='quantile'
        )
        prev_prob_true, prev_prob_pred = calibration_curve(
            prevalent_y, prevalent_scores, n_bins=n_bins, strategy='quantile'
        )
    except Exception as e:
        print(f"WARNING: Could not compute calibration curves: {e}")
        inc_prob_true = inc_prob_pred = None
        prev_prob_true = prev_prob_pred = None

    stats = {
        'model': model,
        'mode': mode,
        'split_seed': split_seed,
        'n_incident': len(incident_scores),
        'n_prevalent': len(prevalent_scores),
        'incident_intercept': inc_intercept,
        'incident_slope': inc_slope,
        'incident_brier': inc_brier,
        'prevalent_intercept': prev_intercept,
        'prevalent_slope': prev_slope,
        'prevalent_brier': prev_brier,
        'intercept_diff': inc_intercept - prev_intercept,
        'slope_diff': inc_slope - prev_slope,
        'brier_diff': inc_brier - prev_brier
    }

    # Print summary
    print(f"\nSample sizes:")
    print(f"  Incident:  n={len(incident_scores)}")
    print(f"  Prevalent: n={len(prevalent_scores)}")

    print(f"\nCalibration Metrics:")
    print(f"\n  Incident:")
    print(f"    Intercept: {inc_intercept:+.3f} (optimal: ~0)")
    print(f"    Slope:     {inc_slope:.3f} (optimal: ~1)")
    print(f"    Brier:     {inc_brier:.4f} (lower better)")

    print(f"\n  Prevalent:")
    print(f"    Intercept: {prev_intercept:+.3f} (optimal: ~0)")
    print(f"    Slope:     {prev_slope:.3f} (optimal: ~1)")
    print(f"    Brier:     {prev_brier:.4f} (lower better)")

    print(f"\n  Differences (Incident - Prevalent):")
    print(f"    Intercept: {stats['intercept_diff']:+.3f}")
    print(f"    Slope:     {stats['slope_diff']:+.3f}")
    print(f"    Brier:     {stats['brier_diff']:+.4f}")

    # Interpretation
    print(f"\nInterpretation:")

    # Intercept interpretation
    if abs(inc_intercept) < 0.1 and abs(prev_intercept) < 0.1:
        print("  Both groups well-calibrated (intercepts near 0)")
    elif abs(prev_intercept) > 0.15:
        direction = "over" if prev_intercept > 0 else "under"
        print(f"  Prevalent cases show systematic {direction}prediction (intercept={prev_intercept:+.2f})")

    # Slope interpretation
    if 0.9 <= inc_slope <= 1.1 and 0.9 <= prev_slope <= 1.1:
        print("  Both groups show good discrimination-calibration relationship")
    elif prev_slope < 0.85:
        print(f"  Prevalent cases show UNDERCONFIDENT predictions (slope={prev_slope:.2f})")
        print("  Model predicts too conservatively for prevalent cases")
    elif prev_slope > 1.15:
        print(f"  Prevalent cases show OVERCONFIDENT predictions (slope={prev_slope:.2f})")
        print("  Model predicts too aggressively for prevalent cases")

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Calibration curves
    if inc_prob_true is not None and prev_prob_true is not None:
        axes[0].plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
        axes[0].plot(inc_prob_pred, inc_prob_true, 'o-',
                    label=f'Incident (slope={inc_slope:.2f})',
                    color='#2E86AB', linewidth=2, markersize=8)
        axes[0].plot(prev_prob_pred, prev_prob_true, 's-',
                    label=f'Prevalent (slope={prev_slope:.2f})',
                    color='#A23B72', linewidth=2, markersize=8)
        axes[0].set_xlabel('Mean Predicted Probability', fontweight='bold')
        axes[0].set_ylabel('Observed Proportion', fontweight='bold')
        axes[0].set_title('Calibration Curves', fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[0].set_xlim([0, 1])
        axes[0].set_ylim([0, 1])

    # Slope comparison
    slopes = [inc_slope, prev_slope]
    colors = ['#2E86AB', '#A23B72']
    labels = ['Incident', 'Prevalent']
    bars = axes[1].bar(labels, slopes, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[1].axhline(1.0, color='green', linestyle='--', linewidth=2, label='Optimal')
    axes[1].set_ylabel('Calibration Slope', fontweight='bold')
    axes[1].set_title('Calibration Slope\n(1.0 = well-calibrated)', fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis='y')
    axes[1].set_ylim([0, max(1.5, max(slopes) * 1.1)])

    # Add value labels on bars
    for bar, val in zip(bars, slopes):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, height,
                    f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

    # Brier score comparison
    briers = [inc_brier, prev_brier]
    bars = axes[2].bar(labels, briers, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[2].set_ylabel('Brier Score', fontweight='bold')
    axes[2].set_title('Brier Score\n(lower = better)', fontweight='bold')
    axes[2].grid(alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, briers):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2, height,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.suptitle(f'{model} - {mode.upper()} Predictions\nCalibration by Case Type',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    plot_path = output_dir / f"calibration_{model}_{mode}_seed{split_seed}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved: {plot_path.name}")
    plt.close()

    return stats


def analyze_feature_bias(
    preds: pd.DataFrame,
    metadata: pd.DataFrame,
    model: str,
    split_seed: int,
    mode: str,
    output_dir: Path,
    config: InvestigationConfig
) -> Dict:
    """Analyze whether features discriminate differently for incident vs prevalent"""

    print("\n" + "=" * 80)
    print(f"FEATURE BIAS ANALYSIS: {model} ({mode.upper()})")
    print("=" * 80)

    # Check for required modules
    if not IMPORTS_OK or auroc is None:
        print("ERROR: auroc function not available")
        print("Please ensure ced_ml package is installed (pip install -e .)")
        return None

    # Load feature importance or selected features
    run_dir = config.results_dir / model / find_latest_run(config.results_dir, model) / f"split_seed{split_seed}"

    # Try multiple possible locations for feature list
    feature_paths = [
        run_dir / "selected_features.csv",
        run_dir / "reports" / "feature_reports" / f"{model}__feature_report_train.csv",
    ]

    selected_features = None
    for feature_path in feature_paths:
        if feature_path.exists():
            try:
                selected_features = pd.read_csv(feature_path)
                print(f"Loaded features from: {feature_path.name}")
                break
            except Exception as e:
                print(f"WARNING: Could not load {feature_path}: {e}")
                continue

    if selected_features is None:
        print(f"WARNING: Could not find feature list in any of:")
        for p in feature_paths:
            print(f"  - {p}")
        print("Feature bias analysis requires selected features from training")
        return None

    # Handle different column names
    if 'feature' in selected_features.columns:
        feature_names = selected_features['feature'].tolist()
    elif 'Feature' in selected_features.columns:
        feature_names = selected_features['Feature'].tolist()
    elif 'protein' in selected_features.columns:
        feature_names = selected_features['protein'].tolist()
    else:
        print(f"ERROR: No 'feature', 'Feature', or 'protein' column found. Columns: {selected_features.columns.tolist()}")
        return None
    print(f"Analyzing {len(feature_names)} selected features")

    # Get protein columns from metadata
    protein_cols = [c for c in metadata.columns if c.endswith('_resid')]
    available_features = [f for f in feature_names if f in protein_cols]

    if len(available_features) == 0:
        print("ERROR: No protein features found in metadata")
        return None

    print(f"Found {len(available_features)} features in metadata")

    # Separate incident, prevalent, and controls
    incident = preds[preds['case_type'] == 'Incident'].copy()
    prevalent = preds[preds['case_type'] == 'Prevalent'].copy()
    controls = preds[preds['case_type'] == 'Controls'].copy()  # Note: 'Controls' with 's'

    if len(incident) == 0 or len(prevalent) == 0:
        print(f"WARNING: Missing case types (Incident: {len(incident)}, Prevalent: {len(prevalent)})")
        return None

    if len(controls) == 0:
        print("WARNING: No controls found, cannot compute feature discrimination")
        return None

    # Map to metadata indices
    if 'idx' in preds.columns:
        incident_idx = incident['idx'].values
        prevalent_idx = prevalent['idx'].values
        control_idx = controls['idx'].values
    else:
        print("ERROR: 'idx' column not found in predictions")
        return None

    # Calculate per-feature AUROC for incident vs control and prevalent vs control
    feature_results = []

    print("\nCalculating per-feature discrimination...")
    for feat in available_features[:50]:  # Limit to top 50 to avoid long runtime
        try:
            # Get feature values
            inc_vals = metadata.loc[incident_idx, feat].values
            prev_vals = metadata.loc[prevalent_idx, feat].values
            ctrl_vals = metadata.loc[control_idx, feat].values

            # Skip if missing data
            if np.any(np.isnan(inc_vals)) or np.any(np.isnan(prev_vals)) or np.any(np.isnan(ctrl_vals)):
                continue

            # Incident vs Control AUROC
            inc_y = np.concatenate([np.ones(len(inc_vals)), np.zeros(len(ctrl_vals))])
            inc_X = np.concatenate([inc_vals, ctrl_vals])
            inc_auc = auroc(inc_y, inc_X)

            # Prevalent vs Control AUROC
            prev_y = np.concatenate([np.ones(len(prev_vals)), np.zeros(len(ctrl_vals))])
            prev_X = np.concatenate([prev_vals, ctrl_vals])
            prev_auc = auroc(prev_y, prev_X)

            # Bias score (positive = incident-biased, negative = prevalent-biased)
            bias_score = inc_auc - prev_auc

            feature_results.append({
                'feature': feat,
                'incident_auc': inc_auc,
                'prevalent_auc': prev_auc,
                'bias_score': bias_score,
                'abs_bias': abs(bias_score)
            })
        except Exception as e:
            print(f"  WARNING: Could not process {feat}: {e}")
            continue

    if len(feature_results) == 0:
        print("ERROR: No features could be analyzed")
        return None

    feat_df = pd.DataFrame(feature_results).sort_values('abs_bias', ascending=False)

    # Summary statistics
    n_incident_biased = sum(feat_df['bias_score'] > 0.05)
    n_prevalent_biased = sum(feat_df['bias_score'] < -0.05)
    n_balanced = len(feat_df) - n_incident_biased - n_prevalent_biased

    mean_inc_auc = feat_df['incident_auc'].mean()
    mean_prev_auc = feat_df['prevalent_auc'].mean()
    mean_bias = feat_df['bias_score'].mean()

    stats = {
        'model': model,
        'mode': mode,
        'split_seed': split_seed,
        'n_features': len(feat_df),
        'n_incident_biased': n_incident_biased,
        'n_prevalent_biased': n_prevalent_biased,
        'n_balanced': n_balanced,
        'pct_incident_biased': n_incident_biased / len(feat_df) * 100,
        'pct_prevalent_biased': n_prevalent_biased / len(feat_df) * 100,
        'mean_incident_auc': mean_inc_auc,
        'mean_prevalent_auc': mean_prev_auc,
        'mean_bias_score': mean_bias
    }

    # Print summary
    print(f"\nFeature Bias Summary:")
    print(f"  Total features analyzed: {len(feat_df)}")
    print(f"  Incident-biased (bias > 0.05):  {n_incident_biased} ({stats['pct_incident_biased']:.1f}%)")
    print(f"  Prevalent-biased (bias < -0.05): {n_prevalent_biased} ({stats['pct_prevalent_biased']:.1f}%)")
    print(f"  Balanced (|bias| ≤ 0.05):        {n_balanced} ({n_balanced/len(feat_df)*100:.1f}%)")

    print(f"\nMean discrimination:")
    print(f"  Incident vs Control:  AUROC = {mean_inc_auc:.3f}")
    print(f"  Prevalent vs Control: AUROC = {mean_prev_auc:.3f}")
    print(f"  Mean bias score:      {mean_bias:+.3f}")

    print(f"\nTop 10 most biased features:")
    print(feat_df.head(10)[['feature', 'incident_auc', 'prevalent_auc', 'bias_score']].to_string(index=False))

    # Interpretation
    print(f"\nInterpretation:")
    if stats['pct_incident_biased'] > 60:
        print(f"  STRONG INCIDENT BIAS: {stats['pct_incident_biased']:.0f}% of features favor incident discrimination")
        print("  Feature selection likely biased toward incident cases")
    elif stats['pct_prevalent_biased'] > 60:
        print(f"  STRONG PREVALENT BIAS: {stats['pct_prevalent_biased']:.0f}% of features favor prevalent discrimination")
        print("  Feature selection likely biased toward prevalent cases")
    else:
        print("  BALANCED: Features discriminate similarly for both case types")

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Scatter: Incident AUC vs Prevalent AUC
    axes[0].scatter(feat_df['incident_auc'], feat_df['prevalent_auc'],
                   alpha=0.6, s=50, color='#2E86AB')
    axes[0].plot([0.5, 1], [0.5, 1], 'k--', label='Equal discrimination', linewidth=2)
    axes[0].set_xlabel('Incident vs Control AUROC', fontweight='bold')
    axes[0].set_ylabel('Prevalent vs Control AUROC', fontweight='bold')
    axes[0].set_title('Feature Discrimination by Case Type', fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_xlim([0.5, 1.0])
    axes[0].set_ylim([0.5, 1.0])

    # Histogram of bias scores
    axes[1].hist(feat_df['bias_score'], bins=30, alpha=0.7,
                color='#2E86AB', edgecolor='black', linewidth=1)
    axes[1].axvline(0, color='green', linestyle='--', linewidth=2, label='No bias')
    axes[1].axvline(mean_bias, color='red', linestyle='-', linewidth=2, label=f'Mean={mean_bias:.3f}')
    axes[1].set_xlabel('Bias Score (Inc AUC - Prev AUC)', fontweight='bold')
    axes[1].set_ylabel('Number of Features', fontweight='bold')
    axes[1].set_title('Distribution of Feature Bias', fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis='y')

    # Bar plot: bias category counts
    categories = ['Incident-biased\n(>0.05)', 'Balanced\n(±0.05)', 'Prevalent-biased\n(<-0.05)']
    counts = [n_incident_biased, n_balanced, n_prevalent_biased]
    colors_bar = ['#2E86AB', '#95C623', '#A23B72']
    bars = axes[2].bar(categories, counts, color=colors_bar, alpha=0.7,
                      edgecolor='black', linewidth=2)
    axes[2].set_ylabel('Number of Features', fontweight='bold')
    axes[2].set_title('Feature Bias Categories', fontweight='bold')
    axes[2].grid(alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, counts):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2, height,
                    f'{val}\n({val/len(feat_df)*100:.1f}%)',
                    ha='center', va='bottom', fontweight='bold')

    plt.suptitle(f'{model} - {mode.upper()} Predictions\nFeature Bias Analysis',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    plot_path = output_dir / f"feature_bias_{model}_{mode}_seed{split_seed}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved: {plot_path.name}")
    plt.close()

    # Save detailed results
    feat_path = output_dir / f"feature_bias_details_{model}_{mode}_seed{split_seed}.csv"
    feat_df.to_csv(feat_path, index=False)
    print(f"Details saved: {feat_path.name}")

    return stats


def run_investigation(
    models: List[str],
    split_seed: int,
    mode: str,
    analyses: List[str],
    run_id: Optional[str] = None
):
    """Run comprehensive investigation"""

    config = InvestigationConfig()

    print("=" * 80)
    print("PREVALENT vs INCIDENT CASE SCORE INVESTIGATION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Models: {', '.join(models)}")
    print(f"  Split seed: {split_seed}")
    print(f"  Mode: {mode.upper()}")
    print(f"  Analyses: {', '.join(analyses)}")
    print(f"  Output: {config.output_dir}")

    all_results = []

    for model in models:
        print(f"\n{'=' * 80}")
        print(f"ANALYZING: {model}")
        print(f"{'=' * 80}")

        # Load data
        data = load_predictions(config, model, split_seed, mode, run_id)
        if data is None:
            print(f"Skipping {model} due to loading errors")
            continue

        preds, subset_metadata, full_metadata = data

        # Run requested analyses
        if 'distributions' in analyses:
            result = analyze_distributions(preds, model, split_seed, mode, config.output_dir)
            if result:
                all_results.append(result)

        if 'calibration' in analyses:
            result = analyze_calibration(preds, model, split_seed, mode, config.output_dir)
            if result:
                all_results.append(result)

        if 'features' in analyses:
            result = analyze_feature_bias(
                preds, full_metadata, model, split_seed, mode, config.output_dir, config
            )
            if result:
                all_results.append(result)

    # Create summary
    if all_results:
        summary = pd.DataFrame(all_results)
        summary_path = config.output_dir / f"summary_{mode}_seed{split_seed}.csv"
        summary.to_csv(summary_path, index=False)

        print("\n" + "=" * 80)
        print("SUMMARY TABLE")
        print("=" * 80)
        print(summary.to_string(index=False))
        print(f"\nSummary saved: {summary_path.name}")

        # Overall conclusions
        print("\n" + "=" * 80)
        print("OVERALL CONCLUSIONS")
        print("=" * 80)

        median_diffs = summary['median_diff'].values
        avg_median_diff = np.mean(median_diffs)
        avg_pct_diff = np.mean(summary['median_pct_diff'].values)
        sig_count = sum(summary['mw_pval'] < 0.05)
        avg_cohens_d = np.mean(summary['cohens_d'].values)
        avg_power = np.mean(summary['power'].values)

        print(f"\nAcross {len(all_results)} models:")
        print(f"  Significant differences: {sig_count}/{len(all_results)} (p<0.05)")
        print(f"  Average median difference: {avg_median_diff:+.3f} ({avg_pct_diff:+.1f}%)")
        print(f"  Average Cohen's d: {avg_cohens_d:.3f}")
        print(f"  Average power: {avg_power:.1%}")

        if all(d > 0 for d in median_diffs):
            print("\n  CONSISTENT PATTERN: Incident cases score HIGHER across all models")
            print("  This is UNEXPECTED and warrants investigation")
        elif all(d < 0 for d in median_diffs):
            print("\n  CONSISTENT PATTERN: Prevalent cases score HIGHER across all models")
            print("  This aligns with clinical expectations")
        else:
            print("\n  MIXED RESULTS: Direction varies by model")

    print("\n" + "=" * 80)
    print("INVESTIGATION COMPLETE")
    print("=" * 80)
    print(f"\nOutputs saved to: {config.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Investigate prevalent vs incident case score differences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # OOF analysis for single model (fastest)
  python investigate.py --mode oof --model LR_EN

  # Test set analysis (requires investigation splits)
  python investigate.py --mode test --model LR_EN

  # All models
  python investigate.py --mode oof --all-models

  # Specific analyses only
  python investigate.py --mode oof --model LR_EN --analyses distributions
        """
    )

    parser.add_argument(
        '--mode',
        choices=['oof', 'test'],
        default='oof',
        help='Analysis mode: oof (out-of-fold) or test (test set)'
    )

    parser.add_argument(
        '--model',
        help='Model to analyze (e.g., LR_EN, RF, XGBoost, LinSVM_cal)'
    )

    parser.add_argument(
        '--all-models',
        action='store_true',
        help='Analyze all available models'
    )

    parser.add_argument(
        '--split-seed',
        type=int,
        default=0,
        help='Split seed to analyze (default: 0)'
    )

    parser.add_argument(
        '--analyses',
        default='distributions,calibration,features',
        help='Comma-separated list of analyses: distributions,calibration,features (default: all three)'
    )

    parser.add_argument(
        '--run-id',
        help='Specific run ID to use (default: latest)'
    )

    args = parser.parse_args()

    # Validate inputs
    config = InvestigationConfig()

    if args.all_models:
        models = config.all_models
    elif args.model:
        models = [args.model]
    else:
        parser.error("Must specify --model or --all-models")

    analyses = [a.strip() for a in args.analyses.split(',')]
    valid_analyses = {'distributions', 'calibration', 'features'}
    invalid = set(analyses) - valid_analyses
    if invalid:
        parser.error(f"Invalid analyses: {invalid}. Choose from: {valid_analyses}")

    # Run investigation
    run_investigation(
        models=models,
        split_seed=args.split_seed,
        mode=args.mode,
        analyses=analyses,
        run_id=args.run_id
    )


if __name__ == "__main__":
    main()
