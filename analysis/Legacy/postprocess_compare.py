#!/usr/bin/env python3
"""
postprocess_compare.py

Aggregates ML results across models or sensitivity analysis runs.

MODES:
  --mode models (default):
    Reads per-(scenario,model) run dirs under results_dir, writes:
      results_dir/COMBINED/core/cv_summary_ranked.csv
      results_dir/COMBINED/core/final_summary_with_test.csv
      results_dir/COMBINED/core/BEST_by_trainCV_PR.csv
      results_dir/COMBINED/core/{scenario}__test_model_comparisons_bootstrap.csv
      results_dir/COMBINED/core/{scenario}__decision_curve__TEST.csv

  --mode sensitivity:
    Aggregates Mann-Whitney sensitivity analysis results across screen_top_n values.
    Expects directory structure: results_dir/screen{N}/model_dirs/...
    Writes:
      results_dir/screen_topn_comparison.csv
      results_dir/screen_topn_comparison.png
      results_dir/screen_topn_recommendation.txt

Assumes shared TEST split was enforced (via save_splits.py + --splits_dir).
If repeated splits are present (split_id varies), comparisons are done per split_id and then aggregated.
"""

# example runs:
# python postprocess_compare.py \
#   --results_dir /path/to/results_faith
#
# python postprocess_compare.py \
#   --results_dir /path/to/results_faith/sensitivity \
#   --mode sensitivity

import os
import re
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

# Import all shared utilities (metrics, plotting, DCA, clinical plotting)
from shared_utils import (
    _mkdir,
    _safe_metric,
    format_ci,
    stratified_bootstrap_diff_ci,
    _apply_plot_metadata,
    _compute_recalibration,
    _binned_logits,
    _dca_thresholds,
    _parse_dca_report_points,
    decision_curve_analysis,
    save_dca_results,
    find_dca_zero_crossing,
    _plot_roc_curve,
    _plot_pr_curve,
    _plot_dca_curve,
    _plot_calibration_curve,
    _plot_risk_distribution,
)

RANDOM_STATE = 0


def compare_incident_prevalent_ttest(df, risk_col, category_col="CeD_comparison", scenario="", model_label=""):
    """
    Perform independent t-test comparing risk scores between incident and prevalent cases.

    Args:
        df: DataFrame with risk scores and case type categorization
        risk_col: Column name for risk scores
        category_col: Column name for case categorization (values: "Incident", "Prevalent")
        scenario: Scenario name for reporting
        model_label: Model label for reporting

    Returns:
        Dictionary with t-test results (t_stat, p_value, mean_incident, mean_prevalent, n_incident, n_prevalent)
    """
    if category_col not in df.columns or risk_col not in df.columns:
        return None

    # Filter for valid data
    mask = df[risk_col].notna() & df[category_col].notna()
    df_valid = df[mask]

    if df_valid.empty:
        return None

    # Separate by case type
    incident_scores = df_valid[df_valid[category_col] == "Incident"][risk_col].values
    prevalent_scores = df_valid[df_valid[category_col] == "Prevalent"][risk_col].values

    # Need at least 2 samples in each group
    if len(incident_scores) < 2 or len(prevalent_scores) < 2:
        return None

    # Perform independent samples t-test
    t_stat, p_value = stats.ttest_ind(incident_scores, prevalent_scores)

    return {
        "scenario": scenario,
        "model": model_label,
        "risk_column": risk_col,
        "n_incident": len(incident_scores),
        "n_prevalent": len(prevalent_scores),
        "mean_incident": float(np.mean(incident_scores)),
        "std_incident": float(np.std(incident_scores)),
        "mean_prevalent": float(np.mean(prevalent_scores)),
        "std_prevalent": float(np.std(prevalent_scores)),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "mean_difference": float(np.mean(incident_scores) - np.mean(prevalent_scores)),
    }


def compare_models_test_bootstrap(scenario, y_true, pred_dict, n_boot=500, seed=0):
    """DEPRECATED: Bootstrap comparisons removed. Splits provide natural variance.

    Kept for backward compatibility only. Returns empty DataFrame.
    Cross-model comparisons now use split-to-split variation instead of resampling.
    """
    return pd.DataFrame()


def net_benefit(y_true, p, threshold):
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)
    pt = float(threshold)
    if pt <= 0.0 or pt >= 1.0 or len(y) == 0:
        return np.nan
    tp = int(((p >= pt) & (y == 1)).sum())
    fp = int(((p >= pt) & (y == 0)).sum())
    n = int(len(y))
    w = pt / (1.0 - pt)
    return (tp / n) - (fp / n) * w


def decision_curve_table(scenario, y_true, pred_dict, max_pt=0.20, step=0.005):
    y = np.asarray(y_true).astype(int)
    prev = float(np.mean(y)) if len(y) else np.nan
    thresholds = np.arange(step, max_pt + 1e-12, step)
    rows = []
    for pt in thresholds:
        rows.append({"scenario": scenario, "threshold": float(pt), "model": "treat_none", "net_benefit": 0.0})
        if np.isfinite(prev):
            w = pt / (1.0 - pt)
            nb_all = prev - (1.0 - prev) * w
        else:
            nb_all = np.nan
        rows.append({"scenario": scenario, "threshold": float(pt), "model": "treat_all", "net_benefit": float(nb_all)})

        for m, p in pred_dict.items():
            rows.append({"scenario": scenario, "threshold": float(pt), "model": m, "net_benefit": float(net_benefit(y, p, pt))})
    return pd.DataFrame(rows)


def summarize_and_rank(df_rep):
    group_cols = ["scenario", "model"]
    if "split_id" in df_rep.columns:
        group_cols = ["scenario", "model", "split_id"]
    g = df_rep.groupby(group_cols, as_index=False).agg(
        AUROC_mean=("AUROC_oof", "mean"),
        AUROC_sd=("AUROC_oof", "std"),
        PR_AUC_mean=("PR_AUC_oof", "mean"),
        PR_AUC_sd=("PR_AUC_oof", "std"),
        Brier_mean=("Brier_oof", "mean"),
        Brier_sd=("Brier_oof", "std"),
    )

    def rank_within_scenario(df_s):
        df_s = df_s.copy()
        df_s["rank_PR"] = df_s["PR_AUC_mean"].rank(ascending=False, method="min")
        df_s["rank_AUROC"] = df_s["AUROC_mean"].rank(ascending=False, method="min")
        df_s["rank_Brier"] = df_s["Brier_mean"].rank(ascending=True, method="min")
        df_s["avg_rank"] = df_s[["rank_PR", "rank_AUROC", "rank_Brier"]].mean(axis=1)
        return df_s.sort_values(["avg_rank", "rank_PR", "rank_AUROC", "rank_Brier"], ascending=True)

    ranked = []
    for scen, df_s in g.groupby("scenario", sort=False):
        ranked.append(rank_within_scenario(df_s))
    return pd.concat(ranked, ignore_index=True)


def _infer_model_from_dir(run_dir: str) -> str:
    base = os.path.basename(run_dir)
    parts = base.split("__")
    if len(parts) > 1:
        return parts[1]
    return ""


def _summarize_numeric(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    if df.empty:
        return df
    df_num = df.copy()
    numeric_cols = df_num.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return pd.DataFrame()
    g = df_num.groupby(group_cols)
    agg = g[numeric_cols].agg(["mean", "std"])
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    agg = agg.reset_index()
    if "split_id" in df.columns:
        n_splits = df.groupby(group_cols)["split_id"].nunique().reset_index(name="n_splits")
        agg = agg.merge(n_splits, on=group_cols, how="left")
    else:
        agg["n_splits"] = df.groupby(group_cols).size().values
    return agg


def _aggregate_learning_curve_runs(lc_frames):
    if not lc_frames:
        return pd.DataFrame()

    per_run = []
    metric_label = ""
    metric_direction = ""
    scoring = ""

    for df in lc_frames:
        if df is None or df.empty or "train_size" not in df.columns:
            continue
        run_id = df.get("run_dir")
        run_label = run_id.iloc[0] if run_id is not None else "run"

        if "train_score_mean" in df.columns and "val_score_mean" in df.columns:
            agg = (
                df.groupby("train_size", as_index=False)
                .agg(train_mean=("train_score_mean", "mean"), val_mean=("val_score_mean", "mean"))
            )
        elif "train_score" in df.columns and "val_score" in df.columns:
            agg = (
                df.groupby("train_size", as_index=False)
                .agg(train_mean=("train_score", "mean"), val_mean=("val_score", "mean"))
            )
        else:
            continue

        agg["run"] = run_label
        per_run.append(agg)

        if not metric_label and "error_metric" in df.columns:
            metric_label = str(df["error_metric"].iloc[0])
        if not metric_direction and "metric_direction" in df.columns:
            metric_direction = str(df["metric_direction"].iloc[0])
        if not scoring and "scoring" in df.columns:
            scoring = str(df["scoring"].iloc[0])

    if not per_run:
        return pd.DataFrame()

    all_df = pd.concat(per_run, ignore_index=True)

    def _ci_lo(x):
        return float(np.percentile(x, 2.5)) if len(x) > 1 else np.nan

    def _ci_hi(x):
        return float(np.percentile(x, 97.5)) if len(x) > 1 else np.nan

    summary = (
        all_df.groupby("train_size", as_index=False)
        .agg(
            train_mean=("train_mean", "mean"),
            train_sd=("train_mean", "std"),
            train_ci_lo=("train_mean", _ci_lo),
            train_ci_hi=("train_mean", _ci_hi),
            val_mean=("val_mean", "mean"),
            val_sd=("val_mean", "std"),
            val_ci_lo=("val_mean", _ci_lo),
            val_ci_hi=("val_mean", _ci_hi),
            n_runs=("run", "nunique"),
        )
    )
    summary["metric_label"] = metric_label or scoring
    summary["metric_direction"] = metric_direction
    summary["scoring"] = scoring
    return summary


def _plot_learning_curve_summary(df, out_path, title, meta_lines=None):
    if df is None or df.empty:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[PLOT] Learning curve failed to import dependencies: {e}")
        return

    x = df["train_size"].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 6))

    # Helper function to safely plot confidence/uncertainty bands
    def _plot_band(mean_col, sd_col, ci_lo_col, ci_hi_col, color, label, alpha_ci=0.12, alpha_sd=0.20):
        mean = np.asarray(df[mean_col], dtype=float)
        sd = np.asarray(df[sd_col], dtype=float)
        ci_lo = np.asarray(df[ci_lo_col], dtype=float)
        ci_hi = np.asarray(df[ci_hi_col], dtype=float)

        # Plot 95% CI band if available
        if np.isfinite(ci_lo).any() and np.isfinite(ci_hi).any():
            valid = np.isfinite(ci_lo) & np.isfinite(ci_hi)
            if valid.sum() > 1:
                ax.fill_between(x[valid], ci_lo[valid], ci_hi[valid],
                               color=color, alpha=alpha_ci, label=f"{label} 95% CI")

        # Plot ±1 SD band if available
        if np.isfinite(sd).any() and np.isfinite(mean).any():
            valid = np.isfinite(mean) & np.isfinite(sd)
            if valid.sum() > 1:
                ax.fill_between(x[valid], (mean - sd)[valid], (mean + sd)[valid],
                               color=color, alpha=alpha_sd, label=f"{label} ±1 SD")

    # Plot bands (train first, then val to layer properly)
    _plot_band("train_mean", "train_sd", "train_ci_lo", "train_ci_hi", "steelblue", "Train")
    _plot_band("val_mean", "val_sd", "val_ci_lo", "val_ci_hi", "darkorange", "Val")

    # Plot individual validation data points if available
    if "val_score" in df.columns:
        val_scores = np.asarray(df["val_score"], dtype=float)
        valid_val = np.isfinite(val_scores)
        if valid_val.any():
            ax.scatter(x[valid_val], val_scores[valid_val],
                      color="darkorange", alpha=0.35, s=20, label="Val points")

    # Plot mean lines with markers
    ax.plot(x, df["train_mean"], color="steelblue", linewidth=2.5, linestyle="--", label="Train mean",
            marker='o', markersize=6, markerfacecolor='steelblue', markeredgecolor='steelblue')
    ax.plot(x, df["val_mean"], color="darkorange", linewidth=2.5, label="Val mean",
            marker='s', markersize=6, markerfacecolor='darkorange', markeredgecolor='darkorange')

    # Format axis labels with metric direction
    metric_label = str(df["metric_label"].iloc[0]) if "metric_label" in df.columns and len(df) else "Score"
    metric_direction = str(df["metric_direction"].iloc[0]) if "metric_direction" in df.columns and len(df) else ""
    ylabel = metric_label.replace("_", " ").upper()
    if metric_direction == "lower_is_better":
        ylabel += " (lower is better)"
    elif metric_direction == "higher_is_better":
        ylabel += " (higher is better)"

    ax.set_xlabel("Training examples", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2)

    ax.legend(fontsize=8, loc="best")

    bottom_margin = _apply_plot_metadata(fig, meta_lines)
    plt.subplots_adjust(left=0.15, right=0.9, top=0.8, bottom=bottom_margin)
    plt.savefig(out_path, dpi=150, pad_inches=0.1)
    plt.close()


# ============================================================
# SENSITIVITY ANALYSIS MODE
# ============================================================
# _compute_recalibration moved to plot_utils.py



SCREEN_MAP = {
    "screen500": 500,
    "screen1000": 1000,
    "screen2000": 2000,
    "screen3000": 3000,
    "screenALL": 0,
}


def discover_sensitivity_dirs(results_dir):
    """Discover screen* subdirectories for sensitivity analysis."""
    screen_dirs = {}
    for name in os.listdir(results_dir):
        if not name.startswith("screen"):
            continue
        p = os.path.join(results_dir, name)
        if os.path.isdir(p):
            screen_top_n = SCREEN_MAP.get(name, -1)
            screen_dirs[name] = {"path": p, "screen_top_n": screen_top_n}
    return screen_dirs


def aggregate_sensitivity_results(results_dir):
    """Aggregate test_metrics.csv from all screen* subdirectories."""
    screen_dirs = discover_sensitivity_dirs(results_dir)
    if not screen_dirs:
        raise SystemExit(f"No screen* directories found in {results_dir}")

    results = []
    for screen_label, info in sorted(screen_dirs.items()):
        screen_path = info["path"]
        screen_top_n = info["screen_top_n"]

        # Find model directories within each screen directory
        for model_dir_name in os.listdir(screen_path):
            model_path = os.path.join(screen_path, model_dir_name)
            if not os.path.isdir(model_path):
                continue

            metrics_file = os.path.join(model_path, "core", "test_metrics.csv")
            settings_file = os.path.join(model_path, "core", "run_settings.json")

            if os.path.exists(metrics_file):
                df_metrics = pd.read_csv(metrics_file)
                if len(df_metrics) > 0:
                    row = df_metrics.iloc[0].to_dict()
                    row["screen_top_n"] = screen_top_n
                    row["screen_label"] = screen_label
                    row["model_dir"] = model_path

                    # Extract model name from directory
                    parts = model_dir_name.split("__")
                    row["model"] = parts[1] if len(parts) > 1 else "unknown"

                    # Get n_proteins from settings if available
                    if os.path.exists(settings_file):
                        import json
                        with open(settings_file) as f:
                            settings = json.load(f)
                            row["n_proteins_used"] = settings.get(
                                "n_proteins_after_screen",
                                settings.get("n_protein_cols")
                            )

                    results.append(row)
                    print(f"  Loaded: {metrics_file}")

    if not results:
        raise SystemExit(f"No test_metrics.csv files found in {results_dir}/screen*/*/core/")

    df = pd.DataFrame(results)
    df = df.sort_values(["screen_top_n", "model"], ascending=True)
    return df


def print_sensitivity_decision(df, output_dir):
    """Print and save decision recommendation for sensitivity analysis."""

    # Get Brier scores - handle different column names
    brier_col = None
    for col in ["Brier", "brier", "Brier_test", "brier_score"]:
        if col in df.columns:
            brier_col = col
            break

    if brier_col is None:
        print("\n⚠️  No Brier score column found - cannot make recommendation")
        return

    # Average across models for each screen_top_n
    avg_by_screen = df.groupby("screen_top_n")[brier_col].mean().reset_index()
    avg_by_screen.columns = ["screen_top_n", "avg_brier"]

    all_row = avg_by_screen[avg_by_screen["screen_top_n"] == 0]
    screened_rows = avg_by_screen[avg_by_screen["screen_top_n"] > 0]

    recommendation_lines = []
    recommendation_lines.append("=" * 70)
    recommendation_lines.append("MANN-WHITNEY SCREENING SENSITIVITY ANALYSIS - RECOMMENDATION")
    recommendation_lines.append("=" * 70)

    if len(all_row) == 0 or len(screened_rows) == 0:
        msg = "⚠️  Insufficient data for comparison (need both ALL and screened results)"
        recommendation_lines.append(msg)
        print("\n".join(recommendation_lines))
        return

    all_brier = all_row["avg_brier"].values[0]
    best_screened = screened_rows.loc[screened_rows["avg_brier"].idxmin()]
    best_brier = best_screened["avg_brier"]
    best_n = int(best_screened["screen_top_n"])

    recommendation_lines.append(f"\nAverage Brier scores across models:")
    for _, row in avg_by_screen.iterrows():
        label = "ALL" if row["screen_top_n"] == 0 else str(int(row["screen_top_n"]))
        recommendation_lines.append(f"  screen_top_n={label:>4}: Brier={row['avg_brier']:.5f}")

    diff = all_brier - best_brier
    diff_pct = (diff / best_brier) * 100 if best_brier != 0 else 0

    recommendation_lines.append(f"\nBest screened: screen_top_n={best_n} (Brier={best_brier:.5f})")
    recommendation_lines.append(f"ALL proteins:  screen_top_n=0  (Brier={all_brier:.5f})")
    recommendation_lines.append(f"Difference: {diff:+.5f} ({diff_pct:+.2f}%)")
    recommendation_lines.append("")

    if diff > 0.001:  # ALL is worse
        recommendation_lines.append("✅ RECOMMENDATION: KEEP SCREENING")
        recommendation_lines.append(f"   Using all proteins worsens Brier by {diff:.5f}")
        recommendation_lines.append(f"   → Set SCREEN_TOP_N={best_n} in CeD_optimized.lsf")
    elif diff < -0.001:  # ALL is better
        recommendation_lines.append("❌ RECOMMENDATION: REMOVE SCREENING")
        recommendation_lines.append(f"   Using all proteins improves Brier by {-diff:.5f}")
        recommendation_lines.append(f"   → Set SCREEN_TOP_N=0 (or FEATURE_SELECT=kbest) in CeD_optimized.lsf")
    else:
        recommendation_lines.append("🔶 RECOMMENDATION: NO CLEAR WINNER")
        recommendation_lines.append(f"   Difference is small ({diff:+.5f})")
        recommendation_lines.append(f"   → Keep screening for computational efficiency (SCREEN_TOP_N={best_n})")

    recommendation_lines.append("=" * 70)

    # Print to console
    print("\n".join(recommendation_lines))

    # Save to file
    rec_path = os.path.join(output_dir, "screen_topn_recommendation.txt")
    with open(rec_path, "w") as f:
        f.write("\n".join(recommendation_lines))
    print(f"\nSaved recommendation to: {rec_path}")


def plot_sensitivity_results(df, output_dir):
    """Generate sensitivity analysis comparison plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Get metric columns
        metrics = []
        for col, title, color in [
            ("AUROC", "AUROC (↑)", "forestgreen"),
            ("PR_AUC", "PR-AUC (↑)", "steelblue"),
            ("Brier", "Brier (↓)", "firebrick"),
        ]:
            # Try different column name variants
            for variant in [col, col.lower(), f"{col}_test", col.replace("_", "")]:
                if variant in df.columns:
                    metrics.append((variant, title, color))
                    break

        if not metrics:
            print("[PLOT] No metric columns found, skipping plot")
            return

        # Average across models
        avg_df = df.groupby("screen_top_n")[
            [m[0] for m in metrics]
        ].mean().reset_index()
        avg_df["label"] = avg_df["screen_top_n"].apply(
            lambda x: "ALL" if x == 0 else str(int(x))
        )
        avg_df = avg_df.sort_values("screen_top_n")

        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
        if len(metrics) == 1:
            axes = [axes]

        x = np.arange(len(avg_df))
        x_labels = avg_df["label"].values

        for ax, (col, title, color) in zip(axes, metrics):
            values = avg_df[col].values
            bars = ax.bar(x, values, color=color, alpha=0.7, edgecolor="black")
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, rotation=45, ha="right")
            ax.set_xlabel("screen_top_n")
            ax.set_ylabel(title.split(" ")[0])
            ax.set_title(title)

            for bar, val in zip(bars, values):
                if np.isfinite(val):
                    ax.annotate(f"{val:.4f}",
                               xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                               ha="center", va="bottom", fontsize=9)

        meta_lines = []
        if len(df) > 0:
            n_models = df["model"].nunique() if "model" in df.columns else None
            base = f"Split=TEST | runs={len(df)}"
            if n_models:
                base += f" | models={int(n_models)}"
            meta_lines.append(base)

            extra = []
            if "n_train" in df.columns:
                extra.append(f"n_train_mean={np.nanmean(df['n_train']):.1f}")
            if "n_test" in df.columns:
                extra.append(f"n_test_mean={np.nanmean(df['n_test']):.1f}")
            if extra:
                meta_lines.append(" | ".join(extra))

        if meta_lines:
            fig.text(0.5, 0.01, "\n".join(meta_lines), ha="center", va="bottom", fontsize=9)
            bottom = 0.08 + 0.03 * max(0, len(meta_lines) - 1)
        plot_path = os.path.join(output_dir, "screen_topn_comparison.png")
        plt.subplots_adjust(left=0.15, right=0.9, top=0.8, bottom=0.15)
        plt.savefig(plot_path, dpi=150, pad_inches=0.1)
        plt.close()
        print(f"Saved plot: {plot_path}")

    except ImportError:
        print("[PLOT] matplotlib not available, skipping plot")
    except Exception as e:
        print(f"[PLOT] Failed to generate plot: {e}")


def run_sensitivity_mode(args):
    """Run sensitivity analysis aggregation."""
    print("=" * 70)
    print("SENSITIVITY ANALYSIS MODE")
    print("=" * 70)
    print(f"Results directory: {args.results_dir}")

    df = aggregate_sensitivity_results(args.results_dir)

    # Save comparison CSV
    csv_path = os.path.join(args.results_dir, "screen_topn_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved comparison CSV: {csv_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    summary_cols = ["screen_label", "model", "n_proteins_used"]
    for col in ["AUROC", "PR_AUC", "Brier"]:
        for variant in [col, col.lower(), f"{col}_test"]:
            if variant in df.columns:
                summary_cols.append(variant)
                break
    available_cols = [c for c in summary_cols if c in df.columns]
    print(df[available_cols].to_string(index=False))

    # Print decision recommendation
    print_sensitivity_decision(df, args.results_dir)

    # Generate plot
    plot_sensitivity_results(df, args.results_dir)

    print(f"\n✓ Sensitivity analysis complete. Results in: {args.results_dir}")


# ============================================================
# MODEL COMPARISON MODE (default behavior)
# ============================================================
def compute_per_model_dca(
    run_dirs,
    dca_threshold_min=0.0005,
    dca_threshold_max=1.0,
    dca_threshold_step=0.001,
    dca_report_points=None,
):
    """
    Compute per-model DCA curves from saved test predictions.

    Writes to same locations as training-time DCA:
      {run_dir}/diagnostics/dca/{scenario}__{model}__dca_curve.csv
      {run_dir}/diagnostics/dca/{scenario}__{model}__dca_summary.json
      {run_dir}/diagnostics/dca/{scenario}__{model}__dca_plot.png

    This enables zero-crossing detection for enhanced plots without
    running DCA during training.

    Args:
        run_dirs: List of model output directories to process
        dca_threshold_min: Minimum threshold (e.g., 0.0005)
        dca_threshold_max: Maximum threshold (e.g., 1.0)
        dca_threshold_step: Step size (e.g., 0.001)
        dca_report_points: Key thresholds for summary (e.g., [0.005, 0.01, 0.02, 0.05])
    """
    thresholds = _dca_thresholds(dca_threshold_min, dca_threshold_max, dca_threshold_step)
    report_pts = dca_report_points or [0.005, 0.01, 0.02, 0.05]

    for run_dir in run_dirs:
        # Find all test prediction files
        pred_dir = os.path.join(run_dir, "preds", "test_preds")
        if not os.path.exists(pred_dir):
            continue

        pred_files = [f for f in os.listdir(pred_dir) if f.endswith(".csv") and "__test_preds__" in f]

        for pred_file in pred_files:
            # Parse scenario and model from filename
            # Format: {scenario}__test_preds__{model}.csv
            match = re.match(r"(.+)__test_preds__(.+)\.csv", pred_file)
            if not match:
                continue
            scenario, model = match.groups()

            # Load predictions
            pred_path = os.path.join(pred_dir, pred_file)
            try:
                df_pred = pd.read_csv(pred_path)
            except Exception as e:
                print(f"[WARN] Could not read {pred_file}: {e}")
                continue

            if "y_true" not in df_pred.columns:
                print(f"[WARN] {pred_file}: missing y_true column")
                continue

            # Determine which probability column to use (priority order)
            prob_col = None
            for col in ["risk_test", "risk_test_adjusted", "risk_test_raw"]:
                if col in df_pred.columns:
                    prob_col = col
                    break

            if prob_col is None:
                print(f"[WARN] {pred_file}: no probability columns found")
                continue

            y_true = df_pred["y_true"].to_numpy()
            y_pred_prob = df_pred[prob_col].to_numpy()

            # Create output directory
            dca_dir = os.path.join(run_dir, "diagnostics", "dca")
            os.makedirs(dca_dir, exist_ok=True)

            # Build metadata
            n_test = len(y_true)
            n_pos = int(y_true.sum())
            prev = float(np.mean(y_true))
            meta_lines = [
                f"Scenario={scenario} | Model={model} | Split=TEST",
                f"n_test={n_test} | n_pos={n_pos} | prev={prev:.4f}",
                f"prob_source={prob_col}",
                "Regenerated by postprocess_compare.py"
            ]

            # Compute and save DCA
            try:
                dca_summary = save_dca_results(
                    y_true,
                    y_pred_prob,
                    out_dir=dca_dir,
                    prefix=f"{scenario}__{model}__",
                    thresholds=thresholds,
                    report_points=report_pts,
                    prevalence_adjustment=None,  # Use sample prevalence
                    meta_lines=meta_lines,
                    plot_dir=dca_dir,
                )

                if dca_summary.get("dca_computed"):
                    beats_range = dca_summary.get('model_beats_all_range', 'N/A')
                    print(f"[DCA] {scenario}/{model}: model beats 'treat all' at {beats_range}")
                else:
                    error_msg = dca_summary.get('error', 'unknown')
                    print(f"[DCA] {scenario}/{model}: computation failed - {error_msg}")
            except Exception as e:
                print(f"[ERROR] DCA computation failed for {scenario}/{model}: {e}")
                continue


def discover_run_dirs(results_dir):
    # expects: results_faith/{Scenario}__{Model}__... (your naming)
    all_dirs = []
    for name in os.listdir(results_dir):
        p = os.path.join(results_dir, name)
        if not os.path.isdir(p):
            continue
        if name == "COMBINED":
            continue
        # require core/test_metrics.csv presence
        if os.path.exists(os.path.join(p, "core", "test_metrics.csv")) and os.path.exists(os.path.join(p, "cv", "cv_repeat_metrics.csv")):
            all_dirs.append(p)
    return sorted(all_dirs)


def run_model_comparison(args):
    """Run the default model comparison mode."""
    run_dirs = discover_run_dirs(args.results_dir)
    if not run_dirs:
        raise SystemExit(f"No run dirs found under {args.results_dir}")

    out_combined = os.path.join(args.results_dir, "COMBINED")
    OUT = {
        "core": _mkdir(os.path.join(out_combined, "core")),
        "cv": _mkdir(os.path.join(out_combined, "cv")),
        "preds": _mkdir(os.path.join(out_combined, "preds")),
    }

    # PHASE 1: Per-model detailed DCA curves (for zero-crossing detection in enhanced plots)
    print("\n[DCA] Computing per-model DCA curves from test predictions...")
    compute_per_model_dca(
        run_dirs=run_dirs,
        dca_threshold_min=args.dca_threshold_min,
        dca_threshold_max=args.dca_threshold_max,
        dca_threshold_step=args.dca_threshold_step,
        dca_report_points=args.dca_report_points if hasattr(args, 'dca_report_points') and args.dca_report_points else None,
    )

    # concat raw tables
    cv_all = []
    test_all = []
    for rd in run_dirs:
        cv_all.append(pd.read_csv(os.path.join(rd, "cv", "cv_repeat_metrics.csv")))
        test_all.append(pd.read_csv(os.path.join(rd, "core", "test_metrics.csv")))

    df_cv = pd.concat(cv_all, ignore_index=True)
    df_test = pd.concat(test_all, ignore_index=True)

    df_cv.to_csv(os.path.join(OUT["cv"], "ALL_cv_repeat_metrics.csv"), index=False)
    df_test.to_csv(os.path.join(OUT["core"], "ALL_test_metrics.csv"), index=False)

    # Check split_id consistency across run dirs (if available)
    # With repeated splits (N_SPLITS > 1), each seed has a different split_id - this is expected.
    # We check that within each seed, all models share the same split_id.
    if "split_id" in df_test.columns:
        n_unique_splits = df_test["split_id"].nunique()
        n_models = df_test["model"].nunique()
        n_runs = len(df_test)
        # If #splits * #models == #runs, this is repeated splits mode (each seed has all models)
        if n_unique_splits > 1 and n_unique_splits * n_models == n_runs:
            print(f"[INFO] Detected repeated splits mode: {n_unique_splits} seeds x {n_models} models = {n_runs} runs")
        elif n_unique_splits > 1:
            # Possible mismatch - warn but don't fail (user may have partial runs)
            print(f"[WARN] Found {n_unique_splits} unique split_ids across {n_runs} runs ({n_models} models). "
                  f"If not using repeated splits, ensure all jobs used the same --splits_dir.")

    # ranking + best model by TRAIN-CV
    df_rank = summarize_and_rank(df_cv)
    df_rank.to_csv(os.path.join(OUT["core"], "cv_summary_ranked.csv"), index=False)

    # final summary: merge ranked CV summary with TEST metrics
    merge_keys = ["scenario", "model"]
    if ("split_id" in df_test.columns) and ("split_id" in df_rank.columns):
        merge_keys = ["scenario", "model", "split_id"]
    final = df_rank.merge(df_test, on=merge_keys, how="left")
    final.to_csv(os.path.join(OUT["core"], "final_summary_with_test.csv"), index=False)

    best_by_scenario = []
    if "split_id" in df_rank.columns:
        for (scen, sid), df_s in df_rank.groupby(["scenario", "split_id"], sort=False):
            best = df_s.sort_values(["PR_AUC_mean", "AUROC_mean", "Brier_mean"], ascending=[False, False, True]).iloc[0].to_dict()
            best_by_scenario.append(best)
    else:
        for scen, df_s in df_rank.groupby("scenario", sort=False):
            best = df_s.sort_values(["PR_AUC_mean", "AUROC_mean", "Brier_mean"], ascending=[False, False, True]).iloc[0].to_dict()
            best_by_scenario.append(best)
    pd.DataFrame(best_by_scenario).to_csv(os.path.join(OUT["core"], "BEST_by_trainCV_PR.csv"), index=False)

    # pairwise TEST comparisons + DCA per scenario (needs preds/test_preds CSVs)
    # We locate preds by scanning each run dir's preds/test_preds folder.
    scenarios = sorted(df_test["scenario"].unique().tolist())
    for scen in scenarios:
        if "split_id" in df_test.columns:
            split_ids = (
                df_test.loc[df_test["scenario"] == scen, "split_id"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            split_ids = sorted(split_ids)
        else:
            split_ids = [None]

        per_split_cmp = []
        per_split_dca = []

        for split_id in split_ids:
            pred_dict = {}
            y_ref = None

            for rd in run_dirs:
                # infer scenario from test_metrics rows
                tm = pd.read_csv(os.path.join(rd, "core", "test_metrics.csv"))
                tm_s = tm[tm["scenario"] == scen]
                if tm_s.empty:
                    continue
                if split_id is not None and "split_id" in tm_s.columns:
                    tm_s = tm_s[tm_s["split_id"].astype(str) == split_id]
                    if tm_s.empty:
                        continue

                # for each model row in this run dir, try to load its test_preds
                for m in tm_s["model"].astype(str).tolist():
                    pred_path = os.path.join(rd, "preds", "test_preds", f"{scen}__test_preds__{m}.csv")
                    if not os.path.exists(pred_path):
                        continue
                    dfp = pd.read_csv(pred_path)
                    if "y_true" not in dfp.columns or "risk_test" not in dfp.columns:
                        continue
                    y = dfp["y_true"].to_numpy(dtype=int)
                    p = dfp["risk_test"].to_numpy(dtype=float)

                    if y_ref is None:
                        y_ref = y
                    else:
                        # shared split sanity: require same y length + same label vector
                        if len(y) != len(y_ref) or not np.array_equal(y, y_ref):
                            sid_msg = f" split_id={split_id}" if split_id is not None else ""
                            print(f"[WARN] TEST mismatch for scenario={scen}{sid_msg} when loading {pred_path}. Skipping.")
                            continue

                    pred_dict[m] = p

            if y_ref is None or len(pred_dict) < 2:
                sid_msg = f" split_id={split_id}" if split_id is not None else ""
                print(f"[WARN] scenario={scen}{sid_msg}: not enough test preds found for comparisons/DCA (need >=2).")
                continue

            # Bootstrap skipped: splits provide natural variance estimates for cross-model comparisons
            # Per-split model differences are aggregated via percentiles (2.5th, 97.5th)
            df_cmp = pd.DataFrame()
            if split_id is not None:
                df_cmp["split_id"] = split_id
            per_split_cmp.append(df_cmp)

            df_dca = decision_curve_table(scen, y_ref, pred_dict, max_pt=args.dca_max_pt, step=args.dca_step)
            if split_id is not None:
                df_dca["split_id"] = split_id
            per_split_dca.append(df_dca)

        if not per_split_cmp:
            print(f"[WARN] scenario={scen}: no valid split_id groups for comparisons/DCA.")
            continue

        df_cmp_all = pd.concat(per_split_cmp, ignore_index=True)
        df_dca_all = pd.concat(per_split_dca, ignore_index=True) if per_split_dca else None

        # If repeated splits, write per-split outputs and aggregate
        if "split_id" in df_cmp_all.columns:
            df_cmp_all.to_csv(
                os.path.join(OUT["core"], f"{scen}__test_model_comparisons_bootstrap_by_split.csv"),
                index=False,
            )
            agg = (
                df_cmp_all.groupby(["scenario", "metric", "model1", "model2"], as_index=False)
                .agg(
                    diff_model1_minus_model2=("diff_model1_minus_model2", "mean"),
                    CI_lo=("diff_model1_minus_model2",
                           lambda x: float(np.percentile(x, 2.5)) if len(x) >= 2 else np.nan),
                    CI_hi=("diff_model1_minus_model2",
                           lambda x: float(np.percentile(x, 97.5)) if len(x) >= 2 else np.nan),
                    n_splits=("split_id", "nunique"),
                )
            )
            agg["diff_95CI"] = agg.apply(
                lambda r: format_ci(
                    r["CI_lo"], r["CI_hi"], decimals=4 if r["metric"] == "Brier" else 3
                ),
                axis=1,
            )
            agg = agg.drop(columns=["CI_lo", "CI_hi"])
            df_cmp_out = agg
        else:
            df_cmp_out = df_cmp_all

        df_cmp_out.to_csv(os.path.join(OUT["core"], f"{scen}__test_model_comparisons_bootstrap.csv"), index=False)

        if df_dca_all is not None:
            if "split_id" in df_dca_all.columns:
                df_dca_all.to_csv(
                    os.path.join(OUT["core"], f"{scen}__decision_curve__TEST_by_split.csv"),
                    index=False,
                )
                dca_agg = (
                    df_dca_all.groupby(["scenario", "threshold", "model"], as_index=False)
                    .agg(
                        net_benefit=("net_benefit", "mean"),
                        net_benefit_sd=("net_benefit", "std"),
                        n_splits=("split_id", "nunique"),
                    )
                )
                df_dca_out = dca_agg
            else:
                df_dca_out = df_dca_all

            df_dca_out.to_csv(os.path.join(OUT["core"], f"{scen}__decision_curve__TEST.csv"), index=False)

    print(f"[OK] Wrote COMBINED outputs to: {out_combined}")


def run_single_model_summary(args):
    """Aggregate splits for a single model into a combined summary."""
    run_dirs = discover_run_dirs(args.results_dir)
    if not run_dirs:
        raise SystemExit(f"No run dirs found under {args.results_dir}")

    if not args.model:
        raise SystemExit("--model is required for --mode single")

    model_key = args.model.strip()
    matched_dirs = []
    for rd in run_dirs:
        dir_model = _infer_model_from_dir(rd)
        if dir_model and dir_model.lower() == model_key.lower():
            matched_dirs.append(rd)
            continue
        tm_path = os.path.join(rd, "core", "test_metrics.csv")
        if os.path.exists(tm_path):
            df_tmp = pd.read_csv(tm_path)
            if "model" in df_tmp.columns:
                if df_tmp["model"].astype(str).str.lower().eq(model_key.lower()).any():
                    matched_dirs.append(rd)
    run_dirs = sorted(set(matched_dirs))
    if not run_dirs:
        raise SystemExit(f"No run dirs found for model '{args.model}' under {args.results_dir}")

    cv_all = []
    test_all = []
    val_all = []
    run_manifest = []
    split_map = {}

    for rd in run_dirs:
        run_name = os.path.basename(rd)
        tm_path = os.path.join(rd, "core", "test_metrics.csv")
        cv_path = os.path.join(rd, "cv", "cv_repeat_metrics.csv")
        if not os.path.exists(tm_path) or not os.path.exists(cv_path):
            raise SystemExit(f"Missing metrics for {run_name} (need test_metrics.csv + cv_repeat_metrics.csv)")

        df_test = pd.read_csv(tm_path)
        df_test["run_dir"] = run_name
        test_all.append(df_test)

        df_cv = pd.read_csv(cv_path)
        df_cv["run_dir"] = run_name
        cv_all.append(df_cv)

        val_path = os.path.join(rd, "core", "val_metrics.csv")
        if os.path.exists(val_path):
            df_val = pd.read_csv(val_path)
            df_val["run_dir"] = run_name
            val_all.append(df_val)

        for _, row in df_test.iterrows():
            scen = row.get("scenario", "")
            split_id = str(row.get("split_id", "")) if "split_id" in df_test.columns else ""
            split_map[(rd, scen)] = split_id
            run_manifest.append({
                "run_dir": run_name,
                "scenario": scen,
                "model": row.get("model", ""),
                "split_id": split_id,
                "n_train": row.get("n_train", np.nan),
                "n_test": row.get("n_test", np.nan),
            })

    df_test = pd.concat(test_all, ignore_index=True)
    df_cv = pd.concat(cv_all, ignore_index=True)
    df_val = pd.concat(val_all, ignore_index=True) if val_all else pd.DataFrame()

    if df_test.empty:
        raise SystemExit("No test metrics found after filtering.")

    model_labels = sorted(df_test["model"].astype(str).unique().tolist()) if "model" in df_test.columns else []
    model_label = model_labels[0] if model_labels else model_key

    if args.expected_splits and int(args.expected_splits) > 0:
        expected = int(args.expected_splits)
        if "split_id" in df_test.columns:
            n_unique = df_test["split_id"].nunique()
            if n_unique != expected or len(run_dirs) != expected:
                raise SystemExit(
                    f"Expected {expected} splits for model {model_label}, found {n_unique} split_ids across {len(run_dirs)} runs."
                )
        else:
            if len(run_dirs) != expected:
                raise SystemExit(
                    f"Expected {expected} runs for model {model_label}, found {len(run_dirs)}."
                )

    out_root = args.single_outdir or os.path.join(args.results_dir, f"{model_label}_COMBINED")
    OUT = {
        "core": _mkdir(os.path.join(out_root, "core")),
        "cv": _mkdir(os.path.join(out_root, "cv")),
        "preds": _mkdir(os.path.join(out_root, "preds")),
        "preds_test": _mkdir(os.path.join(out_root, "preds", "test_preds")),
        "preds_val": _mkdir(os.path.join(out_root, "preds", "val_preds")),
        "preds_controls": _mkdir(os.path.join(out_root, "preds", "controls_oof")),
        "preds_train_oof": _mkdir(os.path.join(out_root, "preds", "train_oof")),
        "preds_plots": _mkdir(os.path.join(out_root, "preds", "plots")),
        "diagnostics_plots": _mkdir(os.path.join(out_root, "diagnostics", "plots")),
        "diagnostics_lc": _mkdir(os.path.join(out_root, "diagnostics", "learning_curve")),
    }

    df_test.to_csv(os.path.join(OUT["core"], "ALL_test_metrics.csv"), index=False)
    df_test.to_csv(os.path.join(OUT["core"], "test_metrics_by_split.csv"), index=False)
    _summarize_numeric(df_test, ["scenario", "model"]).to_csv(
        os.path.join(OUT["core"], "test_metrics_summary_mean_sd.csv"), index=False
    )

    if not df_val.empty:
        df_val.to_csv(os.path.join(OUT["core"], "ALL_val_metrics.csv"), index=False)
        df_val.to_csv(os.path.join(OUT["core"], "val_metrics_by_split.csv"), index=False)
        _summarize_numeric(df_val, ["scenario", "model"]).to_csv(
            os.path.join(OUT["core"], "val_metrics_summary_mean_sd.csv"), index=False
        )

    df_cv.to_csv(os.path.join(OUT["cv"], "ALL_cv_repeat_metrics.csv"), index=False)
    cv_by_split = summarize_and_rank(df_cv)
    cv_by_split.to_csv(os.path.join(OUT["cv"], "cv_summary_by_split.csv"), index=False)

    if not cv_by_split.empty:
        if "split_id" in cv_by_split.columns:
            cv_summary = (
                cv_by_split.groupby(["scenario", "model"], as_index=False)
                .agg(
                    AUROC_mean=("AUROC_mean", "mean"),
                    AUROC_sd=("AUROC_mean", "std"),
                    PR_AUC_mean=("PR_AUC_mean", "mean"),
                    PR_AUC_sd=("PR_AUC_mean", "std"),
                    Brier_mean=("Brier_mean", "mean"),
                    Brier_sd=("Brier_mean", "std"),
                    n_splits=("split_id", "nunique"),
                )
            )
        else:
            cv_summary = (
                cv_by_split.groupby(["scenario", "model"], as_index=False)
                .agg(
                    AUROC_mean=("AUROC_mean", "mean"),
                    AUROC_sd=("AUROC_mean", "std"),
                    PR_AUC_mean=("PR_AUC_mean", "mean"),
                    PR_AUC_sd=("PR_AUC_mean", "std"),
                    Brier_mean=("Brier_mean", "mean"),
                    Brier_sd=("Brier_mean", "std"),
                )
            )
            cv_summary["n_splits"] = len(run_dirs)
        cv_summary.to_csv(os.path.join(OUT["cv"], "cv_summary_mean_sd.csv"), index=False)

    pd.DataFrame(run_manifest).to_csv(os.path.join(OUT["core"], "run_manifest.csv"), index=False)

    scenarios = sorted(df_test["scenario"].dropna().astype(str).unique().tolist())
    lc_frames_by_scen = {scen: [] for scen in scenarios}
    tuning_plots_by_scen = {scen: [] for scen in scenarios}

    for rd in run_dirs:
        run_name = os.path.basename(rd)
        for scen in scenarios:
            lc_path = os.path.join(rd, "diagnostics", "learning_curve", f"{scen}__learning_curve__{model_label}.csv")
            if os.path.exists(lc_path):
                df_lc = pd.read_csv(lc_path)
                df_lc["run_dir"] = run_name
                lc_frames_by_scen[scen].append(df_lc)

            # Collect tuning history plots
            tuning_plot_path = os.path.join(rd, "diagnostics", "plots", f"{scen}__{model_label}__tuning_history.png")
            if os.path.exists(tuning_plot_path):
                tuning_plots_by_scen[scen].append(tuning_plot_path)

    def _build_meta_lines(scenario, model, split_label, y_vals=None, split_ids=None, extra=None):
        lines = [f"Scenario={scenario} | Model={model} | Split={split_label}"]
        if y_vals is not None:
            y_arr = np.asarray(y_vals).astype(int)
            n = int(len(y_arr))
            n_pos = int(y_arr.sum())
            prev = (n_pos / n) if n > 0 else np.nan
            lines.append(f"n={n} pos={n_pos} prev={prev:.4f}" if np.isfinite(prev) else f"n={n} pos={n_pos}")
        elif split_ids is not None:
            n = int(len(split_ids))
            lines.append(f"n={n}")
        if split_ids is not None:
            n_splits = pd.Series(split_ids).dropna().nunique()
            lines.append(f"splits={int(n_splits)}")
        if extra:
            lines.append(extra)
        return lines

    def _pos_label_for_scenario(scenario):
        scen = str(scenario or "").lower()
        if scen == "incidentonly":
            return "Incident"
        if scen == "incidentplusprevalent":
            return "Incident/Prevalent"
        return "Incident CeD"

    # Collect t-test results for saving to CSV
    ttest_results = []

    for scen in scenarios:
        preds_all = []
        val_preds_all = []
        train_oof_all = []
        controls_all = []
        for rd in run_dirs:
            split_id = split_map.get((rd, scen), "")
            run_name = os.path.basename(rd)

            test_path = os.path.join(rd, "preds", "test_preds", f"{scen}__test_preds__{model_label}.csv")
            if os.path.exists(test_path):
                dfp = pd.read_csv(test_path)
                dfp["split_id"] = split_id
                dfp["run_dir"] = run_name
                preds_all.append(dfp)

            val_path = os.path.join(rd, "preds", "val_preds", f"{scen}__val_preds__{model_label}.csv")
            if os.path.exists(val_path):
                dfv = pd.read_csv(val_path)
                dfv["split_id"] = split_id
                dfv["run_dir"] = run_name
                val_preds_all.append(dfv)

            ctrl_path = os.path.join(rd, "preds", "controls_oof", f"{scen}__controls_risk__{model_label}__oof_mean.csv")
            if os.path.exists(ctrl_path):
                dfc = pd.read_csv(ctrl_path)
                dfc["split_id"] = split_id
                dfc["run_dir"] = run_name
                controls_all.append(dfc)

            train_path = os.path.join(rd, "preds", "train_oof", f"{scen}__train_oof__{model_label}.csv")
            if os.path.exists(train_path):
                dft = pd.read_csv(train_path)
                dft["split_id"] = split_id
                dft["run_dir"] = run_name
                train_oof_all.append(dft)

        if preds_all:
            df_test_preds = pd.concat(preds_all, ignore_index=True)
            df_test_preds.to_csv(
                os.path.join(OUT["preds_test"], f"{scen}__test_preds__{model_label}__all_splits.csv"),
                index=False,
            )

            # Load DCA threshold from first available run for enhanced plots
            dca_threshold_test = None
            for rd in run_dirs:
                dca_csv = os.path.join(rd, "diagnostics", "dca", f"{scen}__{model_label}__dca_curve.csv")
                if os.path.exists(dca_csv):
                    try:
                        dca_threshold_test = find_dca_zero_crossing(dca_csv)
                        if dca_threshold_test is not None:
                            print(f"[OK] Loaded DCA threshold for {scen}: {dca_threshold_test:.6f}")
                        break
                    except Exception as e:
                        print(f"[WARN] Could not load DCA threshold: {e}")

            # Compute recalibration parameters if possible
            calib_intercept_test, calib_slope_test = np.nan, np.nan
            calib_intercept_test_adj, calib_slope_test_adj = np.nan, np.nan
            if "y_true" in df_test_preds.columns and "risk_test" in df_test_preds.columns:
                try:
                    calib_intercept_test, calib_slope_test = _compute_recalibration(
                        df_test_preds["y_true"], df_test_preds["risk_test"]
                    )
                except Exception as e:
                    print(f"[WARN] Failed to compute recalibration for {scen}: {e}")

            # Compute separate recalibration for adjusted if available
            if "y_true" in df_test_preds.columns and "risk_test_adjusted" in df_test_preds.columns:
                try:
                    calib_intercept_test_adj, calib_slope_test_adj = _compute_recalibration(
                        df_test_preds["y_true"], df_test_preds["risk_test_adjusted"]
                    )
                except Exception as e:
                    print(f"[WARN] Failed to compute recalibration for adjusted {scen}: {e}")

            if "y_true" in df_test_preds.columns and "risk_test" in df_test_preds.columns:
                meta_test = _build_meta_lines(
                    scen,
                    model_label,
                    "TEST",
                    y_vals=df_test_preds["y_true"],
                    split_ids=df_test_preds.get("split_id"),
                    extra="prob=risk_test",
                )
                _plot_roc_curve(
                    df_test_preds["y_true"],
                    df_test_preds["risk_test"],
                    os.path.join(OUT["diagnostics_plots"], f"{scen}__{model_label}__TEST_roc_curve.png"),
                    title=f"ROC Curve (TEST) - {scen}",
                    subtitle=f"{model_label} | mean ± SD, 95% CI",
                    split_ids=df_test_preds.get("split_id"),
                    meta_lines=meta_test,
                )
                _plot_pr_curve(
                    df_test_preds["y_true"],
                    df_test_preds["risk_test"],
                    os.path.join(OUT["diagnostics_plots"], f"{scen}__{model_label}__TEST_pr_curve.png"),
                    title=f"PR Curve (TEST) - {scen}",
                    subtitle=f"{model_label} | mean ± SD, 95% CI",
                    split_ids=df_test_preds.get("split_id"),
                    meta_lines=meta_test,
                )
                # Generate 5-panel version
                _plot_calibration_curve(
                    df_test_preds["y_true"],
                    df_test_preds["risk_test"],
                    os.path.join(OUT["diagnostics_plots"], f"{scen}__{model_label}__TEST_calibration_5panel.png"),
                    title=f"Calibration (TEST) - {scen}",
                    subtitle=f"{model_label} | mean ± SD, 95% CI",
                    n_bins=args.calib_bins,
                    split_ids=df_test_preds.get("split_id"),
                    meta_lines=meta_test,
                    four_panel=True,
                    calib_intercept=calib_intercept_test,
                    calib_slope=calib_slope_test,
                )
                if "risk_test_adjusted" in df_test_preds.columns:
                    meta_test_adj = _build_meta_lines(
                        scen,
                        model_label,
                        "TEST",
                        y_vals=df_test_preds["y_true"],
                        split_ids=df_test_preds.get("split_id"),
                        extra="prob=risk_test_adjusted",
                    )
                    # Generate 5-panel version for adjusted
                    _plot_calibration_curve(
                        df_test_preds["y_true"],
                        df_test_preds["risk_test_adjusted"],
                        os.path.join(OUT["diagnostics_plots"], f"{scen}__{model_label}__TEST_calibration_figure.png"),
                        title=f"Calibration (TEST, adjusted) - {scen}",
                        subtitle=f"{model_label} | mean ± SD, 95% CI",
                        n_bins=args.calib_bins,
                        split_ids=df_test_preds.get("split_id"),
                        meta_lines=meta_test_adj,
                        four_panel=True,
                        calib_intercept=calib_intercept_test_adj,
                        calib_slope=calib_slope_test_adj,
                    )
                _plot_dca_curve(
                    df_test_preds["y_true"],
                    df_test_preds["risk_test"],
                    os.path.join(OUT["diagnostics_plots"], f"{scen}__{model_label}__TEST_dca_curve.png"),
                    title=f"DCA (TEST) - {scen}",
                    subtitle=f"{model_label} | mean ± SD, 95% CI",
                    max_pt=float(args.dca_max_pt),
                    step=float(args.dca_step),
                    split_ids=df_test_preds.get("split_id"),
                    meta_lines=meta_test,
                )

                _plot_risk_distribution(
                    df_test_preds["y_true"],
                    df_test_preds["risk_test"],
                    os.path.join(OUT["preds_plots"], f"{scen}__{model_label}__TEST_risk_distribution.png"),
                    title=f"Risk distribution (TEST) - {scen}",
                    subtitle="risk_test; combined across splits",
                    meta_lines=meta_test,
                    dca_threshold=dca_threshold_test,
                )
                if "risk_test_adjusted" in df_test_preds.columns:
                    meta_adj = _build_meta_lines(
                        scen,
                        model_label,
                        "TEST",
                        y_vals=df_test_preds["y_true"],
                        split_ids=df_test_preds.get("split_id"),
                        extra="prob=risk_test_adjusted",
                    )
                    _plot_risk_distribution(
                        df_test_preds["y_true"],
                        df_test_preds["risk_test_adjusted"],
                        os.path.join(OUT["preds_plots"], f"{scen}__{model_label}__TEST_risk_distribution_adjusted.png"),
                        title=f"Risk distribution (TEST, adjusted) - {scen}",
                        subtitle="risk_test_adjusted; combined across splits",
                        xlabel="Predicted risk (probability)",
                        meta_lines=meta_adj,
                        dca_threshold=dca_threshold_test,
                    )
                if "risk_test_raw" in df_test_preds.columns:
                    meta_raw = _build_meta_lines(
                        scen,
                        model_label,
                        "TEST",
                        y_vals=df_test_preds["y_true"],
                        split_ids=df_test_preds.get("split_id"),
                        extra="prob=risk_test_raw",
                    )
                    _plot_risk_distribution(
                        df_test_preds["y_true"],
                        df_test_preds["risk_test_raw"],
                        os.path.join(OUT["preds_plots"], f"{scen}__{model_label}__TEST_risk_distribution_raw.png"),
                        title=f"Risk distribution (TEST, raw) - {scen}",
                        subtitle="risk_test_raw; combined across splits",
                        meta_lines=meta_raw,
                        dca_threshold=dca_threshold_test,
                    )
        if val_preds_all:
            df_val_preds = pd.concat(val_preds_all, ignore_index=True)
            df_val_preds.to_csv(
                os.path.join(OUT["preds_val"], f"{scen}__val_preds__{model_label}__all_splits.csv"),
                index=False,
            )
            # Use same DCA threshold as TEST (already loaded above)
            dca_threshold_val = dca_threshold_test

            if "y" in df_val_preds.columns and "p_active" in df_val_preds.columns:
                # Compute recalibration parameters for VAL
                calib_intercept_val, calib_slope_val = np.nan, np.nan
                calib_intercept_val_adj, calib_slope_val_adj = np.nan, np.nan
                try:
                    calib_intercept_val, calib_slope_val = _compute_recalibration(
                        df_val_preds["y"], df_val_preds["p_active"]
                    )
                except Exception as e:
                    print(f"[WARN] Failed to compute recalibration for VAL scenario={scen}: {e}")

                # Compute separate recalibration for adjusted if available
                if "p_adjusted" in df_val_preds.columns:
                    try:
                        calib_intercept_val_adj, calib_slope_val_adj = _compute_recalibration(
                            df_val_preds["y"], df_val_preds["p_adjusted"]
                        )
                    except Exception as e:
                        print(f"[WARN] Failed to compute recalibration for adjusted VAL scenario={scen}: {e}")

                meta_val = _build_meta_lines(
                    scen,
                    model_label,
                    "VAL",
                    y_vals=df_val_preds["y"],
                    split_ids=df_val_preds.get("split_id"),
                    extra="prob=p_active",
                )
                _plot_roc_curve(
                    df_val_preds["y"],
                    df_val_preds["p_active"],
                    os.path.join(OUT["diagnostics_plots"], f"{scen}__{model_label}__VAL_roc_curve.png"),
                    title=f"ROC Curve (VAL) - {scen}",
                    subtitle=f"{model_label} | mean ± SD, 95% CI",
                    split_ids=df_val_preds.get("split_id"),
                    meta_lines=meta_val,
                )
                _plot_pr_curve(
                    df_val_preds["y"],
                    df_val_preds["p_active"],
                    os.path.join(OUT["diagnostics_plots"], f"{scen}__{model_label}__VAL_pr_curve.png"),
                    title=f"PR Curve (VAL) - {scen}",
                    subtitle=f"{model_label} | mean ± SD, 95% CI",
                    split_ids=df_val_preds.get("split_id"),
                    meta_lines=meta_val,
                )
                # Generate 5-panel version for VAL
                _plot_calibration_curve(
                    df_val_preds["y"],
                    df_val_preds["p_active"],
                    os.path.join(OUT["diagnostics_plots"], f"{scen}__{model_label}__VAL_calibration_5panel.png"),
                    title=f"Calibration (VAL) - {scen}",
                    subtitle=f"{model_label} | mean ± SD, 95% CI",
                    n_bins=args.calib_bins,
                    split_ids=df_val_preds.get("split_id"),
                    meta_lines=meta_val,
                    four_panel=True,
                    calib_intercept=calib_intercept_val,
                    calib_slope=calib_slope_val,
                )
                if "p_adjusted" in df_val_preds.columns:
                    meta_val_adj = _build_meta_lines(
                        scen,
                        model_label,
                        "VAL",
                        y_vals=df_val_preds["y"],
                        split_ids=df_val_preds.get("split_id"),
                        extra="prob=p_adjusted",
                    )
                    # Generate 5-panel version for VAL adjusted
                    _plot_calibration_curve(
                        df_val_preds["y"],
                        df_val_preds["p_adjusted"],
                        os.path.join(OUT["diagnostics_plots"], f"{scen}__{model_label}__VAL_calibration_figure.png"),
                        title=f"Calibration (VAL, adjusted) - {scen}",
                        subtitle=f"{model_label} | mean ± SD, 95% CI",
                        n_bins=args.calib_bins,
                        split_ids=df_val_preds.get("split_id"),
                        meta_lines=meta_val_adj,
                        four_panel=True,
                        calib_intercept=calib_intercept_val_adj,
                        calib_slope=calib_slope_val_adj,
                    )
                _plot_dca_curve(
                    df_val_preds["y"],
                    df_val_preds["p_active"],
                    os.path.join(OUT["diagnostics_plots"], f"{scen}__{model_label}__VAL_dca_curve.png"),
                    title=f"DCA (VAL) - {scen}",
                    subtitle=f"{model_label} | mean ± SD, 95% CI",
                    max_pt=float(args.dca_max_pt),
                    step=float(args.dca_step),
                    split_ids=df_val_preds.get("split_id"),
                    meta_lines=meta_val,
                )

                _plot_risk_distribution(
                    df_val_preds["y"],
                    df_val_preds["p_active"],
                    os.path.join(OUT["preds_plots"], f"{scen}__{model_label}__VAL_risk_distribution.png"),
                    title=f"Risk distribution (VAL) - {scen}",
                    subtitle="p_active; combined across splits",
                    meta_lines=meta_val,
                    dca_threshold=dca_threshold_val,
                )
                if "p_adjusted" in df_val_preds.columns:
                    meta_val_adj = _build_meta_lines(
                        scen,
                        model_label,
                        "VAL",
                        y_vals=df_val_preds["y"],
                        split_ids=df_val_preds.get("split_id"),
                        extra="prob=p_adjusted",
                    )
                    _plot_risk_distribution(
                        df_val_preds["y"],
                        df_val_preds["p_adjusted"],
                        os.path.join(OUT["preds_plots"], f"{scen}__{model_label}__VAL_risk_distribution_adjusted.png"),
                        title=f"Risk distribution (VAL, adjusted) - {scen}",
                        subtitle="p_adjusted; combined across splits",
                        xlabel="Predicted risk (probability)",
                        meta_lines=meta_val_adj,
                        dca_threshold=dca_threshold_val,
                    )
                if "p_raw" in df_val_preds.columns:
                    meta_val_raw = _build_meta_lines(
                        scen,
                        model_label,
                        "VAL",
                        y_vals=df_val_preds["y"],
                        split_ids=df_val_preds.get("split_id"),
                        extra="prob=p_raw",
                    )
                    _plot_risk_distribution(
                        df_val_preds["y"],
                        df_val_preds["p_raw"],
                        os.path.join(OUT["preds_plots"], f"{scen}__{model_label}__VAL_risk_distribution_raw.png"),
                        title=f"Risk distribution (VAL, raw) - {scen}",
                        subtitle="p_raw; combined across splits",
                        meta_lines=meta_val_raw,
                        dca_threshold=dca_threshold_val,
                    )
        if train_oof_all:
            df_train_oof = pd.concat(train_oof_all, ignore_index=True)
            df_train_oof.to_csv(
                os.path.join(OUT["preds_train_oof"], f"{scen}__train_oof__{model_label}__all_splits.csv"),
                index=False,
            )
            y_train = None
            if "y_true" in df_train_oof.columns:
                y_train = df_train_oof["y_true"]
            elif "y" in df_train_oof.columns:
                y_train = df_train_oof["y"]

            if y_train is not None:
                pos_label = _pos_label_for_scenario(scen)
                train_cols = [
                    ("risk_train_oof", "active", "", "Predicted risk"),
                    ("risk_train_oof_adjusted", "adjusted", "adjusted", "Predicted risk (probability)"),
                    ("risk_train_oof_raw", "raw", "raw", "Predicted risk"),
                ]
                for col, label, suffix, xlabel in train_cols:
                    if col not in df_train_oof.columns:
                        continue
                    meta_train = _build_meta_lines(
                        scen,
                        model_label,
                        "TRAIN OOF",
                        y_vals=y_train,
                        split_ids=df_train_oof.get("split_id"),
                        extra=f"prob={col}",
                    )
                    suffix_tag = f"_{suffix}" if suffix else ""

                    # Check if CeD_comparison column exists for three-way categorization
                    category_col = None
                    if "CeD_comparison" in df_train_oof.columns:
                        category_col = df_train_oof["CeD_comparison"]

                    _plot_risk_distribution(
                        y_train,
                        df_train_oof[col],
                        os.path.join(OUT["preds_plots"], f"{scen}__{model_label}__TRAIN_OOF_risk_distribution{suffix_tag}.png"),
                        title=f"Risk distribution (TRAIN OOF{', ' + label if suffix else ''}) - {scen}",
                        subtitle=f"{col}; combined across splits",
                        xlabel=xlabel,
                        pos_label=pos_label,
                        meta_lines=meta_train,
                        category_col=category_col,
                    )

                    # Compute t-test comparing incident vs prevalent risk scores
                    ttest_result = compare_incident_prevalent_ttest(
                        df_train_oof,
                        risk_col=col,
                        category_col="CeD_comparison",
                        scenario=scen,
                        model_label=model_label,
                    )
                    if ttest_result is not None:
                        ttest_results.append(ttest_result)
                        print(f"[T-TEST] {scen} | {model_label} | {col}:")
                        print(f"  Incident (n={ttest_result['n_incident']}): mean={ttest_result['mean_incident']:.4f} ± {ttest_result['std_incident']:.4f}")
                        print(f"  Prevalent (n={ttest_result['n_prevalent']}): mean={ttest_result['mean_prevalent']:.4f} ± {ttest_result['std_prevalent']:.4f}")
                        print(f"  Difference: {ttest_result['mean_difference']:.4f}")
                        print(f"  t-statistic: {ttest_result['t_statistic']:.4f}, p-value: {ttest_result['p_value']:.6f}")
        if controls_all:
            df_ctrl = pd.concat(controls_all, ignore_index=True)
            df_ctrl.to_csv(
                os.path.join(OUT["preds_controls"], f"{scen}__controls_risk__{model_label}__all_splits.csv"),
                index=False,
            )
            ctrl_cols = [c for c in df_ctrl.columns if c.startswith("risk_") and "pct" not in c]
            for col in ctrl_cols:
                meta_ctrl = _build_meta_lines(
                    scen,
                    model_label,
                    "TRAIN OOF (controls)",
                    y_vals=None,
                    split_ids=df_ctrl.get("split_id"),
                    extra=f"prob={col}",
                )
                _plot_risk_distribution(
                    None,
                    df_ctrl[col],
                    os.path.join(OUT["preds_plots"], f"{scen}__{model_label}__TRAIN_controls_{col}.png"),
                    title=f"Controls risk distribution (TRAIN OOF) - {scen}",
                    subtitle=f"{col}; combined across splits",
                    meta_lines=meta_ctrl,
                )

        lc_frames = lc_frames_by_scen.get(scen, [])
        lc_summary = _aggregate_learning_curve_runs(lc_frames)
        if not lc_summary.empty:
            lc_csv = os.path.join(OUT["diagnostics_lc"], f"{scen}__{model_label}__learning_curve_summary.csv")
            lc_summary.to_csv(lc_csv, index=False)
            metric_label = lc_summary["metric_label"].iloc[0] if "metric_label" in lc_summary.columns else ""
            metric_dir = lc_summary["metric_direction"].iloc[0] if "metric_direction" in lc_summary.columns else ""
            scoring = lc_summary["scoring"].iloc[0] if "scoring" in lc_summary.columns else ""
            lc_meta = [
                f"Scenario={scen} | Model={model_label} | Split=TRAIN (learning curve)",
                f"runs={int(lc_summary['n_runs'].max()) if 'n_runs' in lc_summary.columns else len(lc_frames)}",
                f"scoring={scoring} | metric={metric_label} | direction={metric_dir}" if (scoring or metric_label or metric_dir) else "",
            ]
            lc_plot = os.path.join(OUT["diagnostics_plots"], f"{scen}__{model_label}__learning_curve.png")
            _plot_learning_curve_summary(
                lc_summary,
                lc_plot,
                title=f"Learning Curve (TRAIN) - {scen}",
                meta_lines=lc_meta,
            )

        # Copy tuning history plots from first available run
        tuning_plots = tuning_plots_by_scen.get(scen, [])
        if tuning_plots:
            src_tuning_plot = tuning_plots[0]  # Use from first seed
            dest_tuning_plot = os.path.join(OUT["diagnostics_plots"], f"{scen}__{model_label}__tuning_history.png")
            try:
                import shutil
                shutil.copy2(src_tuning_plot, dest_tuning_plot)
            except Exception as e:
                print(f"[WARN] Could not copy tuning history plot from {src_tuning_plot}: {e}")

    # Save t-test results to CSV
    if ttest_results:
        df_ttest = pd.DataFrame(ttest_results)
        ttest_csv = os.path.join(OUT["core"], f"{model_label}__incident_vs_prevalent_ttest__train.csv")
        df_ttest.to_csv(ttest_csv, index=False)
        print(f"[OK] Wrote t-test results to: {ttest_csv}")

    print(f"[OK] Wrote single-model outputs to: {out_root}")


def main():
    """Main entry point with mode selection."""
    ap = argparse.ArgumentParser(
        description="Postprocess ML results: compare models or aggregate sensitivity analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Model comparison mode (default):
  python postprocess_compare.py --results_dir results/ML/results_faith_full

  # Sensitivity analysis mode (after running with SENSITIVITY_MODE=1):
  python postprocess_compare.py --mode sensitivity --results_dir results/ML/results_faith_full/sensitivity
"""
    )
    ap.add_argument("--mode", choices=["models", "sensitivity", "single"], default="models",
                    help="Mode: 'models' compares models (default), 'sensitivity' aggregates screen_top_n results")
    ap.add_argument("--results_dir", required=True,
                    help="Base results directory")
    ap.add_argument("--dca_max_pt", type=float, default=0.20,
                    help="Max threshold for cross-model DCA visualization (models mode only)")
    ap.add_argument("--dca_step", type=float, default=0.005,
                    help="Threshold step for cross-model DCA visualization (models mode only)")
    ap.add_argument("--dca_threshold_min", type=float, default=0.0005,
                    help="Minimum threshold for per-model DCA curves (default 0.0005)")
    ap.add_argument("--dca_threshold_max", type=float, default=1.0,
                    help="Maximum threshold for per-model DCA curves (default 1.0)")
    ap.add_argument("--dca_threshold_step", type=float, default=0.001,
                    help="Step size for per-model DCA curves (default 0.001)")
    ap.add_argument("--dca_report_points", type=str, default="0.005,0.01,0.02,0.05",
                    help="Comma-separated thresholds for DCA summary reporting")
    ap.add_argument("--calib_bins", type=int, default=10,
                    help="Number of bins for calibration plots")
    ap.add_argument("--model", type=str, default="",
                    help="Model name for single-mode aggregation (e.g., LR_EN)")
    ap.add_argument("--single_outdir", type=str, default=None,
                    help="Optional output directory for single-mode aggregation")
    ap.add_argument("--expected_splits", type=int, default=0,
                    help="Require this many splits in single-mode aggregation (0 = no check)")
    args = ap.parse_args()

    # Parse DCA report points from string to list
    if hasattr(args, 'dca_report_points') and args.dca_report_points:
        args.dca_report_points = _parse_dca_report_points(args.dca_report_points)

    if args.mode == "sensitivity":
        run_sensitivity_mode(args)
    elif args.mode == "single":
        run_single_model_summary(args)
    else:
        run_model_comparison(args)


if __name__ == "__main__":
    main()
