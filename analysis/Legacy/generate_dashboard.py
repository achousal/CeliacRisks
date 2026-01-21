#!/usr/bin/env python3
"""
generate_dashboard.py

Creates an interactive HTML dashboard with plotly for ML model comparison.

Features:
- Interactive ROC/PR curves with hover tooltips
- Per-split and aggregated performance metrics
- Model category filtering (scenario, model type, config)
- Aggregated mean curves across models
- Hyperparameter tuning history across all splits
- Calibration curves with confidence intervals
- Decision curve analysis comparison
- Performance summary tables with sorting/filtering

Usage:
    python generate_dashboard.py --results_dir results_holdout --outfile dashboard.html
"""

import argparse
import os
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
except ImportError:
    print("ERROR: plotly not installed. Install with: pip install plotly", file=sys.stderr)
    sys.exit(1)

# Color palette for models
MODEL_COLORS = px.colors.qualitative.Set1 + px.colors.qualitative.Set2


def find_model_dirs(results_dir: str) -> List[Path]:
    """Find all model output directories."""
    results_path = Path(results_dir)
    model_dirs = []

    # Look for directories matching pattern: scenario__model__config
    for item in results_path.iterdir():
        if item.is_dir() and "__" in item.name:
            core_dir = item / "core"
            if core_dir.exists() and (core_dir / "test_metrics.csv").exists():
                model_dirs.append(item)

    return sorted(model_dirs)


def load_test_metrics(model_dir: Path) -> Optional[pd.DataFrame]:
    """Load test metrics CSV."""
    metrics_path = model_dir / "core" / "test_metrics.csv"
    if not metrics_path.exists():
        return None
    return pd.read_csv(metrics_path)


def load_test_predictions(model_dir: Path) -> Optional[pd.DataFrame]:
    """Load test predictions if available."""
    preds_dir = model_dir / "preds" / "test_preds"
    if not preds_dir.exists():
        return None

    preds_files = list(preds_dir.glob("*__test_preds__*.csv"))
    if not preds_files:
        return None

    return pd.read_csv(preds_files[0])


def load_val_predictions(model_dir: Path) -> Optional[pd.DataFrame]:
    """Load validation predictions if available."""
    preds_dir = model_dir / "preds" / "val_preds"
    if not preds_dir.exists():
        return None

    preds_files = list(preds_dir.glob("*__val_preds__*.csv"))
    if not preds_files:
        return None

    return pd.read_csv(preds_files[0])


def load_controls_oof(model_dir: Path) -> Optional[pd.DataFrame]:
    """Load controls OOF risk predictions if available."""
    preds_dir = model_dir / "preds" / "controls_oof"
    if not preds_dir.exists():
        return None

    preds_files = list(preds_dir.glob("*__controls_risk__*__oof_mean.csv"))
    if not preds_files:
        return None

    return pd.read_csv(preds_files[0])


def load_train_oof(model_dir: Path) -> Optional[pd.DataFrame]:
    """Load TRAIN OOF predictions (controls + cases) if available."""
    preds_dir = model_dir / "preds" / "train_oof"
    if not preds_dir.exists():
        return None

    preds_files = list(preds_dir.glob("*__train_oof__*.csv"))
    if not preds_files:
        return None

    return pd.read_csv(preds_files[0])


def load_learning_curve(model_dir: Path) -> Optional[pd.DataFrame]:
    """Load learning curve CSV if available."""
    lc_dir = model_dir / "diagnostics" / "learning_curve"
    if not lc_dir.exists():
        return None

    lc_files = list(lc_dir.glob("*__learning_curve__*.csv"))
    if not lc_files:
        return None

    return pd.read_csv(lc_files[0])


def load_dca_curve(model_dir: Path) -> Optional[pd.DataFrame]:
    """Load DCA curve data."""
    dca_dir = model_dir / "diagnostics" / "dca"
    if not dca_dir.exists():
        return None

    dca_files = list(dca_dir.glob("*__dca_curve.csv"))
    if not dca_files:
        return None

    return pd.read_csv(dca_files[0])


def load_calibration_data(model_dir: Path) -> Optional[pd.DataFrame]:
    """Load calibration curve data."""
    calib_dir = model_dir / "diagnostics" / "calibration"
    if not calib_dir.exists():
        return None

    calib_files = list(calib_dir.glob("*__calibration__*.csv"))
    if not calib_files:
        return None

    df = pd.read_csv(calib_files[0])
    # Normalize column names (celiacML_faith.py uses prob_pred/prob_true)
    col_map = {
        "prob_pred": "predicted_probability",
        "prob_true": "observed_frequency",
    }
    df = df.rename(columns=col_map)
    return df


def parse_model_name(dirname: str) -> Dict[str, str]:
    """Extract scenario, model, and config from directory name."""
    parts = dirname.split("__")
    if len(parts) >= 2:
        return {
            "scenario": parts[0],
            "model": parts[1],
            "config": "__".join(parts[2:]) if len(parts) > 2 else ""
        }
    return {"scenario": "", "model": dirname, "config": ""}


def get_unique_categories(all_data: List[Dict]) -> Dict[str, List[str]]:
    """Extract unique scenarios, models, and configs for filtering."""
    scenarios = sorted(set(d["scenario"] for d in all_data if d["scenario"]))
    models = sorted(set(d["model"] for d in all_data if d["model"]))
    configs = sorted(set(d["config"] for d in all_data if d["config"]))
    return {"scenarios": scenarios, "models": models, "configs": configs}


def infer_dashboard_name(all_data: List[Dict], outfile: str) -> str:
    """Infer a readable dashboard name from the model context."""
    models = sorted({d.get("model", "") for d in all_data if d.get("model")})
    if len(models) == 1:
        return f"{models[0]}_Dashboard"
    if len(models) > 1:
        return "Combined_Dashboard"
    return Path(outfile).stem or "Dashboard"


def compute_mean_roc(all_data: List[Dict], group_by: str = "model") -> Dict[str, Dict]:
    """Compute mean ROC curves grouped by model type.

    Returns dict mapping group name to {'fpr': array, 'tpr_mean': array, 'tpr_std': array, 'auc_mean': float, 'auc_std': float}
    """
    from sklearn.metrics import roc_curve, auc

    # Group data
    groups = {}
    for data in all_data:
        if data["predictions"] is None:
            continue
        preds = data["predictions"]
        if "y_true" not in preds.columns or "risk_test" not in preds.columns:
            continue

        key = data.get(group_by, "unknown")
        if key not in groups:
            groups[key] = []
        groups[key].append(preds)

    # Compute mean curves per group
    mean_curves = {}
    base_fpr = np.linspace(0, 1, 100)

    for group_name, preds_list in groups.items():
        tprs = []
        aucs = []

        for preds in preds_list:
            y_true = preds["y_true"].values
            y_pred = preds["risk_test"].values
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)

            # Interpolate to common FPR values
            tpr_interp = np.interp(base_fpr, fpr, tpr)
            tpr_interp[0] = 0.0
            tprs.append(tpr_interp)
            aucs.append(roc_auc)

        if tprs:
            mean_curves[group_name] = {
                "fpr": base_fpr,
                "tpr_mean": np.mean(tprs, axis=0),
                "tpr_std": np.std(tprs, axis=0),
                "tpr_ci_lo": np.nanpercentile(tprs, 2.5, axis=0),
                "tpr_ci_hi": np.nanpercentile(tprs, 97.5, axis=0),
                "auc_mean": np.mean(aucs),
                "auc_std": np.std(aucs),
                "n": len(tprs)
            }

    return mean_curves


def compute_mean_pr(all_data: List[Dict], group_by: str = "model") -> Dict[str, Dict]:
    """Compute mean PR curves grouped by model type."""
    from sklearn.metrics import precision_recall_curve, average_precision_score

    groups = {}
    for data in all_data:
        if data["predictions"] is None:
            continue
        preds = data["predictions"]
        if "y_true" not in preds.columns or "risk_test" not in preds.columns:
            continue

        key = data.get(group_by, "unknown")
        if key not in groups:
            groups[key] = []
        groups[key].append(preds)

    mean_curves = {}
    base_recall = np.linspace(0, 1, 100)

    for group_name, preds_list in groups.items():
        precisions = []
        aps = []

        for preds in preds_list:
            y_true = preds["y_true"].values
            y_pred = preds["risk_test"].values
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            ap = average_precision_score(y_true, y_pred)

            # Interpolate (recall is decreasing, so flip)
            precision_interp = np.interp(base_recall, recall[::-1], precision[::-1])
            precisions.append(precision_interp)
            aps.append(ap)

        if precisions:
            mean_curves[group_name] = {
                "recall": base_recall,
                "precision_mean": np.mean(precisions, axis=0),
                "precision_std": np.std(precisions, axis=0),
                "precision_ci_lo": np.nanpercentile(precisions, 2.5, axis=0),
                "precision_ci_hi": np.nanpercentile(precisions, 97.5, axis=0),
                "ap_mean": np.mean(aps),
                "ap_std": np.std(aps),
                "n": len(precisions)
            }

    return mean_curves


def compute_mean_calibration(all_data: List[Dict], group_by: str = "model") -> Dict[str, Dict]:
    """Compute mean calibration curves grouped by model type."""
    groups = {}
    for data in all_data:
        if data["calibration"] is None:
            continue
        calib = data["calibration"]
        if "predicted_probability" not in calib.columns or "observed_frequency" not in calib.columns:
            continue

        key = data.get(group_by, "unknown")
        if key not in groups:
            groups[key] = []
        groups[key].append(calib)

    mean_curves = {}
    base_pred = np.linspace(0, 1, 10)

    for group_name, calib_list in groups.items():
        observed = []

        for calib in calib_list:
            pred_prob = calib["predicted_probability"].values
            obs_freq = calib["observed_frequency"].values
            # Interpolate
            obs_interp = np.interp(base_pred, pred_prob, obs_freq)
            observed.append(obs_interp)

        if observed:
            mean_curves[group_name] = {
                "predicted": base_pred,
                "observed_mean": np.mean(observed, axis=0),
                "observed_std": np.std(observed, axis=0),
                "observed_ci_lo": np.nanpercentile(observed, 2.5, axis=0),
                "observed_ci_hi": np.nanpercentile(observed, 97.5, axis=0),
                "n": len(observed)
            }

    return mean_curves


def compute_mean_dca(all_data: List[Dict], group_by: str = "model") -> Dict[str, Dict]:
    """Compute mean DCA curves grouped by model type."""
    groups = {}
    for data in all_data:
        if data["dca"] is None:
            continue
        dca = data["dca"]
        if "threshold_pct" not in dca.columns or "net_benefit_model" not in dca.columns:
            continue

        key = data.get(group_by, "unknown")
        if key not in groups:
            groups[key] = []
        groups[key].append(dca)

    mean_curves = {}

    for group_name, dca_list in groups.items():
        # Use thresholds from first DCA as base
        base_thresh = dca_list[0]["threshold_pct"].values
        nb_models = []
        nb_alls = []
        nb_nones = []

        for dca in dca_list:
            nb_model = np.interp(base_thresh, dca["threshold_pct"].values, dca["net_benefit_model"].values)
            nb_all = np.interp(base_thresh, dca["threshold_pct"].values, dca["net_benefit_all"].values)
            nb_none = np.interp(base_thresh, dca["threshold_pct"].values, dca["net_benefit_none"].values)
            nb_models.append(nb_model)
            nb_alls.append(nb_all)
            nb_nones.append(nb_none)

        if nb_models:
            mean_curves[group_name] = {
                "threshold_pct": base_thresh,
                "nb_model_mean": np.mean(nb_models, axis=0),
                "nb_model_std": np.std(nb_models, axis=0),
                "nb_model_ci_lo": np.nanpercentile(nb_models, 2.5, axis=0),
                "nb_model_ci_hi": np.nanpercentile(nb_models, 97.5, axis=0),
                "nb_all_mean": np.mean(nb_alls, axis=0),
                "nb_none_mean": np.mean(nb_nones, axis=0),
                "n": len(nb_models)
            }

    return mean_curves


def build_risk_dataframe(all_data: List[Dict]) -> pd.DataFrame:
    target_col = "CeD_comparison"

    def _normalize_outcome(label: str) -> str:
        if label == "Controls":
            return "Control"
        return label

    rows = []
    for data in all_data:
        scenario = data.get("scenario", "")
        model = data.get("model", "")

        test_df = data.get("predictions")
        if test_df is not None and "y_true" in test_df.columns:
            if target_col in test_df.columns:
                outcomes = test_df[target_col].astype(str).map(_normalize_outcome).values
            else:
                outcomes = np.where(test_df["y_true"].astype(int) == 1, "Incident", "Control")
            test_cols = [("risk_test_adjusted", "adjusted"), ("risk_test_raw", "raw")]
            for col, label in test_cols:
                if col not in test_df.columns:
                    continue
                for outcome, score in zip(outcomes, test_df[col].values):
                    if not np.isfinite(score):
                        continue
                    rows.append({
                        "scenario": scenario,
                        "model": model,
                        "split": "TEST",
                        "prob_type": label,
                        "outcome": outcome,
                        "score": float(score),
                    })

        val_df = data.get("val_predictions")
        if val_df is not None and "y" in val_df.columns:
            val_cols = [("p_adjusted", "adjusted"), ("p_raw", "raw")]
            for col, label in val_cols:
                if col not in val_df.columns:
                    continue
                for y_val, score in zip(val_df["y"].values, val_df[col].values):
                    if not np.isfinite(score):
                        continue
                    rows.append({
                        "scenario": scenario,
                        "model": model,
                        "split": "VAL",
                        "prob_type": label,
                        "outcome": "Incident" if int(y_val) == 1 else "Control",
                        "score": float(score),
                    })

        train_df = data.get("train_oof")
        if train_df is not None:
            if target_col in train_df.columns:
                outcomes = train_df[target_col].astype(str).map(_normalize_outcome).values
            elif "y_true" in train_df.columns:
                outcomes = np.where(train_df["y_true"].astype(int) == 1, "Incident", "Control")
            elif "y" in train_df.columns:
                outcomes = np.where(train_df["y"].astype(int) == 1, "Incident", "Control")
            else:
                outcomes = None

            train_cols = [("risk_train_oof_adjusted", "adjusted"), ("risk_train_oof_raw", "raw")]
            for col, label in train_cols:
                if col not in train_df.columns or outcomes is None:
                    continue
                for outcome, score in zip(outcomes, train_df[col].values):
                    if not np.isfinite(score):
                        continue
                    rows.append({
                        "scenario": scenario,
                        "model": model,
                        "split": "TRAIN",
                        "prob_type": label,
                        "outcome": outcome,
                        "score": float(score),
                    })
        else:
            ctrl_df = data.get("controls_oof")
            if ctrl_df is not None:
                ctrl_cols = []
                for col in ctrl_df.columns:
                    if not col.startswith("risk_") or col.endswith("_pct"):
                        continue
                    if col.endswith("_adjusted"):
                        ctrl_cols.append((col, "adjusted"))
                    elif col.endswith("_raw"):
                        ctrl_cols.append((col, "raw"))
                for col, label in ctrl_cols:
                    for score in ctrl_df[col].values:
                        if not np.isfinite(score):
                            continue
                        rows.append({
                            "scenario": scenario,
                            "model": model,
                            "split": "TRAIN",
                            "prob_type": label,
                            "outcome": "Control",
                            "score": float(score),
                        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def create_risk_distribution_figure(risk_df: pd.DataFrame, split_label: str) -> Optional[go.Figure]:
    df = risk_df[risk_df["split"] == split_label].copy()
    if df.empty:
        return None

    prob_types = [p for p in ["adjusted", "raw"] if p in df["prob_type"].unique().tolist()]
    if not prob_types:
        prob_types = sorted(df["prob_type"].unique().tolist())

    fig = go.Figure()
    trace_probs = []

    for prob in prob_types:
        df_p = df[df["prob_type"] == prob]
        for (model, scenario, outcome), df_g in df_p.groupby(["model", "scenario", "outcome"]):
            fig.add_trace(go.Histogram(
                x=df_g["score"],
                histnorm="probability density",
                opacity=0.45,
                name=f"{model} {outcome} ({scenario})",
                visible=(prob == prob_types[0]),
                nbinsx=60,
            ))
            trace_probs.append(prob)

    buttons = []
    for prob in prob_types:
        vis = [tp == prob for tp in trace_probs]
        buttons.append(dict(
            label=prob,
            method="update",
            args=[
                {"visible": vis},
                {"title": f"Risk score distributions ({split_label}) - {prob}"}
            ],
        ))

    fig.update_layout(
        title=f"Risk score distributions ({split_label}) - {prob_types[0]}",
        xaxis_title="Predicted risk",
        yaxis_title="Density",
        barmode="overlay",
        width=1000,
        height=550,
        template="plotly_white",
        updatemenus=[dict(
            buttons=buttons,
            direction="down",
            x=0.01,
            xanchor="left",
            y=1.15,
            yanchor="top",
        )],
        legend=dict(tracegroupgap=4),
    )
    return fig


def _calibration_bins_from_preds(y_true, y_pred, n_bins=10, bin_strategy="quantile"):
    """Compute binned calibration data.

    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        n_bins: Number of bins
        bin_strategy: 'uniform' (equal-width) or 'quantile' (equal-count, better for imbalanced) bins

    Returns:
        Tuple of (bin_centers, observed_frequencies, counts, predictions) or None
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred).astype(float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if len(y) == 0:
        return None

    if bin_strategy == "quantile":
        quantiles = np.linspace(0, 100, int(n_bins) + 1)
        bins = np.percentile(p, quantiles)
        bins = np.unique(bins)
        if len(bins) < 3:
            bins = np.linspace(0, 1, int(n_bins) + 1)
    else:
        bins = np.linspace(0, 1, int(n_bins) + 1)

    actual_n_bins = len(bins) - 1
    centers = (bins[:-1] + bins[1:]) / 2
    idx = np.digitize(p, bins) - 1
    idx = np.clip(idx, 0, actual_n_bins - 1)
    obs = []
    counts = []
    for i in range(actual_n_bins):
        m = idx == i
        obs.append(np.nan if m.sum() == 0 else float(y[m].mean()))
        counts.append(int(m.sum()))
    return centers, np.array(obs, dtype=float), np.array(counts, dtype=int), p


def _binned_logodds(y_true, y_pred, n_bins=10, bin_strategy="quantile"):
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred).astype(float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if len(y) == 0:
        return None, None

    if bin_strategy == "quantile":
        quantiles = np.linspace(0, 100, int(n_bins) + 1)
        bins = np.percentile(p, quantiles)
        bins = np.unique(bins)
        if len(bins) < 3:
            bins = np.linspace(0, 1, int(n_bins) + 1)
    else:
        bins = np.linspace(0, 1, int(n_bins) + 1)

    eps = 1e-7
    idx = np.digitize(p, bins) - 1
    idx = np.clip(idx, 0, len(bins) - 2)
    xs, ys = [], []
    for i in range(len(bins) - 1):
        m = idx == i
        if not np.any(m):
            continue
        pred_mean = float(np.mean(p[m]))
        obs_mean = float(np.mean(y[m]))
        pred_mean = float(np.clip(pred_mean, eps, 1 - eps))
        obs_mean = float(np.clip(obs_mean, eps, 1 - eps))
        xs.append(np.log(pred_mean / (1 - pred_mean)))
        ys.append(np.log(obs_mean / (1 - obs_mean)))
    if len(xs) < 2:
        return None, None
    order = np.argsort(xs)
    return np.array(xs, dtype=float)[order], np.array(ys, dtype=float)[order]


def _compute_loess_logodds(y_true, y_pred, n_bins=10, bin_strategy="quantile"):
    """Compute LOESS-smoothed log-odds calibration curve (fallback to binned log-odds).

    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities

    Returns:
        Tuple of (loess_x, loess_logit_y) or (None, None) on failure
    """
    eps = 1e-7
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred).astype(float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if len(y) < 10:
        return None, None

    p_clipped = np.clip(p, eps, 1 - eps)
    logit_pred = np.log(p_clipped / (1 - p_clipped))

    sort_idx = np.argsort(logit_pred)
    logit_sorted = logit_pred[sort_idx]
    y_sorted = y[sort_idx]

    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
    except ImportError:
        return _binned_logodds(y, p_clipped, n_bins=n_bins, bin_strategy=bin_strategy)

    try:
        loess_result = lowess(y_sorted, logit_sorted, frac=0.3, return_sorted=True)
        loess_x = loess_result[:, 0]
        loess_y = loess_result[:, 1]
        loess_y_clipped = np.clip(loess_y, eps, 1 - eps)
        loess_logit_y = np.log(loess_y_clipped / (1 - loess_y_clipped))
        return loess_x, loess_logit_y
    except Exception:
        return _binned_logodds(y, p_clipped, n_bins=n_bins, bin_strategy=bin_strategy)


def create_mean_calibration_with_distribution(all_data: List[Dict], prob_type: str, split_label: str = "TEST") -> Optional[go.Figure]:
    """Create four-panel calibration figure:
       - Top-Left: Probability Calibration (Quantile bins)
       - Top-Right: Probability Calibration (Uniform bins)
       - Bottom-Left: Log-Odds Calibration + Recalibration Line
       - Bottom-Right: Prediction Distribution
    """
    groups = {}
    preds_by_model = {}
    for data in all_data:
        if split_label.upper() == "VAL":
            preds = data.get("val_predictions")
            y_col = "y"
            col = "p_adjusted" if prob_type == "adjusted" else "p_raw"
        else:
            preds = data.get("predictions")
            y_col = "y_true"
            col = "risk_test_adjusted" if prob_type == "adjusted" else "risk_test_raw"

        if preds is None or y_col not in preds.columns or col not in preds.columns:
            continue

        key = data.get("model", "unknown")
        groups.setdefault(key, []).append((preds[y_col].values, preds[col].values))
        preds_by_model.setdefault(key, []).extend(preds[col].values.tolist())

    if not groups:
        return None

    # Four-panel subplot: 2x2 grid
    fig = make_subplots(
        rows=2, cols=2,
        shared_xaxes=False,
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
        subplot_titles=[
            "Calibration (Quantile Bins)", "Calibration (Uniform Bins)",
            "Log-Odds Calibration", "Prediction Distribution"
        ]
    )

    # Collect all logit values for axis range in Panel 3
    all_logits = []

    for i, (model_name, runs) in enumerate(sorted(groups.items())):
        # Default to steelblue if single model (as requested), else palette
        if len(groups) == 1:
            color = "steelblue"
        else:
            color = MODEL_COLORS[i % len(MODEL_COLORS)]
        
        # --- Helper for probability panels ---
        def _add_prob_panel(strat, row, col):
            curves = []
            counts_all = []
            for y_true, y_pred in runs:
                res = _calibration_bins_from_preds(y_true, y_pred, n_bins=10, bin_strategy=strat)
                if res is None:
                    continue
                centers, obs, counts, _ = res
                curves.append(obs)
                counts_all.append(counts)

            if not curves:
                return

            curves = np.array(curves, dtype=float)
            counts_all = np.array(counts_all, dtype=float)
            mean_obs = np.nanmean(curves, axis=0)
            sd_obs = np.nanstd(curves, axis=0)
            ci_lo = np.nanpercentile(curves, 2.5, axis=0)
            ci_hi = np.nanpercentile(curves, 97.5, axis=0)
            mean_counts = np.nanmean(counts_all, axis=0)
            
            # CI band
            fig.add_trace(go.Scatter(
                x=np.concatenate([centers, centers[::-1]]),
                y=np.concatenate([np.clip(ci_hi, 0, 1), np.clip(ci_lo, 0, 1)[::-1]]),
                fill="toself",
                fillcolor=color.replace("rgb", "rgba").replace(")", ",0.12)") if "rgb" in color else ("rgba(70, 130, 180, 0.12)" if color == "steelblue" else "rgba(128,128,128,0.12)"),
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
                hoverinfo="skip",
                name=f"{model_name} {strat} CI",
            ), row=row, col=col)

            # SD band
            fig.add_trace(go.Scatter(
                x=np.concatenate([centers, centers[::-1]]),
                y=np.concatenate([np.clip(mean_obs + sd_obs, 0, 1), np.clip(mean_obs - sd_obs, 0, 1)[::-1]]),
                fill="toself",
                fillcolor=color.replace("rgb", "rgba").replace(")", ",0.2)") if "rgb" in color else ("rgba(70, 130, 180, 0.2)" if color == "steelblue" else "rgba(128,128,128,0.2)"),
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
                hoverinfo="skip",
                name=f"{model_name} {strat} SD",
            ), row=row, col=col)
            
            # Mean line with annotations
            hover_text = [f"{model_name} ({strat})<br>Predicted: {c:.3f}<br>Observed: {o:.3f}<br>n≈{int(n)}"
                          for c, o, n in zip(centers, mean_obs, mean_counts)]
            
            # Annotations text
            anno_text = [f"n≈{int(n)}" if n > 0 else "" for n in mean_counts]
            
            fig.add_trace(go.Scatter(
                x=centers,
                y=mean_obs,
                mode="lines+markers+text",
                line=dict(color=color, width=2),
                text=anno_text,
                textposition="top center",
                textfont=dict(size=9, color='darkblue'),
                name=f"{model_name}",
                hovertext=hover_text,
                hoverinfo="text",
                showlegend=(strat == "quantile"), # Only show legend once
            ), row=row, col=col)

        # === Panel 1: Quantile bins ===
        _add_prob_panel("quantile", 1, 1)

        # === Panel 2: Uniform bins ===
        _add_prob_panel("uniform", 1, 2)

        # === Panel 3: Log-odds calibration with LOESS + Recalibration Line ===
        loess_curves_x = []
        loess_curves_y = []
        
        # Calculate recalibration (intercept/slope) across all runs (averaged)
        recal_intercepts = []
        recal_slopes = []

        from sklearn.linear_model import LogisticRegression

        for y_true, y_pred in runs:
            # LOESS computation
            lx, ly = _compute_loess_logodds(y_true, y_pred, n_bins=10)
            if lx is not None and ly is not None:
                loess_curves_x.append(lx)
                loess_curves_y.append(ly)
            y_arr = np.asarray(y_true).astype(int)
            p_arr = np.asarray(y_pred).astype(float)
            mask = np.isfinite(y_arr) & np.isfinite(p_arr)
            p_arr = p_arr[mask]
            y_arr = y_arr[mask]
            
            if len(p_arr) > 0:
                # Logits for range
                eps = 1e-7
                p_clip = np.clip(p_arr, eps, 1 - eps)
                logit_vals = np.log(p_clip / (1 - p_clip))
                all_logits.extend(logit_vals.tolist())
                
                # Recalibration calc
                if len(np.unique(y_arr)) >= 2:
                    try:
                        lr = LogisticRegression(C=1e9, solver="lbfgs")
                        lr.fit(logit_vals.reshape(-1, 1), y_arr)
                        recal_intercepts.append(lr.intercept_[0])
                        recal_slopes.append(lr.coef_[0][0])
                    except Exception:
                        pass

        # Plot Averaged Recalibration Line
        if recal_intercepts:
            mean_int = np.mean(recal_intercepts)
            mean_slope = np.mean(recal_slopes)
            
            # Line range: use -5 to 5 or data range
            line_x = np.linspace(-5, 5, 100)
            line_y = mean_int + mean_slope * line_x
            
            fig.add_trace(go.Scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                line=dict(color=color, width=1.5, dash="dot"),
                name=f"{model_name} Recal (α={mean_int:.2f}, β={mean_slope:.2f})",
                showlegend=False,
                hoverinfo="name"
            ), row=2, col=1)

        # Helper to get rgba fill color (reused logic)
        def _get_fill(c, opacity):
            if c == "steelblue":
                return f"rgba(70, 130, 180, {opacity})"
            elif c == "crimson":
                return f"rgba(220, 20, 60, {opacity})"
            elif c.startswith("rgb"):
                return c.replace("rgb", "rgba").replace(")", f",{opacity})")
            elif c.startswith("#"):
                h = c.lstrip('#')
                return f"rgba({int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}, {opacity})"
            return f"rgba(128,128,128,{opacity})"

        # Plot Aggregated LOESS
        if loess_curves_x:
            # Interpolate to common x for aggregation
            all_x = np.concatenate(loess_curves_x)
            if len(all_x) > 1:
                # Ensure base_x covers the range
                base_x = np.linspace(np.min(all_x), np.max(all_x), 100)
            else:
                base_x = np.array([-5, 5])

            interp_y = []
            for lx, ly in zip(loess_curves_x, loess_curves_y):
                # Ensure lx is strictly increasing for interp, though _compute_loess usually ensures this
                # If duplicates exist, np.interp is fine, but sorting is key
                sort_i = np.argsort(lx)
                interp_y.append(np.interp(base_x, lx[sort_i], ly[sort_i]))

            interp_y = np.array(interp_y)
            mean_y = np.nanmean(interp_y, axis=0)
            sd_y = np.nanstd(interp_y, axis=0)
            
            # Sanity check: if mean_y is all NaN, skip trace
            if not np.all(np.isnan(mean_y)):
                # SD band for log-odds
                fig.add_trace(go.Scatter(
                    x=np.concatenate([base_x, base_x[::-1]]),
                    y=np.concatenate([mean_y + sd_y, (mean_y - sd_y)[::-1]]),
                    fill="toself",
                    fillcolor=_get_fill(color, 0.2),
                    line=dict(color="rgba(255,255,255,0)"),
                    showlegend=False,
                    hoverinfo="skip",
                    name=f"{model_name} logodds SD",
                ), row=2, col=1)

            # Mean LOESS line
            fig.add_trace(go.Scatter(
                x=base_x,
                y=mean_y,
                mode="lines",
                line=dict(color=color, width=2.5),
                name=f"{model_name} LOESS",
                showlegend=False,
                hovertemplate=f"{model_name}<br>logit(p̂): %{{x:.2f}}<br>Empirical logit: %{{y:.2f}}<extra></extra>",
            ), row=2, col=1)

        # === Panel 4: Distribution Histogram ===
        pred_vals = np.array(preds_by_model.get(model_name, []), dtype=float)
        pred_vals = pred_vals[np.isfinite(pred_vals)]
        if len(pred_vals) > 0:
            fig.add_trace(go.Histogram(
                x=pred_vals,
                nbinsx=50,
                opacity=0.45,
                histnorm="probability density",
                name=f"{model_name} dist",
                showlegend=False,
                marker=dict(color=color),
            ), row=2, col=2)

    # Reference lines
    # Panel 1 & 2: Perfect calibration
    for c in [1, 2]:
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="black"),
            name="Perfect calibration",
            showlegend=False
        ), row=1, col=c)

    # Panel 3: Ideal Log-Odds
    # Panel 3: Ideal Log-Odds & Axis Styling
    # Logic: X-axis based on predictions (logits), Y-axis based on Empirical (LOESS)
    # matching postprocess_compare.py logic
    
    # 1. Determine X-range (Predicted Log-Odds)
    if all_logits:
        logit_min = np.percentile(all_logits, 1)
        logit_max = np.percentile(all_logits, 99)
        # Default range if data is weird or too narrow, centered on 0 usually
        # But commonly we want roughly -5 to 5 for log-odds
        logit_min = min(logit_min, -5)
        logit_max = max(logit_max, 4)
    else:
        logit_min, logit_max = -5, 5
    
    logit_range_x = [logit_min - 0.5, logit_max + 0.5]

    # 2. Determine Y-range (Empirical Log-Odds)
    # Start with X-range (identity)
    logit_range_y = list(logit_range_x)
    
    # Expand Y if LOESS goes outside
    if loess_curves_y:
        # Check all LOESS Y values
        all_loess_y = np.concatenate(loess_curves_y)
        all_loess_y = all_loess_y[np.isfinite(all_loess_y)]
        if len(all_loess_y) > 0:
            l_min = np.percentile(all_loess_y, 1)
            l_max = np.percentile(all_loess_y, 99)
            logit_range_y[0] = min(logit_range_y[0], l_min - 0.5)
            logit_range_y[1] = max(logit_range_y[1], l_max + 0.5)

    # Plot Ideal Line across the visual max of both
    ideal_min = min(logit_range_x[0], logit_range_y[0])
    ideal_max = max(logit_range_x[1], logit_range_y[1])
    
    fig.add_trace(go.Scatter(
        x=[ideal_min, ideal_max], y=[ideal_min, ideal_max],
        mode="lines",
        line=dict(dash="dash", color="black"),
        name="Ideal (α=0, β=1)",
        showlegend=False
    ), row=2, col=1)

    fig.update_xaxes(range=logit_range_x, title_text="Predicted Log-Odds", row=2, col=1)
    fig.update_yaxes(range=logit_range_y, title_text="Empirical Log-Odds", row=2, col=1)

    fig.update_layout(
        title=f"Calibration 4-Panel Summary ({split_label}, {prob_type})",
        width=1200,
        height=1000,
        hovermode="closest",
        template="plotly_white",
        barmode="overlay",
    )
    
    # Axes labels
    fig.update_xaxes(range=[-0.02, 1.02], title_text="Predicted Probability", row=1, col=1)
    fig.update_yaxes(range=[-0.02, 1.02], title_text="Observed Frequency", row=1, col=1)
    
    fig.update_xaxes(range=[-0.02, 1.02], title_text="Predicted Probability", row=1, col=2)
    fig.update_yaxes(range=[-0.02, 1.02], title_text="Observed Frequency", row=1, col=2)

    fig.update_xaxes(title_text="Predicted Log-Odds", row=2, col=1)
    fig.update_yaxes(title_text="Observed Log-Odds", row=2, col=1)
    
    fig.update_xaxes(range=[-0.02, 1.02], title_text="Predicted Probability", row=2, col=2)
    fig.update_yaxes(title_text="Density", row=2, col=2)

    return fig


def create_learning_curve_figure(all_data: List[Dict]) -> Optional[go.Figure]:
    groups = {}
    meta = {}
    for data in all_data:
        lc = data.get("learning_curve")
        if lc is None or lc.empty or "train_size" not in lc.columns:
            continue
        key = data.get("model", "unknown")
        run_label = data.get("dir_name", "run")
        df = lc.copy()
        if "train_score_mean" in df.columns and "val_score_mean" in df.columns:
            agg = df.groupby("train_size", as_index=False).agg(
                train_mean=("train_score_mean", "mean"),
                val_mean=("val_score_mean", "mean"),
            )
        elif "train_score" in df.columns and "val_score" in df.columns:
            agg = df.groupby("train_size", as_index=False).agg(
                train_mean=("train_score", "mean"),
                val_mean=("val_score", "mean"),
            )
        else:
            continue
        agg["run"] = run_label
        groups.setdefault(key, []).append(agg)
        if key not in meta:
            meta[key] = {
                "metric_label": str(df["error_metric"].iloc[0]) if "error_metric" in df.columns else "",
                "metric_direction": str(df["metric_direction"].iloc[0]) if "metric_direction" in df.columns else "",
                "scoring": str(df["scoring"].iloc[0]) if "scoring" in df.columns else "",
            }

    if not groups:
        return None

    fig = go.Figure()

    for i, (model_name, runs) in enumerate(sorted(groups.items())):
        # Color scheme logic:
        # - Single model: Train=Steelblue, Val=Crimson (requested style)
        # - Multi model: Use palette colors, distinguish Train/Val by dash style (Train=Dash, Val=Solid)
        if len(groups) == 1:
            c_train = "steelblue"
            c_val = "crimson"
        else:
            base_color = MODEL_COLORS[i % len(MODEL_COLORS)]
            c_train = base_color
            c_val = base_color
        
        all_df = pd.concat(runs, ignore_index=True)
        summary = (
            all_df.groupby("train_size", as_index=False)
            .agg(
                train_mean=("train_mean", "mean"),
                train_sd=("train_mean", "std"),
                train_ci_lo=("train_mean", lambda x: np.percentile(x, 2.5) if len(x) > 1 else np.nan),
                train_ci_hi=("train_mean", lambda x: np.percentile(x, 97.5) if len(x) > 1 else np.nan),
                val_mean=("val_mean", "mean"),
                val_sd=("val_mean", "std"),
                val_ci_lo=("val_mean", lambda x: np.percentile(x, 2.5) if len(x) > 1 else np.nan),
                val_ci_hi=("val_mean", lambda x: np.percentile(x, 97.5) if len(x) > 1 else np.nan),
                n_runs=("run", "nunique"),
            )
        )
        x = summary["train_size"]

        # Helper to get rgba fill color
        def _get_fill(c, opacity):
            if c == "steelblue":
                return f"rgba(70, 130, 180, {opacity})"
            elif c == "crimson":
                return f"rgba(220, 20, 60, {opacity})"
            elif c.startswith("rgb"):
                return c.replace("rgb", "rgba").replace(")", f",{opacity})")
            elif c.startswith("#"):
                # Simple hex to rgba
                h = c.lstrip('#')
                return f"rgba({int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}, {opacity})"
            return f"rgba(128,128,128,{opacity})"

        # Train: Steelblue (or palette), Dashed
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([summary["train_ci_hi"], summary["train_ci_lo"][::-1]]),
            fill="toself",
            fillcolor=_get_fill(c_train, 0.15),
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False,
            hoverinfo="skip",
            name=f"{model_name} train CI",
        ))
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([summary["train_mean"] + summary["train_sd"], (summary["train_mean"] - summary["train_sd"])[::-1]]),
            fill="toself",
            fillcolor=_get_fill(c_train, 0.25),
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False,
            hoverinfo="skip",
            name=f"{model_name} train SD",
        ))
        fig.add_trace(go.Scatter(
            x=x,
            y=summary["train_mean"],
            mode="lines",
            line=dict(color=c_train, width=2.5, dash="dash"),
            name=f"{model_name} Train",
            hovertemplate=f"{model_name} Train<br>Train size: %{{x}}<br>Score: %{{y:.4f}}<extra></extra>",
        ))

        # Val: Red (or palette), Solid
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([summary["val_ci_hi"], summary["val_ci_lo"][::-1]]),
            fill="toself",
            fillcolor=_get_fill(c_val, 0.10),
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False,
            hoverinfo="skip",
            name=f"{model_name} val CI",
        ))
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([summary["val_mean"] + summary["val_sd"], (summary["val_mean"] - summary["val_sd"])[::-1]]),
            fill="toself",
            fillcolor=_get_fill(c_val, 0.15),
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False,
            hoverinfo="skip",
            name=f"{model_name} val SD",
        ))
        fig.add_trace(go.Scatter(
            x=x,
            y=summary["val_mean"],
            mode="lines",
            line=dict(color=c_val, width=2.5),
            name=f"{model_name} Val",
            hovertemplate=f"{model_name} Val<br>Train size: %{{x}}<br>Score: %{{y:.4f}}<extra></extra>",
        ))

    fig.update_layout(
        title="Learning Curves (Train + Val) Summary",
        xaxis_title="Training examples",
        yaxis_title="Score",
        width=900,
        height=600,
        hovermode="closest",
        template="plotly_white"
    )
    return fig


def create_roc_comparison(all_data: List[Dict]) -> go.Figure:
    """Create interactive ROC curve comparison."""
    fig = go.Figure()

    for data in all_data:
        if data["predictions"] is None:
            continue

        preds = data["predictions"]
        if "y_true" not in preds.columns or "risk_test" not in preds.columns:
            continue

        y_true = preds["y_true"].values
        y_pred = preds["risk_test"].values

        # Compute ROC curve
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=f"{data['model']} (AUC={roc_auc:.3f})",
            hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>"
        ))

    # Add diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color="gray"),
        name="Random (AUC=0.500)",
        showlegend=True
    ))

    fig.update_layout(
        title="ROC Curve Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=800,
        height=600,
        hovermode="closest",
        template="plotly_white"
    )

    fig.update_xaxes(range=[-0.02, 1.02])
    fig.update_yaxes(range=[-0.02, 1.02])

    return fig


def create_pr_comparison(all_data: List[Dict]) -> go.Figure:
    """Create interactive PR curve comparison."""
    fig = go.Figure()

    for data in all_data:
        if data["predictions"] is None:
            continue

        preds = data["predictions"]
        if "y_true" not in preds.columns or "risk_test" not in preds.columns:
            continue

        y_true = preds["y_true"].values
        y_pred = preds["risk_test"].values

        # Compute PR curve
        from sklearn.metrics import precision_recall_curve, average_precision_score
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        pr_auc = average_precision_score(y_true, y_pred)

        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode="lines",
            name=f"{data['model']} (AP={pr_auc:.3f})",
            hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>"
        ))

    # Add baseline
    if all_data and all_data[0]["predictions"] is not None:
        baseline = all_data[0]["predictions"]["y_true"].mean()
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[baseline, baseline],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            name=f"Baseline (prevalence={baseline:.4f})",
            showlegend=True
        ))

    fig.update_layout(
        title="Precision-Recall Curve Comparison",
        xaxis_title="Recall (Sensitivity)",
        yaxis_title="Precision (PPV)",
        width=800,
        height=600,
        hovermode="closest",
        template="plotly_white"
    )

    fig.update_xaxes(range=[-0.02, 1.02])
    fig.update_yaxes(range=[-0.02, 1.02])

    return fig


def create_calibration_comparison(all_data: List[Dict]) -> go.Figure:
    """Create interactive calibration curve comparison."""
    fig = go.Figure()

    for data in all_data:
        if data["calibration"] is None:
            continue

        calib = data["calibration"]
        if "predicted_probability" in calib.columns and "observed_frequency" in calib.columns:
            fig.add_trace(go.Scatter(
                x=calib["predicted_probability"],
                y=calib["observed_frequency"],
                mode="lines+markers",
                name=data["model"],
                hovertemplate="Predicted: %{x:.3f}<br>Observed: %{y:.3f}<extra></extra>"
            ))

    # Perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color="black"),
        name="Perfect calibration",
        showlegend=True
    ))

    fig.update_layout(
        title="Calibration Curve Comparison",
        xaxis_title="Predicted Probability",
        yaxis_title="Observed Frequency",
        width=800,
        height=600,
        hovermode="closest",
        template="plotly_white"
    )

    fig.update_xaxes(range=[-0.02, 1.02])
    fig.update_yaxes(range=[-0.02, 1.02])

    return fig


def create_dca_comparison(all_data: List[Dict]) -> go.Figure:
    """Create interactive DCA comparison."""
    fig = go.Figure()

    for data in all_data:
        if data["dca"] is None:
            continue

        dca = data["dca"]
        if "threshold_pct" not in dca.columns:
            continue

        fig.add_trace(go.Scatter(
            x=dca["threshold_pct"],
            y=dca["net_benefit_model"],
            mode="lines",
            name=f"{data['model']} (Model)",
            hovertemplate="Threshold: %{x:.2f}%<br>Net Benefit: %{y:.4f}<extra></extra>"
        ))

    # Add treat-all and treat-none baselines from first DCA
    if all_data and all_data[0]["dca"] is not None:
        dca_ref = all_data[0]["dca"]
        fig.add_trace(go.Scatter(
            x=dca_ref["threshold_pct"],
            y=dca_ref["net_benefit_all"],
            mode="lines",
            line=dict(dash="dash", color="red"),
            name="Treat All",
            showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=dca_ref["threshold_pct"],
            y=dca_ref["net_benefit_none"],
            mode="lines",
            line=dict(dash="dot", color="black"),
            name="Treat None",
            showlegend=True
        ))

    fig.update_layout(
        title="Decision Curve Analysis Comparison",
        xaxis_title="Threshold Probability (%)",
        yaxis_title="Net Benefit",
        width=1000,
        height=600,
        hovermode="closest",
        template="plotly_white"
    )

    return fig


def create_mean_roc_figure(all_data: List[Dict]) -> go.Figure:
    """Create ROC figure with mean curves per model type (with CI bands)."""
    fig = go.Figure()
    mean_curves = compute_mean_roc(all_data, group_by="model")

    for i, (model_name, curve) in enumerate(sorted(mean_curves.items())):
        # Default to steelblue if single model
        if len(mean_curves) == 1:
            color = "steelblue"
        else:
            color = MODEL_COLORS[i % len(MODEL_COLORS)]

        # Add 95% CI band
        fig.add_trace(go.Scatter(
            x=np.concatenate([curve["fpr"], curve["fpr"][::-1]]),
            y=np.concatenate([curve["tpr_ci_hi"], curve["tpr_ci_lo"][::-1]]),
            fill="toself",
            fillcolor=color.replace("rgb", "rgba").replace(")", ",0.12)") if "rgb" in color else ("rgba(70, 130, 180, 0.12)" if color == "steelblue" else "rgba(128,128,128,0.12)"),
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False,
            hoverinfo="skip",
            name=f"{model_name} CI"
        ))

        # Add SD band
        fig.add_trace(go.Scatter(
            x=np.concatenate([curve["fpr"], curve["fpr"][::-1]]),
            y=np.concatenate([
                curve["tpr_mean"] + curve["tpr_std"],
                (curve["tpr_mean"] - curve["tpr_std"])[::-1]
            ]),
            fill="toself",
            fillcolor=color.replace("rgb", "rgba").replace(")", ",0.2)") if "rgb" in color else ("rgba(70, 130, 180, 0.2)" if color == "steelblue" else "rgba(128,128,128,0.2)"),
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False,
            hoverinfo="skip",
            name=f"{model_name} SD"
        ))

        # Add mean line
        fig.add_trace(go.Scatter(
            x=curve["fpr"],
            y=curve["tpr_mean"],
            mode="lines",
            line=dict(color=color, width=2),
            name=f"{model_name} (AUC={curve['auc_mean']:.3f}±{curve['auc_std']:.3f}, n={curve['n']})",
            hovertemplate=f"{model_name}<br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>"
        ))

    # Diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color="gray"),
        name="Random (AUC=0.500)",
        showlegend=True
    ))

    fig.update_layout(
        title="Mean ROC Curves by Model Type (±1 SD, 95% CI)",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=800,
        height=600,
        hovermode="closest",
        template="plotly_white"
    )
    fig.update_xaxes(range=[-0.02, 1.02])
    fig.update_yaxes(range=[-0.02, 1.02])

    return fig


def create_mean_pr_figure(all_data: List[Dict]) -> go.Figure:
    """Create PR figure with mean curves per model type (with CI bands)."""
    fig = go.Figure()
    mean_curves = compute_mean_pr(all_data, group_by="model")

    for i, (model_name, curve) in enumerate(sorted(mean_curves.items())):
        # Default to steelblue if single model
        if len(mean_curves) == 1:
            color = "steelblue"
        else:
            color = MODEL_COLORS[i % len(MODEL_COLORS)]

        # Add 95% CI band
        fig.add_trace(go.Scatter(
            x=np.concatenate([curve["recall"], curve["recall"][::-1]]),
            y=np.concatenate([
                np.clip(curve["precision_ci_hi"], 0, 1),
                np.clip(curve["precision_ci_lo"], 0, 1)[::-1]
            ]),
            fill="toself",
            fillcolor=color.replace("rgb", "rgba").replace(")", ",0.12)") if "rgb" in color else ("rgba(70, 130, 180, 0.12)" if color == "steelblue" else "rgba(128,128,128,0.12)"),
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False,
            hoverinfo="skip",
            name=f"{model_name} CI"
        ))

        # Add SD band
        fig.add_trace(go.Scatter(
            x=np.concatenate([curve["recall"], curve["recall"][::-1]]),
            y=np.concatenate([
                np.clip(curve["precision_mean"] + curve["precision_std"], 0, 1),
                np.clip(curve["precision_mean"] - curve["precision_std"], 0, 1)[::-1]
            ]),
            fill="toself",
            fillcolor=color.replace("rgb", "rgba").replace(")", ",0.2)") if "rgb" in color else ("rgba(70, 130, 180, 0.2)" if color == "steelblue" else "rgba(128,128,128,0.2)"),
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False,
            hoverinfo="skip",
            name=f"{model_name} SD"
        ))

        # Add mean line
        fig.add_trace(go.Scatter(
            x=curve["recall"],
            y=curve["precision_mean"],
            mode="lines",
            line=dict(color=color, width=2),
            name=f"{model_name} (AP={curve['ap_mean']:.3f}±{curve['ap_std']:.3f}, n={curve['n']})",
            hovertemplate=f"{model_name}<br>Recall: %{{x:.3f}}<br>Precision: %{{y:.3f}}<extra></extra>"
        ))

    # Add baseline (prevalence) if available
    if all_data and all_data[0]["predictions"] is not None:
        baseline = all_data[0]["predictions"]["y_true"].mean()
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[baseline, baseline],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            name=f"Baseline (prevalence={baseline:.4f})",
            showlegend=True
        ))

    fig.update_layout(
        title="Mean PR Curves by Model Type (±1 SD, 95% CI)",
        xaxis_title="Recall (Sensitivity)",
        yaxis_title="Precision (PPV)",
        width=800,
        height=600,
        hovermode="closest",
        template="plotly_white"
    )
    fig.update_xaxes(range=[-0.02, 1.02])
    fig.update_yaxes(range=[-0.02, 1.02])

    return fig


def create_mean_calibration_figure(all_data: List[Dict]) -> go.Figure:
    """Create calibration figure with mean curves per model type."""
    fig = go.Figure()
    mean_curves = compute_mean_calibration(all_data, group_by="model")

    for i, (model_name, curve) in enumerate(sorted(mean_curves.items())):
        color = MODEL_COLORS[i % len(MODEL_COLORS)]

        # Add 95% CI band
        fig.add_trace(go.Scatter(
            x=np.concatenate([curve["predicted"], curve["predicted"][::-1]]),
            y=np.concatenate([
                np.clip(curve["observed_ci_hi"], 0, 1),
                np.clip(curve["observed_ci_lo"], 0, 1)[::-1]
            ]),
            fill="toself",
            fillcolor=color.replace("rgb", "rgba").replace(")", ",0.12)") if "rgb" in color else "rgba(128,128,128,0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False,
            hoverinfo="skip",
            name=f"{model_name} CI"
        ))

        # Add SD band
        fig.add_trace(go.Scatter(
            x=np.concatenate([curve["predicted"], curve["predicted"][::-1]]),
            y=np.concatenate([
                np.clip(curve["observed_mean"] + curve["observed_std"], 0, 1),
                np.clip(curve["observed_mean"] - curve["observed_std"], 0, 1)[::-1]
            ]),
            fill="toself",
            fillcolor=color.replace("rgb", "rgba").replace(")", ",0.2)") if "rgb" in color else "rgba(128,128,128,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False,
            hoverinfo="skip",
            name=f"{model_name} SD"
        ))

        # Add mean line
        fig.add_trace(go.Scatter(
            x=curve["predicted"],
            y=curve["observed_mean"],
            mode="lines+markers",
            line=dict(color=color, width=2),
            name=f"{model_name} (n={curve['n']})",
            hovertemplate=f"{model_name}<br>Predicted: %{{x:.3f}}<br>Observed: %{{y:.3f}}<extra></extra>"
        ))

    # Perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color="black"),
        name="Perfect calibration",
        showlegend=True
    ))

    fig.update_layout(
        title="Mean Calibration Curves by Model Type (±1 SD, 95% CI)",
        xaxis_title="Predicted Probability",
        yaxis_title="Observed Frequency",
        width=800,
        height=600,
        hovermode="closest",
        template="plotly_white"
    )
    fig.update_xaxes(range=[-0.02, 1.02])
    fig.update_yaxes(range=[-0.02, 1.02])

    return fig


def create_mean_dca_figure(all_data: List[Dict]) -> go.Figure:
    """Create DCA figure with mean curves per model type."""
    fig = go.Figure()
    mean_curves = compute_mean_dca(all_data, group_by="model")

    for i, (model_name, curve) in enumerate(sorted(mean_curves.items())):
        # Default to steelblue if single model
        if len(mean_curves) == 1:
            color = "steelblue"
        else:
            color = MODEL_COLORS[i % len(MODEL_COLORS)]

        # Add 95% CI band
        fig.add_trace(go.Scatter(
            x=np.concatenate([curve["threshold_pct"], curve["threshold_pct"][::-1]]),
            y=np.concatenate([curve["nb_model_ci_hi"], curve["nb_model_ci_lo"][::-1]]),
            fill="toself",
            fillcolor=color.replace("rgb", "rgba").replace(")", ",0.12)") if "rgb" in color else ("rgba(70, 130, 180, 0.12)" if color == "steelblue" else "rgba(128,128,128,0.12)"),
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False,
            hoverinfo="skip",
            name=f"{model_name} CI"
        ))

        # Add SD band
        fig.add_trace(go.Scatter(
            x=np.concatenate([curve["threshold_pct"], curve["threshold_pct"][::-1]]),
            y=np.concatenate([
                curve["nb_model_mean"] + curve["nb_model_std"],
                (curve["nb_model_mean"] - curve["nb_model_std"])[::-1]
            ]),
            fill="toself",
            fillcolor=color.replace("rgb", "rgba").replace(")", ",0.2)") if "rgb" in color else ("rgba(70, 130, 180, 0.2)" if color == "steelblue" else "rgba(128,128,128,0.2)"),
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False,
            hoverinfo="skip",
            name=f"{model_name} SD"
        ))

        # Add mean line
        fig.add_trace(go.Scatter(
            x=curve["threshold_pct"],
            y=curve["nb_model_mean"],
            mode="lines",
            line=dict(color=color, width=2),
            name=f"{model_name} (n={curve['n']})",
            hovertemplate=f"{model_name}<br>Threshold: %{{x:.2f}}%<br>Net Benefit: %{{y:.4f}}<extra></extra>"
        ))

    # Add treat-all and treat-none from first available curve
    if mean_curves:
        first_curve = list(mean_curves.values())[0]
        fig.add_trace(go.Scatter(
            x=first_curve["threshold_pct"],
            y=first_curve["nb_all_mean"],
            mode="lines",
            line=dict(dash="dash", color="red"),
            name="Treat All",
            showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=first_curve["threshold_pct"],
            y=first_curve["nb_none_mean"],
            mode="lines",
            line=dict(dash="dot", color="black"),
            name="Treat None",
            showlegend=True
        ))

    fig.update_layout(
        title="Mean DCA Curves by Model Type (±1 SD, 95% CI)",
        xaxis_title="Threshold Probability (%)",
        yaxis_title="Net Benefit",
        width=1000,
        height=600,
        hovermode="closest",
        template="plotly_white"
    )

    return fig


def create_metrics_table(all_data: List[Dict]) -> go.Figure:
    """Create interactive metrics comparison table."""
    rows = []

    for data in all_data:
        if data["metrics"] is None:
            continue

        metrics = data["metrics"].iloc[0]
        rows.append({
            "Model": data["model"],
            "AUROC": f"{metrics.get('AUROC_test', np.nan):.3f}",
            "PR-AUC": f"{metrics.get('PR_AUC_test', np.nan):.3f}",
            "Brier": f"{metrics.get('Brier_test', np.nan):.4f}",
            "Cal Intercept": f"{metrics.get('calibration_intercept_test', np.nan):.3f}",
            "Cal Slope": f"{metrics.get('calibration_slope_test', np.nan):.3f}",
            "N Test": int(metrics.get('n_test', 0)),
            "N Pos": int(metrics.get('n_test_pos', 0)),
        })

    df = pd.DataFrame(rows)

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(df.columns),
            fill_color='paleturquoise',
            align='left',
            font=dict(size=12, color='black')
        ),
        cells=dict(
            values=[df[col] for col in df.columns],
            fill_color='lavender',
            align='left',
            font=dict(size=11)
        )
    )])

    fig.update_layout(
        title="Model Performance Summary",
        width=1000,
        height=400
    )

    return fig


def generate_html_dashboard(results_dir: str, outfile: str, plotlyjs: str):
    """Generate interactive HTML dashboard."""
    print(f"Searching for model results in: {results_dir}")
    model_dirs = find_model_dirs(results_dir)

    if not model_dirs:
        print(f"ERROR: No model directories found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(model_dirs)} model directories")

    # Load all data
    all_data = []
    for model_dir in model_dirs:
        info = parse_model_name(model_dir.name)
        print(f"  Loading: {info['model']} ({model_dir.name})")

        all_data.append({
            "dir": model_dir,
            "dir_name": model_dir.name,
            "model": info["model"],
            "scenario": info["scenario"],
            "config": info["config"],
            "metrics": load_test_metrics(model_dir),
            "predictions": load_test_predictions(model_dir),
            "val_predictions": load_val_predictions(model_dir),
            "controls_oof": load_controls_oof(model_dir),
            "train_oof": load_train_oof(model_dir),
            "learning_curve": load_learning_curve(model_dir),
            "dca": load_dca_curve(model_dir),
            "calibration": load_calibration_data(model_dir),
        })

    # Get unique categories for filtering
    categories = get_unique_categories(all_data)

    # Create dashboard components
    print("\nGenerating interactive plots...")

    # Summary metrics table
    table_fig = create_metrics_table(all_data)

    # Individual model ROC curves (all models)
    roc_fig = create_roc_comparison(all_data)

    # Mean ROC curves by model type
    mean_roc_fig = create_mean_roc_figure(all_data)

    # Individual model PR curves (all models)
    pr_fig = create_pr_comparison(all_data)

    # Mean PR curves by model type
    mean_pr_fig = create_mean_pr_figure(all_data)

    # Calibration curves
    calib_fig = create_calibration_comparison(all_data)
    calib_summary_figs = {}
    for split_label in ["TEST", "VAL"]:
        for prob in ["adjusted", "raw"]:
            key = f"{split_label.lower()}-{prob}"
            fig = create_mean_calibration_with_distribution(all_data, prob, split_label=split_label)
            if fig is not None:
                calib_summary_figs[key] = fig
    available_calib_tabs = list(calib_summary_figs.keys())

    # DCA curves
    dca_fig = create_dca_comparison(all_data)
    mean_dca_fig = create_mean_dca_figure(all_data)

    # Learning curves
    learning_curve_fig = create_learning_curve_figure(all_data)

    # Risk score distributions
    risk_df = build_risk_dataframe(all_data)
    risk_figs = {
        "TEST": create_risk_distribution_figure(risk_df, "TEST"),
        "VAL": create_risk_distribution_figure(risk_df, "VAL"),
        "TRAIN": create_risk_distribution_figure(risk_df, "TRAIN"),
    }

    # Count data availability
    n_with_preds = sum(1 for d in all_data if d["predictions"] is not None)
    n_with_calib = sum(1 for d in all_data if d["calibration"] is not None)
    n_with_dca = sum(1 for d in all_data if d["dca"] is not None)
    print(f"  Data availability: {n_with_preds} predictions, {n_with_calib} calibration, {n_with_dca} DCA")

    # Combine into HTML
    print(f"\nWriting dashboard to: {outfile}")

    # Build model info for JavaScript filtering
    model_info_json = json.dumps([
        {"name": d["dir_name"], "model": d["model"], "scenario": d["scenario"], "config": d["config"]}
        for d in all_data
    ])

    def fig_html(fig, include_js=False):
        return fig.to_html(
            full_html=False,
            include_plotlyjs="inline" if include_js else False,
        )

    plotly_inline = plotlyjs == "inline"
    plotly_script_tag = "" if plotly_inline else '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'

    html_parts = []
    display_name = infer_dashboard_name(all_data, outfile)

    html_parts.append(f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{display_name} - Celiac Disease Prediction</title>
    {plotly_script_tag}
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        h2 {{
            color: #555;
            margin-top: 40px;
            border-bottom: 2px solid #ddd;
            padding-bottom: 10px;
        }}
        h3 {{
            color: #666;
            margin-top: 20px;
        }}
        .plot-container {{
            background-color: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .info {{
            background-color: #e7f3fe;
            border-left: 6px solid #2196F3;
            padding: 15px;
            margin: 20px 0;
        }}
        .warning {{
            background-color: #fff3cd;
            border-left: 6px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
        }}
        .filter-panel {{
            background-color: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .filter-group {{
            display: inline-block;
            margin-right: 30px;
            vertical-align: top;
        }}
        .filter-group label {{
            font-weight: bold;
            display: block;
            margin-bottom: 5px;
        }}
        .filter-group select {{
            padding: 8px;
            font-size: 14px;
            min-width: 150px;
        }}
        .checkbox-group {{
            max-height: 150px;
            overflow-y: auto;
            border: 1px solid #ddd;
            padding: 10px;
            border-radius: 4px;
        }}
        .checkbox-group label {{
            display: block;
            padding: 3px 0;
            cursor: pointer;
        }}
        .checkbox-group label:hover {{
            background-color: #f0f0f0;
        }}
        .tab-container {{
            margin: 20px 0;
        }}
        .tab-buttons {{
            display: flex;
            gap: 5px;
            margin-bottom: 10px;
        }}
        .tab-button {{
            padding: 10px 20px;
            border: none;
            background-color: #e0e0e0;
            cursor: pointer;
            border-radius: 4px 4px 0 0;
            font-size: 14px;
        }}
        .tab-button.active {{
            background-color: white;
            border-bottom: 2px solid #2196F3;
        }}
        .tab-content {{
            display: none;
        }}
        .tab-content.active {{
            display: block;
        }}
        .stats-summary {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            margin: 10px 0;
        }}
        .stat-box {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            min-width: 120px;
        }}
        .stat-box .number {{
            font-size: 24px;
            font-weight: bold;
            color: #2196F3;
        }}
        .stat-box .label {{
            font-size: 12px;
            color: #666;
        }}
    </style>
</head>
<body>
    <h1>{display_name}</h1>

    <div class="info">
        <strong>Dataset:</strong> Incident Celiac Disease prediction from proteomics<br>
        <strong>Models:</strong> {", ".join(sorted(set(d["model"] for d in all_data)))}<br>
        <strong>Scenarios:</strong> {", ".join(categories["scenarios"]) if categories["scenarios"] else "N/A"}<br>
        <strong>Generated:</strong> {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
    </div>

    <div class="stats-summary">
        <div class="stat-box">
            <div class="number">{len(all_data)}</div>
            <div class="label">Total Runs</div>
        </div>
        <div class="stat-box">
            <div class="number">{len(set(d["model"] for d in all_data))}</div>
            <div class="label">Model Types</div>
        </div>
        <div class="stat-box">
            <div class="number">{n_with_preds}</div>
            <div class="label">With Predictions</div>
        </div>
        <div class="stat-box">
            <div class="number">{n_with_calib}</div>
            <div class="label">With Calibration</div>
        </div>
        <div class="stat-box">
            <div class="number">{n_with_dca}</div>
            <div class="label">With DCA</div>
        </div>
    </div>

    <div class="filter-panel">
        <h3>Filter Models</h3>
        <div class="filter-group">
            <label>Model Type</label>
            <div class="checkbox-group" id="model-filter">
                {"".join(f'<label><input type="checkbox" value="{m}" checked onchange="updateFilters()"> {m}</label>' for m in sorted(set(d["model"] for d in all_data)))}
            </div>
        </div>
        <div class="filter-group">
            <label>Scenario</label>
            <div class="checkbox-group" id="scenario-filter">
                {"".join(f'<label><input type="checkbox" value="{s}" checked onchange="updateFilters()"> {s}</label>' for s in categories["scenarios"]) if categories["scenarios"] else '<label>No scenarios</label>'}
            </div>
        </div>
        <div class="filter-group">
            <label>Quick Select</label><br>
            <button onclick="selectAll()">Select All</button>
            <button onclick="selectNone()">Select None</button>
        </div>
    </div>
""")

    # Performance Summary Table
    html_parts.append('<h2>Performance Summary</h2>')
    html_parts.append('<div class="plot-container">')
    html_parts.append(fig_html(table_fig, include_js=plotly_inline))
    html_parts.append('</div>')

    # ROC Curves Section with tabs
    html_parts.append('<h2>ROC Curves</h2>')
    html_parts.append('<div class="tab-container">')
    html_parts.append('<div class="tab-buttons">')
    html_parts.append('<button class="tab-button active" onclick="showTab(\'roc\', \'mean\')">Mean by Model Type</button>')
    html_parts.append('<button class="tab-button" onclick="showTab(\'roc\', \'all\')">All Individual Models</button>')
    html_parts.append('</div>')
    html_parts.append('<div id="roc-mean" class="tab-content active">')
    html_parts.append('<div class="plot-container">')
    html_parts.append(fig_html(mean_roc_fig))
    html_parts.append('</div></div>')
    html_parts.append('<div id="roc-all" class="tab-content">')
    html_parts.append('<div class="plot-container">')
    html_parts.append(fig_html(roc_fig))
    html_parts.append('</div></div>')
    html_parts.append('</div>')

    # PR Curves Section with tabs
    html_parts.append('<h2>Precision-Recall Curves</h2>')
    html_parts.append('<div class="tab-container">')
    html_parts.append('<div class="tab-buttons">')
    html_parts.append('<button class="tab-button active" onclick="showTab(\'pr\', \'mean\')">Mean by Model Type</button>')
    html_parts.append('<button class="tab-button" onclick="showTab(\'pr\', \'all\')">All Individual Models</button>')
    html_parts.append('</div>')
    html_parts.append('<div id="pr-mean" class="tab-content active">')
    html_parts.append('<div class="plot-container">')
    html_parts.append(fig_html(mean_pr_fig))
    html_parts.append('</div></div>')
    html_parts.append('<div id="pr-all" class="tab-content">')
    html_parts.append('<div class="plot-container">')
    html_parts.append(fig_html(pr_fig))
    html_parts.append('</div></div>')
    html_parts.append('</div>')

    # Calibration Curves Section with tabs
    html_parts.append('<h2>Calibration Curves</h2>')
    if n_with_calib == 0 and not available_calib_tabs:
        html_parts.append('<div class="warning">No calibration data available. Ensure --save_calibration is enabled during training.</div>')
    else:
        html_parts.append('<div class="tab-container">')
        html_parts.append('<div class="tab-buttons">')
        first_tab = True
        for tab in available_calib_tabs:
            active = " active" if first_tab else ""
            parts = tab.split("-", 1)
            if len(parts) == 2:
                label = f"{parts[0].upper()} {parts[1]}"
            else:
                label = tab
            html_parts.append(
                f"<button class=\"tab-button{active}\" onclick=\"showTab('calib', '{tab}')\">{label}</button>"
            )
            first_tab = False
        if calib_fig is not None:
            active = " active" if first_tab else ""
            html_parts.append(
                f"<button class=\"tab-button{active}\" onclick=\"showTab('calib', 'all')\">All Individual Models</button>"
            )
            first_tab = False
        html_parts.append('</div>')
        first_content = True
        for tab in available_calib_tabs:
            active = " active" if first_content else ""
            html_parts.append(f'<div id="calib-{tab}" class="tab-content{active}">')
            html_parts.append('<div class="plot-container">')
            html_parts.append(fig_html(calib_summary_figs[tab]))
            html_parts.append('</div></div>')
            first_content = False
        if calib_fig is not None:
            active = " active" if first_content else ""
            html_parts.append(f'<div id="calib-all" class="tab-content{active}">')
            html_parts.append('<div class="plot-container">')
            html_parts.append(fig_html(calib_fig))
            html_parts.append('</div></div>')
        html_parts.append('</div>')

    # DCA Curves Section with tabs
    html_parts.append('<h2>Decision Curve Analysis</h2>')
    if n_with_dca == 0:
        html_parts.append('<div class="warning">No DCA data available. Ensure --compute_dca is enabled during training.</div>')
    else:
        html_parts.append('<div class="tab-container">')
        html_parts.append('<div class="tab-buttons">')
        html_parts.append('<button class="tab-button active" onclick="showTab(\'dca\', \'mean\')">Mean by Model Type</button>')
        html_parts.append('<button class="tab-button" onclick="showTab(\'dca\', \'all\')">All Individual Models</button>')
        html_parts.append('</div>')
        html_parts.append('<div id="dca-mean" class="tab-content active">')
        html_parts.append('<div class="plot-container">')
        html_parts.append(fig_html(mean_dca_fig))
        html_parts.append('</div></div>')
        html_parts.append('<div id="dca-all" class="tab-content">')
        html_parts.append('<div class="plot-container">')
        html_parts.append(fig_html(dca_fig))
        html_parts.append('</div></div>')
        html_parts.append('</div>')

    # Learning Curves
    html_parts.append('<h2>Learning Curves (Train + Val)</h2>')
    if learning_curve_fig is None:
        html_parts.append('<div class="warning">No learning curve data available. Ensure --learning_curve is enabled during training.</div>')
    else:
        html_parts.append('<div class="plot-container">')
        html_parts.append(fig_html(learning_curve_fig))
        html_parts.append('</div>')

    # Risk Score Distributions
    html_parts.append('<h2>Risk Score Distributions</h2>')
    available_risk_tabs = [k for k, fig in risk_figs.items() if fig is not None]
    if not available_risk_tabs:
        html_parts.append('<div class="warning">No risk score data available. Ensure --save_test_preds/--save_val_preds/--save_train_oof or --save_controls_oof are enabled.</div>')
    else:
        html_parts.append('<div class="tab-container">')
        html_parts.append('<div class="tab-buttons">')
        for i, tab in enumerate(available_risk_tabs):
            active = " active" if i == 0 else ""
            html_parts.append(
                f"<button class=\"tab-button{active}\" onclick=\"showTab('risk', '{tab.lower()}')\">{tab}</button>"
            )
        html_parts.append('</div>')
        for i, tab in enumerate(available_risk_tabs):
            active = " active" if i == 0 else ""
            html_parts.append(f'<div id="risk-{tab.lower()}" class="tab-content{active}">')
            html_parts.append('<div class="plot-container">')
            html_parts.append(fig_html(risk_figs[tab]))
            html_parts.append('</div></div>')
        html_parts.append('</div>')

    # JavaScript for tabs and filtering
    html_parts.append(f"""
    <script>
        const modelInfo = {model_info_json};

        function showTab(section, tab) {{
            // Hide all tabs in this section
            document.querySelectorAll('[id^="' + section + '-"]').forEach(el => {{
                el.classList.remove('active');
            }});
            // Show selected tab
            document.getElementById(section + '-' + tab).classList.add('active');

            // Update button styles
            const container = document.getElementById(section + '-' + tab).closest('.tab-container');
            container.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');

            // Trigger resize for Plotly
            window.dispatchEvent(new Event('resize'));
        }}

        function getSelectedModels() {{
            const checkboxes = document.querySelectorAll('#model-filter input:checked');
            return Array.from(checkboxes).map(cb => cb.value);
        }}

        function getSelectedScenarios() {{
            const checkboxes = document.querySelectorAll('#scenario-filter input:checked');
            return Array.from(checkboxes).map(cb => cb.value);
        }}

        function updateFilters() {{
            const selectedModels = getSelectedModels();
            const selectedScenarios = getSelectedScenarios();

            // Update all Plotly figures
            document.querySelectorAll('.js-plotly-plot').forEach(plot => {{
                if (plot.data) {{
                    const visibility = plot.data.map(trace => {{
                        // Check if trace name contains any selected model
                        const traceName = trace.name || '';
                        const matchesModel = selectedModels.some(m => traceName.includes(m));
                        // Always show reference lines
                        const isReference = traceName.includes('Random') ||
                                          traceName.includes('Baseline') ||
                                          traceName.includes('Perfect') ||
                                          traceName.includes('Treat All') ||
                                          traceName.includes('Treat None');
                        return (matchesModel || isReference) ? true : 'legendonly';
                    }});
                    Plotly.restyle(plot, {{'visible': visibility}});
                }}
            }});
        }}

        function selectAll() {{
            document.querySelectorAll('.checkbox-group input').forEach(cb => cb.checked = true);
            updateFilters();
        }}

        function selectNone() {{
            document.querySelectorAll('.checkbox-group input').forEach(cb => cb.checked = false);
            updateFilters();
        }}
    </script>
""")

    html_parts.append('</body></html>')

    with open(outfile, 'w') as f:
        f.write('\n'.join(html_parts))

    print(f"Dashboard created: {outfile}")
    print(f"  Open in browser: file://{os.path.abspath(outfile)}")


def main():
    parser = argparse.ArgumentParser(description="Generate interactive HTML dashboard for ML models")
    parser.add_argument("--results_dir", required=True, help="Results directory containing model outputs")
    parser.add_argument("--outfile", default="dashboard.html", help="Output HTML file")
    parser.add_argument("--plotlyjs", choices=["inline", "cdn"], default="inline",
                        help="How to include Plotly JS in HTML (inline works offline)")

    args = parser.parse_args()

    generate_html_dashboard(args.results_dir, args.outfile, args.plotlyjs)


if __name__ == "__main__":
    main()
