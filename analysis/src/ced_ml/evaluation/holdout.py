"""
Holdout evaluation module.

This module provides functionality for evaluating trained models on holdout sets.
It handles:
- Loading holdout indices and model artifacts
- Computing discrimination, calibration, and clinical utility metrics
- Generating predictions with prevalence adjustment
- Saving comprehensive evaluation results
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from ced_ml.data.schema import (
    TARGET_COL,
    CONTROL_LABEL,
    get_scenario_labels,
    get_protein_columns,
)
from ced_ml.data.io import load_data, identify_protein_columns
from ced_ml.metrics import (
    binary_metrics_at_threshold,
    compute_discrimination_metrics,
    compute_brier_score,
    save_dca_results,
    top_risk_capture,
    parse_dca_report_points,
    generate_dca_thresholds,
)
from ced_ml.models.calibration import (
    calibration_intercept_slope,
    expected_calibration_error,
    adjust_probabilities_for_prevalence,
)


def load_holdout_indices(path: str) -> np.ndarray:
    """
    Load holdout indices from CSV file.

    Args:
        path: Path to holdout index CSV with 'idx' column

    Returns:
        Array of holdout indices

    Raises:
        ValueError: If file missing 'idx' column
    """
    df = pd.read_csv(path)
    if "idx" not in df.columns:
        raise ValueError(f"Holdout index file {path} must contain an 'idx' column.")
    return df["idx"].to_numpy(dtype=int)


def load_model_artifact(path: str) -> Dict[str, Any]:
    """
    Load saved model artifact (joblib bundle).

    Args:
        path: Path to model artifact (.joblib file)

    Returns:
        Dictionary containing model and metadata
    """
    return joblib.load(path)


def extract_holdout_data(
    df_filtered: pd.DataFrame,
    X_all: pd.DataFrame,
    y_all: np.ndarray,
    holdout_idx: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Extract holdout subset from full dataset.

    Args:
        df_filtered: Full filtered dataframe
        X_all: Full feature matrix
        y_all: Full target array
        holdout_idx: Indices for holdout set

    Returns:
        (df_holdout, X_holdout, y_holdout) tuple

    Raises:
        ValueError: If holdout indices exceed dataset size
    """
    if len(holdout_idx) > 0 and holdout_idx.max() >= len(df_filtered):
        raise ValueError(
            f"Holdout index exceeds dataset rows "
            f"({holdout_idx.max()} >= {len(df_filtered)})."
        )

    # Convert to int indices for proper indexing
    idx_int = holdout_idx.astype(int) if len(holdout_idx) > 0 else np.array([], dtype=int)

    df_holdout = df_filtered.iloc[idx_int].reset_index(drop=True)
    X_holdout = X_all.iloc[idx_int].reset_index(drop=True)
    y_holdout = y_all[idx_int]

    return df_holdout, X_holdout, y_holdout


def compute_holdout_metrics(
    y_true: np.ndarray,
    proba_eval: np.ndarray,
    bundle: Dict[str, Any],
    scenario: str,
    clinical_points: List[float],
) -> Dict[str, Any]:
    """
    Compute comprehensive holdout metrics.

    Args:
        y_true: True labels for holdout set
        proba_eval: Predicted probabilities
        bundle: Model artifact bundle with metadata
        scenario: Scenario name (e.g., 'IncidentOnly')
        clinical_points: Clinical probability thresholds to evaluate

    Returns:
        Dictionary of holdout metrics
    """
    # Discrimination metrics
    disc_metrics = compute_discrimination_metrics(y_true, proba_eval)
    brier = compute_brier_score(y_true, proba_eval)

    # Calibration metrics
    cal_a, cal_b = calibration_intercept_slope(y_true, proba_eval)
    ece = expected_calibration_error(y_true, proba_eval)

    # Extract threshold metadata
    thresholds_meta = bundle.get("thresholds", {})
    objective_name = thresholds_meta.get("objective_name", "max_f1")
    thr_objective = thresholds_meta.get("objective", 0.5)
    thr_f1 = thresholds_meta.get("max_f1", 0.5)
    thr_spec90 = thresholds_meta.get("spec90", 0.5)
    ctrl_specs = thresholds_meta.get("control_specs", {})

    # Metrics at key thresholds
    m_obj = binary_metrics_at_threshold(y_true, proba_eval, thr_objective)
    m_f1 = binary_metrics_at_threshold(y_true, proba_eval, thr_f1)
    m_spec90 = binary_metrics_at_threshold(y_true, proba_eval, thr_spec90)

    # Prevalence metadata
    prevalence_meta = bundle.get("prevalence", {})
    train_prev = prevalence_meta.get("train_sample", np.nan)
    if not np.isfinite(train_prev):
        train_prev = float(y_true.mean())
    target_prev = prevalence_meta.get("target", train_prev)
    if target_prev is None or not np.isfinite(target_prev):
        target_prev = train_prev
    target_prev = float(np.clip(target_prev, 1e-6, 1.0 - 1e-6))

    # Build metrics dictionary
    metrics = {
        "scenario": scenario,
        "model_name": bundle.get("model_name"),
        "model_label": bundle.get("model_label"),
        "split_id": bundle.get("split_id"),
        "n_holdout": int(len(y_true)),
        "n_holdout_pos": int(y_true.sum()),
        "train_prevalence_sample": float(train_prev) if np.isfinite(train_prev) else np.nan,
        "target_prevalence": float(target_prev),
        "AUROC_holdout": disc_metrics["AUROC"],
        "PR_AUC_holdout": disc_metrics["PR_AUC"],
        "Brier_holdout": float(brier),
        "calibration_intercept_holdout": float(cal_a) if np.isfinite(cal_a) else np.nan,
        "calibration_slope_holdout": float(cal_b) if np.isfinite(cal_b) else np.nan,
        "ECE_holdout": float(ece),
        "thr_objective_name": objective_name,
        "thr_objective": float(thr_objective),
        "precision_holdout_at_thr_objective": float(m_obj["precision"]),
        "recall_holdout_at_thr_objective": float(m_obj["recall"]),
        "specificity_holdout_at_thr_objective": float(m_obj["specificity"]),
        "thr_maxF1": float(thr_f1),
        "f1_holdout_at_thr_maxF1": float(m_f1["f1"]),
        "precision_holdout_at_thr_maxF1": float(m_f1["precision"]),
        "recall_holdout_at_thr_maxF1": float(m_f1["recall"]),
        "thr_spec90": float(thr_spec90),
        "sensitivity_holdout_at_spec90": float(m_spec90["recall"]),
        "specificity_holdout_at_spec90": float(m_spec90["specificity"]),
    }

    # Control specificity thresholds
    for key, val in ctrl_specs.items():
        try:
            thr_val = float(val)
        except Exception:
            continue
        m_ctrl = binary_metrics_at_threshold(y_true, proba_eval, thr_val)
        tag = str(key).replace("0.", "")
        metrics[f"thr_ctrl_{tag}"] = float(thr_val)
        metrics[f"precision_holdout_ctrl_{tag}"] = float(m_ctrl["precision"])
        metrics[f"recall_holdout_ctrl_{tag}"] = float(m_ctrl["recall"])
        metrics[f"specificity_holdout_ctrl_{tag}"] = float(m_ctrl["specificity"])

    # Clinical thresholds
    for thr in clinical_points:
        if not (0.0 < thr < 1.0):
            continue
        m_thr = binary_metrics_at_threshold(y_true, proba_eval, thr)
        tag = f"clin_{str(thr).replace('.', 'p')}"
        metrics[f"{tag}_threshold"] = float(thr)
        metrics[f"{tag}_precision"] = float(m_thr["precision"])
        metrics[f"{tag}_recall"] = float(m_thr["recall"])
        metrics[f"{tag}_specificity"] = float(m_thr["specificity"])
        metrics[f"{tag}_f1"] = float(m_thr["f1"])

    return metrics


def compute_top_risk_capture(
    y_true: np.ndarray,
    proba_eval: np.ndarray,
    top_fracs: List[float],
) -> pd.DataFrame:
    """
    Compute top-risk capture statistics.

    Args:
        y_true: True labels
        proba_eval: Predicted probabilities
        top_fracs: List of top fractions to evaluate (e.g., [0.01, 0.05])

    Returns:
        DataFrame with capture statistics per fraction
    """
    rows = []
    for frac in sorted(top_fracs):
        capture = top_risk_capture(y_true, proba_eval, frac=frac)
        rows.append({"frac": frac, **capture})
    return pd.DataFrame(rows)


def save_holdout_predictions(
    outdir: str,
    holdout_idx: np.ndarray,
    df_holdout: pd.DataFrame,
    y_true: np.ndarray,
    proba_eval: np.ndarray,
    proba_adjusted: np.ndarray,
) -> None:
    """
    Save holdout predictions to CSV.

    Args:
        outdir: Output directory
        holdout_idx: Original holdout indices
        df_holdout: Holdout dataframe
        y_true: True labels
        proba_eval: Predicted probabilities
        proba_adjusted: Prevalence-adjusted probabilities
    """
    out = pd.DataFrame({
        "idx": holdout_idx,
        TARGET_COL: df_holdout[TARGET_COL].astype(str),
        "y_true": y_true.astype(int),
        "risk_holdout": proba_eval,
        "risk_holdout_adjusted": proba_adjusted,
        "risk_holdout_raw": proba_eval,
    })
    out.to_csv(os.path.join(outdir, "holdout_predictions.csv"), index=False)


def evaluate_holdout(
    infile: str,
    holdout_idx_file: str,
    model_artifact_path: str,
    outdir: str,
    scenario: Optional[str] = None,
    compute_dca: bool = False,
    dca_threshold_min: Optional[float] = None,
    dca_threshold_max: Optional[float] = None,
    dca_threshold_step: Optional[float] = None,
    dca_report_points: str = "",
    dca_use_target_prevalence: bool = False,
    save_preds: bool = False,
    toprisk_fracs: str = "0.01",
    target_prevalence: Optional[float] = None,
    clinical_threshold_points: str = "",
    subgroup_min_n: int = 40,
) -> Dict[str, Any]:
    """
    Evaluate trained model on holdout set.

    This is the main entry point for holdout evaluation. It:
    1. Loads the model artifact and holdout data
    2. Generates predictions
    3. Computes comprehensive metrics
    4. Optionally computes DCA and subgroup analyses
    5. Saves all results to disk

    Args:
        infile: Path to full dataset CSV
        holdout_idx_file: Path to holdout indices CSV
        model_artifact_path: Path to trained model artifact (.joblib)
        outdir: Output directory for results
        scenario: Override scenario (if not in artifact)
        compute_dca: Whether to compute decision curve analysis
        dca_threshold_min: Min threshold for DCA
        dca_threshold_max: Max threshold for DCA
        dca_threshold_step: Step size for DCA thresholds
        dca_report_points: Comma-separated thresholds to report
        dca_use_target_prevalence: Use prevalence-adjusted probs for DCA
        save_preds: Save individual predictions to CSV
        toprisk_fracs: Comma-separated top-risk fractions (e.g., "0.01,0.05")
        target_prevalence: Override target prevalence
        clinical_threshold_points: Comma-separated clinical thresholds
        subgroup_min_n: Minimum sample size for subgroup reporting

    Returns:
        Dictionary of holdout metrics
    """
    # Create output directory
    os.makedirs(outdir, exist_ok=True)

    # Load model artifact
    bundle = load_model_artifact(model_artifact_path)
    model = bundle["model"]
    scenario_final = scenario or bundle.get("scenario", "IncidentOnly")

    # Load and filter data
    positive_labels = get_scenario_labels(scenario_final)
    df_raw = load_data(infile)

    # Filter to relevant classes
    keep_labels = [CONTROL_LABEL] + positive_labels
    df_filtered = df_raw[df_raw[TARGET_COL].isin(keep_labels)].copy()

    # Create binary outcome
    df_filtered["y"] = df_filtered[TARGET_COL].isin(positive_labels).astype(int)
    y_all = df_filtered["y"].to_numpy()

    # Identify protein columns
    prot_cols = identify_protein_columns(df_filtered)

    # Create feature matrix (proteins only for now)
    X_all = df_filtered[prot_cols]

    # Extract holdout subset
    holdout_idx = load_holdout_indices(holdout_idx_file)
    df_holdout, X_holdout, y_holdout = extract_holdout_data(
        df_filtered, X_all, y_all, holdout_idx
    )

    # Generate predictions
    proba_eval = np.clip(model.predict_proba(X_holdout)[:, 1], 0.0, 1.0)

    # Prevalence adjustment
    prevalence_meta = bundle.get("prevalence", {})
    train_prev = prevalence_meta.get("train_sample", np.nan)
    if not np.isfinite(train_prev):
        train_prev = float(y_holdout.mean())

    target_prev = target_prevalence if target_prevalence is not None else prevalence_meta.get("target", train_prev)
    if target_prev is None or not np.isfinite(target_prev):
        target_prev = train_prev
    target_prev = float(np.clip(target_prev, 1e-6, 1.0 - 1e-6))

    proba_adjusted = adjust_probabilities_for_prevalence(proba_eval, train_prev, target_prev)

    # Parse clinical thresholds
    clinical_points_src = clinical_threshold_points or bundle.get("args", {}).get("clinical_threshold_points", "")
    clinical_points = sorted({float(t.strip()) for t in (clinical_points_src or "").split(",") if t.strip()})

    # Compute metrics
    metrics = compute_holdout_metrics(
        y_holdout,
        proba_eval,
        bundle,
        scenario_final,
        clinical_points,
    )

    # Save metrics
    pd.DataFrame([metrics]).to_csv(os.path.join(outdir, "holdout_metrics.csv"), index=False)

    # Top-risk capture
    top_fracs = sorted({float(t.strip()) for t in (toprisk_fracs or "").split(",") if t.strip()})
    if top_fracs:
        top_risk_df = compute_top_risk_capture(y_holdout, proba_eval, top_fracs)
        top_risk_df.to_csv(os.path.join(outdir, "holdout_toprisk_capture.csv"), index=False)

    # Save predictions if requested
    if save_preds:
        save_holdout_predictions(
            outdir,
            holdout_idx,
            df_holdout,
            y_holdout,
            proba_eval,
            proba_adjusted,
        )

    # Decision curve analysis
    if compute_dca:
        min_thr = dca_threshold_min if dca_threshold_min is not None else bundle.get("args", {}).get("dca_threshold_min", 0.001)
        max_thr = dca_threshold_max if dca_threshold_max is not None else bundle.get("args", {}).get("dca_threshold_max", 0.10)
        step_thr = dca_threshold_step if dca_threshold_step is not None else bundle.get("args", {}).get("dca_threshold_step", 0.001)

        dca_thresholds = generate_dca_thresholds(min_thr, max_thr, step_thr)
        report_points = parse_dca_report_points(dca_report_points) or parse_dca_report_points(
            bundle.get("args", {}).get("dca_report_points", "")
        )

        dca_dir = os.path.join(outdir, "diagnostics", "dca")
        dca_probs = proba_adjusted if dca_use_target_prevalence else proba_eval
        dca_prev = target_prev if dca_use_target_prevalence else None

        summary = save_dca_results(
            y_holdout,
            dca_probs,
            out_dir=dca_dir,
            prefix="holdout__",
            thresholds=dca_thresholds,
            report_points=report_points,
            prevalence_adjustment=dca_prev,
        )

        with open(os.path.join(dca_dir, "holdout_dca_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    return metrics
