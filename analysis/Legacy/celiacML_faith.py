#!/usr/bin/env python3
"""
celiacML_faith.py (per-scenario, per-model run)

Robust, HPC-friendly clinical risk modeling with:
  - fixed TRAIN/TEST split (optionally shared via --splits_dir)
  - nested tuning on TRAIN (RandomizedSearchCV inner CV)
  - repeated stratified CV on TRAIN with OOF predictions
  - train-only prefilter + train-only screening (optional)
  - stable compact panels + correlation pruning (TRAIN-only)
  - optional bootstrap CIs and calibration / learning curves

User-settable settings:
  - --random_state, --tune_n_jobs, --error_score, --missing_imputer
  - LR/SVM/RF grids + convergence knobs
  - threshold objective (max_f1 / max_fbeta / youden / fixed_spec / fixed_ppv)
  - optional calibration for final LR/RF too
  - panel correlation method and representative tie-break behavior
  - screen_min_n_per_group exposed

References:
  - TRIPOD guidelines for prediction model reporting
  - Steyerberg, E. W. (2019). Clinical Prediction Models (2nd ed.)
  - Van Calster et al. (2019). Calibration: the Achilles heel of predictive analytics

Version: 1.1.0
"""

__version__ = "1.1.0"
__author__ = "CeD ML Pipeline"

import os, sys, json, time, argparse, hashlib, re
import logging
import math
import numbers
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from joblib import parallel_backend
import joblib

import numpy as np
import pandas as pd

import sklearn

from sklearn.base import clone
from sklearn.model_selection import (
    RepeatedStratifiedKFold, train_test_split,
    StratifiedKFold, RandomizedSearchCV, learning_curve
)
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_recall_curve, roc_curve, log_loss,
    precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("WARNING: XGBoost not available. Install with: pip install xgboost")

from scipy import stats
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Import shared utilities (DCA computation moved to postprocess_compare.py for efficiency)
from shared_utils import (
    _mkdir,
    _safe_metric,
    _prevalence,
    compute_distribution_stats,
    prob_metrics,
    format_ci,
    auroc,
    prauc,
    brier,
    logloss,
    compute_metrics_with_cis,
    stratified_bootstrap_ci,
    threshold_max_f1,
    threshold_max_fbeta,
    threshold_youden,
    threshold_for_specificity,
    threshold_for_precision,
    threshold_from_controls,
    binary_metrics_at_threshold,
    top_risk_capture,
    choose_threshold_objective,
    adjust_probabilities_for_prevalence,
    PrevalenceAdjustedModel,
    calibration_intercept_slope,
    calib_intercept_metric,
    calib_slope_metric,
    expected_calibration_error,
    _apply_plot_metadata,
    _compute_recalibration,
    _binned_logits,
    save_dca_results,
    find_dca_zero_crossing,
    load_spec95_threshold,
    _dca_thresholds,
    _plot_roc_curve,
    _plot_pr_curve,
    _plot_prob_calibration_panel,
    _plot_logit_calibration_panel,
    _plot_calibration_curve,
    _plot_risk_distribution,
)

ENHANCED_PLOTS_AVAILABLE = True  # All enhanced plotting functions available from shared_utils

# ----------------------------
# sklearn version helpers (for forward compatibility)
# ----------------------------
def _sklearn_version_tuple(ver: str) -> Tuple[int, int, int]:
    # Robust parse (handles rc/dev suffixes).
    nums = re.findall(r"\d+", ver)
    nums = (nums + ["0", "0", "0"])[:3]
    return (int(nums[0]), int(nums[1]), int(nums[2]))

SKLEARN_VER = _sklearn_version_tuple(getattr(sklearn, "__version__", "0.0.0"))

# ----------------------------
# Defaults / schema
# ----------------------------
RANDOM_STATE = 0
ID_COL = "eid"
TARGET_COL = "CeD_comparison"

META_NUM_COLS = ["age", "BMI"]
CAT_COLS = ["sex", "Genetic ethnic grouping"]  # exact column names
CED_DATE_COL = "CeD_date"
CONTROL_LABEL = "Controls"


# ----------------------------
# Row filtering helper (shared with save_splits.py)
# ----------------------------
def apply_row_filters(
    df: pd.DataFrame,
    drop_uncertain_controls: bool = True,
    dropna_meta_num: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply row filters consistently across split generation and training.

    CRITICAL: This function must be used by BOTH save_splits.py and celiacML_faith.py
    to ensure index alignment. If you modify filtering logic here, the same changes
    apply everywhere automatically.

    Filters applied:
    1. drop_uncertain_controls: Remove Controls that have a CeD_date present
       (these are ambiguous - possibly undiagnosed cases)
    2. dropna_meta_num: Remove rows missing age or BMI
       (required for modeling)

    Args:
        df: DataFrame with CeD_comparison, CeD_date, age, BMI columns
        drop_uncertain_controls: If True, drop Controls with CeD_date present
        dropna_meta_num: If True, drop rows missing age or BMI

    Returns:
        (filtered_df, stats_dict) where stats_dict contains filtering statistics
    """
    stats: Dict[str, Any] = {
        "n_in": len(df),
        "drop_uncertain_controls": drop_uncertain_controls,
        "dropna_meta_num": dropna_meta_num,
        "n_removed_uncertain_controls": 0,
        "n_removed_dropna_meta_num": 0,
        "n_out": 0,
    }

    df2 = df.copy()

    # Filter 1: Drop "uncertain controls" (Controls with CeD_date present)
    if drop_uncertain_controls and (CED_DATE_COL in df2.columns):
        mask_uncertain = (df2[TARGET_COL] == CONTROL_LABEL) & df2[CED_DATE_COL].notna()
        n_uncertain = int(mask_uncertain.sum())
        if n_uncertain > 0:
            df2 = df2.loc[~mask_uncertain].copy()
        stats["n_removed_uncertain_controls"] = n_uncertain

    # Filter 2: Drop rows missing required numeric metadata (age, BMI)
    if dropna_meta_num:
        meta_present = [c for c in META_NUM_COLS if c in df2.columns]
        if meta_present:
            n_before = len(df2)
            df2 = df2.dropna(subset=meta_present).copy()
            stats["n_removed_dropna_meta_num"] = n_before - len(df2)

    df2 = df2.reset_index(drop=True)
    stats["n_out"] = len(df2)

    return df2, stats


# ----------------------------
# Logging helpers (HPC-friendly)
# ----------------------------
_LOGGER: Optional[logging.Logger] = None

def configure_logging(outdir: str, level: str = "INFO", log_filename: str = "run.log") -> logging.Logger:
    """Configure a simple stdout + file logger.

    - Stdout logging is useful for LSF *.out
    - File logging is useful for persistent run provenance in args.outdir
    """
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    logger = logging.getLogger("celiacML_faith")
    logger.setLevel(lvl)
    logger.propagate = False

    # Remove old handlers (safe for repeated calls / notebook imports)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(lvl)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    try:
        if outdir:
            os.makedirs(outdir, exist_ok=True)
            fh = logging.FileHandler(os.path.join(outdir, log_filename))
            fh.setLevel(lvl)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    except Exception:
        # Never hard-fail on logging in HPC contexts
        pass

    return logger

def set_logger(logger: Optional[logging.Logger]) -> None:
    """Set the global logger instance."""
    global _LOGGER
    _LOGGER = logger


def logprint(*args, level: str = "info", flush: bool = True, **kwargs) -> None:
    """
    Unified logging function - PRIMARY logging interface.

    If a logger is configured via configure_logging(), messages are routed through it.
    Otherwise, falls back to print() for compatibility.

    Args:
        *args: Message components (joined with spaces)
        level: Log level (debug, info, warning, error)
        flush: Whether to flush stdout immediately
        **kwargs: Additional arguments passed to print() in fallback mode
    """
    msg = " ".join(str(a) for a in args)
    
    if _LOGGER is not None:
        fn = getattr(_LOGGER, str(level).lower(), _LOGGER.info)
        fn(msg)
    else:
        # Fallback to print when no logger configured
        level_upper = str(level).upper()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {level_upper}: {msg}", flush=flush, **kwargs)


def split_id_from_indices(idx_test: np.ndarray) -> str:
    """Stable split identifier for sanity checks across parallel jobs."""
    idx = np.asarray(idx_test, dtype=int).copy()
    idx.sort()
    return hashlib.md5(idx.tobytes()).hexdigest()


def get_cpus(default: int = 1) -> int:
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    if "LSB_DJOB_NUMPROC" in os.environ:
        return int(os.environ["LSB_DJOB_NUMPROC"])
    return int(default)


# ----------------------------
# Output organization
# ----------------------------
# _mkdir moved to metric_utils.py

def make_outdirs(root: str) -> Dict[str, str]:
    d = {
        "core": _mkdir(os.path.join(root, "core")),
        "cv": _mkdir(os.path.join(root, "cv")),
        "preds_test": _mkdir(os.path.join(root, "preds", "test_preds")),
        "preds_val": _mkdir(os.path.join(root, "preds", "val_preds")),
        "preds_controls": _mkdir(os.path.join(root, "preds", "controls_oof")),
        "preds_train_oof": _mkdir(os.path.join(root, "preds", "train_oof")),
        "preds_plots": _mkdir(os.path.join(root, "preds", "plots")),
        "reports_features": _mkdir(os.path.join(root, "reports", "feature_reports")),
        "reports_stable": _mkdir(os.path.join(root, "reports", "stable_panel")),
        "reports_panels": _mkdir(os.path.join(root, "reports", "panels")),
        "reports_subgroups": _mkdir(os.path.join(root, "reports", "subgroups")),
        "diag": _mkdir(os.path.join(root, "diagnostics")),
        "diag_splits": _mkdir(os.path.join(root, "diagnostics", "splits")),
        "diag_prefilter": _mkdir(os.path.join(root, "diagnostics", "prefilter")),
        "diag_screen": _mkdir(os.path.join(root, "diagnostics", "screening")),
        "diag_calib": _mkdir(os.path.join(root, "diagnostics", "calibration")),
        "diag_lc": _mkdir(os.path.join(root, "diagnostics", "learning_curve")),
        "diag_ci": _mkdir(os.path.join(root, "diagnostics", "test_ci_files")),
        "diag_timing": _mkdir(os.path.join(root, "diagnostics", "timing")),
        "diag_survival": _mkdir(os.path.join(root, "diagnostics", "survival")),
    }
    return d


# ----------------------------
# Parsing helpers
# ----------------------------
def _parse_float_list(s: str) -> List[float]:
    if not s:
        return []
    out = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except Exception:
            continue
    return out


def _parse_int_list(s: str) -> List[int]:
    if not s:
        return []
    out = []
    for x in str(s).split(","):
        x = x.strip()
        if not x:
            continue
        try:
            out.append(int(x))
        except Exception:
            continue
    return out


def _require_int_list(s: str, name: str) -> List[int]:
    values = _parse_int_list(s)
    if not values:
        raise ValueError(f"{name} must be a non-empty comma-separated list (e.g. '200,500').")
    return values


def _parse_str_list(s: str) -> List[str]:
    if not s:
        return []
    out = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if tok:
            out.append(tok)
    return out


def _parse_none_int_float_list(s: str) -> List:
    """
    Parse list where tokens can be:
      - "none" => None
      - int
      - float
      - strings (e.g. "sqrt")
    """
    if not s:
        return []
    out = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        lo = tok.lower()
        if lo in ("none", "null"):
            out.append(None)
            continue
        if lo == "sqrt":
            out.append("sqrt")
            continue
        # try int then float
        try:
            if re.match(r"^[+-]?\d+$", tok):
                out.append(int(tok))
                continue
        except Exception:
            pass
        try:
            out.append(float(tok))
            continue
        except Exception:
            out.append(tok)
    return out



def _coerce_int_or_none_list(vals: List[Any], *, name: str) -> List[Any]:
    """Coerce lists intended for *integer-or-None* sklearn params.

    - Accepts ints and None.
    - Accepts floats that are whole numbers (e.g., 10.0 -> 10).
    - Raises a clear ValueError for non-integer floats/strings.
    """
    out: List[Any] = []
    for v in vals:
        if v is None:
            out.append(None)
            continue
        if isinstance(v, bool):
            # avoid bool being treated as int
            raise ValueError(f"{name}: invalid boolean value {v}")
        if isinstance(v, int):
            out.append(v)
            continue
        if isinstance(v, float):
            if float(v).is_integer():
                out.append(int(v))
                continue
            raise ValueError(f"{name}: expected int or None, got non-integer float {v}")
        if isinstance(v, str):
            vv = v.strip().lower()
            if vv in ("none", "null"):
                out.append(None)
                continue
            try:
                out.append(int(vv))
                continue
            except Exception:
                pass
            try:
                fv = float(vv)
                if float(fv).is_integer():
                    out.append(int(fv))
                    continue
            except Exception:
                pass
            raise ValueError(f"{name}: expected int or None, got '{v}'")
        raise ValueError(f"{name}: expected int or None, got {type(v).__name__}={v}")
    return out


def _coerce_min_samples_leaf_list(vals: List[Any], *, name: str = "rf_min_samples_leaf_grid") -> List[Any]:
    """Coerce min_samples_leaf grid to sklearn-legal types.

    sklearn accepts:
      - int >= 1
      - float in (0, 1.0) (fraction of samples)

    We additionally coerce whole-number floats (e.g., 5.0) to int 5,
    to be robust to CLI parsing that may yield floats.
    """
    out: List[Any] = []
    for v in vals:
        if isinstance(v, bool):
            raise ValueError(f"{name}: invalid boolean value {v}")
        if isinstance(v, int):
            if v < 1:
                raise ValueError(f"{name}: int must be >= 1, got {v}")
            out.append(v)
            continue
        if isinstance(v, float):
            if 0.0 < v < 1.0:
                out.append(float(v))
                continue
            if float(v).is_integer():
                iv = int(v)
                if iv < 1:
                    raise ValueError(f"{name}: int must be >= 1, got {iv}")
                out.append(iv)
                continue
            raise ValueError(f"{name}: float must be in (0,1) or an integer value, got {v}")
        if isinstance(v, str):
            vv = v.strip().lower()
            try:
                iv = int(vv)
                if iv < 1:
                    raise ValueError(f"{name}: int must be >= 1, got {iv}")
                out.append(iv)
                continue
            except Exception:
                pass
            try:
                fv = float(vv)
                if 0.0 < fv < 1.0:
                    out.append(float(fv))
                    continue
                if float(fv).is_integer():
                    iv = int(fv)
                    if iv < 1:
                        raise ValueError(f"{name}: int must be >= 1, got {iv}")
                    out.append(iv)
                    continue
            except Exception:
                pass
            raise ValueError(f"{name}: could not parse '{v}' as int>=1 or float in (0,1)")
        raise ValueError(f"{name}: unsupported type {type(v).__name__}={v}")
    return out

def parse_k_grid(s: str) -> List[int]:
    return _parse_int_list(s)


def parse_tune_n_jobs(s: str, cpus: int) -> Optional[int]:
    """
    Returns:
      None => "auto"
      int  => explicit n_jobs
    """
    if s is None:
        return None
    if isinstance(s, int):
        return int(max(1, s))
    ss = str(s).strip().lower()
    if ss in ("", "auto", "none"):
        return None
    try:
        v = int(float(ss))
        return int(max(1, v))
    except Exception:
        return None


def parse_error_score(s: str):
    """
    RandomizedSearchCV(error_score=...): float or "raise"
    Default in this code: np.nan (as string "nan")
    """
    if s is None:
        return np.nan
    ss = str(s).strip().lower()
    if ss == "raise":
        return "raise"
    if ss in ("nan", "na", "none", ""):
        return np.nan
    try:
        return float(ss)
    except Exception:
        return np.nan


def parse_class_weight_options(s: str) -> List:
    """
    Input examples:
      "none,balanced" => [None, "balanced"]
      "none"          => [None]
      "balanced"      => ["balanced"]
    """
    if not s:
        return [None, "balanced"]
    toks = [t.strip().lower() for t in str(s).split(",") if t.strip()]
    out = []
    for t in toks:
        if t in ("none", "null"):
            out.append(None)
        elif t == "balanced":
            out.append("balanced")
    # if user passes junk, fall back to safe default
    if not out:
        return [None, "balanced"]
    # dedupe while preserving order
    seen = set()
    out2 = []
    for v in out:
        key = str(v)
        if key in seen:
            continue
        seen.add(key)
        out2.append(v)
    return out2


def compute_scale_pos_weight_from_y(y: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    pos = max(1, pos)
    neg = max(1, neg)
    return float(neg / pos)


def resolve_xgb_scale_pos_weight(y: np.ndarray, args) -> float:
    raw = str(args.xgb_scale_pos_weight).strip().lower()
    if raw in ("", "auto"):
        return compute_scale_pos_weight_from_y(y)
    try:
        return float(raw)
    except Exception:
        return compute_scale_pos_weight_from_y(y)


def make_logspace(minv: float, maxv: float, points: int, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    minv = float(minv)
    maxv = float(maxv)
    points = int(points)
    if points < 2:
        points = 2
    # allow user to pass linear values; interpret as powers of 10 if min/max look like 1e-3..1e3
    # Here we treat min/max as actual numbers and build logspace by their log10.
    if minv <= 0 or maxv <= 0:
        return np.logspace(-3, 3, 13)
    a = np.log10(minv)
    b = np.log10(maxv)
    if rng is not None:
        samples = rng.uniform(a, b, size=points)
        return np.power(10.0, samples)
    return np.logspace(a, b, points)


def _randomize_numeric_list(
    values: Sequence[Any],
    rng: Optional[np.random.RandomState],
    *,
    as_int: bool = False,
    min_int: Optional[int] = None,
    max_int: Optional[int] = None,
    unique_int: bool = False,
    log_scale: bool = False,
    min_float: Optional[float] = None,
    max_float: Optional[float] = None,
    perturb_mode: bool = False,
) -> List[Any]:
    """
    Randomize numeric grid values in-place (used when --grid_randomize is enabled).
    Non-numeric entries are preserved.

    If perturb_mode=True, adds random noise to each grid point individually.
    Otherwise, samples uniformly from [min(values), max(values)].
    """
    values_list = list(values)
    if rng is None or len(values_list) <= 1:
        return values_list

    numeric_idx = [i for i, v in enumerate(values_list) if isinstance(v, numbers.Real)]
    if len(numeric_idx) <= 1:
        return values_list

    numeric_vals = [float(values_list[i]) for i in numeric_idx]
    low = min(numeric_vals)
    high = max(numeric_vals)

    if math.isclose(low, high):
        return values_list

    n = len(numeric_idx)
    sampled_vals: List[Any]

    # Perturb mode: add noise to each grid point individually
    if perturb_mode and not as_int:
        sorted_vals = sorted(numeric_vals)
        perturbed = []
        for i, val in enumerate(sorted_vals):
            # Determine perturbation range based on neighbors
            if i == 0:
                # First point: perturb within half the gap to next point
                gap = (sorted_vals[i+1] - val) / 2 if i+1 < len(sorted_vals) else (high - low) * 0.1
                noise = rng.uniform(-gap * 0.5, gap)
            elif i == len(sorted_vals) - 1:
                # Last point: perturb within half the gap from previous point
                gap = (val - sorted_vals[i-1]) / 2
                noise = rng.uniform(-gap, gap * 0.5)
            else:
                # Middle points: perturb within the interval to neighbors
                gap_left = (val - sorted_vals[i-1]) / 2
                gap_right = (sorted_vals[i+1] - val) / 2
                noise = rng.uniform(-gap_left, gap_right)

            perturbed_val = val + noise
            if min_float is not None:
                perturbed_val = max(perturbed_val, float(min_float))
            if max_float is not None:
                perturbed_val = min(perturbed_val, float(max_float))
            perturbed.append(perturbed_val)

        sampled_vals = perturbed
    elif as_int:
        if perturb_mode:
            # Perturb integer values around grid points
            sorted_vals = sorted(numeric_vals)
            perturbed = []
            for i, val in enumerate(sorted_vals):
                int_val = int(val)
                # Determine perturbation range based on neighbors
                if i == 0:
                    # First point: perturb within half the gap to next point
                    gap = int((sorted_vals[i+1] - val) / 2) if i+1 < len(sorted_vals) else max(1, int((high - low) * 0.1))
                    offset = rng.randint(-max(1, gap // 2), gap + 1)
                elif i == len(sorted_vals) - 1:
                    # Last point: perturb within half the gap from previous point
                    gap = int((val - sorted_vals[i-1]) / 2)
                    offset = rng.randint(-gap, max(1, gap // 2) + 1)
                else:
                    # Middle points: perturb within the interval to neighbors
                    gap_left = int((val - sorted_vals[i-1]) / 2)
                    gap_right = int((sorted_vals[i+1] - val) / 2)
                    offset = rng.randint(-gap_left, gap_right + 1)

                perturbed_val = int_val + offset
                if min_int is not None:
                    perturbed_val = max(perturbed_val, int(min_int))
                if max_int is not None:
                    perturbed_val = min(perturbed_val, int(max_int))
                perturbed.append(perturbed_val)

            # Handle unique_int constraint
            if unique_int:
                perturbed = list(dict.fromkeys(perturbed))  # Remove duplicates while preserving order
                # If we lost values due to deduplication, regenerate
                while len(perturbed) < n:
                    lo = int(math.floor(low))
                    hi = int(math.ceil(high))
                    if min_int is not None:
                        lo = max(lo, int(min_int))
                    if max_int is not None:
                        hi = min(hi, int(max_int))
                    new_val = rng.randint(lo, hi + 1)
                    if new_val not in perturbed:
                        perturbed.append(new_val)
            sampled_vals = perturbed
        else:
            lo = int(math.floor(low))
            hi = int(math.ceil(high))
            if min_int is not None:
                lo = max(lo, int(min_int))
            if max_int is not None:
                hi = min(hi, int(max_int))
            if hi < lo:
                hi = lo
            domain = np.arange(lo, hi + 1)
            if domain.size == 0:
                return values_list
            replace = True
            if unique_int and domain.size >= n:
                replace = False
            samples = rng.choice(domain, size=n, replace=replace)
            sampled_vals = [int(x) for x in samples]
    else:
        if log_scale and (low <= 0 or high <= 0):
            log_scale = False

        if log_scale:
            logs = rng.uniform(np.log10(low), np.log10(high), size=n)
            samples = np.power(10.0, logs)
        else:
            samples = rng.uniform(low, high, size=n)

        if min_float is not None:
            samples = np.maximum(samples, float(min_float))
        if max_float is not None:
            samples = np.minimum(samples, float(max_float))
        sampled_vals = [float(x) for x in samples]

    for idx, val in zip(numeric_idx, sampled_vals):
        values_list[idx] = val

    return values_list


# ----------------------------
# Metrics helpers - moved to metric_utils.py
# ----------------------------
# prob_metrics, _safe_metric, stratified_bootstrap_ci imported from metric_utils


def compute_test_cis(y_te: np.ndarray, p_test: np.ndarray, n_boot: int, seed: int):
    ci_auc = stratified_bootstrap_ci(y_te, p_test, auroc, n_boot=n_boot, seed=seed)
    ci_pr  = stratified_bootstrap_ci(y_te, p_test, prauc, n_boot=n_boot, seed=seed)
    ci_br  = stratified_bootstrap_ci(y_te, p_test, brier, n_boot=n_boot, seed=seed)
    return ci_auc, ci_pr, ci_br


# ----------------------------
# Calibration metrics (intercept + slope) + bootstrap
# ----------------------------
# ----------------------------
# Decision Curve Analysis (DCA) for Clinical Utility
# ----------------------------
# DCA functions moved to dca_utils.py (imported above)


def _format_prev(n_pos: int, n_total: int) -> str:
    if n_total <= 0:
        return "NA"
    return f"{(n_pos / n_total):.4f}"


def _format_split_meta(label: str, n: int, n_pos: int, frac: Optional[float] = None) -> str:
    if n <= 0:
        base = f"{label}: n=0"
    else:
        base = f"{label}: n={n} pos={n_pos} prev={_format_prev(n_pos, n)}"
    if frac is not None and np.isfinite(frac):
        base += f" frac={frac:.3f}"
    return base


def _build_split_metadata_lines(
    scen_name: str,
    model_label: str,
    split_label: str,
    split_id: Optional[str],
    n_train: int,
    n_val: int,
    n_test: int,
    n_train_pos: int,
    n_val_pos: int,
    n_test_pos: int,
    risk_prob_source: Optional[str] = None,
    threshold_source: Optional[str] = None,
    extra_parts: Optional[Sequence[str]] = None,
) -> List[str]:
    denom = n_train + n_val + n_test
    train_frac = (n_train / denom) if denom else np.nan
    val_frac = (n_val / denom) if denom else np.nan
    test_frac = (n_test / denom) if denom else np.nan

    header = f"Scenario={scen_name} | Model={model_label} | Split={split_label}"
    if split_id:
        header += f" | split_id={split_id}"

    split_lines = [
        _format_split_meta("Train", n_train, n_train_pos, train_frac),
        _format_split_meta("Val", n_val, n_val_pos, val_frac),
        _format_split_meta("Test", n_test, n_test_pos, test_frac),
    ]

    tail_parts: List[str] = []
    if risk_prob_source:
        tail_parts.append(f"risk_prob={risk_prob_source}")
    if threshold_source:
        tail_parts.append(f"thr_source={threshold_source}")
    if extra_parts:
        tail_parts.extend([p for p in extra_parts if p])
    tail = " | ".join(tail_parts) if tail_parts else ""

    lines = [header] + split_lines
    if tail:
        lines.append(tail)

    return lines

def _coerce_time_series(ser: pd.Series) -> Tuple[pd.Series, str]:
    """Return series coerced to numeric or datetime with a type tag."""
    if np.issubdtype(ser.dtype, np.number):
        return pd.to_numeric(ser, errors="coerce"), "numeric"
    if np.issubdtype(ser.dtype, np.datetime64):
        return pd.to_datetime(ser, errors="coerce"), "datetime"

    dt = pd.to_datetime(ser, errors="coerce")
    dt_valid = int(dt.notna().sum())
    if len(ser) > 0 and dt_valid >= max(3, int(0.7 * len(ser))):
        return dt, "datetime"
    return pd.to_numeric(ser, errors="coerce"), "numeric"


def _compute_survival_time(
    df: pd.DataFrame,
    time_col: str = "",
    start_col: str = "",
    end_col: str = "",
    time_unit: str = "days",
) -> Tuple[Optional[pd.Series], List[str], str]:
    warnings: List[str] = []
    time: Optional[pd.Series] = None
    time_mode = "numeric"

    if time_col:
        if time_col not in df.columns:
            warnings.append(f"time_col '{time_col}' not found")
            return None, warnings
        ser = df[time_col]
        coerced, mode = _coerce_time_series(ser)
        if mode == "datetime":
            warnings.append(
                f"time_col '{time_col}' parsed as datetime; using days since min({time_col})"
            )
            ref = coerced.min()
            time = (coerced - ref).dt.total_seconds() / 86400.0
            time_mode = "datetime"
        else:
            time = coerced
            time_mode = "numeric"
    elif start_col and end_col:
        missing = [c for c in (start_col, end_col) if c not in df.columns]
        if missing:
            warnings.append(f"start/end cols missing: {missing}")
            return None, warnings, time_mode
        start_raw, start_mode = _coerce_time_series(df[start_col])
        end_raw, end_mode = _coerce_time_series(df[end_col])
        if start_mode == "datetime" and end_mode == "datetime":
            time = (end_raw - start_raw).dt.total_seconds() / 86400.0
            time_mode = "datetime"
        elif start_mode == "numeric" and end_mode == "numeric":
            time = pd.to_numeric(end_raw, errors="coerce") - pd.to_numeric(start_raw, errors="coerce")
            time_mode = "numeric"
        else:
            warnings.append("start/end cols have mixed types; cannot compute durations")
            return None, warnings, time_mode
    else:
        warnings.append("no survival time columns provided")
        return None, warnings, time_mode

    if time_unit == "years" and time_mode == "datetime":
        time = time / 365.25
    elif time_unit == "years" and time_mode == "numeric":
        warnings.append("time_unit=years assumes numeric inputs are already in years")
    return time, warnings, time_mode


def _compute_survival_event(
    df: pd.DataFrame,
    default_event: np.ndarray,
    event_col: str = "",
) -> Tuple[pd.Series, List[str]]:
    warnings: List[str] = []
    if event_col:
        if event_col not in df.columns:
            warnings.append(f"event_col '{event_col}' not found; using default event")
            return pd.Series(default_event, index=df.index), warnings
        ser = pd.to_numeric(df[event_col], errors="coerce")
        if ser.notna().sum() == 0:
            warnings.append(f"event_col '{event_col}' not numeric; using default event")
            return pd.Series(default_event, index=df.index), warnings
        return ser, warnings
    return pd.Series(default_event, index=df.index), warnings


def _assign_km_groups(
    scores: pd.Series,
    n_groups: int,
    labels: Optional[List[str]] = None,
) -> Tuple[pd.Series, List[str]]:
    scores = pd.Series(scores, index=scores.index)
    if n_groups <= 1:
        return pd.Series(["All"] * len(scores), index=scores.index), ["All"]

    if n_groups == 2:
        med = float(np.nanmedian(scores))
        base_labels = ["Low risk", "High risk"]
        use_labels = labels if labels and len(labels) == 2 else base_labels
        grp = np.where(scores >= med, use_labels[1], use_labels[0])
        return pd.Series(grp, index=scores.index), use_labels

    try:
        q = pd.qcut(scores, q=n_groups, labels=False, duplicates="drop")
        n_actual = int(np.nanmax(q)) + 1 if q.notna().any() else 0
        base_labels = [f"Q{i+1}" for i in range(n_actual)]
        use_labels = labels if labels and len(labels) == n_actual else base_labels
        grp = q.map(lambda i: use_labels[int(i)] if pd.notna(i) else "Unknown")
        return grp.astype(str), use_labels
    except Exception:
        med = float(np.nanmedian(scores))
        base_labels = ["Low risk", "High risk"]
        use_labels = labels if labels and len(labels) == 2 else base_labels
        grp = np.where(scores >= med, use_labels[1], use_labels[0])
        return pd.Series(grp, index=scores.index), use_labels


def _kaplan_meier_curve(times: np.ndarray, events: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(times)
    times = np.asarray(times)[order]
    events = np.asarray(events)[order].astype(int)

    uniq_times = np.unique(times)
    surv = 1.0
    km_times = [0.0]
    km_surv = [1.0]
    for t in uniq_times:
        at_risk = np.sum(times >= t)
        if at_risk <= 0:
            continue
        d = int(events[times == t].sum())
        surv *= (1.0 - (d / at_risk))
        km_times.append(float(t))
        km_surv.append(float(surv))
    return np.array(km_times), np.array(km_surv)


def _plot_kaplan_meier(
    df_surv: pd.DataFrame,
    out_path: str,
    group_labels: List[str],
    time_unit: str,
    title: str = "Kaplan-Meier Curve",
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    for label in group_labels:
        sub = df_surv[df_surv["km_group"] == label]
        if sub.empty:
            continue
        times, surv = _kaplan_meier_curve(sub["time"].to_numpy(), sub["event"].to_numpy())
        n = len(sub)
        events = int(sub["event"].sum())
        ax.step(times, surv, where="post", linewidth=2, label=f"{label} (n={n}, events={events})")

    ax.set_xlabel(f"Time ({time_unit})", fontsize=12)
    ax.set_ylabel("Survival probability", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.02, 1.02])

    plt.subplots_adjust(left=0.15, right=0.9, top=0.8, bottom=0.15)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.8)
    plt.close()


def _plot_cox_hazard_ratios(
    cox_df: pd.DataFrame,
    out_path: str,
    title: str = "Cox Regression Hazard Ratios",
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if cox_df.empty:
        return

    fig_h = max(3.0, 0.5 * len(cox_df))
    fig, ax = plt.subplots(figsize=(8, fig_h))
    y_pos = np.arange(len(cox_df))

    hr = cox_df["hazard_ratio"].to_numpy(dtype=float)
    lower = cox_df["ci_lower"].to_numpy(dtype=float)
    upper = cox_df["ci_upper"].to_numpy(dtype=float)
    xerr = np.vstack([hr - lower, upper - hr])

    ax.errorbar(hr, y_pos, xerr=xerr, fmt="o", color="black", ecolor="gray", capsize=3)
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cox_df["covariate"].tolist())
    ax.set_xscale("log")
    ax.set_xlabel("Hazard Ratio (log scale)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, axis="x", alpha=0.3)

    plt.subplots_adjust(left=0.2, right=0.9, top=0.8, bottom=0.15)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.8)
    plt.close()


def save_survival_plots(
    df_te: pd.DataFrame,
    y_te: np.ndarray,
    risk_scores: np.ndarray,
    out_dir: str,
    prefix: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"survival_plots": False, "warnings": []}

    time, time_warnings, time_mode = _compute_survival_time(
        df_te,
        time_col=str(getattr(args, "survival_time_col", "") or ""),
        start_col=str(getattr(args, "survival_start_col", "") or ""),
        end_col=str(getattr(args, "survival_end_col", "") or ""),
        time_unit=str(getattr(args, "survival_time_unit", "days")),
    )
    summary["warnings"].extend(time_warnings)
    summary["time_mode"] = time_mode
    if time is None:
        summary["error"] = "missing time data"
        return summary

    event, event_warnings = _compute_survival_event(
        df_te,
        y_te,
        event_col=str(getattr(args, "survival_event_col", "") or ""),
    )
    summary["warnings"].extend(event_warnings)

    df_surv = pd.DataFrame({
        "time": time,
        "event": event,
        "risk": pd.Series(risk_scores, index=df_te.index),
    })
    df_surv = df_surv.replace([np.inf, -np.inf], np.nan)
    df_surv = df_surv.dropna(subset=["time", "event", "risk"]).copy()
    df_surv = df_surv[df_surv["time"] > 0].copy()
    df_surv["event"] = (df_surv["event"] > 0).astype(int)

    if df_surv.empty:
        summary["error"] = "no valid survival rows after filtering"
        return summary

    if df_surv["event"].sum() == 0:
        summary["warnings"].append("no events observed; survival plots skipped")
        return summary

    os.makedirs(out_dir, exist_ok=True)
    time_unit = str(getattr(args, "survival_time_unit", "days"))

    # Kaplan-Meier curve
    try:
        km_groups = int(getattr(args, "km_groups", 2))
        km_labels = _parse_str_list(getattr(args, "km_group_labels", ""))
        km_group, labels = _assign_km_groups(df_surv["risk"], km_groups, km_labels)
        df_surv["km_group"] = km_group
        km_path = os.path.join(out_dir, f"{prefix}kaplan_meier.png")
        _plot_kaplan_meier(
            df_surv,
            km_path,
            group_labels=labels,
            time_unit=time_unit,
            title="Kaplan-Meier Curve by Risk Group",
        )
        summary["kaplan_meier_path"] = km_path
    except Exception as e:
        summary["warnings"].append(f"KM plot failed: {e}")

    # Cox regression hazard ratios
    try:
        cox_covariates = _parse_str_list(getattr(args, "cox_covariates", ""))
        cov_df = pd.DataFrame({"risk_score": df_surv["risk"]}, index=df_surv.index)
        for cov in cox_covariates:
            if cov in df_te.columns:
                cov_df[cov] = df_te.loc[df_surv.index, cov]
            else:
                summary["warnings"].append(f"cox covariate '{cov}' not found; skipping")

        cox_full = pd.concat([df_surv[["time", "event"]], cov_df], axis=1)
        cox_full = cox_full.replace([np.inf, -np.inf], np.nan).dropna()
        if cox_full["event"].sum() <= 0:
            summary["warnings"].append("no events in Cox data; skipping Cox plot")
        else:
            X = pd.get_dummies(cox_full.drop(columns=["time", "event"]), drop_first=True)
            nunique = X.nunique(dropna=True)
            X = X.loc[:, nunique > 1]
            if X.shape[1] == 0:
                summary["warnings"].append("no valid Cox covariates after filtering")
            elif int(cox_full["event"].sum()) <= X.shape[1]:
                summary["warnings"].append("too few events for Cox covariates; skipping Cox plot")
            else:
                model = sm.PHReg(cox_full["time"].to_numpy(), X, status=cox_full["event"].to_numpy())
                res = model.fit(disp=0)
                params = np.asarray(res.params)
                bse = np.asarray(res.bse)
                pvals = np.asarray(getattr(res, "pvalues", np.full_like(params, np.nan)))

                hr = np.exp(params)
                ci_lower = np.exp(params - 1.96 * bse)
                ci_upper = np.exp(params + 1.96 * bse)
                cox_summary = pd.DataFrame({
                    "covariate": X.columns.tolist(),
                    "coef": params,
                    "hazard_ratio": hr,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "p_value": pvals,
                })

                cox_csv = os.path.join(out_dir, f"{prefix}cox_hazard_ratios.csv")
                cox_summary.to_csv(cox_csv, index=False)
                summary["cox_csv_path"] = cox_csv

                cox_plot = os.path.join(out_dir, f"{prefix}cox_hazard_ratios.png")
                _plot_cox_hazard_ratios(cox_summary, cox_plot)
                summary["cox_plot_path"] = cox_plot
    except Exception as e:
        summary["warnings"].append(f"Cox plot failed: {e}")

    summary["survival_plots"] = True
    return summary


def regenerate_plots_from_artifacts(
    run_dir: str,
    force: bool = False,
    plot_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Regenerate diagnostic plots from saved prediction artifacts using shared_utils plotting functions.

    Loads prediction artifacts (test_preds CSVs, test_metrics thresholds, DCA curves) and
    regenerates diagnostic plots without model retraining. Uses centralized plotting functions
    from shared_utils for consistency with postprocess_compare.py and other visualization.

    Args:
        run_dir: Path to model run directory (e.g., results_dir/IncidentPlusPrevalent__RF__5x10...)
        force: If True, overwrite existing plots
        plot_types: List of plot types to regenerate. If None, regenerate all.
                   Options: ["roc", "pr", "calibration", "risk_dist"]

    Returns:
        Dictionary with keys:
            - success: bool indicating if any plots were regenerated
            - plots_regenerated: list of successfully regenerated plot types
            - plots_skipped: list of existing plots (not overwritten)
            - errors: list of error messages encountered
    """
    summary: Dict[str, Any] = {
        "success": False,
        "plots_regenerated": [],
        "plots_skipped": [],
        "errors": [],
    }

    logprint(f"[regen] Regenerating plots from: {run_dir}")

    # ===== LOAD ARTIFACTS =====
    # Load run settings
    settings_path = os.path.join(run_dir, "core", "run_settings.json")
    if not os.path.exists(settings_path):
        err = f"run_settings.json not found at {settings_path}"
        logprint(f"[regen] ERROR: {err}", level="error")
        summary["errors"].append(err)
        return summary

    with open(settings_path, "r") as f:
        settings = json.load(f)

    scenario = settings.get("scenario", "unknown")
    model_name = settings.get("models", ["unknown"])[0] if isinstance(settings.get("models"), list) else "unknown"
    panel_tag = settings.get("panel_tag", "")
    model_label = f"{model_name}__{panel_tag}" if panel_tag else model_name

    logprint(f"[regen] Scenario: {scenario}, Model: {model_label}")

    # Load TEST predictions
    test_preds_path = os.path.join(run_dir, "preds", "test_preds", f"{scenario}__{model_label}.csv")
    if not os.path.exists(test_preds_path):
        err = f"TEST predictions not found at {test_preds_path}"
        logprint(f"[regen] ERROR: {err}", level="error")
        summary["errors"].append(err)
        return summary

    df_test_preds = pd.read_csv(test_preds_path)
    y_test = df_test_preds["y_test"].to_numpy(dtype=int)
    p_test_raw = df_test_preds.get("p_test_raw", df_test_preds["p_test"]).to_numpy(dtype=float)
    p_test_adjusted = df_test_preds.get("p_test_adjusted", df_test_preds["p_test"]).to_numpy(dtype=float)
    p_test = df_test_preds["p_test"].to_numpy(dtype=float)

    logprint(f"[regen] Loaded TEST predictions: n={len(y_test)}, pos={int(y_test.sum())}")

    # Load TEST metrics for thresholds
    test_metrics_path = os.path.join(run_dir, "core", "test_metrics.csv")
    thresholds = {}
    if os.path.exists(test_metrics_path):
        df_metrics = pd.read_csv(test_metrics_path)
        if len(df_metrics) > 0:
            thresholds["youden"] = df_metrics.iloc[0].get("youden_threshold", np.nan)
            thresholds["alpha"] = df_metrics.iloc[0].get("alpha_threshold", np.nan)
            thresholds["spec95"] = df_metrics.iloc[0].get("spec95_threshold", np.nan)
        youden_val = thresholds.get("youden", np.nan)
        if np.isfinite(youden_val):
            logprint(f"[regen] Loaded thresholds: youden={youden_val:.4f}")

    # Load DCA threshold (optional)
    dca_threshold = None
    dca_csv_path = os.path.join(run_dir, "diagnostics", "dca", f"{scenario}__{model_label}__dca_curve.csv")
    if os.path.exists(dca_csv_path):
        try:
            dca_threshold = find_dca_zero_crossing(dca_csv_path)
            if dca_threshold:
                logprint(f"[regen] Loaded DCA zero-crossing: {dca_threshold:.4f}")
        except Exception as e:
            logprint(f"[regen] WARNING: Could not load DCA threshold: {e}", level="warning")

    # ===== BUILD METADATA =====
    n_train = int(settings.get("n_train", 0))
    n_val = int(settings.get("n_val", 0))
    n_test = len(y_test)
    n_train_pos = int(settings.get("n_train_pos", 0))
    n_val_pos = int(settings.get("n_val_pos", 0))
    n_test_pos = int(y_test.sum())
    risk_prob_source = settings.get("risk_prob_source", "raw")
    threshold_source = settings.get("threshold_source", "train_oof")

    meta_lines = _build_split_metadata_lines(
        scen_name=scenario,
        model_label=model_label,
        split_label="TEST",
        split_id=None,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        n_train_pos=n_train_pos,
        n_val_pos=n_val_pos,
        n_test_pos=n_test_pos,
        risk_prob_source=risk_prob_source,
        threshold_source=threshold_source,
    )

    # ===== DETERMINE WHICH PLOTS TO REGENERATE =====
    all_plot_types = ["roc", "pr", "calibration", "risk_dist"]
    if plot_types is None:
        plot_types_to_gen = all_plot_types
    else:
        plot_types_to_gen = [pt.lower() for pt in plot_types if pt.lower() in all_plot_types]

    logprint(f"[regen] Plot types to regenerate: {plot_types_to_gen}")

    # Create output directories
    plots_dir = os.path.join(run_dir, "diagnostics", "plots")
    preds_plots_dir = os.path.join(run_dir, "preds", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(preds_plots_dir, exist_ok=True)

    # ===== REGENERATE PLOTS USING SHARED_UTILS FUNCTIONS =====

    # ROC Curve (using shared_utils._plot_roc_curve)
    if "roc" in plot_types_to_gen:
        roc_path = os.path.join(plots_dir, f"{scenario}__roc__{model_label}.png")
        if force or not os.path.exists(roc_path):
            try:
                _plot_roc_curve(
                    y_test,
                    p_test,
                    roc_path,
                    f"ROC Curve - {scenario} - {model_label}",
                    meta_lines=meta_lines,
                )
                summary["plots_regenerated"].append("roc")
                logprint(f"[regen] ✓ ROC curve")
            except Exception as e:
                err = f"ROC curve failed: {e}"
                summary["errors"].append(err)
                logprint(f"[regen] ✗ {err}", level="error")
        else:
            summary["plots_skipped"].append("roc")
            logprint(f"[regen] - ROC curve (exists, use --force to overwrite)")

    # PR Curve (using shared_utils._plot_pr_curve)
    if "pr" in plot_types_to_gen:
        pr_path = os.path.join(plots_dir, f"{scenario}__pr__{model_label}.png")
        if force or not os.path.exists(pr_path):
            try:
                _plot_pr_curve(
                    y_test,
                    p_test,
                    pr_path,
                    f"Precision-Recall - {scenario} - {model_label}",
                    meta_lines=meta_lines,
                )
                summary["plots_regenerated"].append("pr")
                logprint(f"[regen] ✓ PR curve")
            except Exception as e:
                err = f"PR curve failed: {e}"
                summary["errors"].append(err)
                logprint(f"[regen] ✗ {err}", level="error")
        else:
            summary["plots_skipped"].append("pr")
            logprint(f"[regen] - PR curve (exists, use --force to overwrite)")

    # Calibration Curves (using shared_utils._compute_recalibration + _plot_prob_calibration_panel)
    if "calibration" in plot_types_to_gen:
        # Raw probabilities calibration
        calib_raw_path = os.path.join(plots_dir, f"{scenario}__calibration_raw__{model_label}.png")
        if force or not os.path.exists(calib_raw_path):
            try:
                alpha_raw, beta_raw = _compute_recalibration(y_test, p_test_raw)
                _plot_calibration_curve(
                    y_test, p_test_raw,
                    out_path=calib_raw_path,
                    title=f"Calibration (Raw) - {scenario} - {model_label}",
                    n_bins=10,
                    calib_intercept=alpha_raw if np.isfinite(alpha_raw) else np.nan,
                    calib_slope=beta_raw if np.isfinite(beta_raw) else np.nan,
                    meta_lines=meta_lines,
                )
                summary["plots_regenerated"].append("calibration_raw")
                logprint(f"[regen] ✓ Calibration (raw)")
            except Exception as e:
                err = f"Calibration (raw) failed: {e}"
                summary["errors"].append(err)
                logprint(f"[regen] ✗ {err}", level="error")
        else:
            summary["plots_skipped"].append("calibration_raw")
            logprint(f"[regen] - Calibration (raw) (exists)")

        # Adjusted probabilities calibration
        calib_adj_path = os.path.join(plots_dir, f"{scenario}__calibration_adjusted__{model_label}.png")
        if force or not os.path.exists(calib_adj_path):
            try:
                alpha_adj, beta_adj = _compute_recalibration(y_test, p_test_adjusted)
                _plot_calibration_curve(
                    y_test, p_test_adjusted,
                    out_path=calib_adj_path,
                    title=f"Calibration (Adjusted) - {scenario} - {model_label}",
                    n_bins=10,
                    calib_intercept=alpha_adj if np.isfinite(alpha_adj) else np.nan,
                    calib_slope=beta_adj if np.isfinite(beta_adj) else np.nan,
                    meta_lines=meta_lines,
                )
                summary["plots_regenerated"].append("calibration_adjusted")
                logprint(f"[regen] ✓ Calibration (adjusted)")
            except Exception as e:
                err = f"Calibration (adjusted) failed: {e}"
                summary["errors"].append(err)
                logprint(f"[regen] ✗ {err}", level="error")
        else:
            summary["plots_skipped"].append("calibration_adjusted")
            logprint(f"[regen] - Calibration (adjusted) (exists)")

    # Risk Distribution Plots (using shared_utils._plot_risk_distribution)
    if "risk_dist" in plot_types_to_gen:
        # Risk distribution: active probabilities (with threshold annotations)
        risk_active_path = os.path.join(preds_plots_dir, f"{scenario}__risk_dist_active__{model_label}.png")
        if force or not os.path.exists(risk_active_path):
            try:
                _plot_risk_distribution(
                    y_test,
                    p_test,
                    risk_active_path,
                    f"Risk Distribution (Active) - {scenario} - {model_label}",
                    subtitle="TEST set",
                    dca_threshold=dca_threshold,
                    spec95_threshold=thresholds.get("spec95"),
                    youden_threshold=thresholds.get("youden"),
                    alpha_threshold=thresholds.get("alpha"),
                    meta_lines=meta_lines,
                )
                summary["plots_regenerated"].append("risk_dist_active")
                logprint(f"[regen] ✓ Risk distribution (active)")
            except Exception as e:
                err = f"Risk distribution (active) failed: {e}"
                summary["errors"].append(err)
                logprint(f"[regen] ✗ {err}", level="error")
        else:
            summary["plots_skipped"].append("risk_dist_active")

        # Risk distribution: raw probabilities
        risk_raw_path = os.path.join(preds_plots_dir, f"{scenario}__risk_dist_raw__{model_label}.png")
        if force or not os.path.exists(risk_raw_path):
            try:
                _plot_risk_distribution(
                    y_test,
                    p_test_raw,
                    risk_raw_path,
                    f"Risk Distribution (Raw) - {scenario} - {model_label}",
                    subtitle="TEST set",
                    meta_lines=meta_lines,
                )
                summary["plots_regenerated"].append("risk_dist_raw")
                logprint(f"[regen] ✓ Risk distribution (raw)")
            except Exception as e:
                err = f"Risk distribution (raw) failed: {e}"
                summary["errors"].append(err)
                logprint(f"[regen] ✗ {err}", level="error")
        else:
            summary["plots_skipped"].append("risk_dist_raw")

        # Risk distribution: adjusted probabilities
        risk_adj_path = os.path.join(preds_plots_dir, f"{scenario}__risk_dist_adjusted__{model_label}.png")
        if force or not os.path.exists(risk_adj_path):
            try:
                _plot_risk_distribution(
                    y_test,
                    p_test_adjusted,
                    risk_adj_path,
                    f"Risk Distribution (Adjusted) - {scenario} - {model_label}",
                    subtitle="TEST set",
                    meta_lines=meta_lines,
                )
                summary["plots_regenerated"].append("risk_dist_adjusted")
                logprint(f"[regen] ✓ Risk distribution (adjusted)")
            except Exception as e:
                err = f"Risk distribution (adjusted) failed: {e}"
                summary["errors"].append(err)
                logprint(f"[regen] ✗ {err}", level="error")
        else:
            summary["plots_skipped"].append("risk_dist_adjusted")

    # ===== FINALIZE =====
    summary["success"] = len(summary["plots_regenerated"]) > 0
    logprint(f"[regen] Complete: {len(summary['plots_regenerated'])} regenerated, {len(summary['plots_skipped'])} skipped, {len(summary['errors'])} errors")

    return summary


def _plot_hyperparameter_tuning_history(cv_results_dict: dict, out_path: str,
                                         model_name: str = "Model",
                                         scoring: str = "neg_brier_score",
                                         meta_lines: Optional[Sequence[str]] = None) -> None:
    """Generate hyperparameter tuning history plot from RandomizedSearchCV.cv_results_."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Extract scores
    mean_test_scores = cv_results_dict.get("mean_test_score", [])
    std_test_scores = cv_results_dict.get("std_test_score", [])
    params = cv_results_dict.get("params", [])

    if len(mean_test_scores) == 0:
        return

    # Sort by mean test score
    sorted_indices = np.argsort(mean_test_scores)[::-1]  # descending
    mean_test_scores = np.array(mean_test_scores)[sorted_indices]
    std_test_scores = np.array(std_test_scores)[sorted_indices]

    # Create iteration indices
    iterations = np.arange(len(mean_test_scores))

    # Flip sign if negative score (for visualization)
    score_label = scoring
    metric_is_error = False
    if scoring.startswith("neg_"):
        mean_test_scores = -mean_test_scores
        score_label = scoring.replace("neg_", "")
        metric_is_error = True

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Top panel: Scores vs iteration (sorted by performance)
    ax1.plot(iterations, mean_test_scores, "b-", linewidth=1.5, alpha=0.7, label="Mean CV score")
    ax1.fill_between(iterations,
                     mean_test_scores - std_test_scores,
                     mean_test_scores + std_test_scores,
                     alpha=0.2, color="blue", label="\u00b11 std")

    # Highlight best iteration (min for error metrics, max otherwise)
    best_idx = np.argmin(mean_test_scores) if metric_is_error else np.argmax(mean_test_scores)
    ax1.scatter([best_idx], [mean_test_scores[best_idx]], color="red", s=100,
               zorder=5, marker="*", label=f"Best (iter={best_idx}, score={mean_test_scores[best_idx]:.4f})")

    ax1.set_xlabel("Iteration (sorted by performance)", fontsize=12)
    label_suffix = " (LOWER IS BETTER)" if metric_is_error else ""
    ax1.set_ylabel(f"{score_label.upper()}{label_suffix}", fontsize=12)
    ax1.set_title(f"Hyperparameter Tuning History - {model_name}", fontsize=14)
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Bottom panel: Score distribution
    ax2.hist(mean_test_scores, bins=min(50, len(mean_test_scores) // 2), alpha=0.7,
            color="blue", edgecolor="black")
    ax2.axvline(mean_test_scores[best_idx], color="red", linestyle="--", linewidth=2,
               label=f"Best score: {mean_test_scores[best_idx]:.4f}")
    ax2.set_xlabel(f"{score_label.upper()}{label_suffix}", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Distribution of CV Scores Across Hyperparameter Trials", fontsize=12)
    ax2.legend(loc="best", fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")

    bottom_margin = _apply_plot_metadata(fig, meta_lines)
    plt.subplots_adjust(left=0.15, right=0.9, top=0.8, bottom=bottom_margin)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.8)
    plt.close()


def _plot_learning_curve(train_sizes: np.ndarray,
                         train_scores: np.ndarray,
                         val_scores: np.ndarray,
                         out_path: str,
                         metric_label: str,
                         metric_is_error: bool,
                         meta_lines: Optional[Sequence[str]] = None) -> None:
    """Generate a learning curve plot with per-split scatter and mean lines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_sizes = np.asarray(train_sizes)
    train_scores = np.asarray(train_scores)
    val_scores = np.asarray(val_scores)

    if train_scores.ndim != 2 or val_scores.ndim != 2:
        return

    n_sizes, n_splits = val_scores.shape
    if n_sizes == 0 or n_splits == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter validation scores across splits
    for split_idx in range(n_splits):
        label = "Validation split" if split_idx == 0 else None
        ax.scatter(train_sizes, val_scores[:, split_idx],
                   color="tab:blue", alpha=0.35, s=25, label=label)

    # Mean lines + shaded variability
    val_mean = val_scores.mean(axis=1)
    val_sd = val_scores.std(axis=1)
    train_mean = train_scores.mean(axis=1)
    train_sd = train_scores.std(axis=1)

    # Plot shaded regions (SD bands)
    ax.fill_between(train_sizes, train_mean - train_sd, train_mean + train_sd,
                    color="gray", alpha=0.15, label="Train ±1 SD")
    ax.fill_between(train_sizes, val_mean - val_sd, val_mean + val_sd,
                    color="tab:blue", alpha=0.15, label="Validation ±1 SD")

    # Plot mean lines
    ax.plot(train_sizes, train_mean, color="gray", linestyle="--", linewidth=1.5, label="Train mean")
    ax.plot(train_sizes, val_mean, "b-", linewidth=2, label="Validation mean")

    metric_text = metric_label.replace("_", " ").upper()
    if metric_is_error:
        metric_text += " (LOWER IS BETTER)"

    ax.set_xlabel("Training samples", fontsize=12)
    ax.set_ylabel(metric_text, fontsize=12)
    ax.set_title("Learning Curve", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Set x-axis ticks and labels to training sizes
    ax.set_xticks(train_sizes)
    ax.set_xticklabels([str(int(size)) for size in train_sizes], rotation=45, ha="right", fontsize=9)

    ax.legend(loc="best", fontsize=10)

    bottom_margin = _apply_plot_metadata(fig, meta_lines)
    plt.subplots_adjust(left=0.15, right=0.9, top=0.8, bottom=bottom_margin)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.8)
    plt.close()


def compute_subgroup_metrics_table(
    df: pd.DataFrame,
    y_true: np.ndarray,
    probs: np.ndarray,
    group_col: str,
    min_n: int = 40,
) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame()
    groups = df[group_col].astype(str).replace("nan", "Missing")
    y = np.asarray(y_true).astype(int)
    p = np.asarray(probs).astype(float)
    rows = []
    for grp in sorted(groups.unique()):
        mask = (groups == grp).to_numpy()
        n = int(mask.sum())
        if n < max(1, int(min_n)):
            continue
        y_g = y[mask]
        p_g = p[mask]
        if y_g.size == 0:
            continue
        metrics = prob_metrics(y_g, p_g)
        cal_a, cal_b = calibration_intercept_slope(y_g, p_g)
        rows.append({
            "group": grp,
            "n": n,
            "n_pos": int(y_g.sum()),
            "prevalence": float(_prevalence(y_g)),
            "AUROC": float(metrics["AUROC"]),
            "PR_AUC": float(metrics["PR_AUC"]),
            "Brier": float(metrics["Brier"]),
            "LogLoss": float(metrics.get("LogLoss", np.nan)),
            "calibration_intercept": float(cal_a) if np.isfinite(cal_a) else np.nan,
            "calibration_slope": float(cal_b) if np.isfinite(cal_b) else np.nan,
        })
    return pd.DataFrame(rows)


def save_subgroup_metrics(
    df: pd.DataFrame,
    y_true: np.ndarray,
    probs: np.ndarray,
    group_col: str,
    out_csv: str,
    min_n: int = 40,
) -> pd.DataFrame:
    sub_df = compute_subgroup_metrics_table(df, y_true, probs, group_col, min_n=min_n)
    if len(sub_df) > 0:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        sub_df.to_csv(out_csv, index=False)
    return sub_df


# save_subgroup_dca removed - subgroup DCA moved to postprocess_compare.py


# ----------------------------
# Threshold helpers (TRAIN OOF -> TEST)
# ----------------------------
# ----------------------------
# Preprocessing / dataset
# ----------------------------
def build_preprocessor(num_cols: Optional[List[str]], cat_cols: List[str], missing_strategy: str = "median") -> ColumnTransformer:
    missing_strategy = (missing_strategy or "median").strip().lower()
    if missing_strategy not in ("median", "mean"):
        missing_strategy = "median"

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy=missing_strategy)),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    num_selector = num_cols if num_cols is not None else make_column_selector(dtype_include=np.number)
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_selector),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )


def build_preprocessor_auto(missing_strategy: str = "median") -> ColumnTransformer:
    missing_strategy = (missing_strategy or "median").strip().lower()
    if missing_strategy not in ("median", "mean"):
        missing_strategy = "median"

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy=missing_strategy)),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, make_column_selector(dtype_include=np.number)),
            ("cat", cat_pipe, make_column_selector(dtype_include=["object", "string", "category"])),
        ],
        remainder="drop",
    )


def _usecols_for_celiac(col: str) -> bool:
    if col in (ID_COL, TARGET_COL, CED_DATE_COL): return True
    if col in META_NUM_COLS: return True
    if col in CAT_COLS: return True
    if isinstance(col, str) and col.endswith("_resid"): return True
    return False


def make_dataset(
    df: pd.DataFrame,
    positives: List[str],
    drop_uncertain_controls: bool = True,
    dropna_meta_num: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, List[str], List[str], Dict[str, Any]]:
    """
    Prepare dataset for modeling.

    Key preprocessing steps:
    1. Filter to specified outcome classes (Controls + positives)
    2. Apply row filters via shared helper (uncertain controls, missing metadata)
    3. Create binary outcome variable (y)
    4. Convert numeric columns to float (coerce errors)
    5. Handle missing categorical values as informative "Missing" category

    Missing Data Strategy:
    - Categorical features (sex, ethnicity): NaN → "Missing" category
      Rationale: Missingness may be informative (MNAR). Model learns if
      "unknown ethnicity" correlates with CeD risk. Preserves subjects.
    - Continuous features (BMI, age): Drop rows with missing values
      Rationale: Small data loss, avoids arbitrary imputation.

    CRITICAL: Row filtering uses apply_row_filters() which is shared with
    save_splits.py. This ensures index alignment between split files and
    the training dataset.

    Args:
        df: Raw dataframe with all columns
        positives: List of positive outcome labels (e.g., ["Incident"])
        drop_uncertain_controls: Drop Controls with CeD_date present
        dropna_meta_num: Drop rows missing age or BMI

    Returns:
        (df_filtered, X, y, num_cols, prot_cols, filter_stats)
    """
    # Step 1: Filter to relevant outcome classes
    keep = [CONTROL_LABEL] + positives
    df2 = df[df[TARGET_COL].isin(keep)].copy()

    # Step 2: Apply row filters (MUST match save_splits.py logic)
    df2, filter_stats = apply_row_filters(
        df2,
        drop_uncertain_controls=drop_uncertain_controls,
        dropna_meta_num=dropna_meta_num,
    )

    # Step 3: Create binary outcome
    df2["y"] = df2[TARGET_COL].isin(positives).astype(int)

    # Step 4: Identify protein columns
    prot_cols = [c for c in df2.columns if isinstance(c, str) and c.endswith("_resid")]
    if not prot_cols:
        raise ValueError("No *_resid columns found. Check your column names.")
    num_cols = META_NUM_COLS + prot_cols

    # Step 5: Convert numeric columns
    for c in num_cols:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")

    # Step 6: Handle missing categorical values as "Missing" category
    # Using astype(str).replace('nan', 'Missing') to handle all NaN representations
    # This is more robust than .fillna() with pandas StringDtype
    for c in CAT_COLS:
        df2[c] = df2[c].astype(str).replace('nan', 'Missing')

    X = df2[num_cols + CAT_COLS]
    y = df2["y"].to_numpy()

    return df2, X, y, num_cols, prot_cols, filter_stats


def temporal_order_indices(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        raise ValueError(f"Temporal column '{col}' not found in dataframe.")
    ser = df[col]
    order_vals = None
    try:
        order_vals = pd.to_datetime(ser, errors="coerce")
    except Exception:
        order_vals = None
    if order_vals is None or order_vals.isna().all():
        order_vals = pd.to_numeric(ser, errors="coerce")
    if isinstance(order_vals, pd.Series) and order_vals.isna().all():
        order_vals = pd.Series(np.arange(len(df)), index=df.index)
    if isinstance(order_vals, pd.Series):
        fill_value = order_vals.min()
        if isinstance(fill_value, pd.Timestamp) or np.issubdtype(order_vals.dtype, np.datetime64):
            fill_value = fill_value - pd.Timedelta(days=1) if pd.notna(fill_value) else pd.Timestamp("1970-01-01")
        elif pd.isna(fill_value):
            fill_value = float("-inf")
        order_vals = order_vals.fillna(fill_value)
    tmp = pd.DataFrame({
        "order_val": order_vals,
        "idx": np.arange(len(df)),
    })
    tmp = tmp.sort_values(["order_val", "idx"], kind="mergesort").reset_index(drop=True)
    return tmp["idx"].to_numpy(dtype=int)


def variance_missingness_prefilter(
    X_tr: pd.DataFrame,
    prot_cols: List[str],
    min_nonmissing: float,
    min_var: float,
    strict: bool,
    out_csv: Optional[str] = None
):
    if len(prot_cols) == 0:
        return prot_cols, {}

    Xm = X_tr[prot_cols]
    nonmiss = Xm.notna().mean(axis=0)
    prot_ok = nonmiss[nonmiss >= min_nonmissing].index.tolist()

    report = {
        "n_proteins_total": int(len(prot_cols)),
        "n_pass_nonmissing": int(len(prot_ok)),
        "min_nonmissing": float(min_nonmissing),
        "min_var": float(min_var),
    }

    if len(prot_ok) == 0:
        msg = f"All {len(prot_cols)} proteins failed non-missing filter (min_nonmissing={min_nonmissing})."
        if strict:
            raise ValueError(msg)
        logprint("âš ï¸  WARNING:", msg, "Disabling prefilter (keeping all proteins).", level="warning")
        report["disabled"] = True
        if out_csv:
            pd.DataFrame({
                "protein": prot_cols,
                "nonmissing_frac": nonmiss.reindex(prot_cols).values,
                "passed_nonmissing": [False] * len(prot_cols),
            }).to_csv(out_csv, index=False)
        return prot_cols, report

    var = pd.Series(np.nanvar(X_tr[prot_ok].to_numpy(dtype=float), axis=0), index=prot_ok)
    prot_keep = var[var >= min_var].index.tolist()
    report["n_pass_variance"] = int(len(prot_keep))

    if len(prot_keep) == 0:
        msg = f"All {len(prot_ok)} proteins failed variance filter (min_var={min_var})."
        if strict:
            raise ValueError(msg)
        logprint("âš ï¸  WARNING:", msg, "Disabling variance step (keeping proteins that passed missingness).", level="warning")
        prot_keep = prot_ok
        report["disabled"] = True

    if out_csv:
        df_rep = pd.DataFrame({
            "protein": prot_cols,
            "nonmissing_frac": nonmiss.reindex(prot_cols).values,
        })
        df_rep["passed_nonmissing"] = df_rep["protein"].isin(prot_ok)
        df_rep["variance_train"] = var.reindex(df_rep["protein"]).values
        df_rep["passed_variance"] = df_rep["protein"].isin(prot_keep)
        df_rep.to_csv(out_csv, index=False)

    return prot_keep, report


def _simple_impute_series(s: pd.Series, strategy: str) -> pd.Series:
    strategy = (strategy or "median").strip().lower()
    if strategy == "mean":
        v = s.mean(skipna=True)
    else:
        v = s.median(skipna=True)
    return s.fillna(v)


def screen_proteins_train_only(
    df_tr: pd.DataFrame,
    y_tr: np.ndarray,
    prot_cols: List[str],
    method: str,
    top_n: int,
    out_csv: Optional[str] = None,
    min_n_per_group: int = 10,
) -> List[str]:
    method = (method or "").strip().lower()
    top_n = int(top_n or 0)
    if top_n <= 0 or len(prot_cols) == 0:
        return prot_cols

    y = np.asarray(y_tr).astype(int)
    if np.unique(y).size < 2:
        return prot_cols

    rows = []
    if method == "mannwhitney":
        for p in prot_cols:
            x = pd.to_numeric(df_tr[p], errors="coerce")
            ok = x.notna().to_numpy()
            if ok.sum() < (2 * min_n_per_group):
                continue
            x_ok = x[ok].to_numpy(dtype=float)
            y_ok = y[ok]
            x0 = x_ok[y_ok == 0]
            x1 = x_ok[y_ok == 1]
            if len(x0) < min_n_per_group or len(x1) < min_n_per_group:
                continue
            try:
                try:
                    _, p_mw = stats.mannwhitneyu(x1, x0, alternative="two-sided", method="asymptotic")
                except TypeError:
                    _, p_mw = stats.mannwhitneyu(x1, x0, alternative="two-sided")
                p_mw = float(p_mw)
            except Exception:
                p_mw = np.nan
            delta = float(np.nanmean(x1) - np.nanmean(x0))
            rows.append((p, p_mw, abs(delta), float(np.mean(ok))))

        if not rows:
            keep = prot_cols
            df_s = pd.DataFrame()
        else:
            df_s = pd.DataFrame(rows, columns=["protein", "p_value", "abs_delta", "nonmissing_frac"])
            df_s = df_s.sort_values(["p_value", "abs_delta"], ascending=[True, False], na_position="last")
            keep = df_s["protein"].head(min(top_n, len(df_s))).tolist()

    elif method == "f_classif":
        Xp = df_tr[prot_cols].apply(pd.to_numeric, errors="coerce")
        med = Xp.median(axis=0, skipna=True)
        Ximp = Xp.fillna(med)
        try:
            F, pvals = f_classif(Ximp.to_numpy(dtype=float), y)
            F = np.asarray(F, dtype=float)
            pvals = np.asarray(pvals, dtype=float)
            ok = np.isfinite(F)
            df_s = pd.DataFrame({
                "protein": np.asarray(prot_cols)[ok],
                "F_score": F[ok],
                "p_value": pvals[ok],
                "nonmissing_frac": Xp.notna().mean(axis=0).to_numpy()[ok],
            })
            df_s = df_s.sort_values(["F_score"], ascending=[False], na_position="last")
            keep = df_s["protein"].head(min(top_n, len(df_s))).tolist()
        except Exception:
            keep = prot_cols
            df_s = pd.DataFrame()
    else:
        logprint(f"âš ï¸  WARNING: unknown screen_method='{method}'. Skipping screening.", level="warning")
        keep = prot_cols
        df_s = pd.DataFrame()

    if out_csv:
        if isinstance(df_s, pd.DataFrame) and len(df_s) > 0:
            df_s.to_csv(out_csv, index=False)
        else:
            pd.DataFrame({
                "note": [f"screen_method={method} produced no rows; screening skipped/fallback used"],
                "top_n": [top_n],
                "n_proteins_in": [len(prot_cols)],
                "n_proteins_out": [len(keep)],
            }).to_csv(out_csv, index=False)

    return keep


class TrainOnlyScreenSelector(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible wrapper around screen_proteins_train_only.
    Ensures screening happens within each CV fold to avoid leakage.
    """

    def __init__(
        self,
        prot_cols: Optional[List[str]] = None,
        method: str = "mannwhitney",
        top_n: int = 0,
        min_n_per_group: int = 10,
        diag_csv: Optional[str] = None,
    ):
        # Keep constructor params untouched for sklearn.clone compatibility.
        self.prot_cols = prot_cols
        self.method = method
        self.top_n = int(top_n or 0)
        self.min_n_per_group = int(min_n_per_group)
        self.diag_csv = diag_csv

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("TrainOnlyScreenSelector expects pandas DataFrame input.")
        prot_cols = list(self.prot_cols) if self.prot_cols is not None else None
        if self.top_n <= 0 or prot_cols is None:
            self.selected_proteins_ = list(prot_cols or [])
            self.keep_cols_ = self._build_keep_cols(X, self.selected_proteins_)
            return self

        available = [c for c in (prot_cols or []) if c in X.columns]
        selected = screen_proteins_train_only(
            df_tr=X,
            y_tr=y,
            prot_cols=available,
            method=self.method,
            top_n=self.top_n,
            out_csv=self.diag_csv,
            min_n_per_group=self.min_n_per_group,
        )
        if len(selected) == 0:
            selected = available
        self.selected_proteins_ = selected
        self.keep_cols_ = self._build_keep_cols(X, selected)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("TrainOnlyScreenSelector expects pandas DataFrame input.")
        keep = [c for c in getattr(self, "keep_cols_", []) if c in X.columns]
        return X.loc[:, keep].copy()

    def _build_keep_cols(self, X: pd.DataFrame, proteins: List[str]) -> List[str]:
        keep = []
        for c in META_NUM_COLS:
            if c in X.columns:
                keep.append(c)
        keep.extend([p for p in proteins if p in X.columns])
        for c in CAT_COLS:
            if c in X.columns and c not in keep:
                keep.append(c)
        return keep

# ----------------------------
# Protein-scope KBest selector (real panel sizes)
# ----------------------------
class ProteinKBestSelector(BaseEstimator, TransformerMixin):
    def __init__(self, prot_cols, k=100, missing_strategy="median"):
        self.prot_cols = prot_cols
        self.k = k
        self.missing_strategy = missing_strategy

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        y = np.asarray(y).astype(int)
        prots = [c for c in (list(self.prot_cols) if self.prot_cols is not None else []) if c in X.columns]
        if len(prots) == 0:
            self.selected_proteins_ = []
            return self

        k = int(self.k)
        k = max(1, min(len(prots), k))

        Xp = X[prots].apply(pd.to_numeric, errors="coerce")
        # impute train-fold only with user-selected strategy
        if (self.missing_strategy or "median").strip().lower() == "mean":
            fill = Xp.mean(axis=0, skipna=True)
        else:
            fill = Xp.median(axis=0, skipna=True)
        Ximp = Xp.fillna(fill)

        try:
            F, _ = f_classif(Ximp.to_numpy(dtype=float), y)
            F = np.asarray(F, dtype=float)
            order = np.argsort(F)[::-1]
            sel = [prots[i] for i in order[:k]]
        except Exception:
            v = np.nanvar(Ximp.to_numpy(dtype=float), axis=0)
            order = np.argsort(v)[::-1]
            sel = [prots[i] for i in order[:k]]

        self.selected_proteins_ = sel
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        keep = []
        for c in META_NUM_COLS:
            if c in X.columns: keep.append(c)
        for c in CAT_COLS:
            if c in X.columns: keep.append(c)
        for c in getattr(self, "selected_proteins_", []):
            if c in X.columns: keep.append(c)
        return X.loc[:, keep].copy()


# ----------------------------
# Feature report (â€œpaper styleâ€) TRAIN-only
# ----------------------------
def _median_iqr(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, "")
    med = float(np.median(x))
    q1 = float(np.percentile(x, 25))
    q3 = float(np.percentile(x, 75))
    return (med, f"[{q1:.3f}, {q3:.3f}]")


def feature_report_train(
    df_tr: pd.DataFrame,
    y_tr: np.ndarray,
    proteins: List[str],
    selection_freq: Optional[Dict[str, float]],
    out_csv: str
):
    rows = []
    y = np.asarray(y_tr).astype(int)

    for p in proteins:
        x = pd.to_numeric(df_tr[p], errors="coerce")
        ok = x.notna().to_numpy()
        if ok.sum() < 50 or len(np.unique(y[ok])) < 2:
            rows.append({
                "protein": p,
                "selection_freq": (selection_freq.get(p, np.nan) if selection_freq else np.nan),
                "n_train_nonmissing": int(ok.sum()),
                "missing_pct_train": float(100.0 * (1.0 - ok.mean())),
            })
            continue

        x_ok = x[ok].to_numpy(dtype=float)
        y_ok = y[ok]
        x0, x1 = x_ok[y_ok == 0], x_ok[y_ok == 1]

        mean0, mean1 = float(np.mean(x0)), float(np.mean(x1))
        sd0, sd1 = float(np.std(x0, ddof=1)), float(np.std(x1, ddof=1))
        delta = mean1 - mean0

        n0, n1 = len(x0), len(x1)
        sp = np.sqrt(((n0 - 1) * sd0**2 + (n1 - 1) * sd1**2) / max(1, (n0 + n1 - 2)))
        d = float(delta / sp) if sp > 0 else np.nan

        try:
            _, p_t = stats.ttest_ind(x1, x0, equal_var=False, nan_policy="omit")
            p_t = float(p_t)
        except Exception:
            p_t = np.nan

        try:
            try:
                _, p_mw = stats.mannwhitneyu(x1, x0, alternative="two-sided", method="asymptotic")
            except TypeError:
                _, p_mw = stats.mannwhitneyu(x1, x0, alternative="two-sided")
            p_mw = float(p_mw)
        except Exception:
            p_mw = np.nan

        try:
            auc = roc_auc_score(y_ok, x_ok)
            if auc < 0.5:
                auc = 1.0 - auc
            auc = float(auc)
        except Exception:
            auc = np.nan

        med0, iqr0 = _median_iqr(x0)
        med1, iqr1 = _median_iqr(x1)

        adj_or, adj_ci, adj_p = np.nan, "", np.nan
        try:
            tmp = df_tr.loc[ok, [p, "age", "BMI", "sex", "Genetic ethnic grouping"]].copy()
            tmp = tmp.rename(columns={p: "prot", "Genetic ethnic grouping": "eth"})
            tmp["y"] = y_ok

            prot = pd.to_numeric(tmp["prot"], errors="coerce").to_numpy(dtype=float)
            tmp["prot"] = (prot - np.nanmean(prot)) / (np.nanstd(prot, ddof=1) + 1e-12)

            m = smf.logit("y ~ prot + age + BMI + C(sex) + C(eth)", data=tmp).fit(disp=0)
            beta = float(m.params["prot"])
            se = float(m.bse["prot"])
            adj_p = float(m.pvalues["prot"])
            adj_or = float(np.exp(beta))
            lo, hi = np.exp(beta - 1.96 * se), np.exp(beta + 1.96 * se)
            adj_ci = f"[{lo:.3f}, {hi:.3f}]"
        except Exception:
            pass

        rows.append({
            "protein": p,
            "selection_freq": (selection_freq.get(p, np.nan) if selection_freq else np.nan),
            "n_train_nonmissing": int(ok.sum()),
            "missing_pct_train": float(100.0 * (1.0 - ok.mean())),
            "mean_control": mean0,
            "mean_case": mean1,
            "delta_mean": delta,
            "cohen_d": d,
            "median_control": med0,
            "IQR_control": iqr0,
            "median_case": med1,
            "IQR_case": iqr1,
            "p_welch_t": p_t,
            "p_mannwhitney": p_mw,
            "single_feature_AUROC": auc,
            "adj_OR_perSD": adj_or,
            "adj_OR_CI95": adj_ci,
            "adj_p": adj_p,
        })

    df_out = pd.DataFrame(rows)

    pvals = df_out.get("adj_p", pd.Series(dtype=float)).to_numpy()
    okp = np.isfinite(pvals)
    q = np.full_like(pvals, np.nan, dtype=float)
    if okp.sum() > 0:
        q[okp] = multipletests(pvals[okp], method="fdr_bh")[1]
    df_out["q_fdr_adj"] = q

    pvals = df_out.get("p_mannwhitney", pd.Series(dtype=float)).to_numpy()
    okp = np.isfinite(pvals)
    q = np.full_like(pvals, np.nan, dtype=float)
    if okp.sum() > 0:
        q[okp] = multipletests(pvals[okp], method="fdr_bh")[1]
    df_out["q_fdr_mannwhitney"] = q

    df_out["abs_cohen_d"] = df_out.get("cohen_d", pd.Series(dtype=float)).abs()
    df_out = df_out.sort_values(
        ["selection_freq", "q_fdr_adj", "abs_cohen_d"],
        ascending=[False, True, False],
        na_position="last",
    )
    df_out.to_csv(out_csv, index=False)


def rank_proteins_univariate(df_tr: pd.DataFrame, y_tr: np.ndarray, prot_cols: List[str], top_n: int = 200) -> List[str]:
    y = np.asarray(y_tr).astype(int)
    rows = []
    for p in prot_cols:
        x = pd.to_numeric(df_tr[p], errors="coerce")
        ok = x.notna().to_numpy()
        if ok.sum() < 50 or len(np.unique(y[ok])) < 2:
            continue
        x_ok = x[ok].to_numpy(dtype=float)
        y_ok = y[ok]
        x0, x1 = x_ok[y_ok == 0], x_ok[y_ok == 1]
        if len(x0) < 5 or len(x1) < 5:
            continue
        try:
            try:
                _, p_mw = stats.mannwhitneyu(x1, x0, alternative="two-sided", method="asymptotic")
            except TypeError:
                _, p_mw = stats.mannwhitneyu(x1, x0, alternative="two-sided")
            p_mw = float(p_mw)
        except Exception:
            p_mw = np.nan
        delta = float(np.nanmean(x1) - np.nanmean(x0))
        rows.append((p, p_mw, abs(delta)))

    if not rows:
        return []
    df_r = pd.DataFrame(rows, columns=["protein", "p_mw", "abs_delta"])
    df_r = df_r.sort_values(["p_mw", "abs_delta"], ascending=[True, False], na_position="last")
    return df_r["protein"].head(int(top_n)).tolist()


def univariate_strength_map(df_tr: pd.DataFrame, y_tr: np.ndarray, proteins: List[str]) -> Dict[str, Tuple[float, float]]:
    """
    Returns {protein: (p_mw, abs_delta)} for use in tie-breaks.
    Smaller p_mw is stronger.
    """
    y = np.asarray(y_tr).astype(int)
    out: Dict[str, Tuple[float, float]] = {}
    for p in proteins:
        if p not in df_tr.columns:
            continue
        x = pd.to_numeric(df_tr[p], errors="coerce")
        ok = x.notna().to_numpy()
        if ok.sum() < 30 or len(np.unique(y[ok])) < 2:
            continue
        x_ok = x[ok].to_numpy(dtype=float)
        y_ok = y[ok]
        x0, x1 = x_ok[y_ok == 0], x_ok[y_ok == 1]
        if len(x0) < 5 or len(x1) < 5:
            continue
        try:
            try:
                _, p_mw = stats.mannwhitneyu(x1, x0, alternative="two-sided", method="asymptotic")
            except TypeError:
                _, p_mw = stats.mannwhitneyu(x1, x0, alternative="two-sided")
            p_mw = float(p_mw)
        except Exception:
            p_mw = np.nan
        delta = float(np.nanmean(x1) - np.nanmean(x0))
        out[p] = (p_mw, abs(delta))
    return out


# ----------------------------
# Models
# ----------------------------
def build_models(cpus: int, random_state: int, args, model_name: Optional[str] = None) -> Dict[str, object]:
    models: Dict[str, object] = {}
    want = {model_name} if model_name else None

    def want_model(name: str) -> bool:
        return want is None or name in want

    # LR: avoid estimator-level parallelism; parallelize tuning with --tune_n_jobs
    if want_model("LR_EN") or want_model("LR_L1"):
        lr_common = dict(
            solver="saga",
            C=1.0,
            max_iter=int(args.lr_max_iter),
            tol=float(args.lr_tol),
            random_state=int(random_state),
        )

        # scikit-learn >=1.8 deprecates `penalty`; use `l1_ratio` (0=l2, 1=l1, in-between=elastic-net).
        if SKLEARN_VER >= (1, 8, 0):
            if want_model("LR_EN"):
                models["LR_EN"] = LogisticRegression(l1_ratio=0.5, **lr_common)
            if want_model("LR_L1"):
                models["LR_L1"] = LogisticRegression(l1_ratio=1.0, **lr_common)
        else:
            if want_model("LR_EN"):
                models["LR_EN"] = LogisticRegression(penalty="elasticnet", l1_ratio=0.5, **lr_common)
            if want_model("LR_L1"):
                models["LR_L1"] = LogisticRegression(penalty="l1", **lr_common)

    # Linear SVM + calibration (already probability calibrated)
    if want_model("LinSVM_cal"):
        base_svm = LinearSVC(
            C=1.0,
            class_weight=None,
            random_state=int(random_state),
            max_iter=int(args.svm_max_iter)
        )
        models["LinSVM_cal"] = CalibratedClassifierCV(
            base_svm,
            method=str(args.calibration_method),
            cv=int(args.calibration_cv)
        )

    # RF: estimator-level parallelism is OK (single level)
    if want_model("RF"):
        rf_n_estimators_grid = _require_int_list(args.rf_n_estimators_grid, "rf_n_estimators_grid")
        rf_kwargs = dict(
            n_estimators=int(rf_n_estimators_grid[0]),
            random_state=int(random_state),
            n_jobs=int(max(1, cpus)),
            bootstrap=bool(int(args.rf_bootstrap))
        )
        if args.rf_max_samples is not None and str(args.rf_max_samples).strip() != "":
            # int or float
            try:
                v = float(args.rf_max_samples)
                if v.is_integer():
                    rf_kwargs["max_samples"] = int(v)
                else:
                    rf_kwargs["max_samples"] = float(v)
            except Exception:
                pass

        models["RF"] = RandomForestClassifier(**rf_kwargs)

    # XGBoost: use tree_method from args (default 'hist', 'gpu_hist' for GPU acceleration)
    if want_model("XGBoost") and XGBOOST_AVAILABLE:
        try:
            default_spw = float(args.xgb_scale_pos_weight)
        except Exception:
            default_spw = 1.0
        tree_method = getattr(args, 'xgb_tree_method', 'hist')
        xgb_kwargs = dict(
            n_estimators=int(args.xgb_n_estimators),
            max_depth=int(args.xgb_max_depth),
            learning_rate=float(args.xgb_learning_rate),
            subsample=float(args.xgb_subsample),
            colsample_bytree=float(args.xgb_colsample_bytree),
            scale_pos_weight=default_spw,
            reg_alpha=float(args.xgb_reg_alpha),
            reg_lambda=float(args.xgb_reg_lambda),
            min_child_weight=int(args.xgb_min_child_weight),
            gamma=float(args.xgb_gamma),
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method=tree_method,
            random_state=int(random_state),
            n_jobs=int(max(1, cpus)) if tree_method != "gpu_hist" else 1
        )
        models["XGBoost"] = XGBClassifier(**xgb_kwargs)

    return models


def _calibrated_estimator_param_name() -> str:
    tmp = CalibratedClassifierCV(LinearSVC())
    params = tmp.get_params().keys()
    if "estimator" in params: return "estimator"
    if "base_estimator" in params: return "base_estimator"
    return "estimator"


def _calibrated_cv_param_name() -> str:
    tmp = CalibratedClassifierCV(LinearSVC())
    params = tmp.get_params().keys()
    if "cv" in params:
        return "cv"
    return "cv"


def _maybe_calibrate_final(est, model_name: str, calibrate: bool, method: str, cv: int, random_state: int):
    """
    Optional extra calibration wrapper for LR/RF (SVM is already calibrated).
    This is applied consistently in CV and final training when enabled.
    """
    if not calibrate:
        return est
    if model_name == "LinSVM_cal":
        return est
    # if already CalibratedClassifierCV, don't double-calibrate
    if isinstance(est, CalibratedClassifierCV):
        return est

    try:
        # scikit-learn API differences
        kwargs = {"method": str(method), "cv": int(cv)}
        # newer versions accept estimator=, older accept base_estimator=
        tmp = CalibratedClassifierCV(LinearSVC())
        if "estimator" in tmp.get_params():
            kwargs = {"estimator": est, **kwargs}
        else:
            kwargs = {"base_estimator": est, **kwargs}
        return CalibratedClassifierCV(**kwargs)
    except Exception:
        return est


# ----------------------------
# Hyperparameter distributions
# ----------------------------
def get_param_distributions(
    model_name: str,
    feature_select: str,
    k_grid: List[int],
    kbest_scope: str,
    args,
    xgb_scale_pos_weight: Optional[float] = None,
    grid_rng: Optional[np.random.RandomState] = None,
    randomize_grids: bool = False,
):
    d = {}
    rng = grid_rng if randomize_grids else None

    if feature_select in ("kbest", "hybrid"):
        if not k_grid:
            raise ValueError("feature_select in {kbest,hybrid} requires --k_grid (e.g. 25,50,100,200)")
        if kbest_scope == "protein":
            d["prot_sel__k"] = k_grid
        else:
            d["sel__k"] = k_grid

    # grids
    lr_Cs = make_logspace(args.lr_C_min, args.lr_C_max, args.lr_C_points, rng=rng)
    svm_Cs = make_logspace(args.svm_C_min, args.svm_C_max, args.svm_C_points, rng=rng)

    lr_class_weights = parse_class_weight_options(args.lr_class_weight_options)
    svm_class_weights = parse_class_weight_options(args.svm_class_weight_options)
    rf_class_weights = parse_class_weight_options(args.rf_class_weight_options)
    # Keep class-weight lists deterministic for reproducibility even when numeric grids are randomized.

    if model_name == "LR_L1":
        d.update({"clf__C": lr_Cs, "clf__class_weight": lr_class_weights})
        return d

    if model_name == "LR_EN":
        l1_grid = _parse_float_list(args.lr_l1_ratio_grid) or [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95]
        if rng is not None:
            l1_grid = _randomize_numeric_list(
                l1_grid,
                rng,
                min_float=0.0,
                max_float=1.0,
                perturb_mode=True,
            )
        d.update({
            "clf__C": lr_Cs,
            "clf__l1_ratio": l1_grid,
            "clf__class_weight": lr_class_weights,
        })
        return d

    if model_name == "LinSVM_cal":
        est_key = _calibrated_estimator_param_name()
        d.update({
            f"clf__{est_key}__C": svm_Cs,
            f"clf__{est_key}__class_weight": svm_class_weights,
        })
        return d

    if model_name == "RF":
        n_estimators_grid = _require_int_list(args.rf_n_estimators_grid, "rf_n_estimators_grid")
        if rng is not None:
            n_estimators_grid = _randomize_numeric_list(n_estimators_grid, rng, as_int=True, min_int=1, unique_int=True, perturb_mode=True)

        max_depth_raw = _parse_none_int_float_list(args.rf_max_depth_grid) or [None, 10, 20, 40]
        # Robustness: sklearn requires max_depth to be int or None (floats like 10.0 will fail)
        max_depth = _coerce_int_or_none_list(max_depth_raw, name="rf_max_depth_grid")
        if rng is not None:
            max_depth = _randomize_numeric_list(max_depth, rng, as_int=True, min_int=1, unique_int=True, perturb_mode=True)

        min_leaf_raw = _parse_none_int_float_list(args.rf_min_samples_leaf_grid) or [5, 10, 20]
        # Robustness: sklearn allows int>=1 or float in (0,1); coerce whole-number floats to int
        min_leaf = _coerce_min_samples_leaf_list(min_leaf_raw, name="rf_min_samples_leaf_grid")
        max_feat = _parse_none_int_float_list(args.rf_max_features_grid) or ["sqrt", 0.3]
        if rng is not None:
            # Randomize numeric values in max_features (preserving strings like "sqrt")
            max_feat = _randomize_numeric_list(max_feat, rng, min_float=0.1, max_float=1.0, perturb_mode=True)

        min_split = [2, 5, 10]
        if rng is not None:
            min_split = _randomize_numeric_list(min_split, rng, as_int=True, min_int=2, unique_int=True, perturb_mode=True)

        # Handle max_samples: prefer grid, fall back to single value
        max_samples_grid = None
        if hasattr(args, 'rf_max_samples_grid') and args.rf_max_samples_grid and str(args.rf_max_samples_grid).strip() != "":
            max_samples_raw = _parse_none_int_float_list(args.rf_max_samples_grid)
            if max_samples_raw:
                max_samples_grid = _coerce_min_samples_leaf_list(max_samples_raw, name="rf_max_samples_grid")
                if rng is not None:
                    max_samples_grid = _randomize_numeric_list(max_samples_grid, rng, min_float=0.1, max_float=1.0, perturb_mode=True)
        elif args.rf_max_samples is not None and str(args.rf_max_samples).strip() != "":
            # Fall back to single value
            try:
                v = float(args.rf_max_samples)
                max_samples_grid = [int(v)] if v.is_integer() else [float(v)]
            except Exception:
                pass

        d.update({
            "clf__n_estimators": n_estimators_grid,
            "clf__max_depth": max_depth,
            "clf__min_samples_leaf": min_leaf,
            "clf__min_samples_split": min_split,
            "clf__max_features": max_feat,
            "clf__bootstrap": [bool(int(args.rf_bootstrap))],
            "clf__class_weight": rf_class_weights,
        })

        if max_samples_grid:
            d["clf__max_samples"] = max_samples_grid

        return d

    if model_name == "XGBoost":
        n_estimators_grid = _parse_int_list(args.xgb_n_estimators_grid) or [500, 1000, 2000]
        max_depth_grid = _parse_int_list(args.xgb_max_depth_grid) or [3, 5, 7]
        learning_rate_grid = _parse_float_list(args.xgb_learning_rate_grid) or [0.01, 0.05, 0.1]
        subsample_grid = _parse_float_list(args.xgb_subsample_grid) or [0.7, 0.8, 1.0]
        colsample_grid = _parse_float_list(args.xgb_colsample_bytree_grid) or [0.7, 0.8, 1.0]
        spw_grid = _parse_float_list(args.xgb_scale_pos_weight_grid)
        if not spw_grid:
            if xgb_scale_pos_weight is not None:
                spw_grid = [float(xgb_scale_pos_weight)]
            else:
                try:
                    spw_grid = [float(args.xgb_scale_pos_weight)]
                except Exception:
                    spw_grid = [1.0]

        if rng is not None:
            n_estimators_grid = _randomize_numeric_list(n_estimators_grid, rng, as_int=True, min_int=1, perturb_mode=True)
            max_depth_grid = _randomize_numeric_list(max_depth_grid, rng, as_int=True, min_int=1, unique_int=True, perturb_mode=True)
            learning_rate_grid = _randomize_numeric_list(learning_rate_grid, rng, min_float=1e-4, log_scale=True, perturb_mode=True)
            subsample_grid = _randomize_numeric_list(subsample_grid, rng, min_float=0.1, max_float=1.0, perturb_mode=True)
            colsample_grid = _randomize_numeric_list(colsample_grid, rng, min_float=0.1, max_float=1.0, perturb_mode=True)
            spw_grid = _randomize_numeric_list(spw_grid, rng, min_float=1e-3, perturb_mode=True)

        d.update({
            "clf__n_estimators": n_estimators_grid,
            "clf__max_depth": max_depth_grid,
            "clf__learning_rate": learning_rate_grid,
            "clf__subsample": subsample_grid,
            "clf__colsample_bytree": colsample_grid,
            "clf__scale_pos_weight": spw_grid,
        })
        return d

    return {}


def build_search(
    pipe: Pipeline,
    model_name: str,
    scoring: str,
    inner_folds: int,
    n_iter: int,
    random_state: int,
    feature_select: str,
    k_grid: List[int],
    kbest_scope: str,
    cpus: int,
    args,
    xgb_scale_pos_weight: Optional[float] = None,
    grid_rng: Optional[np.random.RandomState] = None,
    randomize_grids: bool = False,
):
    dists = get_param_distributions(
        model_name,
        feature_select=feature_select,
        k_grid=k_grid,
        kbest_scope=kbest_scope,
        args=args,
        xgb_scale_pos_weight=xgb_scale_pos_weight,
        grid_rng=grid_rng,
        randomize_grids=randomize_grids,
    )
    if not dists:
        return None

    inner_folds = int(inner_folds)
    if inner_folds < 2:
        logprint(f"[tune] WARNING: inner_folds={inner_folds} < 2; skipping hyperparameter search.", level="warning")
        return None

    n_iter = int(n_iter)
    if n_iter < 1:
        logprint(f"[tune] WARNING: n_iter={n_iter} < 1; skipping hyperparameter search.", level="warning")
        return None

    inner_cv = StratifiedKFold(n_splits=int(inner_folds), shuffle=True, random_state=int(random_state))

    # Decide n_jobs for the search
    tune_n_jobs = parse_tune_n_jobs(args.tune_n_jobs, cpus=cpus)
    if tune_n_jobs is None:
        # auto: parallelize search for LR/SVM; for RF use estimator parallelism and keep search single-threaded
        if model_name in ("LR_EN", "LR_L1", "LinSVM_cal"):
            n_jobs = int(max(1, cpus))
        else:
            n_jobs = 1
    else:
        n_jobs = int(max(1, tune_n_jobs))

    n_jobs = int(max(1, min(int(max(1, cpus)), n_jobs)))
    err_score = parse_error_score(args.error_score)

    return RandomizedSearchCV(
        estimator=pipe,
        param_distributions=dists,
        n_iter=int(n_iter),
        scoring=scoring,
        cv=inner_cv,
        n_jobs=n_jobs,
        pre_dispatch=n_jobs,
        refit=True,
        random_state=int(random_state),
        error_score=err_score,
        verbose=0,
    )


def _get_feature_names(pre) -> np.ndarray:
    if hasattr(pre, "get_feature_names_out"):
        return pre.get_feature_names_out()
    raise RuntimeError("sklearn too old: ColumnTransformer.get_feature_names_out not available.")


def _build_pipeline(
    pre,
    clf,
    feature_select: str,
    prot_cols: List[str],
    kbest_scope: str,
    missing_strategy: str,
    screen_kwargs: Optional[Dict[str, Any]] = None,
    k_default: int = 200
) -> Pipeline:
    steps = []
    if feature_select == "hybrid" and screen_kwargs is not None:
        top_n = int(screen_kwargs.get("top_n", 0) or 0)
        if top_n > 0:
            steps.append((
                "screen",
                TrainOnlyScreenSelector(
                    prot_cols=tuple(prot_cols),
                    method=screen_kwargs.get("method", "mannwhitney"),
                    top_n=top_n,
                    min_n_per_group=int(screen_kwargs.get("min_n_per_group", 10)),
                    diag_csv=screen_kwargs.get("diag_csv"),
                )
            ))

    if feature_select in ("kbest", "hybrid") and kbest_scope == "protein":
        steps.append(("prot_sel", ProteinKBestSelector(prot_cols=tuple(prot_cols), k=k_default, missing_strategy=missing_strategy)))
        steps.append(("pre", pre))
    else:
        steps.append(("pre", pre))
        if feature_select in ("kbest", "hybrid"):
            steps.append(("sel", SelectKBest(score_func=f_classif, k=k_default)))
    steps.append(("clf", clf))
    return Pipeline(steps)


def _maybe_set_screen_diag(est, diag_csv: Optional[str]):
    if not diag_csv:
        return est
    if isinstance(est, Pipeline) and "screen" in est.named_steps:
        est.named_steps["screen"].diag_csv = diag_csv
    return est


# ----------------------------
# Selection extraction
# ----------------------------
def _selected_proteins_from_kbest_transformed(fitted_pipe: Pipeline, prot_cols: List[str]) -> set:
    if "sel" not in fitted_pipe.named_steps:
        return set()
    pre = fitted_pipe.named_steps["pre"]
    fn = _get_feature_names(pre)
    support = fitted_pipe.named_steps["sel"].get_support()
    fn_sel = fn[support]
    prot_set = set()
    for name in fn_sel:
        if name.startswith("num__"):
            orig = name[len("num__"):]
            if orig in prot_cols:
                prot_set.add(orig)
    return prot_set


def _selected_proteins_from_kbest_protein_scope(fitted_pipe: Pipeline) -> set:
    if "prot_sel" not in fitted_pipe.named_steps:
        return set()
    sel = getattr(fitted_pipe.named_steps["prot_sel"], "selected_proteins_", [])
    return set(sel)


def _selected_proteins_from_fitted_pipe(fitted_pipe: Pipeline, prot_cols: List[str], coef_thresh: float) -> set:
    pre = fitted_pipe.named_steps["pre"]
    fn = _get_feature_names(pre)
    if "sel" in fitted_pipe.named_steps:
        support = fitted_pipe.named_steps["sel"].get_support()
        fn = fn[support]
    clf = fitted_pipe.named_steps["clf"]
    coefs = clf.coef_.ravel()
    if len(fn) != len(coefs):
        raise RuntimeError(f"Feature-name length ({len(fn)}) != coef length ({len(coefs)}).")
    prot_set = set()
    for name, c in zip(fn, coefs):
        if name.startswith("num__"):
            orig = name[len("num__"):]
            if (orig in prot_cols) and (abs(c) > coef_thresh):
                prot_set.add(orig)
    return prot_set


def _selected_proteins_from_calibrated_linearsvc(fitted_pipe: Pipeline, prot_cols: List[str], coef_thresh: float) -> set:
    pre = fitted_pipe.named_steps["pre"]
    fn = _get_feature_names(pre)
    if "sel" in fitted_pipe.named_steps:
        support = fitted_pipe.named_steps["sel"].get_support()
        fn = fn[support]
    clf = fitted_pipe.named_steps["clf"]
    if not hasattr(clf, "calibrated_classifiers_"):
        return set()

    coefs_list = []
    try:
        for cc in clf.calibrated_classifiers_:
            est = getattr(cc, "estimator", None)
            if est is None:
                continue
            if hasattr(est, "coef_"):
                coefs_list.append(est.coef_.ravel().astype(float))
    except Exception:
        coefs_list = []

    if not coefs_list:
        return set()

    C = np.vstack(coefs_list)
    avg_abs = np.mean(np.abs(C), axis=0)

    if len(fn) != len(avg_abs):
        return set()

    prot_set = set()
    for name, a in zip(fn, avg_abs):
        if name.startswith("num__"):
            orig = name[len("num__"):]
            if (orig in prot_cols) and (a > coef_thresh):
                prot_set.add(orig)
    return prot_set


def _selected_proteins_from_rf_permutation(
    fitted_pipe: Pipeline,
    X_train_fold: pd.DataFrame,
    y_train_fold: np.ndarray,
    prot_cols: List[str],
    scoring: str,
    top_n: int,
    n_repeats: int,
    seed: int,
    min_importance: float = 0.0
) -> set:
    pre = fitted_pipe.named_steps["pre"]
    fn = _get_feature_names(pre)
    if "sel" in fitted_pipe.named_steps:
        support = fitted_pipe.named_steps["sel"].get_support()
        fn = fn[support]

    try:
        pi = permutation_importance(
            fitted_pipe,
            X_train_fold,
            y_train_fold,
            scoring=scoring,
            n_repeats=int(n_repeats),
            random_state=int(seed),
            n_jobs=1
        )
        imp = np.asarray(pi.importances_mean, dtype=float)
    except Exception:
        return set()

    if imp.size != len(fn):
        return set()

    prot_imp: Dict[str, float] = {}
    for name, val in zip(fn, imp):
        if not np.isfinite(val):
            continue
        if name.startswith("num__"):
            orig = name[len("num__"):]
            if orig in prot_cols:
                prot_imp[orig] = prot_imp.get(orig, 0.0) + float(val)

    if not prot_imp:
        return set()

    items = [(p, v) for p, v in prot_imp.items() if v >= float(min_importance)]
    if not items:
        items = list(prot_imp.items())

    items.sort(key=lambda x: x[1], reverse=True)
    keep = [p for p, _ in items[:max(1, int(top_n))]]
    return set(keep)


def _selected_proteins_from_screen_step(fitted_pipe: Pipeline) -> set:
    if "screen" not in fitted_pipe.named_steps:
        return set()
    selected = getattr(fitted_pipe.named_steps["screen"], "selected_proteins_", [])
    return set(selected) if selected is not None else set()


def extract_selected_proteins(
    fitted_pipe: Pipeline,
    model_name: str,
    feature_select: str,
    prot_cols: List[str],
    kbest_scope: str,
    X_train_fold: Optional[pd.DataFrame] = None,
    y_train_fold: Optional[np.ndarray] = None,
    coef_thresh: float = 1e-12,
    scoring: str = "average_precision",
    perm_top_n: int = 200,
    perm_repeats: int = 3,
    perm_min_importance: float = 0.0,
    seed: int = 0
) -> Tuple[set, set]:
    kbest_set = set()
    model_set = set()
    screen_set = _selected_proteins_from_screen_step(fitted_pipe)

    if feature_select in ("kbest", "hybrid"):
        if kbest_scope == "protein":
            kbest_set = _selected_proteins_from_kbest_protein_scope(fitted_pipe)
        else:
            kbest_set = _selected_proteins_from_kbest_transformed(fitted_pipe, prot_cols=prot_cols)

    if feature_select == "kbest":
        model_set = set(kbest_set)

    elif feature_select == "l1_stability":
        if model_name in ("LR_L1", "LR_EN"):
            model_set = _selected_proteins_from_fitted_pipe(fitted_pipe, prot_cols=prot_cols, coef_thresh=coef_thresh)

    elif feature_select == "hybrid":
        if model_name in ("LR_L1", "LR_EN"):
            model_set = _selected_proteins_from_fitted_pipe(fitted_pipe, prot_cols=prot_cols, coef_thresh=coef_thresh)
        elif model_name == "LinSVM_cal":
            model_set = _selected_proteins_from_calibrated_linearsvc(fitted_pipe, prot_cols=prot_cols, coef_thresh=coef_thresh)
        elif model_name == "RF" and X_train_fold is not None and y_train_fold is not None:
            model_set = _selected_proteins_from_rf_permutation(
                fitted_pipe,
                X_train_fold=X_train_fold,
                y_train_fold=y_train_fold,
                prot_cols=prot_cols,
                scoring=scoring,
                top_n=perm_top_n,
                n_repeats=perm_repeats,
                seed=seed,
                min_importance=perm_min_importance
            )

    return kbest_set, model_set, screen_set


# ----------------------------
# CV OOF
# ----------------------------
def oof_by_repeat_tuned(
    pipe: Pipeline,
    model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    prot_cols: List[str],
    n_splits: int,
    n_repeats: int,
    random_state: int,
    scoring: str,
    inner_folds: int,
    n_iter: int,
    feature_select: str,
    k_grid: List[int],
    kbest_scope: str,
    cpus: int,
    coef_thresh: float,
    perm_top_n: int,
    perm_repeats: int,
    perm_min_importance: float,
    args,
    grid_rng: Optional[np.random.RandomState] = None,
):
    n_splits = int(n_splits)
    n_repeats = int(n_repeats)
    if n_repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {n_repeats}.")

    n = len(y)
    preds = np.full((int(n_repeats), n), np.nan, dtype=float)
    best_rows, sel_rows = [], []
    total_outer = int(n_repeats) if n_splits < 2 else int(n_splits) * int(n_repeats)

    split_idx = 0
    t0 = time.perf_counter()

    if n_splits < 2:
        logprint(f"[cv] WARNING: folds={n_splits} < 2; skipping outer CV and using in-sample predictions.", level="warning")
        all_idx = np.arange(n, dtype=int)
        split_iter = ((all_idx, all_idx) for _ in range(int(n_repeats)))
        split_div = 1
    else:
        rskf = RepeatedStratifiedKFold(n_splits=int(n_splits), n_repeats=int(n_repeats), random_state=int(random_state))
        split_iter = rskf.split(X, y)
        split_div = int(n_splits)

    for train_idx, test_idx in split_iter:
        rep = split_idx // split_div
        base = clone(pipe)

        logprint(f"[{time.strftime('%F %T')}] {model_name} outer {split_idx+1}/{total_outer} (rep={rep}) start", flush=True)

        xgb_spw = None
        if model_name == "XGBoost":
            xgb_spw = resolve_xgb_scale_pos_weight(y[train_idx], args)
            try:
                base.set_params(clf__scale_pos_weight=float(xgb_spw))
            except Exception:
                pass

        search = build_search(
            base, model_name,
            scoring=scoring, inner_folds=inner_folds,
            n_iter=n_iter, random_state=random_state,
            feature_select=feature_select, k_grid=k_grid,
            kbest_scope=kbest_scope,
            cpus=cpus,
            args=args,
            xgb_scale_pos_weight=xgb_spw,
            grid_rng=grid_rng,
            randomize_grids=bool(getattr(args, "grid_randomize", False)),
        )

        if search is None:
            base.fit(X.iloc[train_idx], y[train_idx])
            m = base
            best_params, best_score = {}, np.nan
        else:
            if getattr(search, 'n_jobs', 1) and int(search.n_jobs) > 1:
                with parallel_backend('loky', inner_max_num_threads=1):
                    search.fit(X.iloc[train_idx], y[train_idx])
            else:
                search.fit(X.iloc[train_idx], y[train_idx])
            m = search.best_estimator_
            best_params = search.best_params_
            best_score = float(search.best_score_)

        # Optional calibration for LR/RF (SVM already calibrated)
        m = _maybe_calibrate_final(
            est=m,
            model_name=model_name,
            calibrate=bool(int(args.calibrate_final_models)),
            method=str(args.calibration_method),
            cv=int(args.calibration_cv),
            random_state=int(random_state)
        )
        if isinstance(m, CalibratedClassifierCV) and not hasattr(m, "classes_"):
            # fit calibration on train-fold only
            m.fit(X.iloc[train_idx], y[train_idx])

        p = m.predict_proba(X.iloc[test_idx])[:, 1]
        p = np.clip(p, 0.0, 1.0)
        preds[rep, test_idx] = p

        best_rows.append({
            "model": model_name,
            "repeat": rep,
            "outer_split": int(split_idx),
            "best_score_inner": best_score,
            "best_params": json.dumps(best_params, sort_keys=True),
        })

        X_tr_fold = X.iloc[train_idx]
        y_tr_fold = y[train_idx]

        kbest_set, model_set, screen_set = extract_selected_proteins(
            m if not isinstance(m, CalibratedClassifierCV) else m.estimator if hasattr(m, "estimator") else m,
            model_name=model_name,
            feature_select=feature_select,
            prot_cols=prot_cols,
            kbest_scope=kbest_scope,
            X_train_fold=X_tr_fold,
            y_train_fold=y_tr_fold,
            coef_thresh=coef_thresh,
            scoring=scoring,
            perm_top_n=perm_top_n,
            perm_repeats=perm_repeats,
            perm_min_importance=perm_min_importance,
            seed=random_state
        )

        screen_export_set = screen_set if len(screen_set) > 0 else kbest_set
        if (len(model_set) > 0) or (len(screen_export_set) > 0):
            row = {
                "model": model_name,
                "repeat": rep,
                "outer_split": int(split_idx),
                "n_selected_proteins_split": int(len(model_set) if len(model_set) > 0 else len(screen_export_set)),
                "selected_proteins_split": json.dumps(sorted(model_set if len(model_set) > 0 else screen_export_set)),
                "feature_select": feature_select,
                "kbest_scope": kbest_scope,
                "n_screen_selected_proteins_split": int(len(screen_export_set)),
                "screen_selected_proteins_split": json.dumps(sorted(screen_export_set)) if len(screen_export_set) > 0 else "[]",
            }
            sel_rows.append(row)

        split_idx += 1

    total_sec = time.perf_counter() - t0

    for r in range(int(n_repeats)):
        if np.isnan(preds[r]).any():
            raise RuntimeError(f"Repeat {r} has missing OOF predictions (check CV splitting).")

    return preds, total_sec, pd.DataFrame(best_rows), pd.DataFrame(sel_rows)


def save_calibration_csv(y_true, y_pred, out_csv, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=int(n_bins), strategy="uniform")
    pd.DataFrame({"prob_pred": prob_pred, "prob_true": prob_true}).to_csv(out_csv, index=False)


def save_learning_curve_csv(pipe, X, y, out_csv, scoring, cv=5, lc_min_frac=0.3, lc_points=5, seed=0,
                            out_plot: Optional[str] = None,
                            meta_lines: Optional[Sequence[str]] = None):
    train_sizes = np.linspace(float(lc_min_frac), 1.0, int(lc_points))
    sizes, train_scores, val_scores = learning_curve(
        pipe, X, y,
        train_sizes=train_sizes,
        cv=StratifiedKFold(n_splits=int(cv), shuffle=True, random_state=int(seed)),
        scoring=scoring,
        n_jobs=1,
        shuffle=True,
        random_state=int(seed),
    )

    metric_label = scoring
    metric_is_error = False
    if str(scoring).startswith("neg_"):
        metric_label = str(scoring).replace("neg_", "", 1)
        metric_is_error = True
        train_scores = -train_scores
        val_scores = -val_scores

    if (train_scores < 0).any() or (val_scores < 0).any():
        logprint(f"[learning_curve] WARNING: negative values remain after metric normalization ({scoring}).",
                 level="warning")

    train_mean = train_scores.mean(axis=1)
    train_sd = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_sd = val_scores.std(axis=1)
    n_sizes, n_splits = val_scores.shape
    rows = []
    for i in range(n_sizes):
        for split_idx in range(n_splits):
            rows.append({
                "train_size": int(sizes[i]),
                "cv_split": int(split_idx),
                "train_score": float(train_scores[i, split_idx]),
                "val_score": float(val_scores[i, split_idx]),
                "train_score_mean": float(train_mean[i]),
                "train_score_sd": float(train_sd[i]),
                "val_score_mean": float(val_mean[i]),
                "val_score_sd": float(val_sd[i]),
                "scoring": str(scoring),
                "error_metric": str(metric_label),
                "metric_direction": "lower_is_better" if metric_is_error else "higher_is_better",
            })

    pd.DataFrame(rows).to_csv(out_csv, index=False)

    if out_plot:
        try:
            _plot_learning_curve(sizes, train_scores, val_scores, out_plot, metric_label, metric_is_error,
                                 meta_lines=meta_lines)
            logprint(f"[plot] Saved learning curve plot: {out_plot}")
        except Exception as e:
            logprint(f"[plot] WARNING: Failed to generate learning curve plot: {e}", level="warning")


# ----------------------------
# Panel + correlation helpers
# ----------------------------
def save_high_corr_pairs(df_tr: pd.DataFrame, proteins: List[str], out_csv: str, corr_thresh: float = 0.80, corr_method: str = "pearson") -> str:
    prots = [p for p in proteins if p in df_tr.columns]
    if len(prots) < 2:
        pd.DataFrame(columns=["protein1", "protein2", "abs_corr"]).to_csv(out_csv, index=False)
        return out_csv

    X = df_tr[prots].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        X = X.fillna(X.median(axis=0, skipna=True))

    corr_method = (corr_method or "pearson").strip().lower()
    if corr_method not in ("pearson", "spearman"):
        corr_method = "pearson"

    C = X.corr(method=corr_method).abs()
    rows = []
    for i in range(len(prots)):
        for j in range(i + 1, len(prots)):
            v = float(C.iloc[i, j])
            if np.isfinite(v) and v >= float(corr_thresh):
                rows.append({"protein1": prots[i], "protein2": prots[j], "abs_corr": v})

    out = pd.DataFrame(rows).sort_values("abs_corr", ascending=False) if rows else pd.DataFrame(columns=["protein1", "protein2", "abs_corr"])
    out.to_csv(out_csv, index=False)
    return out_csv


def prune_correlated_panel(
    df_tr: pd.DataFrame,
    y_tr: Optional[np.ndarray],
    proteins: List[str],
    selection_freq: Optional[Dict[str, float]],
    corr_thresh: float = 0.80,
    corr_method: str = "pearson",
    rep_tiebreak: str = "freq",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Collapse highly correlated proteins into connected components; keep one representative per component.

    rep_tiebreak:
      - "freq": highest selection_freq wins (default; same as before)
      - "freq_then_univariate": when selection_freq ties, break tie by univariate MW p-value (smaller is better)
    """
    prots = [p for p in proteins if p in df_tr.columns]
    if len(prots) == 0:
        return pd.DataFrame(columns=["component_id","protein","selection_freq","kept","rep_protein","component_size"]), []

    X = df_tr[prots].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        X = X.fillna(X.median(axis=0, skipna=True))

    corr_method = (corr_method or "pearson").strip().lower()
    if corr_method not in ("pearson", "spearman"):
        corr_method = "pearson"

    C = X.corr(method=corr_method).abs().fillna(0.0)

    adj = {p: set() for p in prots}
    for i, p1 in enumerate(prots):
        for j in range(i + 1, len(prots)):
            p2 = prots[j]
            if float(C.loc[p1, p2]) >= float(corr_thresh):
                adj[p1].add(p2)
                adj[p2].add(p1)

    seen = set()
    comps = []
    for p in prots:
        if p in seen:
            continue
        stack = [p]
        comp = []
        seen.add(p)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(sorted(comp))

    rep_tiebreak = (rep_tiebreak or "freq").strip().lower()
    uni_map: Dict[str, Tuple[float, float]] = {}
    if rep_tiebreak == "freq_then_univariate" and y_tr is not None:
        uni_map = univariate_strength_map(df_tr, y_tr, prots)

    rows = []
    kept = []
    for cid, comp in enumerate(comps, start=1):
        def key(p):
            sf = selection_freq.get(p, np.nan) if selection_freq else np.nan
            sf2 = (sf if np.isfinite(sf) else 0.0)
            # primary: higher freq
            k1 = -sf2
            # secondary: univariate p (smaller better) if enabled and tied
            if rep_tiebreak == "freq_then_univariate":
                p_mw, abs_delta = uni_map.get(p, (np.nan, np.nan))
                p_mw2 = (p_mw if np.isfinite(p_mw) else 1.0)
                k2 = p_mw2
                k3 = -(abs_delta if np.isfinite(abs_delta) else 0.0)
                return (k1, k2, k3, p)
            return (k1, p)

        rep = sorted(comp, key=key)[0]
        kept.append(rep)
        for p in comp:
            sf = selection_freq.get(p, np.nan) if selection_freq else np.nan
            rows.append({
                "component_id": cid,
                "protein": p,
                "selection_freq": sf,
                "kept": (p == rep),
                "rep_protein": rep,
                "component_size": len(comp),
            })

    df_map = pd.DataFrame(rows).sort_values(
        ["kept","selection_freq","protein"],
        ascending=[False, False, True],
        na_position="last"
    )
    kept = sorted(set(kept), key=lambda p: (-(selection_freq.get(p, 0.0) if selection_freq else 0.0), p))
    return df_map, kept


def _thr_to_tag(x: float) -> str:
    return str(round(float(x), 4)).replace(".", "p")


def _safe_json_list(s: str) -> List[str]:
    if not isinstance(s, str) or not s:
        return []
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return [str(z) for z in v]
    except Exception:
        return []
    return []


def selection_freq_from_sel_rows(df_sel: pd.DataFrame, col_json: str) -> Dict[str, float]:
    """
    Frequency across *outer splits* (repeatÃ—fold), not per-repeat unions.
    freq = (#splits where protein appears) / (#splits total)
    """
    if df_sel is None or df_sel.empty or col_json not in df_sel.columns:
        return {}
    total = int(len(df_sel))
    if total <= 0:
        return {}
    counts: Dict[str, int] = {}
    for s in df_sel[col_json].tolist():
        prots = set(_safe_json_list(s))
        for p in prots:
            counts[p] = counts.get(p, 0) + 1
    return {p: c / float(total) for p, c in counts.items()}


def build_raw_panel(sel_freq: Dict[str, float], rule: str, N: int, tau: float) -> pd.DataFrame:
    df = pd.DataFrame([{"protein": p, "selection_freq": float(f)} for p, f in sel_freq.items()])
    if df.empty:
        return df
    df = df.sort_values(["selection_freq","protein"], ascending=[False, True]).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1, dtype=int)

    if rule == "freq_ge_tau":
        df = df[df["selection_freq"] >= float(tau)].copy().reset_index(drop=True)

    if len(df) > int(N):
        df = df.head(int(N)).copy()
    return df


def prune_and_refill_to_N(
    df_tr: pd.DataFrame,
    y_tr: Optional[np.ndarray],
    ranked_proteins: List[str],
    selection_freq: Dict[str, float],
    N: int,
    corr_thresh: float,
    pool_limit: int,
    corr_method: str = "pearson",
    rep_tiebreak: str = "freq",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    1) take top-N ranked
    2) prune by corr components (keep representative)
    3) refill from ranked list until size==N, skipping too-correlated candidates (TRAIN-only)
    """
    ranked_proteins = [p for p in ranked_proteins if p in df_tr.columns]
    topN = ranked_proteins[:min(int(N), len(ranked_proteins))]

    df_map, kept = prune_correlated_panel(
        df_tr=df_tr,
        y_tr=y_tr,
        proteins=topN,
        selection_freq=selection_freq,
        corr_thresh=corr_thresh,
        corr_method=corr_method,
        rep_tiebreak=rep_tiebreak
    )
    df_map = df_map.copy()
    if not df_map.empty:
        df_map["representative_flag"] = df_map["kept"].astype(bool)
        df_map["removed_due_to_corr_with"] = np.where(df_map["kept"], "", df_map["rep_protein"])

    kept = list(kept)
    if len(kept) >= int(N):
        return df_map, kept[:int(N)]

    # correlation matrix on pool (TRAIN only)
    pool = ranked_proteins[:min(int(pool_limit), len(ranked_proteins))]
    if not pool:
        return df_map, kept

    X = df_tr[pool].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        X = X.fillna(X.median(axis=0, skipna=True))

    corr_method = (corr_method or "pearson").strip().lower()
    if corr_method not in ("pearson", "spearman"):
        corr_method = "pearson"
    C = X.corr(method=corr_method).abs().fillna(0.0)

    final = list(kept)
    kept_set = set(final)
    for cand in pool:
        if len(final) >= int(N):
            break
        if cand in kept_set:
            continue
        too_corr = False
        for k in final:
            if cand in C.index and k in C.columns and float(C.loc[cand, k]) >= float(corr_thresh):
                too_corr = True
                break
        if too_corr:
            continue
        final.append(cand)
        kept_set.add(cand)

    # add refill-added proteins to map for traceability
    if not df_map.empty and len(final) > len(kept):
        max_cid = int(df_map["component_id"].max()) if "component_id" in df_map.columns else 0
        add_rows = []
        for i, p in enumerate(final[len(kept):], start=1):
            add_rows.append({
                "component_id": max_cid + i,
                "protein": p,
                "selection_freq": float(selection_freq.get(p, np.nan)),
                "kept": True,
                "rep_protein": p,
                "component_size": 1,
                "representative_flag": True,
                "removed_due_to_corr_with": ""
            })
        df_map = pd.concat([df_map, pd.DataFrame(add_rows)], ignore_index=True)

    return df_map, final


def _stable_panel_from_sel_rows(df_sel: pd.DataFrame, n_repeats: int, stability_thresh: float, col_json: str, fallback_top_n: int = 20):
    rep_sets = []
    for r in range(int(n_repeats)):
        sub = df_sel[df_sel["repeat"] == r]
        u = set()
        for s in sub[col_json].tolist():
            try:
                u.update(json.loads(s))
            except Exception:
                continue
        rep_sets.append(u)

    all_prots = sorted(set().union(*rep_sets)) if rep_sets else []
    rows = []
    for p in all_prots:
        freq = sum((p in rs) for rs in rep_sets) / float(n_repeats)
        rows.append({"protein": p, "selection_freq": freq, "kept": freq >= stability_thresh})

    df_panel = pd.DataFrame(rows)
    if len(df_panel) == 0:
        return df_panel, [], rep_sets

    df_panel = df_panel.sort_values(["kept", "selection_freq", "protein"], ascending=[False, False, True])
    kept = df_panel[df_panel["kept"]]["protein"].tolist()

    if len(kept) == 0:
        logprint(f"âš ï¸  WARNING: No proteins met stability_thresh={stability_thresh:.2f}. Falling back to top {fallback_top_n}.", level="warning")
        kept = df_panel.nlargest(min(fallback_top_n, len(df_panel)), "selection_freq")["protein"].tolist()
        df_panel["kept"] = df_panel["protein"].isin(kept)

    return df_panel, kept, rep_sets


def _load_split_indices(splits_dir: str, scenario: str, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Try seed-suffixed files first (for repeated splits), then fallback to non-suffixed
    if seed is not None:
        tr_seed = os.path.join(splits_dir, f"{scenario}_train_idx_seed{seed}.csv")
        va_seed = os.path.join(splits_dir, f"{scenario}_val_idx_seed{seed}.csv")
        te_seed = os.path.join(splits_dir, f"{scenario}_test_idx_seed{seed}.csv")
        if os.path.exists(tr_seed) and os.path.exists(te_seed):
            tr, va, te = tr_seed, va_seed, te_seed
        else:
            # Fallback to non-suffixed (single split mode)
            tr = os.path.join(splits_dir, f"{scenario}_train_idx.csv")
            va = os.path.join(splits_dir, f"{scenario}_val_idx.csv")
            te = os.path.join(splits_dir, f"{scenario}_test_idx.csv")
    else:
        tr = os.path.join(splits_dir, f"{scenario}_train_idx.csv")
        va = os.path.join(splits_dir, f"{scenario}_val_idx.csv")
        te = os.path.join(splits_dir, f"{scenario}_test_idx.csv")

    if not (os.path.exists(tr) and os.path.exists(te)):
        raise FileNotFoundError(f"Missing split files for {scenario} in {splits_dir}: {tr} / {te}")

    idx_train = pd.read_csv(tr)["idx"].to_numpy(dtype=int)
    if os.path.exists(va):
        idx_val = pd.read_csv(va)["idx"].to_numpy(dtype=int)
    else:
        idx_val = np.array([], dtype=int)
    idx_test  = pd.read_csv(te)["idx"].to_numpy(dtype=int)

    # basic sanity
    if np.intersect1d(idx_train, idx_test).size > 0 or np.intersect1d(idx_train, idx_val).size > 0 or np.intersect1d(idx_val, idx_test).size > 0:
        raise ValueError(f"Splits overlap for {scenario} in {splits_dir}.")
    if (
        np.unique(idx_train).size != idx_train.size
        or np.unique(idx_val).size != idx_val.size
        or np.unique(idx_test).size != idx_test.size
    ):
        raise ValueError(f"Duplicate indices found in splits for {scenario} in {splits_dir}.")

    # sort for reproducibility
    idx_train = np.sort(idx_train)
    idx_val = np.sort(idx_val)
    idx_test  = np.sort(idx_test)
    return idx_train, idx_val, idx_test


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    ap.add_argument("--infile", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--scenario", choices=["IncidentOnly", "IncidentPlusPrevalent"], required=True)

    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--models", nargs="+", required=True)  # in parallel mode, pass SINGLE model
    ap.add_argument("--cpus", type=int, default=get_cpus(1))

    ap.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"],
                    help="Logging level for stdout and outdir/run.log")

    ap.add_argument("--test_size", type=float, default=0.20)
    ap.add_argument("--splits_dir", type=str, required=True,
                    help="Directory containing precomputed splits from save_splits.py (required).")
    ap.add_argument("--save_test_preds", action="store_true")
    ap.add_argument("--save_val_preds", action="store_true")
    ap.add_argument("--save_controls_oof", action="store_true")
    ap.add_argument("--save_train_oof", action="store_true",
                    help="Save TRAIN OOF predictions (controls + cases) for risk distribution plots.")

    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--scoring", choices=["average_precision", "roc_auc", "neg_brier_score"], default="average_precision")
    ap.add_argument("--inner_folds", type=int, default=5)
    ap.add_argument("--n_iter", type=int, default=30)

    ap.add_argument("--feature_select", choices=["none", "kbest", "l1_stability", "hybrid"], default="none")

    ap.add_argument("--kbest_scope", choices=["protein", "transformed"], default="protein")
    ap.add_argument("--k_grid", type=str, default="50,100,200,500")
    ap.add_argument("--kbest_max", type=int, default=500)
    ap.add_argument("--stability_thresh", type=float, default=0.70)
    ap.add_argument("--coef_thresh", type=float, default=1e-12)

    ap.add_argument("--screen_method", choices=["mannwhitney", "f_classif"], default="mannwhitney")
    ap.add_argument("--screen_top_n", type=int, default=0)
    ap.add_argument("--screen_min_n_per_group", type=int, default=10,  # NEW
                    help="Minimum sample size per group (cases/controls) for a protein to be eligible for screening.")

    ap.add_argument("--perm_top_n", type=int, default=200)
    ap.add_argument("--perm_repeats", type=int, default=3)
    ap.add_argument("--perm_min_importance", type=float, default=0.0)

    ap.add_argument("--control_spec_targets", type=str, default="0.95,0.99,0.995")
    ap.add_argument("--risk_plot_spec", type=float, default=0.95,
                    help="Target specificity for 'spec' threshold on risk distribution plots (default: 0.95)")
    ap.add_argument("--alpha_specificity", type=float, default=0.90,
                    help="Target specificity for alpha threshold on risk distribution plots (default: 0.90)")
    ap.add_argument("--toprisk_fracs", type=str, default="0.01")

    ap.add_argument("--stable_panel_from_kbest", action="store_true")

    # ---------------- Panels ----------------
    ap.add_argument("--build_panels", action="store_true",
                    help="Build Top-N frequency panels from TRAIN repeated-CV selections, prune correlations on TRAIN, export artifacts, and optionally refit/eval on TEST.")
    ap.add_argument("--panel_sizes", type=str, default="50,100,200",
                    help="Comma-separated panel sizes to build (e.g. 50,100,200 or 50,100,200,500).")
    ap.add_argument("--panel_corr_thresh", type=float, default=0.80,
                    help="Absolute correlation threshold for pruning at panel build time (TRAIN-only).")
    ap.add_argument("--panel_corr_method", choices=["pearson","spearman"], default="pearson",  # NEW
                    help="Correlation method used for panel pruning / redundancy collapse (TRAIN-only).")
    ap.add_argument("--panel_rep_tiebreak", choices=["freq","freq_then_univariate"], default="freq",  # NEW
                    help="Representative selection in correlated components: freq (default) or freq_then_univariate.")
    ap.add_argument("--panel_source", type=str, default="auto", choices=["auto","screen","model"],
                    help="Which selection column to use to build frequencies. auto: hybrid->screen else model.")
    ap.add_argument("--panel_rule", type=str, default="topN", choices=["topN","freq_ge_tau"],
                    help="Panel rule: topN (default) or frequency>=tau (secondary consensus rule).")
    ap.add_argument("--panel_freq_tau", type=float, default=0.70,
                    help="If panel_rule=freq_ge_tau: keep proteins with selection_freq >= tau.")
    ap.add_argument("--panel_pool_factor", type=int, default=10,
                    help="Candidate pool size for refill = max(panel_sizes)*panel_pool_factor.")
    ap.add_argument("--panel_refit", action="store_true",
                    help="If set, refit/tune/evaluate each pruned panel as an additional model entry in test_metrics/cv_repeat_metrics.")
    ap.add_argument("--panel_proteins_csv", type=str, default="",
                    help="Refit mode: CSV with column 'protein' specifying a fixed panel. If provided, runs the base model on exactly those proteins (TRAIN-only tuning; TEST once).")
    ap.add_argument("--panel_tag", type=str, default="",
                    help="Optional label appended to the model name when using --panel_proteins_csv.")
    ap.add_argument("--panel_audit_frac", type=float, default=0.20,
                    help="Fraction of TRAIN data reserved for panel stability/correlation calculations (removed from model fitting to avoid double dipping). Set 0 to disable.")
    ap.add_argument("--panel_stability_mode", type=str, default="audit", choices=["audit","rskf"],
                    help="Panel stability source: audit reserves --panel_audit_frac from TRAIN; rskf uses repeated stratified CV (no audit holdout).")
    ap.add_argument("--panel_audit_seed", type=int, default=2026,
                    help="Random seed for selecting the panel audit subset.")

    ap.add_argument("--test_ci_bootstrap", action="store_true")
    ap.add_argument("--test_ci_best_only", action="store_true")  # kept for compatibility
    ap.add_argument("--n_boot", type=int, default=500)
    ap.add_argument("--write_test_ci_files", action="store_true")

    ap.add_argument("--var_prefilter", action="store_true")
    ap.add_argument("--min_nonmissing", type=float, default=0.50)
    ap.add_argument("--min_var", type=float, default=0.0)
    ap.add_argument("--var_strict", action="store_true")

    ap.add_argument("--save_calibration", action="store_true")
    ap.add_argument("--calib_bins", type=int, default=10)

    # DCA arguments removed - DCA computation moved to postprocess_compare.py
    # This saves ~6 min/model during training and enables flexible DCA re-computation

    ap.add_argument("--learning_curve", action="store_true")
    ap.add_argument("--lc_cv", type=int, default=5)
    ap.add_argument("--lc_min_frac", type=float, default=0.3)
    ap.add_argument("--lc_points", type=int, default=5)
    ap.add_argument("--learning_curve_scope", choices=["best", "all"], default="best")

    ap.add_argument("--feature_reports", choices=["none", "best", "all"], default="all")
    ap.add_argument("--feature_report_max", type=int, default=200)

    ap.add_argument("--stable_corr_thresh", type=float, default=0.80)

    # ---------------- temporal split awareness ----------------
    ap.add_argument("--temporal_split", action="store_true",
                    help="Flag indicating splits were generated with temporal (chronological) ordering. "
                         "When set, train data comes from earlier time periods and test/holdout from later. "
                         "Actual split logic is in save_splits.py; this flag is for logging/metadata.")
    ap.add_argument("--temporal_col", type=str, default="CeD_date",
                    help="Column used for temporal ordering in save_splits.py (for logging/metadata only).")

    # ---------------- cross-cutting / reproducibility / HPC ----------------
    ap.add_argument("--grid_randomize", action="store_true",
                    help="Sample downstream model hyperparameter grids randomly (controlled by --random_state).")
    ap.add_argument("--grid_randomize_k", action="store_true",
                    help="Randomize k_grid values (feature counts). Requires --grid_randomize_k plus --grid_randomize or --random_state for reproducibility.")
    ap.add_argument("--random_state", type=int, default=0)
    ap.add_argument("--tune_n_jobs", type=str, default="auto",
                    help="RandomizedSearchCV(n_jobs=...). Use 'auto' (default) or an integer.")
    ap.add_argument("--error_score", type=str, default="nan",
                    help="RandomizedSearchCV(error_score=...). Use 'nan' (default), 'raise', or a float.")
    ap.add_argument("--missing_imputer", choices=["median", "mean"], default="median",
                    help="Numeric imputer strategy for proteomics + covariates (median robust to skew/outliers).")

    # ---------------- LR knobs ----------------
    ap.add_argument("--lr_max_iter", type=int, default=8000)
    ap.add_argument("--lr_tol", type=float, default=1e-4)
    ap.add_argument("--lr_C_min", type=float, default=1e-3)
    ap.add_argument("--lr_C_max", type=float, default=1e3)
    ap.add_argument("--lr_C_points", type=int, default=13)
    ap.add_argument("--lr_l1_ratio_grid", type=str, default="0.05,0.1,0.2,0.4,0.6,0.8,0.95")
    ap.add_argument("--lr_class_weight_options", type=str, default="none,balanced")

    # ---------------- SVM + calibration knobs ----------------
    ap.add_argument("--svm_max_iter", type=int, default=8000)
    ap.add_argument("--svm_C_min", type=float, default=1e-3)
    ap.add_argument("--svm_C_max", type=float, default=1e3)
    ap.add_argument("--svm_C_points", type=int, default=13)
    ap.add_argument("--svm_class_weight_options", type=str, default="none,balanced")
    ap.add_argument("--calibration_method", choices=["sigmoid", "isotonic"], default="sigmoid")
    ap.add_argument("--calibration_cv", type=int, default=3)

    # ---------------- RF knobs ----------------
    ap.add_argument("--rf_n_estimators", type=int, default=800,
                    help="Legacy RF n_estimators (unused when rf_n_estimators_grid is required).")
    ap.add_argument("--rf_n_estimators_grid", type=str, default="",
                    help="Comma-separated list of n_estimators to tune (required; e.g., '200,400,800').")
    ap.add_argument("--rf_max_depth_grid", type=str, default="none,10,20,40")
    ap.add_argument("--rf_min_samples_leaf_grid", type=str, default="5,10,20")
    ap.add_argument("--rf_max_features_grid", type=str, default="sqrt,0.3")
    ap.add_argument("--rf_bootstrap", type=int, default=1)
    ap.add_argument("--rf_max_samples", type=str, default="")
    ap.add_argument("--rf_max_samples_grid", type=str, default="")
    ap.add_argument("--rf_class_weight_options", type=str, default="none,balanced")

    # ---------------- XGBoost knobs ----------------
    ap.add_argument("--xgb_n_estimators", type=int, default=1000,
                    help="Default number of boosting rounds for XGBoost")
    ap.add_argument("--xgb_n_estimators_grid", type=str, default="500,1000,2000",
                    help="Comma-separated list of n_estimators to tune (e.g., '500,1000,2000')")
    ap.add_argument("--xgb_max_depth", type=int, default=5,
                    help="Default max tree depth for XGBoost")
    ap.add_argument("--xgb_max_depth_grid", type=str, default="3,5,7",
                    help="Comma-separated list of max_depth to tune (e.g., '3,5,7')")
    ap.add_argument("--xgb_learning_rate", type=float, default=0.05,
                    help="Default learning rate (eta) for XGBoost")
    ap.add_argument("--xgb_learning_rate_grid", type=str, default="0.01,0.05,0.1",
                    help="Comma-separated list of learning rates to tune (e.g., '0.01,0.05,0.1')")
    ap.add_argument("--xgb_subsample", type=float, default=0.8,
                    help="Default row subsample ratio for XGBoost")
    ap.add_argument("--xgb_subsample_grid", type=str, default="0.7,0.8,1.0",
                    help="Comma-separated list of subsample ratios to tune (e.g., '0.7,0.8,1.0')")
    ap.add_argument("--xgb_colsample_bytree", type=float, default=0.8,
                    help="Default column subsample ratio per tree for XGBoost")
    ap.add_argument("--xgb_colsample_bytree_grid", type=str, default="0.7,0.8,1.0",
                    help="Comma-separated list of colsample_bytree ratios to tune (e.g., '0.7,0.8,1.0')")
    ap.add_argument("--xgb_scale_pos_weight", type=str, default="auto",
                    help="scale_pos_weight for XGBoost. Use 'auto' to compute n_neg/n_pos per training split or provide a float.")
    ap.add_argument("--xgb_scale_pos_weight_grid", type=str, default="",
                    help="Optional comma-separated list overriding auto scale_pos_weight during tuning (e.g., '100,200,300').")
    ap.add_argument("--xgb_reg_alpha", type=float, default=0.0,
                    help="L1 regularization term on weights (xgboost alpha)")
    ap.add_argument("--xgb_reg_lambda", type=float, default=1.0,
                    help="L2 regularization term on weights (xgboost lambda)")
    ap.add_argument("--xgb_min_child_weight", type=int, default=1,
                    help="Minimum sum of instance weight needed in a child (xgboost)")
    ap.add_argument("--xgb_gamma", type=float, default=0.0,
                    help="Minimum loss reduction required to make a split (xgboost)")
    ap.add_argument("--xgb_tree_method", type=str, default="hist",
                    choices=["auto", "exact", "approx", "hist", "gpu_hist"],
                    help="XGBoost tree construction algorithm. Use 'gpu_hist' for GPU acceleration, 'hist' for CPU.")

    # ---------------- risk-score calibration + threshold objective ----------------
    ap.add_argument("--calibrate_final_models", type=int, default=0,
                    help="0/1. If 1, apply CalibratedClassifierCV to LR/RF as well (SVM already calibrated).")
    ap.add_argument("--threshold_objective", choices=["max_f1","max_fbeta","youden","fixed_spec","fixed_ppv"], default="max_f1")
    ap.add_argument("--fbeta", type=float, default=1.0)
    ap.add_argument("--fixed_spec", type=float, default=0.90)
    ap.add_argument("--fixed_ppv", type=float, default=0.10)
    ap.add_argument("--clinical_threshold_points", type=str, default="0.001,0.002,0.005,0.01,0.02,0.03",
                    help="Comma-separated list of clinical probability thresholds to summarize (e.g., deployment net benefit points).")
    ap.add_argument("--target_prevalence", type=float, default=0.01,
                    help="Target population prevalence for prevalence-shift calibration (e.g., 0.01 for 1%).")
    ap.add_argument(
        "--target_prevalence_source",
        choices=["fixed", "train", "val", "test"],
        default="fixed",
        help="Source for target prevalence used in prevalence-shift calibration. "
             "'fixed' uses --target_prevalence, others compute prevalence from the split.",
    )
    ap.add_argument("--subgroup_min_n", type=int, default=40,
                    help="Minimum sample size per subgroup when generating subgroup metrics (set 0 to disable).")
    ap.add_argument(
        "--risk_prob_source",
        choices=["raw", "adjusted"],
        default="raw",
        help="Probability stream used for thresholds, metrics, and exported artifacts. "
             "'adjusted' applies the prevalence shift defined by --target_prevalence."
    )

    # ---------------- survival plots ----------------
    ap.add_argument("--survival_plots", action="store_true",
                    help="Generate Kaplan-Meier and Cox regression plots on the TEST set.")
    ap.add_argument("--survival_time_col", type=str, default="",
                    help="Column with time-to-event or follow-up time (numeric).")
    ap.add_argument("--survival_start_col", type=str, default="",
                    help="Column for baseline time when deriving durations (date or numeric).")
    ap.add_argument("--survival_end_col", type=str, default="",
                    help="Column for event/censor time when deriving durations (date or numeric).")
    ap.add_argument("--survival_event_col", type=str, default="",
                    help="Event indicator column (1=event, 0=censored). Defaults to model y.")
    ap.add_argument("--survival_time_unit", choices=["days", "years"], default="days",
                    help="Time unit used for survival plots.")
    ap.add_argument("--km_groups", type=int, default=2,
                    help="Number of risk groups for Kaplan-Meier curves (default 2).")
    ap.add_argument("--km_group_labels", type=str, default="",
                    help="Comma-separated labels for KM groups (optional).")
    ap.add_argument("--cox_covariates", type=str, default="",
                    help="Comma-separated covariate columns to include in Cox regression (optional).")

    # ---------------- DCA configuration ----------------
    ap.add_argument("--compute_dca", type=int, default=1,
                    help="Enable DCA computation during training (1=yes, 0=skip). Default 1.")
    ap.add_argument("--dca_threshold_min", type=float, default=0.0005,
                    help="Minimum threshold for DCA curve (default 0.0005 = 0.05%)")
    ap.add_argument("--dca_threshold_max", type=float, default=1.0,
                    help="Maximum threshold for DCA curve (default 1.0 = 100%)")
    ap.add_argument("--dca_threshold_step", type=float, default=0.001,
                    help="Step size for DCA thresholds (default 0.001 = 0.1%)")
    ap.add_argument("--dca_report_points", type=str, default="0.005,0.01,0.02,0.05",
                    help="Comma-separated key thresholds for DCA summary (e.g., '0.005,0.01,0.02,0.05')")

    ap.add_argument(
        "--threshold_source",
        choices=["train_oof", "val"],
        default="train_oof",
        help="Source split used to select thresholds. train_oof uses TRAIN OOF; val uses VAL predictions.",
    )

    # ---------------- resume from checkpoint ----------------
    ap.add_argument(
        "--resume",
        action="store_true",
        help="If a trained model artifact exists in outdir, skip training and resume from evaluation/DCA steps."
    )

    # ---------------- plot regeneration mode ----------------
    ap.add_argument(
        "--regenerate_plots",
        action="store_true",
        help="Regenerate plots from saved artifacts (no retraining). Requires --run_dir."
    )
    ap.add_argument(
        "--run_dir",
        type=str,
        default="",
        help="Path to model run directory (for --regenerate_plots). Example: results_dir/IncidentPlusPrevalent__RF__5x10..."
    )
    ap.add_argument(
        "--plot_types",
        type=str,
        default="",
        help="Comma-separated plot types to regenerate (default: all). Options: roc,pr,calibration,risk_dist"
    )
    ap.add_argument(
        "--force_overwrite",
        action="store_true",
        help="Overwrite existing plots when regenerating (default: skip existing)"
    )

    return ap


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = build_arg_parser()
    return ap.parse_args(argv)

@dataclass
class FaithRunner:
    """Lightweight runner wrapper (keeps CLI stable, improves organization)."""
    args: argparse.Namespace

    def run(self) -> None:
        run_from_args(self.args)



def run_from_args(args: argparse.Namespace) -> None:
    os.makedirs(args.outdir, exist_ok=True)
    OUT = make_outdirs(args.outdir)

    # Pipeline outline:
    #  1) filter/split the dataset so train/dev/holdout all align.
    #  2) configure features (prefiltering, screening, optional panels).
    #  3) perform nested CV to tune + evaluate candidates and pick thresholds.
    #  4) refit tuned models on TRAIN, evaluate on TEST, and export artifacts.

    with open(os.path.join(OUT["core"], "run_settings.json"), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    if not args.tune:
        raise ValueError("This script expects --tune enabled (by design).")

    if len(args.models) != 1:
        raise ValueError("Parallel mode expects exactly ONE model per run: pass e.g. --models LR_EN")

    scen_name = args.scenario
    model_name = args.models[0]
    model_label = model_name
    if args.panel_proteins_csv and args.panel_tag:
        model_label = f"{model_name}__{args.panel_tag}"

    seed = int(args.random_state)
    grid_randomize_main = bool(getattr(args, "grid_randomize", False))
    grid_randomize_k = bool(getattr(args, "grid_randomize_k", False))
    grid_rng = np.random.RandomState(seed + 7919) if (grid_randomize_main or grid_randomize_k) else None

    k_grid_raw = parse_k_grid(args.k_grid) if args.feature_select in ("kbest", "hybrid") else []
    if args.feature_select in ("kbest", "hybrid"):
        k_grid_raw = [k for k in k_grid_raw if k > 0]
        if args.kbest_max and int(args.kbest_max) > 0:
            before = list(k_grid_raw)
            k_grid_raw = [k for k in k_grid_raw if k <= int(args.kbest_max)]
            if before != k_grid_raw:
                logprint(f"[kbest] enforcing --kbest_max={args.kbest_max}. k_grid filtered: {before} -> {k_grid_raw}")

    spec_targets = sorted({float(t.strip()) for t in (args.control_spec_targets or "").split(",") if t.strip()})
    top_fracs = sorted({float(t.strip()) for t in (args.toprisk_fracs or "").split(",") if t.strip()})
    clinical_thresholds = sorted({float(t.strip()) for t in (args.clinical_threshold_points or "").split(",") if t.strip()})

    df = pd.read_csv(args.infile, usecols=_usecols_for_celiac)

    positives = ["Incident"] if scen_name == "IncidentOnly" else ["Incident", "Prevalent"]
    df2, X_all, y_all, _, prot_cols, filter_stats = make_dataset(df, positives=positives)

    logprint(f"[make_dataset] Row filtering applied:")
    logprint(f"  - Input rows: {filter_stats['n_in']:,}")
    logprint(f"  - Removed uncertain controls: {filter_stats['n_removed_uncertain_controls']:,}")
    logprint(f"  - Removed missing metadata: {filter_stats['n_removed_dropna_meta_num']:,}")
    logprint(f"  - Output rows: {filter_stats['n_out']:,}")

    # --- temporal split awareness ---
    if args.temporal_split:
        logprint(f"[temporal] TEMPORAL VALIDATION MODE ENABLED")
        logprint(f"[temporal] Splits ordered by column: {args.temporal_col}")
        logprint(f"[temporal] Train set contains temporally EARLIER samples")
        logprint(f"[temporal] Test/Holdout sets contain temporally LATER samples")
        logprint(f"[temporal] This mimics real-world prospective deployment")
    else:
        logprint(f"[split] Standard (random) stratified split mode")

    # --- shared split ---
    holdout_path = os.path.join(args.splits_dir, f"{scen_name}_HOLDOUT_idx.csv")
    if os.path.exists(holdout_path):
        logprint(f"[split] Holdout mode detected. Excluding holdout set from development data.")
        holdout_idx = pd.read_csv(holdout_path)["idx"].to_numpy(dtype=int)

        all_idx = np.arange(len(y_all))
        dev_mask = np.ones(len(y_all), dtype=bool)
        dev_mask[holdout_idx] = False
        dev_idx = all_idx[dev_mask]

        df2 = df2.iloc[dev_idx].reset_index(drop=True)
        X_all = X_all.iloc[dev_idx].reset_index(drop=True)
        y_all = y_all[dev_idx]

        logprint(f"[split] Full dataset: {len(all_idx):,} samples")
        logprint(f"[split] Holdout set: {len(holdout_idx):,} samples (excluded)")
        logprint(f"[split] Development set: {len(y_all):,} samples (Controls + Incident only)")

    idx_train, idx_val, idx_test = _load_split_indices(args.splits_dir, scen_name, seed=seed)
    if idx_train.max() >= len(y_all) or idx_test.max() >= len(y_all) or (len(idx_val) > 0 and idx_val.max() >= len(y_all)):
        raise ValueError(f"Split indices exceed dataset rows for {scen_name}. Check splits_dir vs infile/filters.")

    split_id = split_id_from_indices(idx_test)

    df_tr = df2.iloc[idx_train].reset_index(drop=True)
    df_val = df2.iloc[idx_val].reset_index(drop=True) if len(idx_val) > 0 else None
    df_te = df2.iloc[idx_test].reset_index(drop=True)
    y_tr = y_all[idx_train]
    y_val = y_all[idx_val] if len(idx_val) > 0 else np.array([], dtype=int)
    y_te = y_all[idx_test]

    sets = ["TRAIN"] * len(df_tr)
    ids = [df_tr[ID_COL]] if ID_COL in df2.columns else []
    targets = [df_tr[TARGET_COL]]
    ys = [y_tr.astype(int)]
    if df_val is not None:
        sets.extend(["VAL"] * len(df_val))
        if ID_COL in df2.columns:
            ids.append(df_val[ID_COL])
        targets.append(df_val[TARGET_COL])
        ys.append(y_val.astype(int))
    sets.extend(["TEST"] * len(df_te))
    if ID_COL in df2.columns:
        ids.append(df_te[ID_COL])
    targets.append(df_te[TARGET_COL])
    ys.append(y_te.astype(int))

    trace = pd.DataFrame({
        "scenario": scen_name,
        "split_id": split_id,
        "set": sets,
        ID_COL: pd.concat(ids, ignore_index=True) if ID_COL in df2.columns else np.arange(len(sets)),
        TARGET_COL: pd.concat(targets, ignore_index=True).astype(str),
        "y": np.concatenate(ys),
    })
    trace.to_csv(os.path.join(OUT["diag_splits"], f"{scen_name}__train_test_split_trace.csv"), index=False)

    X_tr0 = X_all.iloc[idx_train].reset_index(drop=True)
    X_val0 = X_all.iloc[idx_val].reset_index(drop=True) if len(idx_val) > 0 else None
    X_te0 = X_all.iloc[idx_test].reset_index(drop=True)

    logprint(f"\n=== {scen_name} / {model_name} ===")
    logprint(f"ALL n={len(y_all)} pos={int(y_all.sum())} proteins={len(prot_cols)}")
    denom = max(1, (len(y_tr) + len(y_val) + len(y_te)))
    actual_test_frac = len(y_te) / denom
    actual_val_frac = len(y_val) / denom if len(y_val) > 0 else 0.0
    if len(y_val) > 0:
        logprint(
            f"TRAIN n={len(y_tr)} pos={int(y_tr.sum())} | "
            f"VAL n={len(y_val)} pos={int(y_val.sum())} | "
            f"TEST n={len(y_te)} pos={int(y_te.sum())} "
            f"(val_fraction={actual_val_frac:.3f}, test_fraction={actual_test_frac:.3f})"
        )
        logprint(f"Prevalence: TRAIN={_prevalence(y_tr):.4f} | VAL={_prevalence(y_val):.4f} | TEST={_prevalence(y_te):.4f}")
    else:
        logprint(
            f"TRAIN n={len(y_tr)} pos={int(y_tr.sum())} | "
            f"TEST n={len(y_te)} pos={int(y_te.sum())} (test_fraction={actual_test_frac:.3f})"
        )
        logprint(f"Prevalence: TRAIN={_prevalence(y_tr):.4f} | TEST={_prevalence(y_te):.4f}")
    logprint(f"Split ID: {split_id}")

    df_panel_source = df_tr.copy()
    X_panel_source = X_tr0.copy()
    y_panel_source = y_tr.copy()
    panel_mode_enabled = bool(
        args.build_panels
        or args.stable_panel_from_kbest
        or args.panel_refit
        or bool(args.panel_proteins_csv)
    )
    audit_frac = float(max(0.0, args.panel_audit_frac)) if hasattr(args, "panel_audit_frac") else 0.0
    panel_stability_mode = str(getattr(args, "panel_stability_mode", "audit")).lower()
    n_audit = 0
    if panel_mode_enabled:
        if panel_stability_mode == "rskf":

            if int(args.folds) < 2:
                logprint("[panel_audit] folds<2; repeated CV stability estimates will be weak.", level="warning")
            logprint("[panel_audit] RSKF mode: using repeated stratified CV selections (no audit holdout).")
        else:
            if audit_frac > 0.0 and len(df_tr) >= 10:
                rng = np.random.RandomState(int(args.panel_audit_seed))
                n_audit = int(round(audit_frac * len(df_tr)))
                n_audit = min(max(1, n_audit), len(df_tr) - 1)
                audit_idx = rng.choice(len(df_tr), size=n_audit, replace=False)
                mask = np.ones(len(df_tr), dtype=bool)
                mask[audit_idx] = False

                df_panel_source = df_tr.iloc[audit_idx].reset_index(drop=True)
                X_panel_source = X_tr0.iloc[audit_idx].reset_index(drop=True)
                y_panel_source = y_tr[audit_idx]

                df_tr = df_tr.iloc[mask].reset_index(drop=True)
                X_tr0 = X_tr0.iloc[mask].reset_index(drop=True)
                y_tr = y_tr[mask]
            elif audit_frac > 0.0 and len(df_tr) < 10:
                logprint("[panel_audit] TRAIN too small for audit holdout; skipping audit.", level="warning")
            else:
                logprint("[panel_audit] Audit disabled; using full TRAIN for panel diagnostics.")

            logprint(f"[panel_audit] Reserved {n_audit} ({audit_frac:.1%}) TRAIN rows for panel stability diagnostics (excluded from fitting).")

    logprint(f"[train_after_audit] Effective TRAIN rows for tuning/fitting: {len(y_tr)} (pos={int(y_tr.sum())})")
    train_prevalence = _prevalence(y_tr)
    target_prev_source = str(getattr(args, "target_prevalence_source", "fixed")).lower()
    if target_prev_source == "train":
        target_prevalence = train_prevalence
    elif target_prev_source == "val" and len(y_val) > 0:
        target_prevalence = _prevalence(y_val)
    elif target_prev_source == "test":
        target_prevalence = _prevalence(y_te)
    else:
        target_prevalence = float(args.target_prevalence)
        target_prev_source = "fixed"
    target_prevalence = min(max(float(target_prevalence), 1e-6), 1.0 - 1e-6)
    logprint(f"[prevalence] TRAIN sample prevalence={train_prevalence:.4f} | target_prevalence={target_prevalence:.4f} (source={target_prev_source})")

    risk_prob_source = str(getattr(args, "risk_prob_source", "raw")).lower()
    use_adjusted_probs = (risk_prob_source == "adjusted")
    logprint(f"[probabilities] Metrics/thresholds/export will use {'prevalence-adjusted' if use_adjusted_probs else 'raw'} probabilities.")
    threshold_source = str(getattr(args, "threshold_source", "train_oof")).lower()

    prot_cols_use = prot_cols

    if args.var_prefilter and len(prot_cols) > 0:
        pf_csv = os.path.join(OUT["diag_prefilter"], f"{scen_name}__protein_prefilter_report.csv")
        prot_cols_use, _ = variance_missingness_prefilter(
            X_tr0, prot_cols=prot_cols,
            min_nonmissing=args.min_nonmissing,
            min_var=args.min_var,
            strict=args.var_strict,
            out_csv=pf_csv
        )
        logprint(f"[prefilter] proteins kept={len(prot_cols_use)} / {len(prot_cols)} (report: {pf_csv})")

    # ------------------
    # Refit-only mode on a fixed panel list
    # ------------------
    if args.panel_proteins_csv:
        dfp = pd.read_csv(args.panel_proteins_csv)
        if "protein" not in dfp.columns:
            raise ValueError("--panel_proteins_csv must contain a 'protein' column")
        requested = [str(x) for x in dfp["protein"].dropna().tolist()]
        requested = [p for p in requested if p in prot_cols_use]
        if len(requested) == 0:
            raise ValueError("No requested panel proteins were found after prefiltering. Check names / filtering settings.")
        prot_cols_use = requested
        args.feature_select = "none"
        args.screen_top_n = 0

    screen_kwargs = None
    screen_diag_csv = ""
    if args.feature_select == "hybrid" and int(args.screen_top_n) > 0:
        screen_diag_csv = os.path.join(OUT["diag_screen"], f"{scen_name}__screen__{args.screen_method}__top{int(args.screen_top_n)}.csv")
        screen_kwargs = {
            "method": args.screen_method,
            "top_n": int(args.screen_top_n),
            "min_n_per_group": int(args.screen_min_n_per_group),
            "diag_csv": None,
        }
        logprint(f"[screen] hybrid screening enabled: method={args.screen_method} top_n={int(args.screen_top_n)} (report: {screen_diag_csv})")

    if len(prot_cols_use) == 0:
        raise ValueError(f"{scen_name}: prot_cols_use became empty after filtering/screening. Adjust thresholds/top_n.")

    num_cols_use = META_NUM_COLS + prot_cols_use
    X_tr = X_tr0[num_cols_use + CAT_COLS].copy()
    X_val = X_val0[num_cols_use + CAT_COLS].copy() if X_val0 is not None else None
    X_te = X_te0[num_cols_use + CAT_COLS].copy()

    if args.feature_select in ("kbest", "hybrid") and args.kbest_scope == "protein":
        pre = build_preprocessor_auto(missing_strategy=args.missing_imputer)
    else:
        pre = build_preprocessor(num_cols=num_cols_use, cat_cols=CAT_COLS, missing_strategy=args.missing_imputer)

    k_grid = k_grid_raw
    if args.feature_select in ("kbest", "hybrid") and args.kbest_scope == "protein":
        n_prot = len(prot_cols_use)
        k_grid = [k for k in k_grid_raw if k <= n_prot]
        if len(k_grid) == 0:
            raise ValueError(f"k_grid values {k_grid_raw} all exceed available proteins {n_prot}.")
        logprint(f"[kbest/hybrid protein-scope] available proteins={n_prot}. Using k_grid={k_grid}")

    if (
        grid_rng is not None
        and bool(getattr(args, "grid_randomize_k", False))
        and args.feature_select in ("kbest", "hybrid")
        and len(k_grid) > 1
    ):
        # Use linear-space randomization for k (feature count)
        # Linear space ensures uniform coverage across all feature counts
        k_min, k_max = min(k_grid), max(k_grid)
        n_k = len(k_grid)
        randomized_k = np.round(grid_rng.uniform(k_min, k_max, size=n_k)).astype(int)
        randomized_k = np.unique(randomized_k)  # Remove duplicates

        # If deduplication reduced count, regenerate missing values
        while len(randomized_k) < n_k:
            new_val = int(np.round(grid_rng.uniform(k_min, k_max)))
            if new_val not in randomized_k:
                randomized_k = np.append(randomized_k, new_val)
        randomized_k = sorted(randomized_k.tolist())

        if args.kbest_scope == "protein":
            max_k = len(prot_cols_use)
            randomized_k = [k for k in randomized_k if k <= max_k]
            if not randomized_k:
                randomized_k = k_grid
        k_grid = randomized_k
        logprint(f"[grid_randomize_k] randomized k_grid (full log-space) -> {k_grid}")

    base_models = build_models(args.cpus, seed, args=args, model_name=model_name)
    if model_name not in base_models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(base_models.keys())}")

    clf = base_models[model_name]
    pipe = _build_pipeline(
        pre=pre,
        clf=clf,
        feature_select=args.feature_select,
        prot_cols=prot_cols_use,
        kbest_scope=args.kbest_scope,
        missing_strategy=args.missing_imputer,
        screen_kwargs=screen_kwargs
    )

    xgb_scale_pos_weight_train = None
    if model_name == "XGBoost":
        xgb_scale_pos_weight_train = resolve_xgb_scale_pos_weight(y_tr, args)
        try:
            pipe.set_params(clf__scale_pos_weight=float(xgb_scale_pos_weight_train))
        except Exception:
            pass

    # ------------------
    # Resume checkpoint: skip training if model artifact exists
    # ------------------
    model_artifact_path = os.path.join(OUT["core"], f"{scen_name}__{model_label}__final_model.joblib")
    resumed_from_checkpoint = False

    if getattr(args, "resume", False) and os.path.exists(model_artifact_path):
        logprint(f"[resume] Found existing model artifact: {model_artifact_path}")
        try:
            bundle = joblib.load(model_artifact_path)
            final_model = bundle["model_raw"]
            export_model = bundle["model"]
            prot_cols_use = bundle.get("prot_cols", prot_cols_use)

            # Restore thresholds from saved bundle
            thr_data = bundle.get("thresholds", {})
            thr_obj_name = thr_data.get("objective_name", "max_f1")
            thr_obj = float(thr_data.get("objective", 0.5))
            thr_f1 = float(thr_data.get("max_f1", 0.5))
            thr_spec90 = float(thr_data.get("spec90", 0.5))
            thr_ctrl_specs = {float(k): float(v) for k, v in thr_data.get("control_specs", {}).items()}

            # Restore prevalence info
            prev_data = bundle.get("prevalence", {})
            train_prevalence = float(prev_data.get("train_sample", train_prevalence))

            # Set placeholder values for training metrics (not available in resume mode)
            best_score_train = np.nan
            best_params_train = bundle.get("args", {}).get("best_params_inner_train", {})
            if isinstance(best_params_train, str):
                try:
                    best_params_train = json.loads(best_params_train)
                except Exception:
                    best_params_train = {}

            # Check if we need to update prot_cols_use for X_te
            num_cols_use = META_NUM_COLS + prot_cols_use
            if X_val0 is not None:
                X_val = X_val0[num_cols_use + CAT_COLS].copy()
            X_te = X_te0[num_cols_use + CAT_COLS].copy()

            resumed_from_checkpoint = True
            logprint(f"[resume] Successfully loaded model, skipping CV and training.")
            logprint(f"[resume] Restored thresholds: objective={thr_obj_name}={thr_obj:.4f}, f1={thr_f1:.4f}, spec90={thr_spec90:.4f}")
        except Exception as e:
            logprint(f"[resume] Failed to load checkpoint ({e}), proceeding with full training.")
            resumed_from_checkpoint = False

    if not resumed_from_checkpoint:
        # ------------------
        # CV (nested tuning) + OOF
        # ------------------
        preds_by_rep, cv_sec, df_best, df_sel = oof_by_repeat_tuned(
            pipe, model_name, X_tr, y_tr,
            prot_cols=prot_cols_use,
            n_splits=args.folds, n_repeats=args.repeats, random_state=seed,
            scoring=args.scoring, inner_folds=args.inner_folds, n_iter=args.n_iter,
            feature_select=args.feature_select, k_grid=k_grid,
            kbest_scope=args.kbest_scope,
            cpus=args.cpus,
            coef_thresh=args.coef_thresh,
            perm_top_n=args.perm_top_n,
            perm_repeats=args.perm_repeats,
            perm_min_importance=args.perm_min_importance,
            args=args,
            grid_rng=grid_rng,
        )
        p_oof_mean_raw = np.clip(preds_by_rep.mean(axis=0), 0.0, 1.0)
        p_oof_mean_adjusted = adjust_probabilities_for_prevalence(
            p_oof_mean_raw,
            train_prevalence,
            target_prevalence,
        )
        p_oof_mean_active = p_oof_mean_adjusted if use_adjusted_probs else p_oof_mean_raw

        # threshold objective
        thr_obj_name, thr_obj = choose_threshold_objective(
            y_tr,
            p_oof_mean_active,
            objective=args.threshold_objective,
            fbeta=args.fbeta,
            fixed_spec=args.fixed_spec,
            fixed_ppv=args.fixed_ppv
        )

        thr_f1 = threshold_max_f1(y_tr, p_oof_mean_active)
        thr_spec90 = threshold_for_specificity(y_tr, p_oof_mean_active, target_spec=0.90)

        p_ctrl_oof = p_oof_mean_active[y_tr == 0]
        thr_ctrl_specs = {s: threshold_from_controls(p_ctrl_oof, s) for s in spec_targets}

        # write CV tables
        df_best.insert(0, "scenario", scen_name)
        df_best.to_csv(os.path.join(OUT["cv"], "best_params_per_split.csv"), index=False)

        if len(df_sel) > 0:
            df_sel.insert(0, "scenario", scen_name)
            df_sel.to_csv(os.path.join(OUT["cv"], "selected_proteins_per_outer_split.csv"), index=False)

        # CV repeat metrics (OOF within each repeat)
        cv_rows = []
        for r in range(int(args.repeats)):
            p_raw = np.clip(preds_by_rep[r], 0.0, 1.0)
            if use_adjusted_probs:
                p_for_metrics = adjust_probabilities_for_prevalence(
                    p_raw,
                    train_prevalence,
                    target_prevalence,
                )
            else:
                p_for_metrics = p_raw
            mm = prob_metrics(y_tr, p_for_metrics)
            cv_rows.append({
                "scenario": scen_name,
                "model": model_label,
                "split_id": split_id,
                "repeat": r,
                "folds": args.folds,
                "repeats": args.repeats,
                "n_train": int(len(y_tr)),
                "n_train_pos": int(y_tr.sum()),
                "n_test": int(len(y_te)),
                "n_test_pos": int(y_te.sum()),
                "n_proteins": int(len(prot_cols_use)),
                "AUROC_oof": float(mm["AUROC"]),
                "PR_AUC_oof": float(mm["PR_AUC"]),
                "Brier_oof": float(mm["Brier"]),
                "cv_seconds": float(cv_sec),
                "tuned": True,
                "inner_scoring": args.scoring,
                "inner_folds": int(args.inner_folds),
                "n_iter": int(args.n_iter),
                "feature_select": args.feature_select,
                "kbest_scope": args.kbest_scope,
                "screen_method": args.screen_method if args.feature_select == "hybrid" else "",
                "screen_top_n": int(args.screen_top_n) if args.feature_select == "hybrid" else 0,
                "var_prefilter": bool(args.var_prefilter),
                "min_nonmissing": float(args.min_nonmissing) if args.var_prefilter else np.nan,
                "min_var": float(args.min_var) if args.var_prefilter else np.nan,
                "random_state": int(seed),
                "tune_n_jobs": str(args.tune_n_jobs),
                "error_score": str(args.error_score),
                "missing_imputer": str(args.missing_imputer),
                "calibrate_final_models": int(args.calibrate_final_models),
            })
        pd.DataFrame(cv_rows).to_csv(os.path.join(OUT["cv"], "cv_repeat_metrics.csv"), index=False)

        # ------------------
        # Final fit on TRAIN (tuned) + TEST eval
        # ------------------
        t_fit0 = time.perf_counter()
        search = build_search(
            pipe, model_name, scoring=args.scoring, inner_folds=args.inner_folds,
            n_iter=args.n_iter, random_state=seed,
            feature_select=args.feature_select, k_grid=k_grid,
            kbest_scope=args.kbest_scope,
            cpus=args.cpus,
            args=args,
            xgb_scale_pos_weight=xgb_scale_pos_weight_train,
            grid_rng=grid_rng,
            randomize_grids=bool(getattr(args, "grid_randomize", False)),
        )
        if search is None:
            final_model = clone(pipe)
            final_model = _maybe_set_screen_diag(final_model, screen_diag_csv)
            final_model.fit(X_tr, y_tr)
            best_params_train, best_score_train = {}, np.nan
        else:
            if getattr(search, 'n_jobs', 1) and int(search.n_jobs) > 1:
                with parallel_backend('loky', inner_max_num_threads=1):
                    search.fit(X_tr, y_tr)
            else:
                search.fit(X_tr, y_tr)
            best_params_train = search.best_params_
            best_score_train = float(search.best_score_)
            best_estimator = search.best_estimator_

            # Save hyperparameter tuning history plot
            try:
                tuning_plot_path = os.path.join(OUT["diag"], "plots", f"{scen_name}__{model_name}__tuning_history.png")
                os.makedirs(os.path.dirname(tuning_plot_path), exist_ok=True)
                tuning_meta = [
                    f"Scenario={scen_name} | Model={model_label} | Split=TRAIN (tuning) | split_id={split_id}",
                    _format_split_meta("Train", int(len(y_tr)), int(y_tr.sum())),
                    f"inner_folds={int(args.inner_folds)} | n_iter={int(args.n_iter)} | scoring={args.scoring}",
                ]
                _plot_hyperparameter_tuning_history(
                    search.cv_results_,
                    tuning_plot_path,
                    model_name=model_label,
                    scoring=args.scoring,
                    meta_lines=tuning_meta,
                )
                logprint(f"[plot] Saved hyperparameter tuning history: {tuning_plot_path}")
            except Exception as e:
                logprint(f"[plot] WARNING: Failed to generate tuning history plot: {e}")

            final_model = clone(best_estimator)
            final_model = _maybe_set_screen_diag(final_model, screen_diag_csv)
            final_model.fit(X_tr, y_tr)
        t_fit1 = time.perf_counter()

        if screen_diag_csv and isinstance(final_model, Pipeline) and "screen" in final_model.named_steps:
            selected_screen = getattr(final_model.named_steps["screen"], "selected_proteins_", [])
            logprint(f"[screen] final screening kept {len(selected_screen)} proteins (report: {screen_diag_csv})")

        # Optional calibration for LR/RF (SVM already calibrated)
        final_model = _maybe_calibrate_final(
            est=final_model,
            model_name=model_name,
            calibrate=bool(int(args.calibrate_final_models)),
            method=str(args.calibration_method),
            cv=int(args.calibration_cv),
            random_state=int(seed)
        )
        if isinstance(final_model, CalibratedClassifierCV) and not hasattr(final_model, "classes_"):
            final_model.fit(X_tr, y_tr)

    p_val_raw = None
    p_val_adjusted = None
    p_val_active = None
    if X_val is not None:
        p_val_raw = np.clip(final_model.predict_proba(X_val)[:, 1], 0.0, 1.0)
        p_val_adjusted = adjust_probabilities_for_prevalence(
            p_val_raw,
            train_prevalence,
            target_prevalence,
        )
        p_val_active = p_val_adjusted if use_adjusted_probs else p_val_raw

        if threshold_source == "val" and len(y_val) > 0:
            logprint("[thresholds] Selecting thresholds on VAL predictions.")
            thr_obj_name, thr_obj = choose_threshold_objective(
                y_val,
                p_val_active,
                objective=args.threshold_objective,
                fbeta=args.fbeta,
                fixed_spec=args.fixed_spec,
                fixed_ppv=args.fixed_ppv
            )
            thr_f1 = threshold_max_f1(y_val, p_val_active)
            thr_spec90 = threshold_for_specificity(y_val, p_val_active, target_spec=0.90)
            p_ctrl_val = p_val_active[y_val == 0]
            thr_ctrl_specs = {s: threshold_from_controls(p_ctrl_val, s) for s in spec_targets}
    elif threshold_source == "val":
        logprint("[thresholds] WARNING: VAL split not available; using TRAIN OOF thresholds.")

    p_test_raw = np.clip(final_model.predict_proba(X_te)[:, 1], 0.0, 1.0)
    prevalence_wrapper = PrevalenceAdjustedModel(
        base_model=final_model,
        sample_prevalence=train_prevalence,
        target_prevalence=target_prevalence
    )
    p_test_adjusted = np.asarray(prevalence_wrapper.predict_proba(X_te), dtype=float)[:, 1]
    export_model = prevalence_wrapper if use_adjusted_probs else final_model
    p_test_active = p_test_adjusted if use_adjusted_probs else p_test_raw

    if p_val_active is not None and len(y_val) > 0:
        m_obj_val = binary_metrics_at_threshold(y_val, p_val_active, thr_obj)
        m_f1_val = binary_metrics_at_threshold(y_val, p_val_active, thr_f1)
        m_90_val = binary_metrics_at_threshold(y_val, p_val_active, thr_spec90)
        m_ctrl_val = {s: binary_metrics_at_threshold(y_val, p_val_active, thr_ctrl_specs[s]) for s in spec_targets}

        mt_val = prob_metrics(y_val, p_val_active)
        cal_a_val, cal_b_val = calibration_intercept_slope(y_val, p_val_active)

        val_row = {
            "scenario": scen_name,
            "model": model_label,
            "split_id": split_id,
            "val_size": float(actual_val_frac),
            "n_train": int(len(y_tr)),
            "n_train_pos": int(y_tr.sum()),
            "n_val": int(len(y_val)),
            "n_val_pos": int(y_val.sum()),
            "n_proteins": int(len(prot_cols_use)),
            "AUROC_val": float(mt_val["AUROC"]),
            "PR_AUC_val": float(mt_val["PR_AUC"]),
            "Brier_val": float(mt_val["Brier"]),
            "tuned": True,
            "feature_select": args.feature_select,
            "kbest_scope": args.kbest_scope,
            "screen_method": args.screen_method if args.feature_select == "hybrid" else "",
            "screen_top_n": int(args.screen_top_n) if args.feature_select == "hybrid" else 0,
            "var_prefilter": bool(args.var_prefilter),
            "best_score_inner_train": best_score_train,
            "best_params_inner_train": json.dumps(best_params_train, sort_keys=True),
            "threshold_source": str(threshold_source),
            "target_prevalence_source": str(target_prev_source),

            # objective threshold
            "thr_train_oof_objective_name": str(thr_obj_name),
            "thr_train_oof_objective": float(thr_obj),
            "precision_val_at_thr_objective": float(m_obj_val["precision"]),
            "recall_val_at_thr_objective": float(m_obj_val["recall"]),
            "f1_val_at_thr_objective": float(m_obj_val["f1"]),
            "specificity_val_at_thr_objective": float(m_obj_val["specificity"]),
            "fp_val_at_thr_objective": int(m_obj_val["fp"]),

            "thr_train_oof_maxF1": float(thr_f1),
            "precision_val_at_thr_maxF1": float(m_f1_val["precision"]),
            "recall_val_at_thr_maxF1": float(m_f1_val["recall"]),
            "f1_val_at_thr_maxF1": float(m_f1_val["f1"]),
            "specificity_val_at_thr_maxF1": float(m_f1_val["specificity"]),
            "fp_val_at_thr_maxF1": int(m_f1_val["fp"]),

            "thr_train_oof_spec90": float(thr_spec90),
            "sensitivity_val_at_spec90": float(m_90_val["recall"]),
            "specificity_val_at_spec90": float(m_90_val["specificity"]),
            "fp_val_at_spec90": int(m_90_val["fp"]),

            "calibration_intercept_val": float(cal_a_val) if np.isfinite(cal_a_val) else np.nan,
            "calibration_slope_val": float(cal_b_val) if np.isfinite(cal_b_val) else np.nan,

            "risk_prob_source": str(args.risk_prob_source),
            "random_state": int(seed),
            "tune_n_jobs": str(args.tune_n_jobs),
            "error_score": str(args.error_score),
            "missing_imputer": str(args.missing_imputer),
            "calibrate_final_models": int(args.calibrate_final_models),
            "threshold_objective": str(args.threshold_objective),
            "fbeta": float(args.fbeta),
            "fixed_spec": float(args.fixed_spec),
            "fixed_ppv": float(args.fixed_ppv),
        }

        for s in spec_targets:
            tag = f"spec{str(s).replace('0.', '')}"
            val_row[f"thr_train_oof_{tag}_ctrl"] = float(thr_ctrl_specs[s])
            val_row[f"precision_val_at_{tag}_ctrl"] = float(m_ctrl_val[s]["precision"])
            val_row[f"recall_val_at_{tag}_ctrl"] = float(m_ctrl_val[s]["recall"])
            val_row[f"f1_val_at_{tag}_ctrl"] = float(m_ctrl_val[s]["f1"])
            val_row[f"specificity_val_at_{tag}_ctrl"] = float(m_ctrl_val[s]["specificity"])
            val_row[f"fp_val_at_{tag}_ctrl"] = int(m_ctrl_val[s]["fp"])

        for thr in clinical_thresholds:
            if not (0.0 < thr < 1.0):
                continue
            m_thr = binary_metrics_at_threshold(y_val, p_val_active, thr)
            tag = f"clin_{str(thr).replace('.', 'p')}"
            val_row[f"{tag}_threshold"] = float(thr)
            val_row[f"{tag}_precision"] = float(m_thr["precision"])
            val_row[f"{tag}_recall"] = float(m_thr["recall"])
            val_row[f"{tag}_specificity"] = float(m_thr["specificity"])
            val_row[f"{tag}_f1"] = float(m_thr["f1"])

        for frac in top_fracs:
            trc = top_risk_capture(y_val, p_val_active, frac=frac)
            key = f"top{int(round(frac*100))}pct"
            val_row[f"{key}_n"] = int(trc["n_top"])
            val_row[f"{key}_cases_in_top"] = int(trc["cases_in_top"])
            val_row[f"{key}_case_capture"] = float(trc["case_capture"]) if np.isfinite(trc["case_capture"]) else np.nan

        pd.DataFrame([val_row]).to_csv(os.path.join(OUT["core"], "val_metrics.csv"), index=False)

        if getattr(args, "save_val_preds", False):
            val_pred_path = os.path.join(OUT["preds_val"], f"{scen_name}__val_preds__{model_name}.csv")
            pd.DataFrame({
                ID_COL: df_val[ID_COL] if df_val is not None and ID_COL in df2.columns else np.arange(len(y_val)),
                "y": y_val.astype(int),
                "p_raw": p_val_raw,
                "p_adjusted": p_val_adjusted,
                "p_active": p_val_active,
            }).to_csv(val_pred_path, index=False)

    # metrics at thresholds
    m_obj = binary_metrics_at_threshold(y_te, p_test_active, thr_obj)
    m_f1  = binary_metrics_at_threshold(y_te, p_test_active, thr_f1)
    m_90  = binary_metrics_at_threshold(y_te, p_test_active, thr_spec90)
    m_ctrl = {s: binary_metrics_at_threshold(y_te, p_test_active, thr_ctrl_specs[s]) for s in spec_targets}

    mt = prob_metrics(y_te, p_test_active)
    cal_a, cal_b = calibration_intercept_slope(y_te, p_test_active)

    test_row = {
        "scenario": scen_name,
        "model": model_label,
        "split_id": split_id,
        "test_size": float(actual_test_frac),
        "n_train": int(len(y_tr)),
        "n_train_pos": int(y_tr.sum()),
        "n_test": int(len(y_te)),
        "n_test_pos": int(y_te.sum()),
        "n_proteins": int(len(prot_cols_use)),
        "AUROC_test": float(mt["AUROC"]),
        "PR_AUC_test": float(mt["PR_AUC"]),
        "Brier_test": float(mt["Brier"]),
        "tuned": True,
        "feature_select": args.feature_select,
        "kbest_scope": args.kbest_scope,
        "screen_method": args.screen_method if args.feature_select == "hybrid" else "",
        "screen_top_n": int(args.screen_top_n) if args.feature_select == "hybrid" else 0,
        "var_prefilter": bool(args.var_prefilter),
        "best_score_inner_train": best_score_train,
        "best_params_inner_train": json.dumps(best_params_train, sort_keys=True),
        "threshold_source": str(threshold_source),
        "target_prevalence_source": str(target_prev_source),

        # objective threshold
        "thr_train_oof_objective_name": str(thr_obj_name),
        "thr_train_oof_objective": float(thr_obj),
        "precision_test_at_thr_objective": float(m_obj["precision"]),
        "recall_test_at_thr_objective": float(m_obj["recall"]),
        "f1_test_at_thr_objective": float(m_obj["f1"]),
        "specificity_test_at_thr_objective": float(m_obj["specificity"]),
        "fp_test_at_thr_objective": int(m_obj["fp"]),

        "thr_train_oof_maxF1": float(thr_f1),
        "precision_test_at_thr_maxF1": float(m_f1["precision"]),
        "recall_test_at_thr_maxF1": float(m_f1["recall"]),
        "f1_test_at_thr_maxF1": float(m_f1["f1"]),
        "specificity_test_at_thr_maxF1": float(m_f1["specificity"]),
        "fp_test_at_thr_maxF1": int(m_f1["fp"]),

        "thr_train_oof_spec90": float(thr_spec90),
        "sensitivity_test_at_spec90": float(m_90["recall"]),
        "specificity_test_at_spec90": float(m_90["specificity"]),
        "fp_test_at_spec90": int(m_90["fp"]),

        "calibration_intercept_test": float(cal_a) if np.isfinite(cal_a) else np.nan,
        "calibration_slope_test": float(cal_b) if np.isfinite(cal_b) else np.nan,

        "cv_seconds": float(cv_sec),
        "final_fit_seconds": float(t_fit1 - t_fit0),
        "n_proteins_used": int(len(prot_cols_use)),
        "risk_prob_source": str(args.risk_prob_source),

        # run meta
        "random_state": int(seed),
        "tune_n_jobs": str(args.tune_n_jobs),
        "error_score": str(args.error_score),
        "missing_imputer": str(args.missing_imputer),
        "calibrate_final_models": int(args.calibrate_final_models),
        "threshold_objective": str(args.threshold_objective),
        "fbeta": float(args.fbeta),
        "fixed_spec": float(args.fixed_spec),
        "fixed_ppv": float(args.fixed_ppv),
    }

    for s in spec_targets:
        tag = f"spec{str(s).replace('0.', '')}"
        test_row[f"thr_train_oof_{tag}_ctrl"] = float(thr_ctrl_specs[s])
        test_row[f"precision_test_at_{tag}_ctrl"] = float(m_ctrl[s]["precision"])
        test_row[f"recall_test_at_{tag}_ctrl"] = float(m_ctrl[s]["recall"])
        test_row[f"f1_test_at_{tag}_ctrl"] = float(m_ctrl[s]["f1"])
        test_row[f"specificity_test_at_{tag}_ctrl"] = float(m_ctrl[s]["specificity"])
        test_row[f"fp_test_at_{tag}_ctrl"] = int(m_ctrl[s]["fp"])

    for thr in clinical_thresholds:
        if not (0.0 < thr < 1.0):
            continue
        m_thr = binary_metrics_at_threshold(y_te, p_test_active, thr)
        tag = f"clin_{str(thr).replace('.', 'p')}"
        test_row[f"{tag}_threshold"] = float(thr)
        test_row[f"{tag}_precision"] = float(m_thr["precision"])
        test_row[f"{tag}_recall"] = float(m_thr["recall"])
        test_row[f"{tag}_specificity"] = float(m_thr["specificity"])
        test_row[f"{tag}_f1"] = float(m_thr["f1"])

    for frac in top_fracs:
        trc = top_risk_capture(y_te, p_test_active, frac=frac)
        key = f"top{int(round(frac*100))}pct"
        test_row[f"{key}_n"] = int(trc["n_top"])
        test_row[f"{key}_cases_in_top"] = int(trc["cases_in_top"])
        test_row[f"{key}_case_capture"] = float(trc["case_capture"]) if np.isfinite(trc["case_capture"]) else np.nan

    if args.test_ci_bootstrap:
        ci_auc, ci_pr, ci_br = compute_test_cis(y_te, p_test_active, n_boot=args.n_boot, seed=seed)
        test_row["AUROC_test_95CI"] = format_ci(ci_auc[0], ci_auc[1], decimals=3)
        test_row["PR_AUC_test_95CI"] = format_ci(ci_pr[0], ci_pr[1], decimals=3)
        test_row["Brier_test_95CI"] = format_ci(ci_br[0], ci_br[1], decimals=4)

        ci_a = stratified_bootstrap_ci(y_te, p_test_active, calib_intercept_metric, n_boot=args.n_boot, seed=seed)
        ci_b = stratified_bootstrap_ci(y_te, p_test_active, calib_slope_metric, n_boot=args.n_boot, seed=seed)
        test_row["calibration_intercept_test_95CI"] = format_ci(ci_a[0], ci_a[1], decimals=3)
        test_row["calibration_slope_test_95CI"] = format_ci(ci_b[0], ci_b[1], decimals=3)

        if args.write_test_ci_files:
            ci_csv = os.path.join(OUT["diag_ci"], f"{scen_name}__test_CI__{model_name}.csv")
            pd.DataFrame([{
                "scenario": scen_name,
                "model": model_label,
                "AUROC_test": float(mt["AUROC"]),
                "AUROC_test_95CI": test_row.get("AUROC_test_95CI", ""),
                "PR_AUC_test": float(mt["PR_AUC"]),
                "PR_AUC_test_95CI": test_row.get("PR_AUC_test_95CI", ""),
                "Brier_test": float(mt["Brier"]),
                "Brier_test_95CI": test_row.get("Brier_test_95CI", ""),
                "calibration_intercept_test": test_row.get("calibration_intercept_test", np.nan),
                "calibration_intercept_test_95CI": test_row.get("calibration_intercept_test_95CI", ""),
                "calibration_slope_test": test_row.get("calibration_slope_test", np.nan),
                "calibration_slope_test_95CI": test_row.get("calibration_slope_test_95CI", ""),
                "n_boot": int(args.n_boot),
            }]).to_csv(ci_csv, index=False)

    pd.DataFrame([test_row]).to_csv(os.path.join(OUT["core"], "test_metrics.csv"), index=False)

    # preds exports
    if args.save_test_preds:
        out_te = pd.DataFrame({
            ID_COL: df_te[ID_COL].to_numpy() if ID_COL in df_te.columns else np.arange(len(df_te)),
            TARGET_COL: df_te[TARGET_COL].astype(str).to_numpy(),
            "y_true": y_te.astype(int),
            "risk_test": p_test_active,
            "risk_test_pct": 100.0 * p_test_active,
            "risk_test_adjusted": p_test_adjusted,
            "risk_test_adjusted_pct": 100.0 * p_test_adjusted,
            "risk_test_raw": p_test_raw,
            "risk_test_raw_pct": 100.0 * p_test_raw,
            "yhat_test_thr_objective": (p_test_active >= thr_obj).astype(int),
            "yhat_test_thr_maxF1": (p_test_active >= thr_f1).astype(int),
            "yhat_test_thr_spec90": (p_test_active >= thr_spec90).astype(int),
        })
        for s in spec_targets:
            tag = f"spec{str(s).replace('0.', '')}"
            out_te[f"yhat_test_thr_{tag}_ctrl"] = (p_test_active >= thr_ctrl_specs[s]).astype(int)
        out_te.to_csv(os.path.join(OUT["preds_test"], f"{scen_name}__test_preds__{model_label}.csv"), index=False)

    if args.save_train_oof:
        p_oof_active = locals().get("p_oof_mean_active")
        p_oof_raw = locals().get("p_oof_mean_raw")
        p_oof_adj = locals().get("p_oof_mean_adjusted")
        if p_oof_active is None:
            logprint("[WARN] TRAIN OOF predictions unavailable; skipping --save_train_oof export.")
        else:
            p_oof_raw = p_oof_raw if p_oof_raw is not None else np.full_like(p_oof_active, np.nan, dtype=float)
            p_oof_adj = p_oof_adj if p_oof_adj is not None else np.full_like(p_oof_active, np.nan, dtype=float)
            out_train = pd.DataFrame({
                ID_COL: df_tr[ID_COL].to_numpy() if ID_COL in df_tr.columns else np.arange(len(df_tr)),
                TARGET_COL: df_tr[TARGET_COL].astype(str).to_numpy(),
                "y_true": y_tr.astype(int),
                "risk_train_oof": p_oof_active,
                "risk_train_oof_pct": 100.0 * p_oof_active,
                "risk_train_oof_adjusted": p_oof_adj,
                "risk_train_oof_adjusted_pct": 100.0 * p_oof_adj,
                "risk_train_oof_raw": p_oof_raw,
                "risk_train_oof_raw_pct": 100.0 * p_oof_raw,
                "split_id": split_id,
            })
            out_train.to_csv(os.path.join(OUT["preds_train_oof"], f"{scen_name}__train_oof__{model_label}.csv"), index=False)

    if args.save_controls_oof:
        controls_mask = (df_tr[TARGET_COL] == "Controls").to_numpy()
        out_risk = pd.DataFrame({
            ID_COL: df_tr.loc[controls_mask, ID_COL].to_numpy() if ID_COL in df_tr.columns else np.arange(int(controls_mask.sum())),
            f"risk_{model_name}_oof_mean": p_oof_mean_active[controls_mask],
            f"risk_{model_name}_oof_mean_pct": 100.0 * p_oof_mean_active[controls_mask],
            f"risk_{model_name}_oof_mean_raw": p_oof_mean_raw[controls_mask],
            f"risk_{model_name}_oof_mean_raw_pct": 100.0 * p_oof_mean_raw[controls_mask],
            f"risk_{model_name}_oof_mean_adjusted": p_oof_mean_adjusted[controls_mask],
            f"risk_{model_name}_oof_mean_adjusted_pct": 100.0 * p_oof_mean_adjusted[controls_mask],
        })
        out_risk.to_csv(os.path.join(OUT["preds_controls"], f"{scen_name}__controls_risk__{model_label}__oof_mean.csv"), index=False)

    # calibration / LC exports (kept)
    if args.save_calibration:
        save_calibration_csv(
            y_te, p_test_active,
            os.path.join(OUT["diag_calib"], f"{scen_name}__calibration__{model_name}.csv"),
            n_bins=args.calib_bins
        )

    # ========================================
    # PER-SPLIT DIAGNOSTIC PLOTS
    # ========================================
    plots_dir = os.path.join(OUT["diag"], "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_meta = _build_split_metadata_lines(
        scen_name=scen_name,
        model_label=model_label,
        split_label="TEST",
        split_id=str(split_id),
        n_train=int(len(y_tr)),
        n_val=int(len(y_val)),
        n_test=int(len(y_te)),
        n_train_pos=int(y_tr.sum()),
        n_val_pos=int(y_val.sum()) if len(y_val) > 0 else 0,
        n_test_pos=int(y_te.sum()),
        risk_prob_source=str(risk_prob_source),
        threshold_source=str(threshold_source),
    )

    # ROC Curve
    try:
        roc_path = os.path.join(plots_dir, f"{scen_name}__{model_name}__roc_curve.png")
        _plot_roc_curve(y_te, p_test_active, roc_path, title=f"ROC Curve - {model_label}",
                       meta_lines=plot_meta)
        logprint(f"[plot] Saved ROC curve: {roc_path}")
    except Exception as e:
        logprint(f"[plot] WARNING: Failed to generate ROC curve: {e}")

    # PR Curve
    try:
        pr_path = os.path.join(plots_dir, f"{scen_name}__{model_name}__pr_curve.png")
        _plot_pr_curve(y_te, p_test_active, pr_path, title=f"PR Curve - {model_label}",
                      meta_lines=plot_meta)
        logprint(f"[plot] Saved PR curve: {pr_path}")
    except Exception as e:
        logprint(f"[plot] WARNING: Failed to generate PR curve: {e}")

    # TEST Calibration Curve (4-panel) - generate raw and adjusted versions
    try:
        # Calculate recalibration metrics (locally for this run)
        calib_intercept_test, calib_slope_test = _compute_recalibration(y_te, p_test_active)

        calib_meta = plot_meta + [f"calib_bins={args.calib_bins}"]

        # Generate 4-panel version for raw predictions
        calib_path_test_4panel = os.path.join(plots_dir, f"{scen_name}__{model_name}__TEST_calibration_raw.png")
        _plot_calibration_curve(y_te, p_test_active, calib_path_test_4panel,
                                title=f"Calibration (TEST) - {scen_name}",
                                subtitle=f"{model_label}",
                                n_bins=args.calib_bins,
                                calib_intercept=calib_intercept_test,
                                calib_slope=calib_slope_test,
                                meta_lines=calib_meta,
                                bin_strategy="uniform",
                                four_panel=True)
        logprint(f"[plot] Saved TEST 4-panel calibration curve (raw): {calib_path_test_4panel}")

        # Generate 4-panel version for adjusted predictions
        if p_test_adjusted is not None:
            calib_path_test_4panel_adj = os.path.join(plots_dir, f"{scen_name}__{model_name}__TEST_calibration_adj.png")
            _plot_calibration_curve(y_te, p_test_adjusted, calib_path_test_4panel_adj,
                                    title=f"Calibration (TEST, adjusted) - {scen_name}",
                                    subtitle=f"{model_label}",
                                    n_bins=args.calib_bins,
                                    calib_intercept=calib_intercept_test,
                                    calib_slope=calib_slope_test,
                                    meta_lines=calib_meta,
                                    bin_strategy="uniform",
                                    four_panel=True)
            logprint(f"[plot] Saved TEST 4-panel calibration curve (adjusted): {calib_path_test_4panel_adj}")

    except Exception as e:
        logprint(f"[plot] WARNING: Failed to generate TEST calibration plots: {e}")

    # Survival plots (Kaplan-Meier + Cox)
    if args.survival_plots:
        try:
            surv_summary = save_survival_plots(
                df_te=df_te,
                y_te=y_te,
                risk_scores=p_test_active,
                out_dir=OUT["diag_survival"],
                prefix=f"{scen_name}__{model_name}__",
                args=args,
            )
            for warn in surv_summary.get("warnings", []):
                logprint(f"[survival] WARNING: {warn}", level="warning")
            if "error" in surv_summary:
                logprint(f"[survival] WARNING: {surv_summary['error']}", level="warning")
            if surv_summary.get("kaplan_meier_path"):
                logprint(f"[plot] Saved Kaplan-Meier curve: {surv_summary['kaplan_meier_path']}")
            if surv_summary.get("cox_plot_path"):
                logprint(f"[plot] Saved Cox hazard ratios: {surv_summary['cox_plot_path']}")
        except Exception as e:
            logprint(f"[survival] WARNING: Failed to generate survival plots: {e}")

    # Decision Curve Analysis (DCA) - compute for individual run
    # Enables DCA thresholds in risk plots and provides backup for postprocessing
    if args.compute_dca:
        try:
            dca_dir = os.path.join(OUT["diag"], "dca")
            os.makedirs(dca_dir, exist_ok=True)

            # Parse DCA report points from command line
            dca_report_pts = [float(x.strip()) for x in args.dca_report_points.split(",") if x.strip()]

            # Generate DCA thresholds using configured range (wide range: 0.05% to 100%)
            dca_thresholds_wide = _dca_thresholds(
                args.dca_threshold_min,
                args.dca_threshold_max,
                args.dca_threshold_step
            )

            dca_meta = [
                f"Scenario={scen_name} | Model={model_name} | Split=TEST",
                f"n={len(y_te)} pos={int(y_te.sum())} prev={np.mean(y_te):.4f}" if len(y_te) > 0 else "",
            ]
            dca_summary = save_dca_results(
                y_te,
                p_test_active,
                out_dir=dca_dir,
                prefix=f"{scen_name}__{model_name}__",
                thresholds=dca_thresholds_wide,  # Use wide range (0.0005 to 1.0)
                report_points=dca_report_pts,
                meta_lines=dca_meta,
                plot_dir=os.path.join(OUT["diag"], "plots"),
            )
            if dca_summary.get("dca_computed", False):
                logprint(f"[DCA] Computed: {dca_summary.get('dca_csv_path', '?')}")
                if dca_summary.get("dca_plot_path"):
                    logprint(f"[DCA] Saved plot: {dca_summary.get('dca_plot_path')}")
            else:
                logprint("[DCA] WARNING: DCA computation failed")
        except Exception as e:
            logprint(f"[DCA] WARNING: Failed to compute DCA: {e}")
    else:
        logprint("[DCA] Skipped (--compute_dca=0)")

    # Risk score distributions (preds/plots)
    preds_plot_dir = OUT["preds_plots"]
    pos_label = "Incident" if scen_name == "IncidentOnly" else "Incident/Prevalent"
    case_label = pos_label.lower()

    def _auto_xlim(scores: Optional[np.ndarray]) -> Optional[Tuple[float, float]]:
        if scores is None:
            return None
        max_score = float(np.nanmax(scores)) if len(scores) else np.nan
        if not np.isfinite(max_score) or max_score <= 0:
            return None
        return (0.0, min(1.0, max_score * 1.05))

    # Extract thresholds and metrics for enhanced plotting
    dca_thr_test = None
    spec95_thr_test = None
    youden_thr_test = None
    alpha_thr_test = None
    metrics_at_thr_test = None
    test_category_col = None

    if ENHANCED_PLOTS_AVAILABLE:
        # Try to load DCA threshold (zero-crossing) - may not exist if DCA disabled during training
        dca_dir = os.path.join(OUT["diag"], "dca")
        dca_curve_path = os.path.join(dca_dir, f"{scen_name}__{model_name}__dca_curve.csv")
        if os.path.exists(dca_curve_path):
            dca_thr_test = find_dca_zero_crossing(dca_curve_path)
            if dca_thr_test is not None:
                logprint(f"[plot] DCA zero-crossing threshold: {dca_thr_test:.4f}")
            else:
                logprint("[plot] DCA curve exists but no zero-crossing in threshold range (model benefit always positive/negative)")
                dca_thr_test = None  # Risk plot won't show DCA line
        else:
            logprint("[plot] DCA curve not found (will be generated by postprocess_compare.py)")
            dca_thr_test = None  # Graceful fallback

        # Load spec threshold from test_metrics.csv (configurable via --risk_plot_spec, default 0.95)
        # Extract from test_row which was populated earlier
        risk_plot_spec = getattr(args, 'risk_plot_spec', 0.95)
        # Map target specificity to column key (e.g., 0.95 -> spec95_ctrl, 0.99 -> spec99_ctrl, etc.)
        spec_int = int(round(risk_plot_spec * 100))
        spec_key = f"thr_train_oof_spec{spec_int}_ctrl"
        if spec_key in test_row:
            spec95_thr_test = test_row[spec_key]

        # Extract metrics at thresholds from test_row
        if spec95_thr_test is not None:
            spec_label = f"spec{spec_int}"
            metrics_at_thr_test = {
                spec_label: {
                    'sensitivity': test_row.get(f'recall_test_at_spec{spec_int}_ctrl', np.nan),
                    'precision': test_row.get(f'precision_test_at_spec{spec_int}_ctrl', np.nan),
                    'fp': test_row.get(f'fp_test_at_spec{spec_int}_ctrl', np.nan),
                    'n_celiac': int(y_te.sum()),
                }
            }

        # If DCA threshold exists, extract metrics at that threshold
        if dca_thr_test is not None:
            m_dca = binary_metrics_at_threshold(y_te, p_test_active, dca_thr_test)
            if metrics_at_thr_test is None:
                metrics_at_thr_test = {}
            metrics_at_thr_test['dca'] = {
                'sensitivity': m_dca['recall'],
                'precision': m_dca['precision'],
                'fp': m_dca['fp'],
                'n_celiac': int(y_te.sum()),
            }

        # Compute Youden threshold
        youden_thr_test = threshold_youden(y_te, p_test_active)
        if metrics_at_thr_test is None:
            metrics_at_thr_test = {}
        m_youden = binary_metrics_at_threshold(y_te, p_test_active, youden_thr_test)
        metrics_at_thr_test['youden'] = {
            'sensitivity': m_youden['recall'],
            'precision': m_youden['precision'],
            'fp': m_youden['fp'],
            'n_celiac': int(y_te.sum()),
        }

        # Compute Alpha threshold (configurable target specificity, default 90%)
        alpha_spec = getattr(args, 'alpha_specificity', 0.90)
        alpha_thr_test = threshold_for_specificity(y_te, p_test_active, target_spec=alpha_spec)
        m_alpha = binary_metrics_at_threshold(y_te, p_test_active, alpha_thr_test)
        metrics_at_thr_test['alpha'] = {
            'sensitivity': m_alpha['recall'],
            'precision': m_alpha['precision'],
            'fp': m_alpha['fp'],
            'n_celiac': int(y_te.sum()),
        }

        # Create category column for incident/prevalent split if available
        if TARGET_COL in df_te.columns:
            test_category_col = df_te[TARGET_COL].to_numpy()
            # Map to standard labels
            test_category_col = np.array([
                'Incident' if x == 'Incident' else ('Prevalent' if x == 'Prevalent' else 'Controls')
                for x in test_category_col
            ])

    try:
        _plot_risk_distribution(
            y_te,
            p_test_active,
            os.path.join(preds_plot_dir, f"{scen_name}__{model_name}__TEST_risk_distribution.png"),
            title=f"Risk score distribution (TEST) - {scen_name}",
            subtitle=f"risk_test (active); controls vs {case_label} cases",
            pos_label=pos_label,
            meta_lines=plot_meta,
            category_col=test_category_col,
            dca_threshold=dca_thr_test,
            spec95_threshold=spec95_thr_test,
            youden_threshold=youden_thr_test if ENHANCED_PLOTS_AVAILABLE else None,
            alpha_threshold=alpha_thr_test if ENHANCED_PLOTS_AVAILABLE else None,
            metrics_at_thresholds=metrics_at_thr_test,
            target_spec=risk_plot_spec,
        )
    except Exception as e:
        logprint(f"[plot] WARNING: Failed to generate TEST risk distribution: {e}")

    try:
        _plot_risk_distribution(
            y_te,
            p_test_adjusted,
            os.path.join(preds_plot_dir, f"{scen_name}__{model_name}__TEST_risk_distribution_adjusted.png"),
            title=f"Risk score distribution (TEST, adjusted) - {scen_name}",
            subtitle=f"risk_test_adjusted; controls vs {case_label} cases",
            xlabel="Predicted risk (probability)",
            pos_label=pos_label,
            meta_lines=plot_meta,
            x_limits=_auto_xlim(p_test_adjusted),
            category_col=test_category_col,
            dca_threshold=dca_thr_test,
            spec95_threshold=spec95_thr_test,
            youden_threshold=youden_thr_test if ENHANCED_PLOTS_AVAILABLE else None,
            alpha_threshold=alpha_thr_test if ENHANCED_PLOTS_AVAILABLE else None,
            metrics_at_thresholds=metrics_at_thr_test,
            target_spec=risk_plot_spec,
        )
    except Exception as e:
        logprint(f"[plot] WARNING: Failed to generate TEST adjusted risk distribution: {e}")

    try:
        _plot_risk_distribution(
            y_te,
            p_test_raw,
            os.path.join(preds_plot_dir, f"{scen_name}__{model_name}__TEST_risk_distribution_raw.png"),
            title=f"Risk score distribution (TEST, raw) - {scen_name}",
            subtitle=f"risk_test_raw; controls vs {case_label} cases",
            pos_label=pos_label,
            meta_lines=plot_meta,
            category_col=test_category_col,
            dca_threshold=dca_thr_test,
            spec95_threshold=spec95_thr_test,
            youden_threshold=youden_thr_test if ENHANCED_PLOTS_AVAILABLE else None,
            alpha_threshold=alpha_thr_test if ENHANCED_PLOTS_AVAILABLE else None,
            metrics_at_thresholds=metrics_at_thr_test,
            target_spec=risk_plot_spec,
        )
    except Exception as e:
        logprint(f"[plot] WARNING: Failed to generate TEST raw risk distribution: {e}")

    if p_val_active is not None and len(y_val) > 0:
        val_plot_meta = _build_split_metadata_lines(
            scen_name=scen_name,
            model_label=model_label,
            split_label="VAL",
            split_id=str(split_id),
            n_train=int(len(y_tr)),
            n_val=int(len(y_val)),
            n_test=int(len(y_te)),
            n_train_pos=int(y_tr.sum()),
            n_val_pos=int(y_val.sum()),
            n_test_pos=int(y_te.sum()),
            risk_prob_source=str(risk_prob_source),
            threshold_source=str(threshold_source),
        )

        # Extract VAL enhanced plotting parameters
        val_category_col = None
        dca_thr_val = None
        spec95_thr_val = None
        youden_thr_val = None
        alpha_thr_val = None
        metrics_at_thr_val = None

        if ENHANCED_PLOTS_AVAILABLE:
            # Use same thresholds as TEST (thresholds are from TRAIN, applied to all splits)
            spec95_thr_val = spec95_thr_test
            dca_thr_val = dca_thr_test

            # Create category column for VAL
            if TARGET_COL in df_val.columns:
                val_category_col = df_val[TARGET_COL].to_numpy()
                val_category_col = np.array([
                    'Incident' if x == 'Incident' else ('Prevalent' if x == 'Prevalent' else 'Controls')
                    for x in val_category_col
                ])

            # Compute metrics at thresholds for VAL
            if spec95_thr_val is not None:
                spec_label = f"spec{spec_int}"
                metrics_at_thr_val = {
                    spec_label: {
                        'sensitivity': val_row.get(f'recall_val_at_spec{spec_int}_ctrl', np.nan),
                        'precision': val_row.get(f'precision_val_at_spec{spec_int}_ctrl', np.nan),
                        'fp': val_row.get(f'fp_val_at_spec{spec_int}_ctrl', np.nan),
                        'n_celiac': int(y_val.sum()),
                    }
                }

            if dca_thr_val is not None:
                m_dca_val = binary_metrics_at_threshold(y_val, p_val_active, dca_thr_val)
                if metrics_at_thr_val is None:
                    metrics_at_thr_val = {}
                metrics_at_thr_val['dca'] = {
                    'sensitivity': m_dca_val['recall'],
                    'precision': m_dca_val['precision'],
                    'fp': m_dca_val['fp'],
                    'n_celiac': int(y_val.sum()),
                }

            # Compute Youden threshold for VAL
            youden_thr_val = threshold_youden(y_val, p_val_active)
            if metrics_at_thr_val is None:
                metrics_at_thr_val = {}
            m_youden_val = binary_metrics_at_threshold(y_val, p_val_active, youden_thr_val)
            metrics_at_thr_val['youden'] = {
                'sensitivity': m_youden_val['recall'],
                'precision': m_youden_val['precision'],
                'fp': m_youden_val['fp'],
                'n_celiac': int(y_val.sum()),
            }

            # Compute Alpha threshold for VAL
            alpha_spec = getattr(args, 'alpha_specificity', 0.90)
            alpha_thr_val = threshold_for_specificity(y_val, p_val_active, target_spec=alpha_spec)
            m_alpha_val = binary_metrics_at_threshold(y_val, p_val_active, alpha_thr_val)
            metrics_at_thr_val['alpha'] = {
                'sensitivity': m_alpha_val['recall'],
                'precision': m_alpha_val['precision'],
                'fp': m_alpha_val['fp'],
                'n_celiac': int(y_val.sum()),
            }

        try:
            _plot_risk_distribution(
                y_val,
                p_val_active,
                os.path.join(preds_plot_dir, f"{scen_name}__{model_name}__VAL_risk_distribution.png"),
                title=f"Risk score distribution (VAL) - {scen_name}",
                subtitle=f"p_active; controls vs {case_label} cases",
                pos_label=pos_label,
                meta_lines=val_plot_meta,
                category_col=val_category_col,
                dca_threshold=dca_thr_val,
                spec95_threshold=spec95_thr_val,
                youden_threshold=youden_thr_val if ENHANCED_PLOTS_AVAILABLE else None,
                alpha_threshold=alpha_thr_val if ENHANCED_PLOTS_AVAILABLE else None,
                metrics_at_thresholds=metrics_at_thr_val,
                target_spec=risk_plot_spec,
            )
        except Exception as e:
            logprint(f"[plot] WARNING: Failed to generate VAL risk distribution: {e}")

        if p_val_adjusted is not None:
            try:
                _plot_risk_distribution(
                    y_val,
                    p_val_adjusted,
                    os.path.join(preds_plot_dir, f"{scen_name}__{model_name}__VAL_risk_distribution_adjusted.png"),
                    title=f"Risk score distribution (VAL, adjusted) - {scen_name}",
                    subtitle=f"p_adjusted; controls vs {case_label} cases",
                    xlabel="Predicted risk (probability)",
                    pos_label=pos_label,
                    meta_lines=val_plot_meta,
                    x_limits=_auto_xlim(p_val_adjusted),
                    category_col=val_category_col,
                    dca_threshold=dca_thr_val,
                    spec95_threshold=spec95_thr_val,
                    youden_threshold=youden_thr_val if ENHANCED_PLOTS_AVAILABLE else None,
                    alpha_threshold=alpha_thr_val if ENHANCED_PLOTS_AVAILABLE else None,
                    metrics_at_thresholds=metrics_at_thr_val,
                    target_spec=risk_plot_spec,
                )
            except Exception as e:
                logprint(f"[plot] WARNING: Failed to generate VAL adjusted risk distribution: {e}")

        if p_val_raw is not None:
            try:
                _plot_risk_distribution(
                    y_val,
                    p_val_raw,
                    os.path.join(preds_plot_dir, f"{scen_name}__{model_name}__VAL_risk_distribution_raw.png"),
                    title=f"Risk score distribution (VAL, raw) - {scen_name}",
                    subtitle=f"p_raw; controls vs {case_label} cases",
                    pos_label=pos_label,
                    meta_lines=val_plot_meta,
                    category_col=val_category_col,
                    dca_threshold=dca_thr_val,
                    spec95_threshold=spec95_thr_val,
                    youden_threshold=youden_thr_val if ENHANCED_PLOTS_AVAILABLE else None,
                    alpha_threshold=alpha_thr_val if ENHANCED_PLOTS_AVAILABLE else None,
                    metrics_at_thresholds=metrics_at_thr_val,
                    target_spec=risk_plot_spec,
                )
            except Exception as e:
                logprint(f"[plot] WARNING: Failed to generate VAL raw risk distribution: {e}")

        # VAL Calibration Plots (4-panel) - generate raw and adjusted versions
        try:
            # Use previously calculated mt_val["AUROC"] if available or just recalibrate
            calib_intercept_val, calib_slope_val = _compute_recalibration(y_val, p_val_active)

            calib_val_meta = val_plot_meta + [f"calib_bins={args.calib_bins}"]

            # 4-panel VAL calibration - raw predictions
            calib_val_4panel_path = os.path.join(plots_dir, f"{scen_name}__{model_name}__VAL_calibration_raw.png")
            _plot_calibration_curve(y_val, p_val_active, calib_val_4panel_path,
                                   title=f"Calibration (VAL) - {scen_name}",
                                   subtitle=f"{model_label}",
                                   n_bins=args.calib_bins,
                                   calib_intercept=calib_intercept_val,
                                   calib_slope=calib_slope_val,
                                   meta_lines=calib_val_meta,
                                   bin_strategy="uniform",
                                   four_panel=True)
            logprint(f"[plot] Saved VAL 4-panel calibration curve (raw): {calib_val_4panel_path}")

            # 4-panel VAL calibration - adjusted predictions
            if p_val_adjusted is not None:
                calib_val_4panel_adj_path = os.path.join(plots_dir, f"{scen_name}__{model_name}__VAL_calibration_adj.png")
                _plot_calibration_curve(y_val, p_val_adjusted, calib_val_4panel_adj_path,
                                       title=f"Calibration (VAL, adjusted) - {scen_name}",
                                       subtitle=f"{model_label}",
                                       n_bins=args.calib_bins,
                                       calib_intercept=calib_intercept_val,
                                       calib_slope=calib_slope_val,
                                       meta_lines=calib_val_meta,
                                       bin_strategy="uniform",
                                       four_panel=True)
                logprint(f"[plot] Saved VAL 4-panel calibration curve (adjusted): {calib_val_4panel_adj_path}")

        except Exception as e:
            logprint(f"[plot] WARNING: Failed to generate VAL calibration curve: {e}")

    p_oof_active = locals().get("p_oof_mean_active")
    p_oof_raw = locals().get("p_oof_mean_raw")
    p_oof_adj = locals().get("p_oof_mean_adjusted")
    if p_oof_active is not None and len(y_tr) > 0:
        train_meta = [
            f"Scenario={scen_name} | Model={model_label} | Split=TRAIN OOF",
            _format_split_meta("Train", int(len(y_tr)), int(y_tr.sum())),
        ]

        # Extract TRAIN enhanced plotting parameters
        train_category_col = None
        dca_thr_train = None
        spec95_thr_train = None
        youden_thr_train = None
        alpha_thr_train = None
        metrics_at_thr_train = None

        if ENHANCED_PLOTS_AVAILABLE:
            # Use same thresholds (from TRAIN, applied to all splits)
            spec95_thr_train = spec95_thr_test
            dca_thr_train = dca_thr_test

            # Create category column for TRAIN (includes prevalent!)
            if TARGET_COL in df_tr.columns:
                train_category_col = df_tr[TARGET_COL].to_numpy()
                train_category_col = np.array([
                    'Incident' if x == 'Incident' else ('Prevalent' if x == 'Prevalent' else 'Controls')
                    for x in train_category_col
                ])

            # Compute metrics at thresholds for TRAIN
            if spec95_thr_train is not None:
                m_spec95_train = binary_metrics_at_threshold(y_tr, p_oof_active, spec95_thr_train)
                spec_label = f"spec{spec_int}"
                metrics_at_thr_train = {
                    spec_label: {
                        'sensitivity': m_spec95_train['recall'],
                        'precision': m_spec95_train['precision'],
                        'fp': m_spec95_train['fp'],
                        'n_celiac': int(y_tr.sum()),
                    }
                }

            if dca_thr_train is not None:
                m_dca_train = binary_metrics_at_threshold(y_tr, p_oof_active, dca_thr_train)
                if metrics_at_thr_train is None:
                    metrics_at_thr_train = {}
                metrics_at_thr_train['dca'] = {
                    'sensitivity': m_dca_train['recall'],
                    'precision': m_dca_train['precision'],
                    'fp': m_dca_train['fp'],
                    'n_celiac': int(y_tr.sum()),
                }

            # Compute Youden threshold for TRAIN
            youden_thr_train = threshold_youden(y_tr, p_oof_active)
            if metrics_at_thr_train is None:
                metrics_at_thr_train = {}
            m_youden_train = binary_metrics_at_threshold(y_tr, p_oof_active, youden_thr_train)
            metrics_at_thr_train['youden'] = {
                'sensitivity': m_youden_train['recall'],
                'precision': m_youden_train['precision'],
                'fp': m_youden_train['fp'],
                'n_celiac': int(y_tr.sum()),
            }

            # Compute Alpha threshold for TRAIN
            alpha_spec = getattr(args, 'alpha_specificity', 0.90)
            alpha_thr_train = threshold_for_specificity(y_tr, p_oof_active, target_spec=alpha_spec)
            m_alpha_train = binary_metrics_at_threshold(y_tr, p_oof_active, alpha_thr_train)
            metrics_at_thr_train['alpha'] = {
                'sensitivity': m_alpha_train['recall'],
                'precision': m_alpha_train['precision'],
                'fp': m_alpha_train['fp'],
                'n_celiac': int(y_tr.sum()),
            }

        try:
            _plot_risk_distribution(
                y_tr,
                p_oof_active,
                os.path.join(preds_plot_dir, f"{scen_name}__{model_name}__TRAIN_OOF_risk_distribution.png"),
                title=f"Risk score distribution (TRAIN OOF) - {scen_name}",
                subtitle=f"oof_mean (active); controls vs {case_label} cases",
                pos_label=pos_label,
                meta_lines=train_meta,
                category_col=train_category_col,
                dca_threshold=dca_thr_train,
                spec95_threshold=spec95_thr_train,
                youden_threshold=youden_thr_train if ENHANCED_PLOTS_AVAILABLE else None,
                alpha_threshold=alpha_thr_train if ENHANCED_PLOTS_AVAILABLE else None,
                metrics_at_thresholds=metrics_at_thr_train,
                target_spec=risk_plot_spec,
            )
        except Exception as e:
            logprint(f"[plot] WARNING: Failed to generate TRAIN OOF risk distribution: {e}")

        if p_oof_adj is not None:
            try:
                _plot_risk_distribution(
                    y_tr,
                    p_oof_adj,
                    os.path.join(preds_plot_dir, f"{scen_name}__{model_name}__TRAIN_OOF_risk_distribution_adjusted.png"),
                    title=f"Risk score distribution (TRAIN OOF, adjusted) - {scen_name}",
                    subtitle=f"oof_mean_adjusted; controls vs {case_label} cases",
                    xlabel="Predicted risk (probability)",
                    pos_label=pos_label,
                    meta_lines=train_meta,
                    x_limits=_auto_xlim(p_oof_adj),
                    category_col=train_category_col,
                    dca_threshold=dca_thr_train,
                    spec95_threshold=spec95_thr_train,
                    youden_threshold=youden_thr_train if ENHANCED_PLOTS_AVAILABLE else None,
                    alpha_threshold=alpha_thr_train if ENHANCED_PLOTS_AVAILABLE else None,
                    metrics_at_thresholds=metrics_at_thr_train,
                    target_spec=risk_plot_spec,
                )
            except Exception as e:
                logprint(f"[plot] WARNING: Failed to generate TRAIN OOF adjusted distribution: {e}")

        if p_oof_raw is not None:
            try:
                _plot_risk_distribution(
                    y_tr,
                    p_oof_raw,
                    os.path.join(preds_plot_dir, f"{scen_name}__{model_name}__TRAIN_OOF_risk_distribution_raw.png"),
                    title=f"Risk score distribution (TRAIN OOF, raw) - {scen_name}",
                    subtitle=f"oof_mean_raw; controls vs {case_label} cases",
                    pos_label=pos_label,
                    meta_lines=train_meta,
                    category_col=train_category_col,
                    dca_threshold=dca_thr_train,
                    spec95_threshold=spec95_thr_train,
                    youden_threshold=youden_thr_train if ENHANCED_PLOTS_AVAILABLE else None,
                    alpha_threshold=alpha_thr_train if ENHANCED_PLOTS_AVAILABLE else None,
                    metrics_at_thresholds=metrics_at_thr_train,
                    target_spec=risk_plot_spec,
                )
            except Exception as e:
                logprint(f"[plot] WARNING: Failed to generate TRAIN OOF raw distribution: {e}")

        controls_mask = (y_tr == 0)
        n_controls = int(controls_mask.sum())
        controls_meta = [
            f"Scenario={scen_name} | Model={model_label} | Split=TRAIN OOF (controls)",
            _format_split_meta("Controls", n_controls, 0),
        ]
        try:
            _plot_risk_distribution(
                None,
                p_oof_active[controls_mask],
                os.path.join(preds_plot_dir, f"{scen_name}__{model_name}__TRAIN_OOF_controls_risk_distribution.png"),
                title=f"Control risk distribution (TRAIN OOF) - {scen_name}",
                subtitle="oof_mean (active)",
                meta_lines=controls_meta,
            )
        except Exception as e:
            logprint(f"[plot] WARNING: Failed to generate TRAIN OOF control distribution: {e}")

        if p_oof_adj is not None:
            try:
                _plot_risk_distribution(
                    None,
                    p_oof_adj[controls_mask],
                    os.path.join(preds_plot_dir, f"{scen_name}__{model_name}__TRAIN_OOF_controls_risk_distribution_adjusted.png"),
                    title=f"Control risk distribution (TRAIN OOF, adjusted) - {scen_name}",
                    subtitle="oof_mean_adjusted",
                    xlabel="Predicted risk (probability)",
                    meta_lines=controls_meta,
                    x_limits=_auto_xlim(p_oof_adj[controls_mask]),
                )
            except Exception as e:
                logprint(f"[plot] WARNING: Failed to generate TRAIN OOF adjusted control distribution: {e}")

        if p_oof_raw is not None:
            try:
                _plot_risk_distribution(
                    None,
                    p_oof_raw[controls_mask],
                    os.path.join(preds_plot_dir, f"{scen_name}__{model_name}__TRAIN_OOF_controls_risk_distribution_raw.png"),
                    title=f"Control risk distribution (TRAIN OOF, raw) - {scen_name}",
                    subtitle="oof_mean_raw",
                    meta_lines=controls_meta,
                )
            except Exception as e:
                logprint(f"[plot] WARNING: Failed to generate TRAIN OOF raw control distribution: {e}")

    subgroup_metrics_path = ""
    if int(args.subgroup_min_n) > 0:
        subgroup_metrics_path = os.path.join(OUT["reports_subgroups"], f"{scen_name}__{model_label}__test_subgroup_metrics.csv")
        subgroup_df = save_subgroup_metrics(
            df_te,
            y_te,
            p_test_active,
            group_col="Genetic ethnic grouping",
            out_csv=subgroup_metrics_path,
            min_n=int(args.subgroup_min_n),
        )
        # Subgroup DCA moved to postprocess_compare.py

    if args.learning_curve:
        lc_csv = os.path.join(OUT["diag_lc"], f"{scen_name}__learning_curve__{model_name}.csv")
        lc_plot = os.path.join(plots_dir, f"{scen_name}__learning_curve__{model_name}.png")
        lc_meta = [
            f"Scenario={scen_name} | Model={model_label} | Split=TRAIN (learning curve) | split_id={split_id}",
            _format_split_meta("Train", int(len(y_tr)), int(y_tr.sum())),
            f"cv={int(args.lc_cv)} | scoring={args.scoring}",
        ]
        save_learning_curve_csv(
            clone(final_model),
            X_tr, y_tr, lc_csv,
            scoring=args.scoring, cv=args.lc_cv,
            lc_min_frac=args.lc_min_frac, lc_points=args.lc_points, seed=seed,
            out_plot=lc_plot,
            meta_lines=lc_meta,
        )

    # feature reports (kept)
    if args.feature_reports != "none":
        sel_freq: Dict[str, float] = {}
        sel_path = os.path.join(OUT["cv"], "selected_proteins_per_outer_split.csv")
        if os.path.exists(sel_path):
            df_sel_all = pd.read_csv(sel_path)
            total_splits = df_sel_all["outer_split"].nunique()
            if total_splits > 0 and "selected_proteins_split" in df_sel_all.columns:
                counts: Dict[str, int] = {}
                for s in df_sel_all["selected_proteins_split"].tolist():
                    try:
                        for p in json.loads(s):
                            counts[p] = counts.get(p, 0) + 1
                    except Exception:
                        pass
                sel_freq = {p: c / total_splits for p, c in counts.items()}

        if sel_freq:
            proteins_ranked = sorted(sel_freq.keys(), key=lambda p: (-sel_freq[p], p))
            proteins_to_report = proteins_ranked[:min(args.feature_report_max, len(proteins_ranked))]
        else:
            proteins_to_report = rank_proteins_univariate(df_tr=df_tr, y_tr=y_tr, prot_cols=prot_cols_use, top_n=args.feature_report_max)

        if proteins_to_report:
            out_feat = os.path.join(OUT["reports_features"], f"{scen_name}__{model_name}__feature_report_train.csv")
            feature_report_train(df_tr, y_tr, proteins_to_report, sel_freq if sel_freq else None, out_feat)

    # ------------------
    # Stable panel refit (kept; corr-method forwarded)
    # ------------------
    if args.stable_panel_from_kbest and model_name == "LR_EN" and args.feature_select in ("kbest", "hybrid"):
        sel_path = os.path.join(OUT["cv"], "selected_proteins_per_outer_split.csv")
        if not os.path.exists(sel_path):
            logprint("[stable_kbest] No selection rows file; skipping.")
        else:
            df_sel_lr = pd.read_csv(sel_path)
            col_json = "selected_proteins_split"  # Always use model selection, not screening set

            df_panel, kept_prots, _ = _stable_panel_from_sel_rows(
                df_sel_lr, n_repeats=args.repeats,
                stability_thresh=args.stability_thresh,
                col_json=col_json,
                fallback_top_n=20
            )
            kept_prots = [p for p in kept_prots if p in set(prot_cols_use)]
            panel_csv = os.path.join(OUT["reports_stable"], f"{scen_name}__stable_panel__KBest__{col_json}.csv")
            df_panel.to_csv(panel_csv, index=False)

            corr_csv = os.path.join(OUT["reports_stable"], f"{scen_name}__stable_panel__KBest__{col_json}__highcorr_pairs.csv")
            save_high_corr_pairs(
                df_tr=df_panel_source,
                proteins=kept_prots,
                out_csv=corr_csv,
                corr_thresh=args.stable_corr_thresh,
                corr_method=args.panel_corr_method
            )

            comp_csv = os.path.join(OUT["reports_stable"], f"{scen_name}__stable_panel__KBest__{col_json}__corr_components.csv")
            sel_lookup = None
            if ("protein" in df_panel.columns) and ("selection_freq" in df_panel.columns):
                sel_lookup = {r["protein"]: float(r["selection_freq"]) for _, r in df_panel.iterrows() if pd.notnull(r["protein"])}

            df_comp, kept_pruned = prune_correlated_panel(
                df_tr=df_panel_source,
                y_tr=y_panel_source,
                proteins=kept_prots,
                selection_freq=sel_lookup,
                corr_thresh=args.stable_corr_thresh,
                corr_method=args.panel_corr_method,
                rep_tiebreak=args.panel_rep_tiebreak
            )
            df_comp.to_csv(comp_csv, index=False)

            pruned_list_csv = os.path.join(OUT["reports_stable"], f"{scen_name}__stable_panel__KBest__{col_json}__pruned_proteins.csv")
            pd.DataFrame({"protein": kept_pruned}).to_csv(pruned_list_csv, index=False)

            if len(kept_pruned) < len(kept_prots):
                logprint(f"[stable_kbest] pruned correlated panel: {len(kept_prots)} -> {len(kept_pruned)} (thr={args.stable_corr_thresh}, method={args.panel_corr_method})")

    # ------------------
    # General panel extraction (all models) + TRAIN-only corr prune + refill + optional refit
    # ------------------
    if args.build_panels and (not args.panel_proteins_csv):
        sel_path = os.path.join(OUT["cv"], "selected_proteins_per_outer_split.csv")
        if not os.path.exists(sel_path):
            logprint("[panels] selected_proteins_per_outer_split.csv not found; skipping panel build.")
        else:
            df_sel_all = pd.read_csv(sel_path)

            if args.panel_source == "auto":
                col_json = "screen_selected_proteins_split" if (args.feature_select == "hybrid" and "screen_selected_proteins_split" in df_sel_all.columns) else "selected_proteins_split"
            elif args.panel_source == "screen":
                col_json = "screen_selected_proteins_split"
            else:
                col_json = "selected_proteins_split"

            if col_json not in df_sel_all.columns:
                logprint(f"[panels] selection column '{col_json}' not found in {sel_path}; skipping panel build.")
            else:
                sel_freq = selection_freq_from_sel_rows(df_sel_all, col_json=col_json)
                if not sel_freq:
                    logprint("[panels] No selection frequencies computed (empty selections). Skipping panel build.")
                else:
                    ranked = sorted(sel_freq.keys(), key=lambda p: (-sel_freq[p], p))
                    panel_sizes = _parse_int_list(args.panel_sizes) or [50, 100, 200]
                    pool_limit = int(max(panel_sizes) * int(args.panel_pool_factor))
                    thr_tag = _thr_to_tag(args.panel_corr_thresh)

                    for N in panel_sizes:
                        df_raw = build_raw_panel(sel_freq, rule=args.panel_rule, N=int(N), tau=float(args.panel_freq_tau))
                        raw_csv = os.path.join(OUT["reports_panels"], f"{scen_name}__{model_name}__N{int(N)}__panel_raw_{args.panel_rule}.csv")
                        df_raw.to_csv(raw_csv, index=False)

                        df_pruned, final_panel = prune_and_refill_to_N(
                            df_tr=df_panel_source,
                            y_tr=y_panel_source,
                            ranked_proteins=ranked,
                            selection_freq=sel_freq,
                            N=int(N),
                            corr_thresh=float(args.panel_corr_thresh),
                            pool_limit=pool_limit,
                            corr_method=args.panel_corr_method,
                            rep_tiebreak=args.panel_rep_tiebreak
                        )
                        pr_csv = os.path.join(OUT["reports_panels"], f"{scen_name}__{model_name}__N{int(N)}__panel_prunedCorr{thr_tag}.csv")
                        df_pruned.to_csv(pr_csv, index=False)

                        manifest = {
                            "scenario": scen_name,
                            "base_model": model_name,
                            "panel_size_N": int(N),
                            "selection_source_col": col_json,
                            "panel_rule": args.panel_rule,
                            "panel_freq_tau": float(args.panel_freq_tau),
                            "corr_method": str(args.panel_corr_method),
                            "corr_thresh": float(args.panel_corr_thresh),
                            "rep_tiebreak": str(args.panel_rep_tiebreak),
                            "pool_limit": int(pool_limit),
                            "seed": int(seed),
                            "split_id": str(split_id),
                        }
                        man_json = os.path.join(OUT["reports_panels"], f"{scen_name}__{model_name}__N{int(N)}__panel_manifest.json")
                        with open(man_json, "w") as f:
                            json.dump(manifest, f, indent=2, sort_keys=True)

                        # Panel refit optional
                        if not args.panel_refit:
                            continue
                        if not final_panel:
                            logprint(f"[panels] N={int(N)}: panel empty after pruning/refill; skipping refit.")
                            continue

                        # NOTE: For brevity and runtime safety, panel_refit keeps the prior behavior

    model_artifact = os.path.join(OUT["core"], f"{scen_name}__{model_label}__final_model.joblib")
    try:
        bundle = {
            "model": export_model,
            "model_raw": final_model,
            "model_name": model_name,
            "model_label": model_label,
            "scenario": scen_name,
            "split_id": split_id,
            "prot_cols": prot_cols_use,
            "cat_cols": CAT_COLS,
            "meta_num_cols": META_NUM_COLS,
            "args": vars(args),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "prevalence": {
                "train_sample": float(train_prevalence) if np.isfinite(train_prevalence) else np.nan,
                "target": float(target_prevalence),
                "target_source": str(target_prev_source),
            },
            "probability_stream": risk_prob_source,
            "thresholds": {
                "objective_name": thr_obj_name,
                "objective": float(thr_obj),
                "max_f1": float(thr_f1),
                "spec90": float(thr_spec90),
                "control_specs": {str(k): float(v) for k, v in thr_ctrl_specs.items()},
                "spec_targets": spec_targets,
            },
        }
        joblib.dump(bundle, model_artifact)
        logprint(f"[artifact] Saved final model bundle: {model_artifact}")
    except Exception as exc:
        logprint(f"[artifact] WARNING: failed to save model bundle ({exc})", level="warning")

    logprint(f"\nWrote: {args.outdir}")
    if os.path.exists(os.path.join(OUT["core"], "val_metrics.csv")):
        logprint(f"  core/val_metrics.csv")
    logprint(f"  core/test_metrics.csv")
    logprint(f"  cv/cv_repaeat_metrics.csv")
    logprint(f"  cv/best_params_per_split.csv")
    if os.path.exists(os.path.join(OUT["cv"], "selected_proteins_per_outer_split.csv")):
        logprint(f"  cv/selected_proteins_per_outer_split.csv")


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point that wires argparse + logging before running FaithRunner."""
    args = parse_args(argv)

    # Handle plot regeneration mode (early exit before normal pipeline)
    if args.regenerate_plots:
        if not args.run_dir:
            raise ValueError("--regenerate_plots requires --run_dir to specify the model run directory")

        # Configure logging to a temporary directory (or use run_dir if available)
        log_dir = args.run_dir if os.path.isdir(args.run_dir) else os.path.dirname(args.run_dir) or "."
        logger = configure_logging(log_dir, level=getattr(args, "log_level", "INFO"), log_filename="regenerate_plots.log")
        set_logger(logger)

        # Parse plot_types if provided
        plot_types = None
        if args.plot_types:
            plot_types = [pt.strip() for pt in args.plot_types.split(",") if pt.strip()]

        # Regenerate plots
        result = regenerate_plots_from_artifacts(
            run_dir=args.run_dir,
            force=args.force_overwrite,
            plot_types=plot_types,
        )

        # Log summary
        if result["success"]:
            logprint(f"[regen] SUCCESS: Regenerated {len(result['plots_regenerated'])} plot(s)")
            if result["plots_skipped"]:
                logprint(f"[regen] Skipped {len(result['plots_skipped'])} existing plot(s) (use --force_overwrite to regenerate)")
        else:
            logprint(f"[regen] FAILED: {len(result['errors'])} error(s)", level="error")
            for err in result["errors"]:
                logprint(f"[regen]   - {err}", level="error")
            sys.exit(1)

        sys.exit(0)

    # Normal training/evaluation pipeline
    # Create output directory early so logs and downstream writes succeed.
    os.makedirs(args.outdir, exist_ok=True)

    # Logging (stdout + run.log in outdir)
    logger = configure_logging(args.outdir, level=getattr(args, "log_level", "INFO"))
    set_logger(logger)

    FaithRunner(args).run()


if __name__ == "__main__":
    main()
