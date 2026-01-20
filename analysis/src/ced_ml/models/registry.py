"""Model registry and hyperparameter grid definitions.

This module provides:
- Model instantiation (RF, XGBoost, LinSVM, LogisticRegression)
- Hyperparameter grid generation for RandomizedSearchCV
- sklearn version compatibility handling

References:
- scikit-learn 1.8+ deprecates penalty= in LogisticRegression (use l1_ratio=)
- XGBoost tree_method controls CPU vs GPU acceleration
"""

import math
import numbers
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except (ImportError, Exception):
    XGBOOST_AVAILABLE = False
    XGBClassifier = None  # type: ignore

from ..config.schema import TrainingConfig


# ----------------------------
# sklearn version compatibility
# ----------------------------
def _sklearn_version_tuple(ver: str) -> Tuple[int, int, int]:
    """Parse sklearn version string (robust to rc/dev suffixes)."""
    nums = re.findall(r"\d+", ver)
    nums = (nums + ["0", "0", "0"])[:3]
    return (int(nums[0]), int(nums[1]), int(nums[2]))


SKLEARN_VER = _sklearn_version_tuple(getattr(sklearn, "__version__", "0.0.0"))


# ----------------------------
# Parameter grid utilities
# ----------------------------
def _parse_float_list(s: str) -> List[float]:
    """Parse comma-separated float values."""
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
    """Parse comma-separated integer values."""
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
    """Parse and validate non-empty integer list."""
    values = _parse_int_list(s)
    if not values:
        raise ValueError(
            f"{name} must be a non-empty comma-separated list (e.g. '200,500')."
        )
    return values


def _parse_none_int_float_list(s: str) -> List:
    """Parse list with mixed types: None, int, float, or strings like 'sqrt'."""
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
    """Coerce values to int or None (sklearn max_depth parameter).

    Accepts:
    - int
    - None
    - float with integer value (e.g., 10.0 -> 10)

    Raises:
        ValueError: For non-integer floats or invalid strings
    """
    out: List[Any] = []
    for v in vals:
        if v is None:
            out.append(None)
            continue
        if isinstance(v, bool):
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


def _coerce_min_samples_leaf_list(
    vals: List[Any], *, name: str = "rf_min_samples_leaf_grid"
) -> List[Any]:
    """Coerce min_samples_leaf grid to sklearn-compatible types.

    sklearn accepts:
    - int >= 1
    - float in (0, 1.0) (fraction of samples)

    Also coerces whole-number floats (e.g., 5.0 -> 5) for CLI robustness.
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
            raise ValueError(
                f"{name}: float must be in (0,1) or an integer value, got {v}"
            )
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
            raise ValueError(
                f"{name}: could not parse '{v}' as int>=1 or float in (0,1)"
            )
        raise ValueError(f"{name}: unsupported type {type(v).__name__}={v}")
    return out


def parse_class_weight_options(s: str) -> List:
    """Parse class_weight options.

    Examples:
        "none,balanced" -> [None, "balanced"]
        "balanced" -> ["balanced"]
        "" -> [None, "balanced"] (default)
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
    # fallback if user passes invalid input
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


def make_logspace(
    minv: float, maxv: float, points: int, rng: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """Generate log-spaced values for regularization parameters.

    Args:
        minv: Minimum value (e.g., 1e-3)
        maxv: Maximum value (e.g., 1e3)
        points: Number of grid points
        rng: Optional RNG for randomized grids

    Returns:
        Array of log-spaced values
    """
    minv = float(minv)
    maxv = float(maxv)
    points = int(points)
    if points < 2:
        points = 2
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
    """Randomize numeric grid values (used with --grid_randomize).

    Non-numeric entries are preserved.

    Args:
        values: Grid values to randomize
        rng: Random state for reproducibility
        as_int: Coerce to integers
        min_int/max_int: Bounds for integer values
        unique_int: Remove duplicates in integer grids
        log_scale: Use log-scale sampling
        min_float/max_float: Bounds for float values
        perturb_mode: Add noise to each grid point individually

    Returns:
        Randomized grid values
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

    # Perturb mode: add noise around each grid point
    if perturb_mode and not as_int:
        sorted_vals = sorted(numeric_vals)
        perturbed = []
        for i, val in enumerate(sorted_vals):
            # Determine perturbation range based on neighbors
            if i == 0:
                gap = (
                    (sorted_vals[i + 1] - val) / 2
                    if i + 1 < len(sorted_vals)
                    else (high - low) * 0.1
                )
                noise = rng.uniform(-gap * 0.5, gap)
            elif i == len(sorted_vals) - 1:
                gap = (val - sorted_vals[i - 1]) / 2
                noise = rng.uniform(-gap, gap * 0.5)
            else:
                gap_left = (val - sorted_vals[i - 1]) / 2
                gap_right = (sorted_vals[i + 1] - val) / 2
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
            sorted_vals = sorted(numeric_vals)
            perturbed = []
            for i, val in enumerate(sorted_vals):
                int_val = int(val)
                if i == 0:
                    gap = (
                        int((sorted_vals[i + 1] - val) / 2)
                        if i + 1 < len(sorted_vals)
                        else max(1, int((high - low) * 0.1))
                    )
                    offset = rng.randint(-max(1, gap // 2), gap + 1)
                elif i == len(sorted_vals) - 1:
                    gap = int((val - sorted_vals[i - 1]) / 2)
                    offset = rng.randint(-gap, max(1, gap // 2) + 1)
                else:
                    gap_left = int((val - sorted_vals[i - 1]) / 2)
                    gap_right = int((sorted_vals[i + 1] - val) / 2)
                    offset = rng.randint(-gap_left, gap_right + 1)

                perturbed_val = int_val + offset
                if min_int is not None:
                    perturbed_val = max(perturbed_val, int(min_int))
                if max_int is not None:
                    perturbed_val = min(perturbed_val, int(max_int))
                perturbed.append(perturbed_val)

            if unique_int:
                perturbed = list(dict.fromkeys(perturbed))
                while len(perturbed) < n:
                    extra = rng.randint(low, high + 1)
                    if min_int is not None:
                        extra = max(extra, int(min_int))
                    if max_int is not None:
                        extra = min(extra, int(max_int))
                    if extra not in perturbed:
                        perturbed.append(extra)

            sampled_vals = perturbed
        else:
            # Uniform sampling for integers
            low_int = int(low) if min_int is None else int(min_int)
            high_int = int(high) if max_int is None else int(max_int)
            if unique_int:
                sampled_vals = list(
                    rng.choice(range(low_int, high_int + 1), size=n, replace=False)
                )
            else:
                sampled_vals = [
                    int(rng.randint(low_int, high_int + 1)) for _ in range(n)
                ]
    else:
        # Float sampling
        if log_scale:
            if low <= 0 or high <= 0:
                sampled_vals = [float(v) for v in numeric_vals]
            else:
                log_low = np.log10(low)
                log_high = np.log10(high)
                samples = rng.uniform(log_low, log_high, size=n)
                sampled_vals = [float(np.power(10.0, s)) for s in samples]
        else:
            sampled_vals = [float(rng.uniform(low, high)) for _ in range(n)]

        # Apply bounds
        if min_float is not None:
            sampled_vals = [max(float(v), float(min_float)) for v in sampled_vals]
        if max_float is not None:
            sampled_vals = [min(float(v), float(max_float)) for v in sampled_vals]

    # Replace numeric values in original positions
    out = list(values_list)
    for idx, sampled_val in zip(numeric_idx, sampled_vals):
        out[idx] = sampled_val

    return out


def compute_scale_pos_weight_from_y(y: np.ndarray) -> float:
    """Compute XGBoost scale_pos_weight from class distribution."""
    y = np.asarray(y).astype(int)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    pos = max(1, pos)
    neg = max(1, neg)
    return float(neg / pos)


# ----------------------------
# Model builders
# ----------------------------
def build_logistic_regression(
    solver: str = "saga",
    C: float = 1.0,
    max_iter: int = 2000,
    tol: float = 1e-4,
    random_state: int = 42,
    l1_ratio: float = 0.5,
    penalty: str = "elasticnet",
) -> LogisticRegression:
    """Build Logistic Regression estimator (sklearn 1.8+ compatible).

    Args:
        solver: Optimization algorithm
        C: Inverse regularization strength
        max_iter: Maximum iterations
        tol: Convergence tolerance
        random_state: Random seed
        l1_ratio: ElasticNet mixing (0=L2, 1=L1)
        penalty: Penalty type (ignored in sklearn >=1.8)

    Returns:
        Configured LogisticRegression estimator
    """
    lr_common = {
        "solver": solver,
        "C": C,
        "max_iter": int(max_iter),
        "tol": float(tol),
        "random_state": int(random_state),
    }

    # sklearn >=1.8 deprecates penalty=, uses l1_ratio
    if SKLEARN_VER >= (1, 8, 0):
        return LogisticRegression(l1_ratio=l1_ratio, **lr_common)
    else:
        return LogisticRegression(penalty=penalty, l1_ratio=l1_ratio, **lr_common)


def build_linear_svm_calibrated(
    C: float = 1.0,
    max_iter: int = 2000,
    calibration_method: str = "sigmoid",
    calibration_cv: int = 5,
    random_state: int = 42,
) -> CalibratedClassifierCV:
    """Build calibrated LinearSVC estimator.

    LinearSVC + CalibratedClassifierCV provides probability estimates.

    Args:
        C: Inverse regularization strength
        max_iter: Maximum iterations
        calibration_method: 'sigmoid' or 'isotonic'
        calibration_cv: CV folds for calibration
        random_state: Random seed

    Returns:
        CalibratedClassifierCV wrapping LinearSVC
    """
    base_svm = LinearSVC(
        C=C, class_weight=None, random_state=int(random_state), max_iter=int(max_iter)
    )
    return CalibratedClassifierCV(
        base_svm, method=str(calibration_method), cv=int(calibration_cv)
    )


def build_random_forest(
    n_estimators: int = 500,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 5,
    min_samples_split: int = 2,
    max_features: str = "sqrt",
    max_samples: Optional[float] = None,
    bootstrap: bool = True,
    random_state: int = 42,
    n_jobs: int = 1,
) -> RandomForestClassifier:
    """Build Random Forest classifier.

    Args:
        n_estimators: Number of trees
        max_depth: Maximum tree depth (None = unlimited)
        min_samples_leaf: Minimum samples per leaf
        min_samples_split: Minimum samples to split
        max_features: Features per split ('sqrt', int, or float)
        max_samples: Samples per tree (None = all)
        bootstrap: Whether to use bootstrap sampling
        random_state: Random seed
        n_jobs: Parallel jobs

    Returns:
        Configured RandomForestClassifier
    """
    rf_kwargs = {
        "n_estimators": int(n_estimators),
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "min_samples_split": min_samples_split,
        "max_features": max_features,
        "bootstrap": bool(bootstrap),
        "random_state": int(random_state),
        "n_jobs": int(max(1, n_jobs)),
    }

    if max_samples is not None:
        try:
            v = float(max_samples)
            rf_kwargs["max_samples"] = int(v) if v.is_integer() else float(v)
        except Exception:
            pass

    return RandomForestClassifier(**rf_kwargs)


def build_xgboost(
    n_estimators: int = 1000,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    scale_pos_weight: float = 1.0,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    min_child_weight: int = 1,
    gamma: float = 0.0,
    tree_method: str = "hist",
    random_state: int = 42,
    n_jobs: int = 1,
) -> XGBClassifier:
    """Build XGBoost classifier.

    Args:
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Step size shrinkage
        subsample: Row sampling fraction
        colsample_bytree: Column sampling fraction
        scale_pos_weight: Balancing of positive/negative weights
        reg_alpha: L1 regularization
        reg_lambda: L2 regularization
        min_child_weight: Minimum sum of instance weight
        gamma: Minimum loss reduction
        tree_method: 'hist', 'gpu_hist', etc.
        random_state: Random seed
        n_jobs: Parallel jobs (1 for GPU)

    Returns:
        Configured XGBClassifier

    Raises:
        ImportError: If XGBoost not installed
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost not available. Install with: pip install xgboost")

    return XGBClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=float(learning_rate),
        subsample=float(subsample),
        colsample_bytree=float(colsample_bytree),
        scale_pos_weight=float(scale_pos_weight),
        reg_alpha=float(reg_alpha),
        reg_lambda=float(reg_lambda),
        min_child_weight=int(min_child_weight),
        gamma=float(gamma),
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method=tree_method,
        random_state=int(random_state),
        n_jobs=int(max(1, n_jobs)) if tree_method != "gpu_hist" else 1,
    )


def build_models(
    model_name: str,
    config: TrainingConfig,
    random_state: int = 42,
    n_jobs: int = 1,
) -> object:
    """Build a single model estimator.

    Args:
        model_name: Model identifier ('LR_EN', 'LR_L1', 'LinSVM_cal', 'RF', 'XGBoost')
        config: Training configuration
        random_state: Random seed
        n_jobs: CPU cores for RF/XGBoost

    Returns:
        sklearn-compatible estimator

    Raises:
        ValueError: If model_name is unknown
        ImportError: If XGBoost requested but not installed
    """
    if model_name == "LR_EN":
        return build_logistic_regression(
            solver=config.lr.solver,
            C=1.0,
            max_iter=config.lr.max_iter,
            tol=1e-4,  # LRConfig doesn't have tol field
            random_state=random_state,
            l1_ratio=0.5,
            penalty="elasticnet",
        )

    elif model_name == "LR_L1":
        return build_logistic_regression(
            solver=config.lr.solver,
            C=1.0,
            max_iter=config.lr.max_iter,
            tol=1e-4,  # LRConfig doesn't have tol field
            random_state=random_state,
            l1_ratio=1.0,
            penalty="l1",
        )

    elif model_name == "LinSVM_cal":
        return build_linear_svm_calibrated(
            C=1.0,
            max_iter=config.svm.max_iter,
            calibration_method=config.calibration.method,
            calibration_cv=config.calibration.cv,
            random_state=random_state,
        )

    elif model_name == "RF":
        # Get first value from n_estimators list for default model
        n_est = config.rf.n_estimators[0] if config.rf.n_estimators else 100
        return build_random_forest(
            n_estimators=n_est,
            random_state=random_state,
            n_jobs=int(max(1, n_jobs)),
            # RF config doesn't have bootstrap/max_samples in new schema
        )

    elif model_name == "XGBoost":
        # Get first values from lists for default model
        n_est = config.xgboost.n_estimators[0] if config.xgboost.n_estimators else 100
        max_d = config.xgboost.max_depth[0] if config.xgboost.max_depth else 5
        lr = config.xgboost.learning_rate[0] if config.xgboost.learning_rate else 0.05
        sub = config.xgboost.subsample[0] if config.xgboost.subsample else 0.8
        col = (
            config.xgboost.colsample_bytree[0]
            if config.xgboost.colsample_bytree
            else 0.8
        )
        spw = 1.0  # Default, will be computed later
        return build_xgboost(
            n_estimators=n_est,
            max_depth=max_d,
            learning_rate=lr,
            subsample=sub,
            colsample_bytree=col,
            scale_pos_weight=spw,
            reg_alpha=config.xgboost.reg_alpha[0] if config.xgboost.reg_alpha else 0.0,
            reg_lambda=(
                config.xgboost.reg_lambda[0] if config.xgboost.reg_lambda else 1.0
            ),
            min_child_weight=(
                config.xgboost.min_child_weight[0]
                if config.xgboost.min_child_weight
                else 1
            ),
            gamma=config.xgboost.gamma[0] if config.xgboost.gamma else 0.0,
            tree_method=config.xgboost.tree_method,
            random_state=random_state,
            n_jobs=(
                int(max(1, n_jobs)) if config.xgboost.tree_method != "gpu_hist" else 1
            ),
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")


# ----------------------------
# Hyperparameter grids
# ----------------------------
def get_param_distributions(
    model_name: str,
    config: TrainingConfig,
    feature_select: str,
    k_grid: List[int],
    kbest_scope: str,
    xgb_scale_pos_weight: Optional[float] = None,
    grid_rng: Optional[np.random.RandomState] = None,
    randomize_grids: bool = False,
) -> Dict[str, List]:
    """Generate hyperparameter distributions for RandomizedSearchCV.

    Args:
        model_name: Model identifier
        config: Training configuration
        feature_select: 'kbest', 'hybrid', or 'none'
        k_grid: Feature selection grid
        kbest_scope: 'protein' or 'transformed'
        xgb_scale_pos_weight: Class balancing for XGBoost
        grid_rng: RNG for grid randomization
        randomize_grids: Whether to randomize grids

    Returns:
        Dictionary of parameter name -> value list
    """
    d = {}
    rng = grid_rng if randomize_grids else None

    # Feature selection grids
    if feature_select in ("kbest", "hybrid"):
        if not k_grid:
            raise ValueError("feature_select in {kbest,hybrid} requires k_grid")
        if kbest_scope == "protein":
            d["prot_sel__k"] = k_grid
        else:
            d["sel__k"] = k_grid

    # Get hyperparameter grids from config
    lr_Cs = config.lr.C if hasattr(config.lr, "C") else [0.001, 0.01, 0.1, 1.0, 10.0]
    svm_Cs = config.svm.C if hasattr(config.svm, "C") else [0.01, 0.1, 1.0, 10.0]

    # Convert class_weight strings to lists
    lr_class_weight = [config.lr.class_weight] if config.lr.class_weight else [None]
    svm_class_weight = [config.svm.class_weight] if config.svm.class_weight else [None]
    rf_class_weight = [config.rf.class_weight] if config.rf.class_weight else [None]

    # Model-specific grids
    if model_name == "LR_L1":
        d.update({"clf__C": lr_Cs, "clf__class_weight": lr_class_weight})
        return d

    if model_name == "LR_EN":
        l1_grid = (
            config.lr.l1_ratio
            if hasattr(config.lr, "l1_ratio")
            else [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95]
        )
        if rng is not None:
            l1_grid = _randomize_numeric_list(
                l1_grid, rng, min_float=0.0, max_float=1.0, perturb_mode=True
            )
        d.update(
            {
                "clf__C": lr_Cs,
                "clf__l1_ratio": l1_grid,
                "clf__class_weight": lr_class_weight,
            }
        )
        return d

    if model_name == "LinSVM_cal":
        # Handle sklearn API differences
        est_key = (
            "estimator"
            if "estimator" in CalibratedClassifierCV(LinearSVC()).get_params()
            else "base_estimator"
        )
        d.update(
            {
                f"clf__{est_key}__C": svm_Cs,
                f"clf__{est_key}__class_weight": svm_class_weight,
            }
        )
        return d

    if model_name == "RF":
        n_estimators_grid = (
            config.rf.n_estimators
            if hasattr(config.rf, "n_estimators")
            else [100, 300, 500]
        )
        if rng is not None:
            n_estimators_grid = _randomize_numeric_list(
                n_estimators_grid,
                rng,
                as_int=True,
                min_int=1,
                unique_int=True,
                perturb_mode=True,
            )

        max_depth = (
            config.rf.max_depth
            if hasattr(config.rf, "max_depth")
            else [None, 10, 20, 40]
        )
        if rng is not None:
            max_depth = _randomize_numeric_list(
                max_depth,
                rng,
                as_int=True,
                min_int=1,
                unique_int=True,
                perturb_mode=True,
            )

        min_leaf = (
            config.rf.min_samples_leaf
            if hasattr(config.rf, "min_samples_leaf")
            else [1, 2, 4]
        )
        min_split = (
            config.rf.min_samples_split
            if hasattr(config.rf, "min_samples_split")
            else [2, 5, 10]
        )
        max_feat = (
            config.rf.max_features
            if hasattr(config.rf, "max_features")
            else ["sqrt", 0.5]
        )

        if rng is not None:
            max_feat = _randomize_numeric_list(
                max_feat, rng, min_float=0.1, max_float=1.0, perturb_mode=True
            )
            min_split = _randomize_numeric_list(
                min_split,
                rng,
                as_int=True,
                min_int=2,
                unique_int=True,
                perturb_mode=True,
            )

        d.update(
            {
                "clf__n_estimators": n_estimators_grid,
                "clf__max_depth": max_depth,
                "clf__min_samples_leaf": min_leaf,
                "clf__min_samples_split": min_split,
                "clf__max_features": max_feat,
                "clf__class_weight": rf_class_weight,
            }
        )

        return d

    if model_name == "XGBoost":
        n_estimators_grid = (
            config.xgboost.n_estimators
            if hasattr(config.xgboost, "n_estimators")
            else [100, 300, 500]
        )
        max_depth_grid = (
            config.xgboost.max_depth
            if hasattr(config.xgboost, "max_depth")
            else [3, 5, 7]
        )
        learning_rate_grid = (
            config.xgboost.learning_rate
            if hasattr(config.xgboost, "learning_rate")
            else [0.01, 0.05, 0.1]
        )
        subsample_grid = (
            config.xgboost.subsample
            if hasattr(config.xgboost, "subsample")
            else [0.7, 0.8, 1.0]
        )
        colsample_grid = (
            config.xgboost.colsample_bytree
            if hasattr(config.xgboost, "colsample_bytree")
            else [0.7, 0.8, 1.0]
        )
        spw_grid = (
            [float(xgb_scale_pos_weight)] if xgb_scale_pos_weight is not None else [1.0]
        )

        if rng is not None:
            n_estimators_grid = _randomize_numeric_list(
                n_estimators_grid, rng, as_int=True, min_int=1, perturb_mode=True
            )
            max_depth_grid = _randomize_numeric_list(
                max_depth_grid,
                rng,
                as_int=True,
                min_int=1,
                unique_int=True,
                perturb_mode=True,
            )
            learning_rate_grid = _randomize_numeric_list(
                learning_rate_grid,
                rng,
                min_float=1e-4,
                log_scale=True,
                perturb_mode=True,
            )
            subsample_grid = _randomize_numeric_list(
                subsample_grid, rng, min_float=0.1, max_float=1.0, perturb_mode=True
            )
            colsample_grid = _randomize_numeric_list(
                colsample_grid, rng, min_float=0.1, max_float=1.0, perturb_mode=True
            )
            spw_grid = _randomize_numeric_list(
                spw_grid, rng, min_float=1e-3, perturb_mode=True
            )

        d.update(
            {
                "clf__n_estimators": n_estimators_grid,
                "clf__max_depth": max_depth_grid,
                "clf__learning_rate": learning_rate_grid,
                "clf__subsample": subsample_grid,
                "clf__colsample_bytree": colsample_grid,
                "clf__scale_pos_weight": spw_grid,
            }
        )
        return d

    return {}
