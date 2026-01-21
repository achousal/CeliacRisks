"""
Hyperparameter search space definitions for all models.

Provides:
- Parameter distributions for RandomizedSearchCV
- Grid randomization for sensitivity analysis
- Model-specific tuning ranges
"""

from typing import Dict, List, Optional, Union

import numpy as np

from ..config import TrainingConfig


def get_param_distributions(
    model_name: str,
    config: TrainingConfig,
    xgb_spw: Optional[float] = None,
    grid_rng: Optional[np.random.Generator] = None,
) -> Dict[str, List]:
    """
    Get parameter distribution for RandomizedSearchCV.

    Args:
        model_name: Model identifier (RF, XGBoost, LR_EN, LR_L1, LinSVM_cal)
        config: TrainingConfiguration object
        xgb_spw: XGBoost scale_pos_weight (optional override)
        grid_rng: Optional RNG for grid randomization (sensitivity analysis)

    Returns:
        Dictionary mapping parameter names to value lists
        Empty dict if model has no hyperparameters to tune
    """
    param_dists = {}
    randomize = grid_rng is not None

    # Feature selection parameters (if applicable)
    feature_select = config.features.feature_select
    if feature_select in ("kbest", "hybrid"):
        k_grid = config.features.k_grid
        if not k_grid:
            raise ValueError(f"feature_select={feature_select} requires features.k_grid")

        # Always use 'sel' step name regardless of kbest_scope
        param_dists["sel__k"] = k_grid

    # Model-specific parameters
    if model_name in ("LR_EN", "LR_L1"):
        param_dists.update(_get_lr_params(config, randomize, grid_rng))

    elif model_name == "LinSVM_cal":
        param_dists.update(_get_svm_params(config, randomize, grid_rng))

    elif model_name == "RF":
        param_dists.update(_get_rf_params(config, randomize, grid_rng))

    elif model_name == "XGBoost":
        param_dists.update(_get_xgb_params(config, xgb_spw, randomize, grid_rng))

    return param_dists


def _get_lr_params(
    config: TrainingConfig, randomize: bool, rng: Optional[np.random.Generator]
) -> Dict[str, List]:
    """Logistic Regression hyperparameters."""
    # C values (inverse regularization strength)
    C_grid = _make_logspace(
        config.lr.C_min,
        config.lr.C_max,
        config.lr.C_points,
        rng=rng if randomize else None,
    )

    # Class weights
    class_weight_options = _parse_class_weight_options(config.lr.class_weight_options)

    params = {"clf__C": C_grid}
    if class_weight_options:
        params["clf__class_weight"] = class_weight_options

    return params


def _get_svm_params(
    config: TrainingConfig, randomize: bool, rng: Optional[np.random.Generator]
) -> Dict[str, List]:
    """Linear SVM hyperparameters (wrapped in CalibratedClassifierCV)."""
    # C values
    C_grid = _make_logspace(
        config.svm.C_min,
        config.svm.C_max,
        config.svm.C_points,
        rng=rng if randomize else None,
    )

    # Class weights
    class_weight_options = _parse_class_weight_options(config.svm.class_weight_options)

    # Parameter prefix depends on sklearn version
    # Newer: estimator__C, older: base_estimator__C
    # Use estimator__ (modern sklearn)
    params = {"clf__estimator__C": C_grid}
    if class_weight_options:
        params["clf__estimator__class_weight"] = class_weight_options

    return params


def _get_rf_params(
    config: TrainingConfig, randomize: bool, rng: Optional[np.random.Generator]
) -> Dict[str, List]:
    """Random Forest hyperparameters."""
    n_estimators_grid = config.rf.n_estimators_grid.copy()
    max_depth_grid = config.rf.max_depth_grid.copy()
    min_samples_split_grid = config.rf.min_samples_split_grid.copy()
    min_samples_leaf_grid = config.rf.min_samples_leaf_grid.copy()
    max_features_grid = config.rf.max_features_grid.copy()

    if randomize and rng:
        n_estimators_grid = _randomize_int_list(n_estimators_grid, rng, min_val=10)
        max_depth_grid = _randomize_int_list(max_depth_grid, rng, min_val=1, unique=True)
        min_samples_split_grid = _randomize_int_list(min_samples_split_grid, rng, min_val=2)
        min_samples_leaf_grid = _randomize_int_list(min_samples_leaf_grid, rng, min_val=1)
        max_features_grid = _randomize_float_list(max_features_grid, rng, min_val=0.1, max_val=1.0)

    params = {
        "clf__n_estimators": n_estimators_grid,
        "clf__max_depth": max_depth_grid,
        "clf__min_samples_split": min_samples_split_grid,
        "clf__min_samples_leaf": min_samples_leaf_grid,
        "clf__max_features": max_features_grid,
    }

    # Class weights
    class_weight_options = _parse_class_weight_options(config.rf.class_weight_options)
    if class_weight_options:
        params["clf__class_weight"] = class_weight_options

    return params


def _get_xgb_params(
    config: TrainingConfig,
    xgb_spw: Optional[float],
    randomize: bool,
    rng: Optional[np.random.Generator],
) -> Dict[str, List]:
    """XGBoost hyperparameters."""
    n_estimators_grid = config.xgboost.n_estimators_grid.copy()
    max_depth_grid = config.xgboost.max_depth_grid.copy()
    learning_rate_grid = config.xgboost.learning_rate_grid.copy()
    subsample_grid = config.xgboost.subsample_grid.copy()
    colsample_grid = config.xgboost.colsample_bytree_grid.copy()

    # Scale pos weight grid
    if xgb_spw is not None:
        # Use fold-specific value +/- 20%
        spw_grid = [xgb_spw * 0.8, xgb_spw, xgb_spw * 1.2]
    else:
        spw_grid = config.xgboost.scale_pos_weight_grid.copy()

    if randomize and rng:
        n_estimators_grid = _randomize_int_list(n_estimators_grid, rng, min_val=1)
        max_depth_grid = _randomize_int_list(max_depth_grid, rng, min_val=1, unique=True)
        learning_rate_grid = _randomize_float_list(
            learning_rate_grid, rng, min_val=1e-4, log_scale=True
        )
        subsample_grid = _randomize_float_list(subsample_grid, rng, min_val=0.1, max_val=1.0)
        colsample_grid = _randomize_float_list(colsample_grid, rng, min_val=0.1, max_val=1.0)
        spw_grid = _randomize_float_list(spw_grid, rng, min_val=1e-3)

    return {
        "clf__n_estimators": n_estimators_grid,
        "clf__max_depth": max_depth_grid,
        "clf__learning_rate": learning_rate_grid,
        "clf__subsample": subsample_grid,
        "clf__colsample_bytree": colsample_grid,
        "clf__scale_pos_weight": spw_grid,
    }


def _make_logspace(
    min_val: float,
    max_val: float,
    n_points: int,
    rng: Optional[np.random.Generator] = None,
) -> List[float]:
    """
    Create log-spaced grid.

    Args:
        min_val: Minimum value (e.g. 1e-4)
        max_val: Maximum value (e.g. 1e4)
        n_points: Number of points
        rng: Optional RNG for perturbation

    Returns:
        List of float values
    """
    if n_points < 1:
        return []

    if n_points == 1:
        return [float(np.sqrt(min_val * max_val))]  # Geometric mean

    # Standard log-spaced grid
    grid = np.logspace(np.log10(min_val), np.log10(max_val), num=n_points).tolist()

    # Optional perturbation
    if rng:
        grid = [float(v * rng.uniform(0.8, 1.2)) for v in grid]
        grid = [max(min_val, min(max_val, v)) for v in grid]

    return grid


def _parse_class_weight_options(options_str: str) -> List:
    """
    Parse class_weight options string.

    Format: "balanced,{0:1,1:5},{0:1,1:10}"

    Returns:
        List of class_weight values (None, 'balanced', or dict)
    """
    if not options_str or options_str.strip() == "":
        return [None]

    options = []

    # Split carefully to avoid breaking {k:v,k:v} dicts
    parts = []
    current = []
    in_dict = False

    for char in options_str + ",":  # Add trailing comma to flush last part
        if char == "{":
            in_dict = True
            current.append(char)
        elif char == "}":
            in_dict = False
            current.append(char)
        elif char == "," and not in_dict:
            if current:
                parts.append("".join(current))
                current = []
        else:
            current.append(char)

    # Parse each part
    for opt in parts:
        opt = opt.strip()
        if opt == "":
            continue
        elif opt == "None":
            options.append(None)
        elif opt == "balanced":
            options.append("balanced")
        elif opt.startswith("{"):
            # Parse dict: {0:1,1:5}
            try:
                weight_dict = {}
                opt = opt.strip("{}")
                for pair in opt.split(","):
                    k, v = pair.split(":")
                    weight_dict[int(k.strip())] = float(v.strip())
                options.append(weight_dict)
            except Exception:
                pass

    return options if options else [None]


def _randomize_int_list(
    values: List[int],
    rng: np.random.Generator,
    min_val: int = 1,
    unique: bool = False,
) -> List[int]:
    """
    Perturb integer grid values for sensitivity analysis.

    Args:
        values: Original grid values (may contain None)
        rng: Random number generator
        min_val: Minimum allowed value
        unique: If True, ensure all values are unique

    Returns:
        Perturbed grid values (None values preserved as-is)
    """
    if not values:
        return []

    perturbed = []
    for v in values:
        if v is None:
            # Preserve None values (e.g., for RF max_depth=None)
            perturbed.append(None)
        else:
            # Perturb by +/- 20%
            delta = max(1, int(v * 0.2))
            new_val = v + rng.integers(-delta, delta + 1)
            new_val = max(min_val, new_val)
            perturbed.append(new_val)

    if unique:
        # Separate None from numeric values for uniqueness
        none_values = [v for v in perturbed if v is None]
        numeric_values = sorted({v for v in perturbed if v is not None})
        perturbed = none_values + numeric_values

    return perturbed


def _randomize_float_list(
    values: List[Union[str, float]],
    rng: np.random.Generator,
    min_val: float = 0.0,
    max_val: float = np.inf,
    log_scale: bool = False,
) -> List[Union[str, float]]:
    """
    Perturb float grid values for sensitivity analysis.

    Non-numeric values (e.g., "sqrt", "log2") are passed through unchanged.

    Args:
        values: Original grid values
        rng: Random number generator
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        log_scale: If True, perturb in log space

    Returns:
        Perturbed grid values
    """
    if not values:
        return []

    perturbed = []
    for v in values:
        # Skip non-numeric values (e.g., "sqrt", "log2" for max_features)
        if not isinstance(v, (int, float)):
            perturbed.append(v)
            continue

        if log_scale and v > 0:
            # Perturb in log space
            log_v = np.log10(v)
            log_delta = 0.2  # +/- 0.2 in log space
            new_log_v = log_v + rng.uniform(-log_delta, log_delta)
            new_val = 10**new_log_v
        else:
            # Perturb by +/- 20%
            new_val = v * rng.uniform(0.8, 1.2)

        new_val = max(min_val, min(max_val, new_val))
        perturbed.append(float(new_val))

    return perturbed


# ============================================================================
# Optuna Parameter Distribution Conversion
# ============================================================================


def get_param_distributions_optuna(
    model_name: str,
    config: TrainingConfig,
    xgb_spw: Optional[float] = None,
) -> Dict[str, Dict]:
    """
    Convert sklearn param distributions to Optuna suggest specs.

    Optuna uses a different format for specifying search spaces:
    - int: {"type": "int", "low": min, "high": max, "log": bool}
    - float: {"type": "float", "low": min, "high": max, "log": bool}
    - categorical: {"type": "categorical", "choices": [list]}

    Args:
        model_name: Model identifier (RF, XGBoost, LR_EN, LR_L1, LinSVM_cal)
        config: TrainingConfiguration object
        xgb_spw: XGBoost scale_pos_weight (optional override)

    Returns:
        Dictionary mapping parameter names to Optuna suggest specs
    """
    # Get sklearn-style distributions first
    sklearn_dists = get_param_distributions(model_name, config, xgb_spw=xgb_spw)

    if not sklearn_dists:
        return {}

    # Convert each parameter
    optuna_dists = {}
    for name, values in sklearn_dists.items():
        spec = _to_optuna_spec(name, values)
        if spec is not None:
            optuna_dists[name] = spec

    return optuna_dists


def _to_optuna_spec(name: str, values: List) -> Optional[Dict]:
    """
    Convert a single sklearn parameter grid to Optuna spec.

    Args:
        name: Parameter name
        values: List of possible values

    Returns:
        Optuna suggest spec dict, or None if conversion fails
    """
    if not values:
        return None

    # Handle single value
    if len(values) == 1:
        return {"type": "categorical", "choices": values}

    # Check if all values are integers
    if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        return {
            "type": "int",
            "low": min(values),
            "high": max(values),
            "log": _is_log_spaced(values),
        }

    # Check if all values are numeric (int or float, but not all int)
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        return {
            "type": "float",
            "low": float(min(values)),
            "high": float(max(values)),
            "log": _is_log_spaced(values),
        }

    # Default to categorical
    return {"type": "categorical", "choices": values}


def _is_log_spaced(values: List) -> bool:
    """
    Heuristically detect if values are log-spaced.

    Uses ratio consistency: if consecutive ratios are similar,
    values are likely log-spaced.

    Args:
        values: List of numeric values

    Returns:
        True if values appear to be log-spaced
    """
    if len(values) < 3:
        return False

    # Filter to positive values only
    positive = [v for v in values if isinstance(v, (int, float)) and v > 0]
    if len(positive) < 3:
        return False

    sorted_vals = sorted(positive)

    # Compute consecutive ratios
    ratios = []
    for i in range(len(sorted_vals) - 1):
        if sorted_vals[i] > 0:
            ratios.append(sorted_vals[i + 1] / sorted_vals[i])

    if not ratios:
        return False

    # Check if ratios are relatively consistent (log-spaced characteristic)
    mean_ratio = np.mean(ratios)
    if mean_ratio <= 1.5:
        # Ratios too close to 1 = linear spacing
        return False

    # Check variance in ratios
    ratio_std = np.std(ratios)
    return ratio_std / mean_ratio < 0.5  # Low relative variance = log-spaced
