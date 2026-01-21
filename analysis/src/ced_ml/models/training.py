"""
Nested cross-validation orchestration for model training.

Provides:
- Out-of-fold (OOF) prediction generation with repeated stratified CV
- Nested hyperparameter tuning (outer CV + inner GridSearchCV/RandomizedSearchCV)
- Feature selection tracking across CV folds
- Protein selection extraction from fitted models
- Optional post-hoc calibration
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from ..config import TrainingConfig

logger = logging.getLogger(__name__)


def oof_predictions_with_nested_cv(
    pipeline: Pipeline,
    model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    protein_cols: List[str],
    config: TrainingConfig,
    random_state: int,
    grid_rng: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, float, pd.DataFrame, pd.DataFrame]:
    """
    Generate out-of-fold predictions using nested cross-validation.

    Outer CV: Repeated stratified K-fold for robust OOF predictions
    Inner CV: Hyperparameter tuning via RandomizedSearchCV

    Args:
        pipeline: Unfitted sklearn pipeline (pre + feature selection + clf)
        model_name: Model identifier (RF, XGBoost, LR_EN, LinSVM_cal)
        X: Training features (N x D)
        y: Training labels (N,)
        protein_cols: List of protein column names
        config: TrainingConfiguration object
        random_state: Random seed
        grid_rng: Optional RNG for grid randomization

    Returns:
        preds: OOF predictions (n_repeats x N) - each row is one repeat's predictions
        elapsed_sec: Training time in seconds
        best_params_df: DataFrame with best hyperparameters per fold
        selected_proteins_df: DataFrame with selected proteins per fold

    Raises:
        RuntimeError: If any repeat has missing OOF predictions (CV split bug)
    """
    n_splits = config.cv.folds
    n_repeats = config.cv.repeats

    if n_repeats < 1:
        raise ValueError(f"cv.repeats must be >= 1, got {n_repeats}")

    n_samples = len(y)
    preds = np.full((n_repeats, n_samples), np.nan, dtype=float)
    best_params_rows: List[Dict[str, Any]] = []
    selected_proteins_rows: List[Dict[str, Any]] = []

    total_outer_folds = n_repeats if n_splits < 2 else n_splits * n_repeats
    split_idx = 0
    t0 = time.perf_counter()

    # Setup outer CV splitter
    if n_splits < 2:
        logger.warning(
            f"[cv] WARNING: folds={n_splits} < 2; skipping outer CV, using in-sample predictions."
        )
        all_indices = np.arange(n_samples, dtype=int)
        split_iterator = ((all_indices, all_indices) for _ in range(n_repeats))
        split_divisor = 1
    else:
        rskf = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )
        split_iterator = rskf.split(X, y)
        split_divisor = n_splits

    # Outer CV loop
    for train_idx, test_idx in split_iterator:
        repeat_num = split_idx // split_divisor
        base_pipeline = clone(pipeline)

        logger.warning(
            f"[{time.strftime('%F %T')}] {model_name} outer fold {split_idx+1}/{total_outer_folds} "
            f"(repeat={repeat_num}) start"
        )

        # Handle XGBoost scale_pos_weight
        xgb_spw = None
        if model_name == "XGBoost":
            xgb_spw = _compute_xgb_scale_pos_weight(y[train_idx], config)
            try:
                base_pipeline.set_params(clf__scale_pos_weight=float(xgb_spw))
            except Exception:
                pass

        # Build inner CV hyperparameter search
        search = _build_hyperparameter_search(
            base_pipeline, model_name, config, random_state, xgb_spw, grid_rng
        )

        # Fit model (with or without search)
        if search is None:
            base_pipeline.fit(X.iloc[train_idx], y[train_idx])
            fitted_model = base_pipeline
            best_params, best_score = {}, np.nan
        else:
            # Use loky backend for multi-threaded search to avoid thread oversubscription
            if getattr(search, "n_jobs", 1) and int(search.n_jobs) > 1:
                with parallel_backend("loky", inner_max_num_threads=1):
                    search.fit(X.iloc[train_idx], y[train_idx])
            else:
                search.fit(X.iloc[train_idx], y[train_idx])

            fitted_model = search.best_estimator_
            best_params = search.best_params_
            best_score = float(search.best_score_)

        # Optional post-hoc calibration (LR/RF only, SVM already calibrated)
        fitted_model = _maybe_apply_calibration(
            fitted_model,
            model_name,
            config,
            X.iloc[train_idx],
            y[train_idx],
            random_state,
        )

        # Generate OOF predictions for this fold
        proba = fitted_model.predict_proba(X.iloc[test_idx])[:, 1]
        proba = np.clip(proba, 0.0, 1.0)
        preds[repeat_num, test_idx] = proba

        # Record best hyperparameters
        best_params_rows.append(
            {
                "model": model_name,
                "repeat": repeat_num,
                "outer_split": split_idx,
                "best_score_inner": best_score,
                "best_params": json.dumps(best_params, sort_keys=True),
            }
        )

        # Extract selected proteins from this fold
        selected_proteins = _extract_selected_proteins_from_fold(
            fitted_model,
            model_name,
            protein_cols,
            config,
            X.iloc[train_idx],
            y[train_idx],
            random_state,
        )

        if selected_proteins:
            selected_proteins_rows.append(
                {
                    "model": model_name,
                    "repeat": repeat_num,
                    "outer_split": split_idx,
                    "n_selected_proteins": len(selected_proteins),
                    "selected_proteins": json.dumps(sorted(selected_proteins)),
                }
            )

        split_idx += 1

    elapsed_sec = time.perf_counter() - t0

    # Validate: no missing OOF predictions
    for r in range(n_repeats):
        if np.isnan(preds[r]).any():
            raise RuntimeError(f"Repeat {r} has missing OOF predictions. Check CV splitting logic.")

    return (
        preds,
        elapsed_sec,
        pd.DataFrame(best_params_rows),
        pd.DataFrame(selected_proteins_rows),
    )


def _compute_xgb_scale_pos_weight(y_train: np.ndarray, config: TrainingConfig) -> float:
    """
    Compute XGBoost scale_pos_weight parameter from training labels.

    Default: ratio of negatives to positives (auto class balancing)
    Override: config.models.xgboost.scale_pos_weight if set

    Args:
        y_train: Training labels (0/1)
        config: TrainingConfiguration object

    Returns:
        scale_pos_weight value (>= 1.0)
    """
    # Check if user specified explicit value
    spw_config = config.models.xgboost.scale_pos_weight
    if spw_config is not None and spw_config > 0:
        return float(spw_config)

    # Auto: ratio of negatives to positives
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))

    if n_pos == 0:
        logger.warning("[xgb] No positive samples in training fold; using spw=1.0")
        return 1.0

    spw = float(n_neg) / float(n_pos)
    return max(1.0, spw)


def _build_hyperparameter_search(
    pipeline: Pipeline,
    model_name: str,
    config: TrainingConfig,
    random_state: int,
    xgb_spw: Optional[float],
    grid_rng: Optional[np.random.RandomState],
):
    """
    Build RandomizedSearchCV for inner CV hyperparameter tuning.

    Returns None if:
    - Model has no hyperparameters to tune
    - config.cv.inner_folds < 2 (tuning disabled)
    - config.cv.n_iter < 1 (tuning disabled)

    Args:
        pipeline: Base pipeline to tune
        model_name: Model identifier
        config: TrainingConfiguration object
        random_state: Random seed
        xgb_spw: XGBoost scale_pos_weight (if applicable)
        grid_rng: Optional RNG for grid randomization

    Returns:
        RandomizedSearchCV object or None
    """
    from sklearn.model_selection import RandomizedSearchCV

    from .hyperparams import get_param_distributions

    # Get parameter distributions for this model
    param_dists = get_param_distributions(model_name, config, xgb_spw=xgb_spw, grid_rng=grid_rng)

    if not param_dists:
        return None

    # Validate inner CV settings
    inner_folds = config.cv.inner_folds
    if inner_folds < 2:
        logger.info(
            f"[tune] WARNING: inner_folds={inner_folds} < 2; skipping hyperparameter search."
        )
        return None

    n_iter = config.cv.n_iter
    if n_iter < 1:
        logger.warning(f"[tune] WARNING: n_iter={n_iter} < 1; skipping hyperparameter search.")
        return None

    # Build inner CV splitter
    inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=random_state)

    # Determine parallelization strategy
    n_jobs = _get_search_n_jobs(model_name, config)

    return RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dists,
        n_iter=n_iter,
        scoring=config.cv.scoring,
        cv=inner_cv,
        n_jobs=n_jobs,
        pre_dispatch=n_jobs,
        refit=True,
        random_state=random_state,
        error_score="raise",  # Fail fast on errors
        verbose=0,
    )


def _get_search_n_jobs(model_name: str, config: TrainingConfig) -> int:
    """
    Determine n_jobs for RandomizedSearchCV.

    Strategy:
    - LR/SVM: Parallelize search (models are single-threaded)
    - RF/XGBoost: Keep search single-threaded (estimators use internal parallelism)

    Args:
        model_name: Model identifier
        config: TrainingConfiguration object

    Returns:
        n_jobs value (>= 1)
    """
    tune_n_jobs = config.compute.tune_n_jobs
    cpus = config.compute.cpus

    if tune_n_jobs is not None:
        # Explicit override
        return max(1, min(cpus, tune_n_jobs))

    # Auto strategy
    if model_name in ("LR_EN", "LR_L1", "LinSVM_cal"):
        # Parallelize search for single-threaded models
        return max(1, cpus)
    else:
        # Keep search single-threaded for models with internal parallelism
        return 1


def _maybe_apply_calibration(
    estimator,
    model_name: str,
    config: TrainingConfig,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    random_state: int,
):
    """
    Apply optional post-hoc probability calibration.

    Rules:
    - LinSVM_cal: Already calibrated (skip)
    - LR/RF/XGBoost: Apply CalibratedClassifierCV if config.calibration.enabled=True
    - Already wrapped: Skip double-calibration

    Args:
        estimator: Fitted estimator or pipeline
        model_name: Model identifier
        config: TrainingConfiguration object
        X_train: Training features (for calibration CV)
        y_train: Training labels (for calibration CV)
        random_state: Random seed

    Returns:
        Calibrated or original estimator
    """
    if not config.calibration.enabled:
        return estimator

    # SVM is already calibrated
    if model_name == "LinSVM_cal":
        return estimator

    # Don't double-calibrate
    if isinstance(estimator, CalibratedClassifierCV):
        return estimator

    # Wrap with calibration
    calibrated = CalibratedClassifierCV(
        estimator=estimator, method=config.calibration.method, cv=config.calibration.cv
    )

    # Fit on training fold
    calibrated.fit(X_train, y_train)
    return calibrated


def _extract_selected_proteins_from_fold(
    fitted_model,
    model_name: str,
    protein_cols: List[str],
    config: TrainingConfig,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    random_state: int,
) -> List[str]:
    """
    Extract selected proteins from a fitted model/pipeline.

    Strategy depends on model type and feature selection method:
    - Linear models (LR/SVM): Extract proteins with |coef| > threshold
    - Tree models (RF/XGBoost): Use permutation importance (if enabled)
    - K-best selection: Extract from SelectKBest step
    - Screening: Extract from TrainOnlyScreenSelector step

    Args:
        fitted_model: Fitted pipeline or calibrated estimator
        model_name: Model identifier
        protein_cols: List of protein column names
        config: TrainingConfiguration object
        X_train: Training fold features (for permutation importance)
        y_train: Training fold labels (for permutation importance)
        random_state: Random seed

    Returns:
        List of selected protein names (sorted)
    """
    # Unwrap CalibratedClassifierCV if needed
    if isinstance(fitted_model, CalibratedClassifierCV):
        if hasattr(fitted_model, "estimator"):
            pipeline = fitted_model.estimator
        else:
            # Older sklearn uses base_estimator
            pipeline = getattr(fitted_model, "base_estimator", fitted_model)
    else:
        pipeline = fitted_model

    if not isinstance(pipeline, Pipeline):
        return []

    selected_proteins = set()

    # Strategy 1: Extract from screening step (if present)
    if "screen" in pipeline.named_steps:
        screen_selected = getattr(pipeline.named_steps["screen"], "selected_proteins_", [])
        if screen_selected:
            selected_proteins.update(screen_selected)

    # Strategy 2: Extract from K-best selection (if present)
    feature_select = config.features.feature_select

    if feature_select in ("kbest", "hybrid"):
        kbest_scope = config.features.kbest_scope

        if kbest_scope == "protein" and "prot_sel" in pipeline.named_steps:
            # Protein-level K-best
            prot_sel_proteins = getattr(pipeline.named_steps["prot_sel"], "selected_proteins_", [])
            if prot_sel_proteins:
                selected_proteins.update(prot_sel_proteins)

        elif "sel" in pipeline.named_steps:
            # Transformed-space K-best
            kbest_proteins = _extract_from_kbest_transformed(pipeline, protein_cols)
            if kbest_proteins:
                selected_proteins.update(kbest_proteins)

    # Strategy 3: Extract from model coefficients (linear models)
    if feature_select in ("l1_stability", "hybrid"):
        model_proteins = _extract_from_model_coefficients(
            pipeline, model_name, protein_cols, config
        )
        if model_proteins:
            selected_proteins.update(model_proteins)

    # Strategy 4: Permutation importance for RF (if enabled and hybrid mode)
    if (
        feature_select == "hybrid"
        and model_name == "RF"
        and config.features.rf_use_permutation
    ):
        perm_proteins = _extract_from_rf_permutation(
            pipeline, X_train, y_train, protein_cols, config, random_state
        )
        if perm_proteins:
            selected_proteins.update(perm_proteins)

    return sorted(selected_proteins)


def _extract_from_kbest_transformed(pipeline: Pipeline, protein_cols: List[str]) -> set:
    """Extract protein names from SelectKBest in transformed space."""
    if "sel" not in pipeline.named_steps:
        return set()

    # Get feature names from preprocessing step
    pre = pipeline.named_steps["pre"]
    if not hasattr(pre, "get_feature_names_out"):
        return set()

    feature_names = pre.get_feature_names_out()
    support = pipeline.named_steps["sel"].get_support()
    selected_names = feature_names[support]

    # Extract protein columns (num__ prefix from ColumnTransformer)
    proteins = set()
    for name in selected_names:
        if name.startswith("num__"):
            orig = name[len("num__") :]
            if orig in protein_cols:
                proteins.add(orig)

    return proteins


def _extract_from_model_coefficients(
    pipeline: Pipeline, model_name: str, protein_cols: List[str], config: TrainingConfig
) -> set:
    """Extract protein names from linear model coefficients."""
    coef_thresh = config.features.coef_threshold

    # Get feature names
    pre = pipeline.named_steps["pre"]
    if not hasattr(pre, "get_feature_names_out"):
        return set()

    feature_names = pre.get_feature_names_out()

    # Apply K-best mask if present
    if "sel" in pipeline.named_steps:
        support = pipeline.named_steps["sel"].get_support()
        feature_names = feature_names[support]

    # Extract coefficients
    clf = pipeline.named_steps["clf"]

    # Handle CalibratedClassifierCV wrapper for LinSVM
    if model_name == "LinSVM_cal" and hasattr(clf, "calibrated_classifiers_"):
        # Average coefficients across calibration folds
        coefs_list = []
        for cc in clf.calibrated_classifiers_:
            est = getattr(cc, "estimator", None)
            if est and hasattr(est, "coef_"):
                coefs_list.append(est.coef_.ravel())

        if not coefs_list:
            return set()

        coefs = np.mean(np.vstack(coefs_list), axis=0)

    elif hasattr(clf, "coef_"):
        # Standard linear model
        coefs = clf.coef_.ravel()

    else:
        return set()

    # Sanity check
    if len(feature_names) != len(coefs):
        logger.warning(
            f"[extract] WARNING: Feature names length ({len(feature_names)}) != "
            f"coef length ({len(coefs)}); skipping extraction"
        )
        return set()

    # Extract proteins with |coef| > threshold
    proteins = set()
    for name, c in zip(feature_names, coefs):
        if name.startswith("num__"):
            orig = name[len("num__") :]
            if orig in protein_cols and abs(c) > coef_thresh:
                proteins.add(orig)

    return proteins


def _extract_from_rf_permutation(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    protein_cols: List[str],
    config: TrainingConfig,
    random_state: int,
) -> set:
    """Extract important proteins from RF using permutation importance."""
    from sklearn.inspection import permutation_importance

    # Get feature names
    pre = pipeline.named_steps["pre"]
    if not hasattr(pre, "get_feature_names_out"):
        return set()

    feature_names = pre.get_feature_names_out()

    # Apply K-best mask if present
    if "sel" in pipeline.named_steps:
        support = pipeline.named_steps["sel"].get_support()
        feature_names = feature_names[support]

    # Compute permutation importance
    try:
        perm_result = permutation_importance(
            pipeline,
            X_train,
            y_train,
            scoring=config.cv.scoring,
            n_repeats=config.features.rf_perm_repeats,
            random_state=random_state,
            n_jobs=1,  # Already inside parallel context
        )
        importances = perm_result.importances_mean
    except Exception as e:
        logger.warning(f"[perm] WARNING: Permutation importance failed: {e}")
        return set()

    # Sanity check
    if len(feature_names) != len(importances):
        return set()

    # Aggregate protein importances (sum across transformed features)
    protein_importance: Dict[str, float] = {}
    for name, imp in zip(feature_names, importances):
        if not np.isfinite(imp):
            continue
        if name.startswith("num__"):
            orig = name[len("num__") :]
            if orig in protein_cols:
                protein_importance[orig] = protein_importance.get(orig, 0.0) + float(imp)

    if not protein_importance:
        return set()

    # Filter by minimum importance
    min_imp = config.features.rf_perm_min_importance
    filtered = [(p, v) for p, v in protein_importance.items() if v >= min_imp]

    if not filtered:
        # If all below threshold, keep all
        filtered = list(protein_importance.items())

    # Sort by importance and take top N
    filtered.sort(key=lambda x: x[1], reverse=True)
    top_n = config.features.rf_perm_top_n
    top_proteins = [p for p, _ in filtered[:top_n]]

    return set(top_proteins)
