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
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from ..config import TrainingConfig
from ..metrics.scorers import get_scorer

if TYPE_CHECKING:
    from .calibration import OOFCalibrator

logger = logging.getLogger(__name__)


def get_model_n_iter(model_name: str, config: TrainingConfig) -> int:
    """
    Get n_iter for a model, using model-specific override or global fallback.

    Args:
        model_name: Model identifier (LR_EN, LR_L1, LinSVM_cal, RF, XGBoost)
        config: TrainingConfig object

    Returns:
        n_iter value (>= 1)
    """
    model_configs = {
        "LR_EN": config.lr,
        "LR_L1": config.lr,
        "LinSVM_cal": config.svm,
        "RF": config.rf,
        "XGBoost": config.xgboost,
    }
    model_cfg = model_configs.get(model_name)
    if model_cfg is not None and getattr(model_cfg, "n_iter", None) is not None:
        return model_cfg.n_iter
    return config.cv.n_iter


def _convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list | tuple):
        return [_convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def oof_predictions_with_nested_cv(
    pipeline: Pipeline,
    model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    protein_cols: list[str],
    config: TrainingConfig,
    random_state: int,
    grid_rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float, pd.DataFrame, pd.DataFrame, "OOFCalibrator | None"]:
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
               If calibration strategy is "oof_posthoc", predictions are calibrated.
        elapsed_sec: Training time in seconds
        best_params_df: DataFrame with best hyperparameters per fold
        selected_proteins_df: DataFrame with selected proteins per fold
        oof_calibrator: OOFCalibrator instance if strategy is "oof_posthoc", else None.
                        Use this calibrator for val/test predictions.

    Raises:
        RuntimeError: If any repeat has missing OOF predictions (CV split bug)
    """
    from .calibration import OOFCalibrator

    n_splits = config.cv.folds
    n_repeats = config.cv.repeats
    calibration_strategy = config.calibration.get_strategy_for_model(model_name)

    if n_repeats < 1:
        raise ValueError(f"cv.repeats must be >= 1, got {n_repeats}")

    n_samples = len(y)
    preds = np.full((n_repeats, n_samples), np.nan, dtype=float)
    best_params_rows: list[dict[str, Any]] = []
    selected_proteins_rows: list[dict[str, Any]] = []

    # Validate outer CV folds
    if n_splits < 2:
        raise ValueError(
            f"cv.folds must be >= 2 for cross-validation, got {n_splits}. "
            "In-sample predictions defeat the purpose of CV and lead to overfitting."
        )

    total_outer_folds = n_splits * n_repeats
    split_idx = 0
    t0 = time.perf_counter()

    # Setup outer CV splitter
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
            except Exception as e:
                logger.warning(
                    f"Failed to set scale_pos_weight={xgb_spw} for XGBoost in fold {split_idx}: {e}. "
                    "Model will train with default class weighting."
                )

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

        # Optional post-hoc calibration (only for per_fold strategy)
        # For oof_posthoc, calibration happens after CV loop completes
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
        best_params_row = {
            "model": model_name,
            "repeat": repeat_num,
            "outer_split": split_idx,
            "best_score_inner": best_score,
            "best_params": json.dumps(_convert_numpy_types(best_params), sort_keys=True),
        }

        # Add Optuna-specific metadata if available
        if (
            search is not None
            and hasattr(search, "study_")
            and search.study_ is not None
            and hasattr(search, "n_trials_")
        ):
            best_params_row["optuna_n_trials"] = search.n_trials_
            best_params_row["optuna_sampler"] = config.optuna.sampler
            best_params_row["optuna_pruner"] = config.optuna.pruner

        best_params_rows.append(best_params_row)

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

    # Handle oof_posthoc calibration: fit calibrator on pooled OOF predictions
    oof_calibrator = None
    if calibration_strategy == "oof_posthoc":
        logger.info(
            f"Fitting OOF calibrator (method={config.calibration.method}) on pooled OOF predictions..."
        )
        # Use mean across repeats for calibration fitting
        mean_oof_preds = np.nanmean(preds, axis=0)
        oof_calibrator = OOFCalibrator(method=config.calibration.method)
        oof_calibrator.fit(mean_oof_preds, y)
        logger.info("OOF calibrator fitted successfully")

        # Apply calibration to OOF predictions for consistent reporting
        for r in range(n_repeats):
            preds[r] = oof_calibrator.transform(preds[r])

    return (
        preds,
        elapsed_sec,
        pd.DataFrame(best_params_rows),
        pd.DataFrame(selected_proteins_rows),
        oof_calibrator,
    )


def _compute_xgb_scale_pos_weight(y_train: np.ndarray, config: TrainingConfig) -> float:
    """
    Compute XGBoost scale_pos_weight parameter from training labels.

    Default: ratio of negatives to positives (auto class balancing)
    Override: config.xgboost.scale_pos_weight_grid[0] if explicitly set

    Args:
        y_train: Training labels (0/1)
        config: TrainingConfiguration object

    Returns:
        scale_pos_weight value (>= 1.0)
    """
    # Check if user specified explicit value via scale_pos_weight_grid
    spw_grid = getattr(config.xgboost, "scale_pos_weight_grid", None)
    if spw_grid and len(spw_grid) == 1 and spw_grid[0] > 0:
        return float(spw_grid[0])

    # Auto: ratio of negatives to positives
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))

    if n_pos == 0:
        logger.warning("[xgb] No positive samples in training fold; using spw=1.0")
        return 1.0

    spw = float(n_neg) / float(n_pos)
    return max(1.0, spw)


def _scoring_to_direction(scoring: str) -> str:
    """
    Infer Optuna optimization direction from sklearn scoring metric.

    Args:
        scoring: sklearn scoring string (e.g., "roc_auc", "average_precision")

    Returns:
        "maximize" or "minimize"
    """
    # Metrics that should be maximized
    maximize_metrics = {
        "roc_auc",
        "average_precision",
        "f1",
        "f1_weighted",
        "f1_micro",
        "f1_macro",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "jaccard",
        "tpr_at_fpr",
    }

    # neg_* metrics are also maximized (sklearn convention)
    if scoring in maximize_metrics or scoring.startswith("neg_"):
        return "maximize"

    return "maximize"  # Default to maximize for unknown metrics  # Default to maximize for unknown metrics


def _build_hyperparameter_search(
    pipeline: Pipeline,
    model_name: str,
    config: TrainingConfig,
    random_state: int,
    xgb_spw: float | None,
    grid_rng: np.random.Generator | None,
):
    """
    Build hyperparameter search object (Optuna or RandomizedSearchCV).

    Returns None if:
    - Model has no hyperparameters to tune
    - config.cv.inner_folds < 2 (tuning disabled)
    - n_iter < 1 (tuning disabled, for RandomizedSearchCV; uses get_model_n_iter())

    Args:
        pipeline: Base pipeline to tune
        model_name: Model identifier
        config: TrainingConfiguration object
        random_state: Random seed
        xgb_spw: XGBoost scale_pos_weight (if applicable)
        grid_rng: Optional RNG for grid randomization

    Returns:
        OptunaSearchCV, RandomizedSearchCV, or None
    """
    # Validate inner CV settings
    inner_folds = config.cv.inner_folds
    if inner_folds < 2:
        logger.info(
            f"[tune] WARNING: inner_folds={inner_folds} < 2; skipping hyperparameter search."
        )
        return None

    # Build inner CV splitter
    inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=random_state)

    # === Optuna path ===
    if config.optuna.enabled:
        from .hyperparams import get_param_distributions_optuna
        from .optuna_search import OptunaSearchCV, optuna_available

        if not optuna_available():
            logger.warning(
                "[optuna] Optuna not installed; falling back to RandomizedSearchCV. "
                "Install with: pip install optuna"
            )
        else:
            # Get Optuna-format parameter distributions
            param_dists = get_param_distributions_optuna(model_name, config, xgb_spw=xgb_spw)

            if not param_dists:
                logger.info(f"[optuna] No tunable params for {model_name}; skipping search.")
                return None

            # Determine direction from config or scoring metric
            direction = config.optuna.direction
            if direction is None:
                direction = _scoring_to_direction(config.cv.scoring)

            logger.info(
                f"[optuna] Using Optuna: {config.optuna.n_trials} trials, "
                f"sampler={config.optuna.sampler}, direction={direction}"
            )

            # Get scorer (handles custom scorers like tpr_at_fpr)
            scorer = get_scorer(config.cv.scoring, target_fpr=config.cv.scoring_target_fpr)

            return OptunaSearchCV(
                estimator=pipeline,
                param_distributions=param_dists,
                n_trials=config.optuna.n_trials,
                timeout=config.optuna.timeout,
                scoring=scorer,
                cv=inner_cv,
                n_jobs=config.optuna.n_jobs,
                random_state=random_state,
                refit=True,
                direction=direction,
                sampler=config.optuna.sampler,
                sampler_seed=config.optuna.sampler_seed,
                pruner=config.optuna.pruner,
                pruner_n_startup_trials=config.optuna.pruner_n_startup_trials,
                pruner_percentile=config.optuna.pruner_percentile,
                storage=config.optuna.storage,
                study_name=config.optuna.study_name,
                load_if_exists=config.optuna.load_if_exists,
                verbose=0,
            )

    # === RandomizedSearchCV path (default) ===
    from sklearn.model_selection import RandomizedSearchCV

    from .hyperparams import get_param_distributions

    n_iter = get_model_n_iter(model_name, config)
    if n_iter < 1:
        logger.warning(f"[tune] WARNING: n_iter={n_iter} < 1; skipping hyperparameter search.")
        return None

    logger.info(f"[tune] {model_name}: using n_iter={n_iter}")

    # Get sklearn-format parameter distributions
    param_dists = get_param_distributions(model_name, config, xgb_spw=xgb_spw, grid_rng=grid_rng)

    if not param_dists:
        return None

    # Determine parallelization strategy
    n_jobs = _get_search_n_jobs(model_name, config)

    # Get scorer (handles custom scorers like tpr_at_fpr)
    scorer = get_scorer(config.cv.scoring, target_fpr=config.cv.scoring_target_fpr)

    return RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dists,
        n_iter=n_iter,
        scoring=scorer,
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
    Apply optional post-hoc probability calibration (per-fold strategy only).

    This function is called during the CV loop. It only applies calibration
    when the strategy is "per_fold". For "oof_posthoc" and "none" strategies,
    it returns the estimator unchanged.

    Rules:
    - strategy="none": Return estimator unchanged
    - strategy="oof_posthoc": Return estimator unchanged (calibration happens post-CV)
    - strategy="per_fold": Apply CalibratedClassifierCV (current behavior)
    - LinSVM_cal: Already calibrated (skip regardless of strategy)
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
    # Get effective strategy for this model
    strategy = config.calibration.get_strategy_for_model(model_name)

    # For none or oof_posthoc, return unchanged
    if strategy in ("none", "oof_posthoc"):
        return estimator

    # per_fold strategy: apply CalibratedClassifierCV
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
    protein_cols: list[str],
    config: TrainingConfig,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    random_state: int,
) -> list[str]:
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
    if feature_select == "hybrid" and model_name == "RF" and config.features.rf_use_permutation:
        perm_proteins = _extract_from_rf_permutation(
            pipeline, X_train, y_train, protein_cols, config, random_state
        )
        if perm_proteins:
            selected_proteins.update(perm_proteins)

    return sorted(selected_proteins)


def _extract_from_kbest_transformed(pipeline: Pipeline, protein_cols: list[str]) -> set:
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

    # Extract protein columns (handle different feature naming patterns)
    proteins = set()
    for name in selected_names:
        # Handle different feature naming patterns:
        # - {protein} (plain name, verbose_feature_names_out=False)
        # - num__{protein} (legacy/standard sklearn ColumnTransformer with prefix)
        # - {protein}_resid (ResidualTransformer output)
        if name in protein_cols:
            proteins.add(name)
        elif name.startswith("num__"):
            orig = name[len("num__") :]
            if orig in protein_cols:
                proteins.add(orig)
        elif name.endswith("_resid"):
            orig = name[: -len("_resid")]
            if orig in protein_cols:
                proteins.add(orig)

    return proteins


def _extract_from_model_coefficients(
    pipeline: Pipeline, model_name: str, protein_cols: list[str], config: TrainingConfig
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
    for name, c in zip(feature_names, coefs, strict=False):
        # Handle different feature naming patterns:
        # - num__{protein} (legacy/standard sklearn ColumnTransformer)
        # - {protein}_resid (ResidualTransformer output)
        if name.startswith("num__"):
            orig = name[len("num__") :]
        elif name.endswith("_resid"):
            orig = name[: -len("_resid")]
        else:
            continue

        if orig in protein_cols and abs(c) > coef_thresh:
            proteins.add(orig)

    return proteins


def _extract_from_rf_permutation(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    protein_cols: list[str],
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
        # Get scorer (handles custom scorers like tpr_at_fpr)
        scorer = get_scorer(config.cv.scoring, target_fpr=config.cv.scoring_target_fpr)

        perm_result = permutation_importance(
            pipeline,
            X_train,
            y_train,
            scoring=scorer,
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
    protein_importance: dict[str, float] = {}
    for name, imp in zip(feature_names, importances, strict=False):
        if not np.isfinite(imp):
            continue

        # Handle different feature naming patterns:
        # - num__{protein} (legacy/standard sklearn ColumnTransformer)
        # - {protein}_resid (ResidualTransformer output)
        if name.startswith("num__"):
            orig = name[len("num__") :]
        elif name.endswith("_resid"):
            orig = name[: -len("_resid")]
        else:
            continue

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
