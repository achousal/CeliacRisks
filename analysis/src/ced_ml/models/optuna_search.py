"""
Optuna-based hyperparameter search wrapper.

Provides OptunaSearchCV - a drop-in replacement for sklearn's RandomizedSearchCV
that uses Optuna for more efficient hyperparameter optimization.

Features:
- Compatible sklearn interface (best_estimator_, best_params_, best_score_, cv_results_)
- Supports TPE, Random, CMA-ES, and Grid samplers
- Supports Median, Percentile, and Hyperband pruners
- Graceful fallback when optuna is not installed
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold, cross_val_score

logger = logging.getLogger(__name__)

# Default seed used when neither sampler_seed nor random_state is provided.
# Named constant to avoid magic number and make the behavior explicit.
_DEFAULT_SEED_FALLBACK = 0

# Attempt optuna import with graceful fallback
_OPTUNA_AVAILABLE = False
try:
    import optuna
    from optuna.pruners import (
        HyperbandPruner,
        MedianPruner,
        NopPruner,
        PercentilePruner,
    )
    from optuna.samplers import (
        CmaEsSampler,
        GridSampler,
        RandomSampler,
        TPESampler,
    )

    _OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None  # type: ignore[assignment]


def optuna_available() -> bool:
    """Check if optuna is installed and available."""
    return _OPTUNA_AVAILABLE


class OptunaSearchCV(BaseEstimator):
    """
    Optuna-based hyperparameter search with sklearn-compatible interface.

    Drop-in replacement for RandomizedSearchCV using Optuna's efficient
    hyperparameter optimization algorithms (TPE, CMA-ES, etc.).

    Parameters
    ----------
    estimator : BaseEstimator
        The sklearn estimator or pipeline to tune.
    param_distributions : dict
        Parameter search space. Each value should be a dict with:
        - type: "int", "float", or "categorical"
        - For int/float: low, high, log (optional bool)
        - For categorical: choices (list)
    n_trials : int, default=100
        Number of optimization trials.
    timeout : float, optional
        Stop study after this many seconds.
    scoring : str or callable, default="accuracy"
        Scoring metric for cross-validation.
    cv : int or CV splitter, default=5
        Cross-validation strategy.
    n_jobs : int, default=1
        Number of parallel jobs for CV (not study parallelization).
    random_state : int, optional
        Random seed for CV splitting. Also used as sampler seed if sampler_seed
        is not explicitly provided.
    refit : bool, default=True
        Whether to refit best estimator on full training data.
    direction : {"minimize", "maximize"}, default="maximize"
        Optimization direction.
    sampler : {"tpe", "random", "cmaes", "grid"}, default="tpe"
        Optuna sampler type.
    sampler_seed : int, optional
        Seed for the Optuna sampler. If None, uses random_state instead.
        Setting this explicitly allows different seeds for CV splitting vs
        hyperparameter sampling.
    pruner : {"median", "percentile", "hyperband", "none"}, default="hyperband"
        Optuna pruner type. HyperbandPruner is recommended for TPE sampler.
    pruner_n_startup_trials : int, default=5
        Number of trials before pruning starts.
    pruner_percentile : float, default=25.0
        Percentile threshold for PercentilePruner.
    storage : str, optional
        Optuna storage URL (e.g., "sqlite:///study.db").
    study_name : str, optional
        Name for the Optuna study.
    load_if_exists : bool, default=False
        Load existing study if it exists.
    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    best_params_ : dict
        Best hyperparameters found.
    best_score_ : float
        Best CV score achieved.
    best_estimator_ : BaseEstimator
        Estimator fitted with best parameters (if refit=True).
    cv_results_ : dict
        Dictionary with trial results (compatible with sklearn).
    study_ : optuna.Study
        The underlying Optuna study object.
    n_trials_ : int
        Number of completed trials.

    Notes
    -----
    Pruning limitation: This implementation uses sklearn's cross_val_score,
    which does not report intermediate values during CV fold evaluation.
    As a result, pruning only takes effect between full CV evaluations (i.e.,
    between trials), not within a trial's CV folds. For full pruning benefits,
    consider using Optuna's native integration or reporting intermediate
    values manually.

    TPE + Hyperband: When using TPE sampler with HyperbandPruner, Optuna
    recommends at least 40 trials for the TPE startup to gather sufficient
    observations. A warning is logged if n_trials < 40 with this combination.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        param_distributions: dict[str, Any],
        *,
        n_trials: int = 100,
        timeout: float | None = None,
        scoring: str | Callable = "accuracy",
        cv: int | Any = 5,
        n_jobs: int = 1,
        random_state: int | None = None,
        refit: bool = True,
        direction: Literal["minimize", "maximize"] = "maximize",
        sampler: Literal["tpe", "random", "cmaes", "grid"] = "tpe",
        sampler_seed: int | None = None,
        pruner: Literal["median", "percentile", "hyperband", "none"] = "hyperband",
        pruner_n_startup_trials: int = 5,
        pruner_percentile: float = 25.0,
        storage: str | None = None,
        study_name: str | None = None,
        load_if_exists: bool = False,
        verbose: int = 0,
    ):
        if not optuna_available():
            raise ImportError("Optuna is not installed. Install with: pip install optuna")

        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_trials = n_trials
        self.timeout = timeout
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.refit = refit
        self.direction = direction
        self.sampler = sampler
        self.sampler_seed = sampler_seed
        self.pruner = pruner
        self.pruner_n_startup_trials = pruner_n_startup_trials
        self.pruner_percentile = pruner_percentile
        self.storage = storage
        self.study_name = study_name
        self.load_if_exists = load_if_exists
        self.verbose = verbose

        # Attributes set during fit
        self.best_params_: dict[str, Any] = {}
        self.best_score_: float = np.nan
        self.best_estimator_: BaseEstimator | None = None
        self.cv_results_: dict[str, list] = {}
        self.study_: optuna.Study | None = None
        self.n_trials_: int = 0

    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create Optuna sampler based on configuration."""
        # Determine seed with explicit precedence
        if self.sampler_seed is not None:
            seed = self.sampler_seed
        elif self.random_state is not None:
            seed = self.random_state
        else:
            seed = _DEFAULT_SEED_FALLBACK
            logger.warning(
                "Both sampler_seed and random_state are None. "
                "Defaulting to seed=%d for determinism. "
                "Consider setting random_state explicitly for reproducibility.",
                _DEFAULT_SEED_FALLBACK,
            )

        if self.sampler == "tpe":
            return TPESampler(seed=seed)
        elif self.sampler == "random":
            return RandomSampler(seed=seed)
        elif self.sampler == "cmaes":
            return CmaEsSampler(seed=seed)
        elif self.sampler == "grid":
            # Grid sampler requires explicit search space
            search_space = self._build_grid_search_space()
            return GridSampler(search_space, seed=seed)
        else:
            raise ValueError(f"Unknown sampler: {self.sampler}")

    def _build_grid_search_space(self) -> dict[str, list]:
        """Build search space for GridSampler from param_distributions."""
        search_space = {}
        for name, spec in self.param_distributions.items():
            if spec.get("type") == "categorical":
                search_space[name] = spec["choices"]
            elif spec.get("type") in ("int", "float"):
                # Create a small grid for continuous params
                low, high = spec["low"], spec["high"]
                if spec.get("log", False):
                    if low <= 0:
                        raise ValueError(
                            f"Cannot use log scale for param {name}: low={low} must be > 0"
                        )
                    values = np.logspace(np.log10(low), np.log10(high), num=5).tolist()
                else:
                    values = np.linspace(low, high, num=5).tolist()
                if spec["type"] == "int":
                    values = sorted({int(v) for v in values})
                search_space[name] = values
            else:
                raise ValueError(f"Cannot build grid for param {name}: {spec}")
        return search_space

    def _create_pruner(self) -> optuna.pruners.BasePruner:
        """Create Optuna pruner based on configuration."""
        if self.pruner == "median":
            return MedianPruner(n_startup_trials=self.pruner_n_startup_trials)
        elif self.pruner == "percentile":
            return PercentilePruner(
                percentile=self.pruner_percentile,
                n_startup_trials=self.pruner_n_startup_trials,
            )
        elif self.pruner == "hyperband":
            return HyperbandPruner()
        elif self.pruner == "none":
            return NopPruner()
        else:
            raise ValueError(f"Unknown pruner: {self.pruner}")

    def _suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Suggest hyperparameters for a trial based on param_distributions."""
        params = {}
        for name, spec in self.param_distributions.items():
            param_type = spec.get("type", "categorical")

            if param_type == "int":
                params[name] = trial.suggest_int(
                    name,
                    spec["low"],
                    spec["high"],
                    log=spec.get("log", False),
                )
            elif param_type == "float":
                params[name] = trial.suggest_float(
                    name,
                    spec["low"],
                    spec["high"],
                    log=spec.get("log", False),
                )
            elif param_type == "categorical":
                # Handle unhashable types (dicts) by converting to/from tuples
                choices = spec["choices"]
                hashable_choices = []
                for choice in choices:
                    if isinstance(choice, dict):
                        # Convert dict to tuple of tuples for hashing
                        hashable_choices.append(tuple(sorted(choice.items())))
                    else:
                        hashable_choices.append(choice)

                suggested = trial.suggest_categorical(name, hashable_choices)

                # Convert back to dict if needed
                if isinstance(suggested, tuple) and all(
                    isinstance(item, tuple) and len(item) == 2 for item in suggested
                ):
                    params[name] = dict(suggested)
                else:
                    params[name] = suggested
            else:
                raise ValueError(f"Unknown param type for {name}: {param_type}")

        return params

    def fit(self, X, y, **fit_params) -> OptunaSearchCV:
        """
        Run Optuna hyperparameter optimization.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        **fit_params : dict
            Additional fit parameters (currently unused).

        Returns
        -------
        self : OptunaSearchCV
            Fitted instance.
        """
        # Keep X as-is (DataFrame or array) for pipeline compatibility
        # ColumnTransformer with string column names requires DataFrame input
        X_arr = X
        # Convert y to array for safe indexing
        y_arr = np.asarray(y)

        # Setup CV splitter
        if isinstance(self.cv, int):
            cv_splitter = StratifiedKFold(
                n_splits=self.cv,
                shuffle=True,
                random_state=self.random_state,
            )
        else:
            cv_splitter = self.cv

        # Create sampler and pruner
        sampler = self._create_sampler()
        pruner = self._create_pruner()

        # Warn about TPE + Hyperband needing sufficient trials for startup
        if self.sampler == "tpe" and self.pruner == "hyperband" and self.n_trials < 40:
            logger.warning(
                f"[optuna] TPE sampler with HyperbandPruner typically needs 40+ trials "
                f"for effective optimization (n_trials={self.n_trials}). "
                "Consider increasing n_trials or using sampler='random' for fewer trials."
            )

        # Set optuna verbosity
        if self.verbose == 0:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        elif self.verbose == 1:
            optuna.logging.set_verbosity(optuna.logging.INFO)
        else:
            optuna.logging.set_verbosity(optuna.logging.DEBUG)

        # Create or load study
        self.study_ = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=self.load_if_exists,
            direction=self.direction,
            sampler=sampler,
            pruner=pruner,
        )

        # Validate existing study compatibility
        if self.load_if_exists and len(self.study_.trials) > 0:
            if self.study_.direction.name.lower() != self.direction:
                logger.warning(
                    f"[optuna] Loaded study has direction={self.study_.direction.name.lower()}, "
                    f"but requested direction={self.direction}. Using existing study's direction."
                )

        # Create objective with CV splitter
        def objective(trial: optuna.Trial) -> float:
            params = self._suggest_params(trial)
            estimator = clone(self.estimator)
            try:
                estimator.set_params(**params)
            except ValueError as e:
                logger.warning(f"[optuna] Invalid params {params}: {e}")
                raise optuna.TrialPruned() from e

            try:
                scores = cross_val_score(
                    estimator,
                    X_arr,
                    y_arr,
                    cv=cv_splitter,
                    scoring=self.scoring,
                    n_jobs=self.n_jobs,
                )
                return float(np.mean(scores))
            except Exception as e:
                logger.warning(f"[optuna] CV failed for params {params}: {e}")
                raise optuna.TrialPruned() from e

        # Run optimization
        self.study_.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=self.verbose > 0,
        )

        # Extract results
        self.n_trials_ = len(self.study_.trials)

        # Check if any trials completed successfully
        completed_trials = [
            t for t in self.study_.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if not completed_trials:
            raise RuntimeError(
                f"All {self.n_trials_} Optuna trials failed. "
                "Check logs for error messages. This may indicate incompatible "
                "hyperparameters, insufficient data, or other issues."
            )

        self.best_params_ = self.study_.best_params
        self.best_score_ = self.study_.best_value

        # Build cv_results_ for sklearn compatibility
        self.cv_results_ = self._build_cv_results()

        # Refit best estimator
        if self.refit:
            self.best_estimator_ = clone(self.estimator)
            self.best_estimator_.set_params(**self.best_params_)
            self.best_estimator_.fit(X_arr, y_arr)

        logger.info(
            f"[optuna] Completed {self.n_trials_} trials. " f"Best score: {self.best_score_:.4f}"
        )

        return self

    def _build_cv_results(self) -> dict[str, list]:
        """Build sklearn-compatible cv_results_ from Optuna study."""
        if self.study_ is None:
            return {}

        results: dict[str, list] = {
            "mean_test_score": [],
            "rank_test_score": [],
            "params": [],
        }

        # Add param columns
        for param_name in self.param_distributions.keys():
            results[f"param_{param_name}"] = []

        # Populate from trials
        for trial in self.study_.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                results["mean_test_score"].append(trial.value)
                results["params"].append(trial.params)

                for param_name in self.param_distributions.keys():
                    results[f"param_{param_name}"].append(trial.params.get(param_name))

        # Compute ranks
        if results["mean_test_score"]:
            scores = np.array(results["mean_test_score"])
            if self.direction == "maximize":
                ranks = np.argsort(np.argsort(-scores)) + 1
            else:
                ranks = np.argsort(np.argsort(scores)) + 1
            results["rank_test_score"] = ranks.tolist()

        return results

    def predict(self, X):
        """Predict using the best estimator."""
        if self.best_estimator_ is None:
            raise ValueError("Estimator not fitted. Call fit() first.")
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        """Predict probabilities using the best estimator."""
        if self.best_estimator_ is None:
            raise ValueError("Estimator not fitted. Call fit() first.")
        if not hasattr(self.best_estimator_, "predict_proba"):
            raise AttributeError(
                f"The estimator {type(self.best_estimator_).__name__} does not have a predict_proba method."
            )
        return self.best_estimator_.predict_proba(X)

    def score(self, X, y):
        """Return the score of the best estimator on the given data."""
        if self.best_estimator_ is None:
            raise ValueError("Estimator not fitted. Call fit() first.")
        return self.best_estimator_.score(X, y)

    def get_trials_dataframe(self):
        """Return trials as a pandas DataFrame (convenience wrapper)."""
        if self.study_ is None:
            raise ValueError("Study not created. Call fit() first.")
        return self.study_.trials_dataframe()
