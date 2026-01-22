"""Model stacking ensemble for combining base model predictions.

This module provides a StackingEnsemble class that trains a meta-learner
on out-of-fold (OOF) predictions from multiple base models to improve
overall predictive performance.

Architecture:
    1. Base models are trained independently (via standard training pipeline)
    2. OOF predictions from each base model are collected
    3. Meta-learner (Logistic Regression with L2) is trained on stacked OOF predictions
    4. Final predictions combine base model outputs through the meta-learner

Expected improvement: +2-5% AUROC over best single model.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class BaseModelBundle:
    """Container for base model information needed for stacking.

    Attributes:
        model_name: Identifier (e.g., 'LR_EN', 'RF')
        model_path: Path to saved model bundle
        oof_predictions: OOF predictions array (n_repeats x n_samples)
        val_predictions: Validation set predictions
        test_predictions: Test set predictions
        train_indices: Training sample indices
        val_indices: Validation sample indices
        test_indices: Test sample indices
    """

    model_name: str
    model_path: Path | None = None
    oof_predictions: np.ndarray | None = None
    val_predictions: np.ndarray | None = None
    test_predictions: np.ndarray | None = None
    train_indices: np.ndarray | None = None
    val_indices: np.ndarray | None = None
    test_indices: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class StackingEnsemble(BaseEstimator, ClassifierMixin):
    """Stacking ensemble that combines base model predictions via a meta-learner.

    The meta-learner is trained on out-of-fold (OOF) predictions from multiple
    base models. This approach prevents information leakage by using predictions
    that were generated when each sample was in the validation fold.

    Attributes:
        base_model_names: List of base model identifiers
        meta_model: Fitted meta-learner (LogisticRegression)
        base_models: Dict mapping model name to loaded model bundle
        scaler: Optional feature scaler for meta-learner input
        classes_: Class labels [0, 1]
        is_fitted_: Whether the ensemble has been fitted

    Example:
        >>> ensemble = StackingEnsemble(
        ...     base_model_names=['LR_EN', 'RF', 'XGBoost'],
        ...     meta_penalty='l2',
        ...     meta_C=1.0
        ... )
        >>> ensemble.fit_from_oof(oof_dict, y_train)
        >>> test_proba = ensemble.predict_proba_from_base_preds(test_preds_dict)
    """

    def __init__(
        self,
        base_model_names: list[str] | None = None,
        meta_penalty: str = "l2",
        meta_C: float = 1.0,
        meta_max_iter: int = 1000,
        meta_solver: str = "lbfgs",
        use_probabilities: bool = True,
        scale_meta_features: bool = True,
        calibrate_meta: bool = True,
        calibration_cv: int = 5,
        random_state: int | None = None,
    ):
        """Initialize stacking ensemble.

        Args:
            base_model_names: List of base model identifiers to include
            meta_penalty: Regularization penalty for meta-learner ('l2', 'l1', 'elasticnet')
            meta_C: Inverse regularization strength for meta-learner
            meta_max_iter: Max iterations for meta-learner convergence
            meta_solver: Solver for logistic regression meta-learner
            use_probabilities: Use probabilities (True) or logits (False) as meta features
            scale_meta_features: Whether to standardize meta-learner input
            calibrate_meta: Whether to calibrate meta-learner predictions
            calibration_cv: CV folds for meta-learner calibration
            random_state: Random seed for reproducibility
        """
        self.base_model_names = base_model_names or []
        self.meta_penalty = meta_penalty
        self.meta_C = meta_C
        self.meta_max_iter = meta_max_iter
        self.meta_solver = meta_solver
        self.use_probabilities = use_probabilities
        self.scale_meta_features = scale_meta_features
        self.calibrate_meta = calibrate_meta
        self.calibration_cv = calibration_cv
        self.random_state = random_state

        # Will be set during fitting
        self.meta_model: LogisticRegression | CalibratedClassifierCV | None = None
        self.scaler: StandardScaler | None = None
        self.base_models: dict[str, Any] = {}
        self.classes_ = np.array([0, 1])
        self.is_fitted_ = False
        self._feature_names: list[str] = []

    def _build_meta_features(
        self,
        oof_dict: dict[str, np.ndarray],
        aggregate_repeats: bool = True,
    ) -> np.ndarray:
        """Build meta-feature matrix from OOF predictions.

        Args:
            oof_dict: Dict mapping model name to OOF predictions
                      Each value is (n_repeats x n_samples) or (n_samples,)
            aggregate_repeats: Whether to average across CV repeats

        Returns:
            Meta-feature matrix (n_samples x n_base_models)
        """
        features = []
        self._feature_names = []

        for model_name in self.base_model_names:
            if model_name not in oof_dict:
                raise ValueError(f"Missing OOF predictions for base model: {model_name}")

            preds = oof_dict[model_name]
            preds = np.asarray(preds)

            # Handle multi-repeat OOF predictions
            if preds.ndim == 2 and aggregate_repeats:
                # Average across repeats (axis 0)
                preds = np.nanmean(preds, axis=0)
            elif preds.ndim == 2:
                # Use first repeat only
                preds = preds[0, :]

            # Convert to logits if requested
            if not self.use_probabilities:
                preds = np.clip(preds, 1e-7, 1 - 1e-7)
                preds = np.log(preds / (1 - preds))

            features.append(preds)
            self._feature_names.append(f"oof_{model_name}")

        return np.column_stack(features)

    def fit_from_oof(
        self,
        oof_dict: dict[str, np.ndarray],
        y_train: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> StackingEnsemble:
        """Fit meta-learner on OOF predictions from base models.

        Args:
            oof_dict: Dict mapping model name to OOF predictions
                      Shape: (n_repeats x n_train) or (n_train,)
            y_train: Training labels
            sample_weight: Optional sample weights

        Returns:
            self (fitted ensemble)
        """
        logger.info(f"Fitting stacking ensemble with {len(self.base_model_names)} base models")

        # Build meta-feature matrix
        X_meta = self._build_meta_features(oof_dict, aggregate_repeats=True)
        y = np.asarray(y_train)

        # Validate shapes
        if X_meta.shape[0] != len(y):
            raise ValueError(
                f"Shape mismatch: X_meta has {X_meta.shape[0]} samples, "
                f"y_train has {len(y)} samples"
            )

        # Check for NaN values (can occur if base model had missing OOF predictions)
        nan_mask = np.isnan(X_meta).any(axis=1)
        if nan_mask.any():
            n_nan = nan_mask.sum()
            logger.warning(f"Dropping {n_nan} samples with NaN meta-features")
            X_meta = X_meta[~nan_mask]
            y = y[~nan_mask]
            if sample_weight is not None:
                sample_weight = sample_weight[~nan_mask]

        # Scale meta-features if requested
        if self.scale_meta_features:
            self.scaler = StandardScaler()
            X_meta = self.scaler.fit_transform(X_meta)

        # Build meta-learner
        base_meta = LogisticRegression(
            penalty=self.meta_penalty if self.meta_penalty != "none" else None,
            C=self.meta_C,
            max_iter=self.meta_max_iter,
            solver=self.meta_solver,
            random_state=self.random_state,
            class_weight="balanced",
        )

        # Optionally wrap in calibration
        if self.calibrate_meta and len(y) >= 2 * self.calibration_cv:
            self.meta_model = CalibratedClassifierCV(
                estimator=base_meta,
                method="isotonic",
                cv=self.calibration_cv,
            )
        else:
            self.meta_model = base_meta

        # Fit meta-learner
        logger.info(
            f"Training meta-learner on {X_meta.shape[0]} samples, {X_meta.shape[1]} features"
        )
        self.meta_model.fit(X_meta, y, sample_weight=sample_weight)
        self.is_fitted_ = True

        logger.info("Stacking ensemble fitted successfully")
        return self

    def predict_proba_from_base_preds(
        self,
        preds_dict: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Predict class probabilities using base model predictions.

        Args:
            preds_dict: Dict mapping model name to predictions (n_samples,)

        Returns:
            Probability matrix (n_samples, 2)
        """
        if not self.is_fitted_:
            raise RuntimeError("Ensemble not fitted. Call fit_from_oof first.")

        # Build meta-features from base model predictions
        features = []
        for model_name in self.base_model_names:
            if model_name not in preds_dict:
                raise ValueError(f"Missing predictions for base model: {model_name}")

            preds = np.asarray(preds_dict[model_name])

            # Handle 2D predictions (take positive class column)
            if preds.ndim == 2:
                preds = preds[:, 1]

            # Convert to logits if needed
            if not self.use_probabilities:
                preds = np.clip(preds, 1e-7, 1 - 1e-7)
                preds = np.log(preds / (1 - preds))

            features.append(preds)

        X_meta = np.column_stack(features)

        # Scale if scaler was fitted
        if self.scaler is not None:
            X_meta = self.scaler.transform(X_meta)

        # Predict with meta-learner
        return self.meta_model.predict_proba(X_meta)

    def predict_from_base_preds(
        self,
        preds_dict: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Predict class labels using base model predictions.

        Args:
            preds_dict: Dict mapping model name to predictions

        Returns:
            Predicted class labels
        """
        proba = self.predict_proba_from_base_preds(preds_dict)
        return (proba[:, 1] >= 0.5).astype(int)

    def fit(self, X: np.ndarray, y: np.ndarray) -> StackingEnsemble:
        """Sklearn-compatible fit method (not recommended for stacking).

        For proper stacking, use fit_from_oof() with pre-computed OOF predictions.
        This method is provided for sklearn pipeline compatibility only.

        Args:
            X: Feature matrix (assumed to be stacked OOF predictions)
            y: Target labels

        Returns:
            self
        """
        logger.warning(
            "Using fit() directly assumes X contains stacked OOF predictions. "
            "For proper stacking, use fit_from_oof() instead."
        )
        # Assume X is already the meta-feature matrix
        if self.scale_meta_features:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

        base_meta = LogisticRegression(
            penalty=self.meta_penalty if self.meta_penalty != "none" else None,
            C=self.meta_C,
            max_iter=self.meta_max_iter,
            solver=self.meta_solver,
            random_state=self.random_state,
            class_weight="balanced",
        )

        if self.calibrate_meta and len(y) >= 2 * self.calibration_cv:
            self.meta_model = CalibratedClassifierCV(
                estimator=base_meta,
                method="isotonic",
                cv=self.calibration_cv,
            )
        else:
            self.meta_model = base_meta

        self.meta_model.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Meta-feature matrix (stacked base model predictions)

        Returns:
            Probability matrix (n_samples, 2)
        """
        if not self.is_fitted_:
            raise RuntimeError("Ensemble not fitted.")

        if self.scaler is not None:
            X = self.scaler.transform(X)

        return self.meta_model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Meta-feature matrix

        Returns:
            Predicted class labels
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def get_meta_model_coef(self) -> dict[str, float]:
        """Get meta-learner coefficients for interpretability.

        Returns:
            Dict mapping base model name to coefficient
        """
        if not self.is_fitted_:
            raise RuntimeError("Ensemble not fitted.")

        # Handle calibrated wrapper
        meta = self.meta_model
        if hasattr(meta, "estimator"):
            # CalibratedClassifierCV stores base estimator
            # Try to get average coefficients from calibrated estimators
            if hasattr(meta, "calibrated_classifiers_"):
                coefs = []
                for cc in meta.calibrated_classifiers_:
                    if hasattr(cc, "estimator") and hasattr(cc.estimator, "coef_"):
                        coefs.append(cc.estimator.coef_[0])
                if coefs:
                    avg_coef = np.mean(coefs, axis=0)
                    return dict(zip(self._feature_names, avg_coef, strict=False))

        # Direct access to LogisticRegression coefficients
        if hasattr(meta, "coef_"):
            return dict(zip(self._feature_names, meta.coef_[0], strict=False))

        return {}

    def save(self, path: Path | str) -> None:
        """Save ensemble to disk.

        Args:
            path: Output path for joblib file
        """
        path = Path(path)
        bundle = {
            "ensemble": self,
            "base_model_names": self.base_model_names,
            "meta_penalty": self.meta_penalty,
            "meta_C": self.meta_C,
            "is_fitted": self.is_fitted_,
            "feature_names": self._feature_names,
        }
        joblib.dump(bundle, path)
        logger.info(f"Ensemble saved to: {path}")

    @classmethod
    def load(cls, path: Path | str) -> StackingEnsemble:
        """Load ensemble from disk.

        Args:
            path: Path to saved ensemble file

        Returns:
            Loaded StackingEnsemble instance
        """
        path = Path(path)
        bundle = joblib.load(path)
        ensemble = bundle["ensemble"]
        logger.info(f"Ensemble loaded from: {path}")
        return ensemble


def collect_oof_predictions(
    results_dir: Path,
    base_models: list[str],
    split_seed: int,
    scenario: str = "IncidentOnly",
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Collect OOF predictions from trained base models.

    Args:
        results_dir: Root results directory
        base_models: List of base model names to collect
        split_seed: Split seed to identify correct subdirectory
        scenario: Scenario name (for filtering)

    Returns:
        oof_dict: Dict mapping model name to OOF predictions
        y_train: Training labels (from first model)
        train_idx: Training indices (from first model)

    Raises:
        FileNotFoundError: If OOF predictions file not found for any model
    """
    oof_dict = {}
    y_train = None
    train_idx = None

    for model_name in base_models:
        # Look for OOF predictions file
        model_dir = results_dir / model_name / f"split_{split_seed}"
        oof_path = model_dir / "preds" / "train_oof" / f"train_oof__{model_name}.csv"

        if not oof_path.exists():
            raise FileNotFoundError(f"OOF predictions not found: {oof_path}")

        # Load OOF predictions
        oof_df = pd.read_csv(oof_path)

        # Extract predictions (may have multiple repeat columns)
        prob_cols = [c for c in oof_df.columns if c.startswith("y_prob")]
        if not prob_cols:
            raise ValueError(f"No probability columns found in {oof_path}")

        # Stack all repeat predictions
        preds = oof_df[prob_cols].values.T  # (n_repeats x n_samples)
        oof_dict[model_name] = preds

        # Get labels and indices from first model
        if y_train is None:
            y_train = oof_df["y_true"].values
            train_idx = oof_df["idx"].values

        logger.info(f"Loaded OOF predictions for {model_name}: shape {preds.shape}")

    return oof_dict, y_train, train_idx


def collect_split_predictions(
    results_dir: Path,
    base_models: list[str],
    split_seed: int,
    split_name: str = "test",
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Collect validation or test predictions from trained base models.

    Args:
        results_dir: Root results directory
        base_models: List of base model names
        split_seed: Split seed to identify correct subdirectory
        split_name: 'val' or 'test'

    Returns:
        preds_dict: Dict mapping model name to predictions
        y_true: True labels
        indices: Sample indices
    """
    preds_dict = {}
    y_true = None
    indices = None

    for model_name in base_models:
        model_dir = results_dir / model_name / f"split_{split_seed}"

        if split_name == "val":
            pred_path = model_dir / "preds" / "val_preds" / f"val_preds__{model_name}.csv"
        else:
            pred_path = model_dir / "preds" / "test_preds" / f"test_preds__{model_name}.csv"

        if not pred_path.exists():
            raise FileNotFoundError(f"Predictions not found: {pred_path}")

        pred_df = pd.read_csv(pred_path)
        preds_dict[model_name] = pred_df["y_prob"].values

        if y_true is None:
            y_true = pred_df["y_true"].values
            indices = pred_df["idx"].values

        logger.info(f"Loaded {split_name} predictions for {model_name}")

    return preds_dict, y_true, indices


def train_stacking_ensemble(
    results_dir: Path,
    base_models: list[str],
    split_seed: int,
    meta_penalty: str = "l2",
    meta_C: float = 1.0,
    calibrate_meta: bool = True,
    random_state: int = 42,
) -> tuple[StackingEnsemble, dict[str, Any]]:
    """Train a stacking ensemble from pre-computed base model outputs.

    This is the main entry point for ensemble training. It:
    1. Collects OOF predictions from base models
    2. Trains the meta-learner
    3. Generates ensemble predictions on val/test sets
    4. Computes ensemble metrics

    Args:
        results_dir: Root results directory containing base model outputs
        base_models: List of base model names to stack
        split_seed: Split seed for identifying model outputs
        meta_penalty: Regularization penalty for meta-learner
        meta_C: Inverse regularization strength
        calibrate_meta: Whether to calibrate meta-learner
        random_state: Random seed

    Returns:
        ensemble: Fitted StackingEnsemble
        results: Dict containing predictions and metrics
    """
    logger.info(f"Training stacking ensemble with base models: {base_models}")

    # Collect OOF predictions
    oof_dict, y_train, train_idx = collect_oof_predictions(results_dir, base_models, split_seed)

    # Create and fit ensemble
    ensemble = StackingEnsemble(
        base_model_names=base_models,
        meta_penalty=meta_penalty,
        meta_C=meta_C,
        calibrate_meta=calibrate_meta,
        random_state=random_state,
    )
    ensemble.fit_from_oof(oof_dict, y_train)

    # Collect val/test predictions and generate ensemble predictions
    results = {
        "base_models": base_models,
        "split_seed": split_seed,
        "meta_penalty": meta_penalty,
        "meta_C": meta_C,
    }

    # Validation set
    try:
        val_preds_dict, y_val, val_idx = collect_split_predictions(
            results_dir, base_models, split_seed, "val"
        )
        val_proba = ensemble.predict_proba_from_base_preds(val_preds_dict)
        results["val_proba"] = val_proba[:, 1]
        results["y_val"] = y_val
        results["val_idx"] = val_idx
    except FileNotFoundError as e:
        logger.warning(f"Could not load validation predictions: {e}")

    # Test set
    try:
        test_preds_dict, y_test, test_idx = collect_split_predictions(
            results_dir, base_models, split_seed, "test"
        )
        test_proba = ensemble.predict_proba_from_base_preds(test_preds_dict)
        results["test_proba"] = test_proba[:, 1]
        results["y_test"] = y_test
        results["test_idx"] = test_idx
    except FileNotFoundError as e:
        logger.warning(f"Could not load test predictions: {e}")

    # Get meta-model coefficients
    results["meta_coef"] = ensemble.get_meta_model_coef()

    return ensemble, results


def save_ensemble_results(
    ensemble: StackingEnsemble,
    results: dict[str, Any],
    output_dir: Path,
    scenario: str = "IncidentOnly",
) -> None:
    """Save ensemble model and results to disk.

    Args:
        ensemble: Fitted stacking ensemble
        results: Results dict from train_stacking_ensemble
        output_dir: Output directory
        scenario: Scenario name
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save ensemble model
    ensemble_path = output_dir / "ENSEMBLE__final_model.joblib"
    ensemble.save(ensemble_path)

    # Save predictions
    if "val_proba" in results and results["val_proba"] is not None:
        val_df = pd.DataFrame(
            {
                "idx": results["val_idx"],
                "y_true": results["y_val"],
                "y_prob": results["val_proba"],
            }
        )
        val_path = output_dir / "val_preds__ENSEMBLE.csv"
        val_df.to_csv(val_path, index=False)
        logger.info(f"Validation predictions saved: {val_path}")

    if "test_proba" in results and results["test_proba"] is not None:
        test_df = pd.DataFrame(
            {
                "idx": results["test_idx"],
                "y_true": results["y_test"],
                "y_prob": results["test_proba"],
            }
        )
        test_path = output_dir / "test_preds__ENSEMBLE.csv"
        test_df.to_csv(test_path, index=False)
        logger.info(f"Test predictions saved: {test_path}")

    # Save metadata
    meta = {
        "base_models": results["base_models"],
        "split_seed": results["split_seed"],
        "meta_penalty": results["meta_penalty"],
        "meta_C": results["meta_C"],
        "meta_coef": results.get("meta_coef", {}),
        "scenario": scenario,
    }
    meta_path = output_dir / "ensemble_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Ensemble metadata saved: {meta_path}")
