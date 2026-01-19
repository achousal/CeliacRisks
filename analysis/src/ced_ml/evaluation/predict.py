"""Prediction generation utilities for trained models.

This module handles:
- Generating predictions on validation/test/holdout sets
- Prevalence-adjusted probability recalibration
- Prediction export to CSV
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union
from sklearn.pipeline import Pipeline

from ..models.prevalence import PrevalenceAdjustedModel


def generate_predictions(
    model: Union[Pipeline, PrevalenceAdjustedModel],
    X: pd.DataFrame,
    return_raw: bool = True,
) -> np.ndarray:
    """Generate probability predictions from a trained model.

    Parameters
    ----------
    model : Pipeline or PrevalenceAdjustedModel
        Trained model to generate predictions.
    X : pd.DataFrame
        Feature matrix (n_samples, n_features).
    return_raw : bool, default=True
        If True, return raw probabilities from model.
        If False and model is PrevalenceAdjustedModel, return adjusted probabilities.

    Returns
    -------
    np.ndarray
        Predicted probabilities for positive class, shape (n_samples,).
        Values clipped to [0.0, 1.0].

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    >>> pipe.fit(X_train, y_train)
    >>> preds = generate_predictions(pipe, X_test)
    """
    if isinstance(model, PrevalenceAdjustedModel):
        if return_raw:
            proba = model.base_model.predict_proba(X)[:, 1]
        else:
            proba = model.predict_proba(X)[:, 1]
    else:
        proba = model.predict_proba(X)[:, 1]

    return np.clip(proba, 0.0, 1.0)


def generate_predictions_with_adjustment(
    model: Union[Pipeline, PrevalenceAdjustedModel],
    X: pd.DataFrame,
    train_prevalence: float,
    target_prevalence: float,
) -> Dict[str, np.ndarray]:
    """Generate raw and prevalence-adjusted predictions.

    Parameters
    ----------
    model : Pipeline or PrevalenceAdjustedModel
        Trained model (typically the base model without adjustment).
    X : pd.DataFrame
        Feature matrix (n_samples, n_features).
    train_prevalence : float
        Sample prevalence in training set (post-downsampling).
    target_prevalence : float
        Target prevalence for deployment context.

    Returns
    -------
    dict
        Dictionary with keys:
        - "raw": Raw model probabilities, shape (n_samples,)
        - "adjusted": Prevalence-adjusted probabilities, shape (n_samples,)

    Examples
    --------
    >>> preds = generate_predictions_with_adjustment(
    ...     model=final_model,
    ...     X=X_test,
    ...     train_prevalence=0.15,
    ...     target_prevalence=0.003
    ... )
    >>> print(preds["raw"].mean(), preds["adjusted"].mean())
    """
    from ..models.prevalence import adjust_probabilities_for_prevalence

    if isinstance(model, PrevalenceAdjustedModel):
        p_raw = model.base_model.predict_proba(X)[:, 1]
    else:
        p_raw = model.predict_proba(X)[:, 1]

    p_raw = np.clip(p_raw, 0.0, 1.0)

    p_adjusted = adjust_probabilities_for_prevalence(
        probs=p_raw,
        sample_prev=train_prevalence,
        target_prev=target_prevalence,
    )

    return {
        "raw": p_raw,
        "adjusted": p_adjusted,
    }


def export_predictions(
    predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    out_path: str,
    ids: Optional[np.ndarray] = None,
    target_col: Optional[pd.Series] = None,
    active_key: str = "adjusted",
    percentile: bool = True,
) -> None:
    """Export predictions to CSV.

    Parameters
    ----------
    predictions : dict
        Dictionary with prediction arrays (e.g., "raw", "adjusted").
    y_true : np.ndarray
        True binary labels, shape (n_samples,).
    out_path : str
        Output CSV file path.
    ids : np.ndarray, optional
        Subject IDs. If None, uses sequential integers.
    target_col : pd.Series, optional
        Original target column values (e.g., "Control", "Incident").
    active_key : str, default="adjusted"
        Key in predictions dict to use as primary risk score.
    percentile : bool, default=True
        If True, include percentile columns (risk * 100).

    Examples
    --------
    >>> preds = {"raw": p_raw, "adjusted": p_adj}
    >>> export_predictions(
    ...     predictions=preds,
    ...     y_true=y_test,
    ...     out_path="predictions/test_preds.csv",
    ...     ids=df_test["studyID"].values,
    ...     target_col=df_test["caseStatus"],
    ...     active_key="adjusted"
    ... )
    """
    n = len(y_true)
    if ids is None:
        ids = np.arange(n)

    data = {
        "id": ids,
        "y_true": y_true.astype(int),
    }

    if target_col is not None:
        data["target"] = target_col.astype(str).values

    for key, vals in predictions.items():
        if len(vals) != n:
            raise ValueError(f"Prediction array '{key}' has length {len(vals)}, expected {n}")
        data[f"risk_{key}"] = vals
        if percentile:
            data[f"risk_{key}_pct"] = 100.0 * vals

    if active_key in predictions:
        data["risk"] = predictions[active_key]
        if percentile:
            data["risk_pct"] = 100.0 * predictions[active_key]

    df = pd.DataFrame(data)
    df.to_csv(out_path, index=False)


def predict_on_validation(
    model: Pipeline,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    train_prevalence: float,
    target_prevalence: float,
    use_adjusted: bool = True,
) -> Dict[str, Any]:
    """Generate predictions on validation set with prevalence adjustment.

    Parameters
    ----------
    model : Pipeline
        Trained model (base model without prevalence adjustment).
    X_val : pd.DataFrame
        Validation feature matrix.
    y_val : np.ndarray
        Validation true labels.
    train_prevalence : float
        Training sample prevalence.
    target_prevalence : float
        Target prevalence for adjustment.
    use_adjusted : bool, default=True
        If True, use adjusted probabilities as primary predictions.

    Returns
    -------
    dict
        Dictionary with keys:
        - "raw": Raw predictions
        - "adjusted": Adjusted predictions
        - "active": Primary predictions (adjusted if use_adjusted=True, else raw)
        - "n": Number of samples
        - "n_pos": Number of positive cases
        - "prevalence": Observed prevalence

    Examples
    --------
    >>> val_preds = predict_on_validation(
    ...     model=final_model,
    ...     X_val=X_val,
    ...     y_val=y_val,
    ...     train_prevalence=0.15,
    ...     target_prevalence=0.003,
    ...     use_adjusted=True
    ... )
    >>> print(f"Active predictions: {val_preds['active'][:5]}")
    """
    preds = generate_predictions_with_adjustment(
        model=model,
        X=X_val,
        train_prevalence=train_prevalence,
        target_prevalence=target_prevalence,
    )

    return {
        "raw": preds["raw"],
        "adjusted": preds["adjusted"],
        "active": preds["adjusted"] if use_adjusted else preds["raw"],
        "n": len(y_val),
        "n_pos": int(y_val.sum()),
        "prevalence": float(y_val.mean()),
    }


def predict_on_test(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    train_prevalence: float,
    target_prevalence: float,
    use_adjusted: bool = True,
) -> Dict[str, Any]:
    """Generate predictions on test set with prevalence adjustment.

    Identical to predict_on_validation but semantically distinct for test set.

    Parameters
    ----------
    model : Pipeline
        Trained model (base model without prevalence adjustment).
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : np.ndarray
        Test true labels.
    train_prevalence : float
        Training sample prevalence.
    target_prevalence : float
        Target prevalence for adjustment.
    use_adjusted : bool, default=True
        If True, use adjusted probabilities as primary predictions.

    Returns
    -------
    dict
        Dictionary with prediction arrays and metadata (see predict_on_validation).

    Examples
    --------
    >>> test_preds = predict_on_test(
    ...     model=final_model,
    ...     X_test=X_test,
    ...     y_test=y_test,
    ...     train_prevalence=0.15,
    ...     target_prevalence=0.003,
    ...     use_adjusted=True
    ... )
    """
    preds = generate_predictions_with_adjustment(
        model=model,
        X=X_test,
        train_prevalence=train_prevalence,
        target_prevalence=target_prevalence,
    )

    return {
        "raw": preds["raw"],
        "adjusted": preds["adjusted"],
        "active": preds["adjusted"] if use_adjusted else preds["raw"],
        "n": len(y_test),
        "n_pos": int(y_test.sum()),
        "prevalence": float(y_test.mean()),
    }


def predict_on_holdout(
    model: Union[Pipeline, PrevalenceAdjustedModel],
    X_holdout: pd.DataFrame,
    y_holdout: np.ndarray,
    train_prevalence: Optional[float] = None,
    target_prevalence: Optional[float] = None,
) -> Dict[str, Any]:
    """Generate predictions on holdout set.

    If model is PrevalenceAdjustedModel, uses embedded prevalence settings.
    Otherwise, requires explicit train_prevalence and target_prevalence.

    Parameters
    ----------
    model : Pipeline or PrevalenceAdjustedModel
        Trained model with or without prevalence adjustment.
    X_holdout : pd.DataFrame
        Holdout feature matrix.
    y_holdout : np.ndarray
        Holdout true labels.
    train_prevalence : float, optional
        Training sample prevalence (required if model is Pipeline).
    target_prevalence : float, optional
        Target prevalence (required if model is Pipeline).

    Returns
    -------
    dict
        Dictionary with prediction arrays and metadata.

    Raises
    ------
    ValueError
        If model is Pipeline and prevalence parameters are missing.

    Examples
    --------
    >>> holdout_preds = predict_on_holdout(
    ...     model=prevalence_adjusted_model,
    ...     X_holdout=X_holdout,
    ...     y_holdout=y_holdout
    ... )
    """
    if isinstance(model, PrevalenceAdjustedModel):
        p_raw = model.base_model.predict_proba(X_holdout)[:, 1]
        p_raw = np.clip(p_raw, 0.0, 1.0)
        p_adjusted = model.predict_proba(X_holdout)[:, 1]
        p_adjusted = np.clip(p_adjusted, 0.0, 1.0)

        return {
            "raw": p_raw,
            "adjusted": p_adjusted,
            "active": p_adjusted,
            "n": len(y_holdout),
            "n_pos": int(y_holdout.sum()),
            "prevalence": float(y_holdout.mean()),
        }
    else:
        if train_prevalence is None or target_prevalence is None:
            raise ValueError(
                "train_prevalence and target_prevalence required for Pipeline models"
            )

        preds = generate_predictions_with_adjustment(
            model=model,
            X=X_holdout,
            train_prevalence=train_prevalence,
            target_prevalence=target_prevalence,
        )

        return {
            "raw": preds["raw"],
            "adjusted": preds["adjusted"],
            "active": preds["adjusted"],
            "n": len(y_holdout),
            "n_pos": int(y_holdout.sum()),
            "prevalence": float(y_holdout.mean()),
        }
