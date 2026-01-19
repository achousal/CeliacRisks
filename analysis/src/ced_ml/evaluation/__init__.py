"""Evaluation module for model performance assessment."""

from ced_ml.evaluation.reports import OutputDirectories, ResultsWriter
from ced_ml.evaluation.predict import (
    generate_predictions,
    generate_predictions_with_adjustment,
    export_predictions,
    predict_on_validation,
    predict_on_test,
    predict_on_holdout,
)

from ced_ml.evaluation.holdout import (
    evaluate_holdout,
    load_holdout_indices,
    load_model_artifact,
    extract_holdout_data,
    compute_holdout_metrics,
    compute_top_risk_capture,
    save_holdout_predictions,
)

__all__ = [
    "OutputDirectories",
    "ResultsWriter",
    "generate_predictions",
    "generate_predictions_with_adjustment",
    "export_predictions",
    "predict_on_validation",
    "predict_on_test",
    "predict_on_holdout",
    "evaluate_holdout",
    "load_holdout_indices",
    "load_model_artifact",
    "extract_holdout_data",
    "compute_holdout_metrics",
    "compute_top_risk_capture",
    "save_holdout_predictions",
]
