"""
CeD-ML: Machine Learning Pipeline for Celiac Disease Risk Prediction

A modular, reproducible ML pipeline for predicting incident Celiac Disease
risk from proteomics biomarkers.
"""

__version__ = "1.0.0"
__author__ = "Andres Chousal"
__license__ = "MIT"

from ced_ml import (
    cli,
    config,
    data,
    evaluation,
    features,
    metrics,
    models,
    plotting,
    utils,
)

__all__ = [
    "__version__",
    "cli",
    "config",
    "data",
    "evaluation",
    "features",
    "metrics",
    "models",
    "plotting",
    "utils",
]
