"""
CeD-ML: Machine Learning Pipeline for Celiac Disease Risk Prediction

A modular, reproducible ML pipeline for predicting incident Celiac Disease
risk from proteomics biomarkers.
"""

# Enable pandas Copy-on-Write for pandas 3.0 compatibility and better memory efficiency
# See: https://pandas.pydata.org/docs/user_guide/copy_on_write.html
import pandas as pd

pd.options.mode.copy_on_write = True

__version__ = "1.0.0"
__author__ = "Andres Chousal"
__license__ = "MIT"

from ced_ml import (  # noqa: E402
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
