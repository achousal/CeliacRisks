"""
Serialization utilities for models and results.
"""

import json
import logging
import pickle
import warnings
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)


def save_joblib(obj: Any, path: str | Path, compress: int = 3):
    """Save object using joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path, compress=compress)


def load_joblib(path: str | Path, check_versions: bool = True) -> Any:
    """
    Load object using joblib with optional version checking.

    Args:
        path: Path to joblib file
        check_versions: If True and object is a model bundle with version metadata,
            warn if sklearn/pandas/numpy versions differ from current environment

    Returns:
        Loaded object

    Warns:
        UserWarning if versions mismatch and check_versions=True
    """
    obj = joblib.load(path)

    # Check for version metadata if this is a model bundle
    if check_versions and isinstance(obj, dict) and "versions" in obj:
        try:
            import numpy as np
            import pandas as pd
            import sklearn

            saved_versions = obj["versions"]
            current_versions = {
                "sklearn": sklearn.__version__,
                "pandas": pd.__version__,
                "numpy": np.__version__,
            }

            mismatches = []
            for lib, saved_ver in saved_versions.items():
                current_ver = current_versions.get(lib)
                if current_ver and saved_ver != current_ver:
                    mismatches.append(f"{lib}: saved={saved_ver}, current={current_ver}")

            if mismatches:
                warnings.warn(
                    f"Model bundle version mismatch in {Path(path).name}:\n"
                    + "\n".join(f"  - {m}" for m in mismatches)
                    + "\nPredictions may be inconsistent.",
                    UserWarning,
                    stacklevel=2,
                )
        except Exception as e:
            logger.debug(f"Could not check versions for {path}: {e}")

    return obj


def save_json(obj: Any, path: str | Path, indent: int = 2):
    """Save object as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, default=str)


def load_json(path: str | Path) -> Any:
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def save_pickle(obj: Any, path: str | Path):
    """Save object using pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path) -> Any:
    """Load pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)
