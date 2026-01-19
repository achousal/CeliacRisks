"""
Serialization utilities for models and results.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Union

import joblib


def save_joblib(obj: Any, path: Union[str, Path], compress: int = 3):
    """Save object using joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path, compress=compress)


def load_joblib(path: Union[str, Path]) -> Any:
    """Load object using joblib."""
    return joblib.load(path)


def save_json(obj: Any, path: Union[str, Path], indent: int = 2):
    """Save object as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, default=str)


def load_json(path: Union[str, Path]) -> Any:
    """Load JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_pickle(obj: Any, path: Union[str, Path]):
    """Save object using pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Union[str, Path]) -> Any:
    """Load pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)
