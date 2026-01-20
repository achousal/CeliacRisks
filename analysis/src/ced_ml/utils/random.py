"""
Random seed management for reproducibility.
"""

import random

import numpy as np


def set_random_seed(seed: int):
    """
    Set random seed for all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)

    # Sklearn doesn't have global seed, use random_state parameter
    # XGBoost uses numpy's random state


def get_cv_seed(base_seed: int, fold_idx: int, repeat_idx: int = 0) -> int:
    """
    Generate deterministic seed for CV fold.

    Args:
        base_seed: Base random seed
        fold_idx: Fold index (0-based)
        repeat_idx: Repeat index (0-based)

    Returns:
        Deterministic seed for this fold/repeat combination
    """
    return base_seed + (repeat_idx * 1000) + fold_idx
