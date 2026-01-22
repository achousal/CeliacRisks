"""
Random seed management for reproducibility.

Provides utilities for deterministic RNG seeding, including an optional
SEED_GLOBAL environment variable for single-threaded reproducibility debugging.
"""

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


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


def apply_seed_global() -> int | None:
    """
    Check SEED_GLOBAL environment variable and apply global seeding if set.

    When SEED_GLOBAL is set to an integer value, seeds Python's random module
    and NumPy's legacy global RNG. This is intended for single-threaded
    reproducibility debugging only; production runs should use explicit
    per-component seeds via config files.

    Returns:
        The seed value applied, or None if SEED_GLOBAL was not set or invalid.

    Examples:
        >>> import os
        >>> os.environ["SEED_GLOBAL"] = "42"
        >>> seed = apply_seed_global()
        >>> seed
        42
        >>> del os.environ["SEED_GLOBAL"]
    """
    seed_str = os.environ.get("SEED_GLOBAL")
    if seed_str is None:
        return None

    seed_str = seed_str.strip()
    if not seed_str:
        return None

    try:
        seed = int(seed_str)
    except ValueError:
        logger.warning(
            "SEED_GLOBAL environment variable has non-integer value '%s'; ignoring.",
            seed_str,
        )
        return None

    if seed < 0 or seed > 2**32 - 1:
        logger.warning(
            "SEED_GLOBAL=%d out of valid range [0, 2^32-1]; ignoring.",
            seed,
        )
        return None

    set_random_seed(seed)
    logger.info("SEED_GLOBAL=%d applied (global RNG seeded for reproducibility).", seed)
    return seed


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
