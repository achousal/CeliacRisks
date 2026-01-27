"""
Screening results caching to eliminate redundant computations.

Caches screening results by (X_train_hash, y_train_hash, method, top_n)
to avoid re-running expensive univariate tests when the same data is screened
multiple times within a single training run.

Thread-safe for concurrent access (important for parallel CV).
"""

import hashlib
import logging
from dataclasses import dataclass
from threading import Lock

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScreeningKey:
    """Immutable key for screening cache lookups."""

    X_hash: str  # Hash of X_train data
    y_hash: str  # Hash of y_train data
    protein_cols_hash: str  # Hash of protein columns list
    method: str  # 'mannwhitney', 'fstat', etc.
    top_n: int  # Number of top features requested


class ScreeningCache:
    """
    Thread-safe cache for screening results.

    Stores (selected_proteins, screening_stats) tuples keyed by screening parameters.
    Automatically clears between different training runs by detecting data changes.
    """

    def __init__(self):
        self._cache: dict[ScreeningKey, tuple[list[str], pd.DataFrame]] = {}
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def _hash_array(self, arr: np.ndarray) -> str:
        """Compute stable hash of numpy array."""
        return hashlib.sha256(arr.tobytes()).hexdigest()[:16]

    def _hash_dataframe(self, df: pd.DataFrame, cols: list[str]) -> str:
        """Compute stable hash of DataFrame subset."""
        # Sort columns for consistency
        sorted_cols = sorted(cols)
        subset = df[sorted_cols].to_numpy()
        return self._hash_array(subset)

    def _hash_list(self, lst: list[str]) -> str:
        """Compute stable hash of string list."""
        joined = "|".join(sorted(lst))
        return hashlib.sha256(joined.encode()).hexdigest()[:16]

    def make_key(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        protein_cols: list[str],
        method: str,
        top_n: int,
    ) -> ScreeningKey:
        """Create cache key from screening parameters."""
        X_hash = self._hash_dataframe(X_train, protein_cols)
        y_hash = self._hash_array(np.asarray(y_train))
        protein_cols_hash = self._hash_list(protein_cols)

        return ScreeningKey(
            X_hash=X_hash,
            y_hash=y_hash,
            protein_cols_hash=protein_cols_hash,
            method=method,
            top_n=top_n,
        )

    def get(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        protein_cols: list[str],
        method: str,
        top_n: int,
    ) -> tuple[list[str], pd.DataFrame] | None:
        """
        Retrieve cached screening results if available.

        Returns:
            (selected_proteins, screening_stats) if cached, else None
        """
        key = self.make_key(X_train, y_train, protein_cols, method, top_n)

        with self._lock:
            result = self._cache.get(key)
            if result is not None:
                self._hits += 1
                logger.debug(
                    f"[screening_cache] HIT: {method} top_n={top_n} "
                    f"(hits={self._hits}, misses={self._misses})"
                )
            else:
                self._misses += 1
            return result

    def put(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        protein_cols: list[str],
        method: str,
        top_n: int,
        selected: list[str],
        stats: pd.DataFrame,
    ) -> None:
        """Store screening results in cache."""
        key = self.make_key(X_train, y_train, protein_cols, method, top_n)

        with self._lock:
            self._cache[key] = (selected, stats.copy())
            logger.debug(
                f"[screening_cache] STORED: {method} top_n={top_n} "
                f"({len(selected)} proteins, {len(stats)} stats)"
            )

    def clear(self) -> None:
        """Clear all cached entries (useful for testing)."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            logger.debug("[screening_cache] CLEARED")

    def stats(self) -> dict[str, int]:
        """Return cache statistics."""
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._cache),
                "hit_rate": self._hits / max(1, self._hits + self._misses),
            }


# Global singleton instance
_SCREENING_CACHE = ScreeningCache()


def get_screening_cache() -> ScreeningCache:
    """Get the global screening cache instance."""
    return _SCREENING_CACHE
