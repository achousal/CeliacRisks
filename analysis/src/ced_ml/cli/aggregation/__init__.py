"""Aggregation submodules for split results processing."""

from ced_ml.cli.aggregation.discovery import (
    discover_ensemble_dirs,
    discover_split_dirs,
)

__all__ = [
    "discover_ensemble_dirs",
    "discover_split_dirs",
]
