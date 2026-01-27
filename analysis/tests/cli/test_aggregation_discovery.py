"""Tests for aggregation discovery module."""

import logging

import pytest
from ced_ml.cli.aggregation.discovery import (
    discover_ensemble_dirs,
    discover_split_dirs,
)


@pytest.fixture
def tmp_results_dir(tmp_path):
    """Create temporary results directory structure."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Create split directories
    for i in [0, 1, 5, 10]:
        (results_dir / f"split_seed{i}").mkdir()

    # Create ensemble directories
    ensemble_dir = results_dir / "ENSEMBLE"
    ensemble_dir.mkdir()
    for i in [0, 1]:
        (ensemble_dir / f"split_seed{i}").mkdir()
    for i in [5, 10]:
        (ensemble_dir / f"split_{i}").mkdir()

    # Create a file (should be ignored)
    (results_dir / "split_seed99.txt").touch()

    return results_dir


def test_discover_split_dirs_basic(tmp_results_dir):
    """Test basic split directory discovery."""
    dirs = discover_split_dirs(tmp_results_dir)
    assert len(dirs) == 4
    assert all(d.is_dir() for d in dirs)
    # Check sorted order
    assert [int(d.name.replace("split_seed", "")) for d in dirs] == [0, 1, 5, 10]


def test_discover_split_dirs_with_logger(tmp_results_dir, caplog):
    """Test split directory discovery with logger."""
    logger = logging.getLogger("test")
    logger.setLevel(logging.DEBUG)
    with caplog.at_level(logging.DEBUG):
        dirs = discover_split_dirs(tmp_results_dir, logger=logger)
    assert len(dirs) == 4
    assert "Discovered 4 split directories" in caplog.text


def test_discover_split_dirs_empty(tmp_path):
    """Test discovery with no split directories."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    dirs = discover_split_dirs(results_dir)
    assert len(dirs) == 0


def test_discover_ensemble_dirs_basic(tmp_results_dir):
    """Test basic ensemble directory discovery."""
    dirs = discover_ensemble_dirs(tmp_results_dir)
    assert len(dirs) == 4
    assert all(d.is_dir() for d in dirs)
    # Check sorted order
    seeds = []
    for d in dirs:
        if d.name.startswith("split_seed"):
            seeds.append(int(d.name.replace("split_seed", "")))
        elif d.name.startswith("split_"):
            seeds.append(int(d.name.replace("split_", "")))
    assert seeds == [0, 1, 5, 10]


def test_discover_ensemble_dirs_no_ensemble(tmp_path):
    """Test discovery when no ENSEMBLE directory exists."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    dirs = discover_ensemble_dirs(results_dir)
    assert len(dirs) == 0


def test_discover_ensemble_dirs_with_logger(tmp_results_dir, caplog):
    """Test ensemble directory discovery with logger."""
    logger = logging.getLogger("test")
    logger.setLevel(logging.DEBUG)
    with caplog.at_level(logging.DEBUG):
        dirs = discover_ensemble_dirs(tmp_results_dir, logger=logger)
    assert len(dirs) == 4
    assert "Discovered 4 ENSEMBLE split directories" in caplog.text


def test_discover_ensemble_dirs_empty_ensemble(tmp_path):
    """Test discovery with empty ENSEMBLE directory."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "ENSEMBLE").mkdir()
    dirs = discover_ensemble_dirs(results_dir)
    assert len(dirs) == 0
