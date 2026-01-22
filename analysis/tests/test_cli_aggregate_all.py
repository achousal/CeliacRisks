"""
Tests for the aggregate_all CLI module.
"""

import json
from unittest.mock import patch

import pytest
from ced_ml.cli.aggregate_all import (
    check_run_completion,
    discover_runs,
    run_aggregate_all,
)


@pytest.fixture
def mock_results_tree(tmp_path):
    """Create a mock results directory structure."""
    # Create results root
    results_root = tmp_path / "results"
    results_root.mkdir()

    # Create model directories
    for model in ["LR_EN", "RF"]:
        model_dir = results_root / model
        model_dir.mkdir()

        # Create run directories
        run_dir = model_dir / "run_20260122_120000"
        run_dir.mkdir()

        # Create split directories with completion markers
        for seed in range(3):
            split_dir = run_dir / f"split_seed{seed}"
            split_dir.mkdir()
            core_dir = split_dir / "core"
            core_dir.mkdir()
            # Create completion marker
            (core_dir / "test_metrics.csv").write_text("metric,value\nAUROC,0.85\n")

    return results_root


@pytest.fixture
def mock_incomplete_results(tmp_path):
    """Create a mock results directory with incomplete runs."""
    results_root = tmp_path / "results"
    results_root.mkdir()

    model_dir = results_root / "XGBoost"
    model_dir.mkdir()
    run_dir = model_dir / "run_20260122_130000"
    run_dir.mkdir()

    # Only 2 of 3 splits complete
    for seed in range(2):
        split_dir = run_dir / f"split_seed{seed}"
        split_dir.mkdir()
        core_dir = split_dir / "core"
        core_dir.mkdir()
        (core_dir / "test_metrics.csv").write_text("metric,value\nAUROC,0.80\n")

    # Third split exists but incomplete
    split_dir = run_dir / "split_seed2"
    split_dir.mkdir()
    (split_dir / "core").mkdir()  # No test_metrics.csv

    # Add metadata indicating expected splits
    (run_dir / "run_metadata.json").write_text(json.dumps({"n_splits": 3}))

    return results_root


class TestDiscoverRuns:
    """Tests for discover_runs function."""

    def test_discovers_model_run_directories(self, mock_results_tree):
        """Should find all model/run directories with splits."""
        runs = discover_runs(mock_results_tree)

        assert len(runs) == 2
        models = {r["model"] for r in runs}
        assert models == {"LR_EN", "RF"}

    def test_returns_split_dirs(self, mock_results_tree):
        """Should include split directories for each run."""
        runs = discover_runs(mock_results_tree)

        for run in runs:
            assert len(run["split_dirs"]) == 3
            assert all(d.name.startswith("split_seed") for d in run["split_dirs"])

    def test_skips_aggregated_directory(self, mock_results_tree):
        """Should skip 'aggregated' directory."""
        (mock_results_tree / "aggregated").mkdir()
        runs = discover_runs(mock_results_tree)

        models = {r["model"] for r in runs}
        assert "aggregated" not in models

    def test_skips_logs_directory(self, mock_results_tree):
        """Should skip 'logs' directory."""
        (mock_results_tree / "logs").mkdir()
        runs = discover_runs(mock_results_tree)

        models = {r["model"] for r in runs}
        assert "logs" not in models

    def test_skips_non_run_directories(self, mock_results_tree):
        """Should skip directories not matching run_* pattern."""
        model_dir = mock_results_tree / "LR_EN"
        (model_dir / "old_results").mkdir()
        (model_dir / "backup").mkdir()

        runs = discover_runs(mock_results_tree)
        lr_runs = [r for r in runs if r["model"] == "LR_EN"]

        assert len(lr_runs) == 1
        assert lr_runs[0]["run_id"] == "run_20260122_120000"


class TestCheckRunCompletion:
    """Tests for check_run_completion function."""

    def test_complete_run(self, mock_results_tree):
        """Should detect complete runs."""
        runs = discover_runs(mock_results_tree)
        status = check_run_completion(runs[0])

        assert status["is_complete"] is True
        assert status["completed_splits"] == 3
        assert status["expected_splits"] == 3

    def test_incomplete_run(self, mock_incomplete_results):
        """Should detect incomplete runs."""
        runs = discover_runs(mock_incomplete_results)
        status = check_run_completion(runs[0])

        assert status["is_complete"] is False
        assert status["completed_splits"] == 2
        assert status["expected_splits"] == 3

    def test_detects_already_aggregated(self, mock_results_tree):
        """Should detect if aggregation already done."""
        runs = discover_runs(mock_results_tree)

        # Create aggregated directory
        run_dir = runs[0]["run_dir"]
        agg_dir = run_dir / "aggregated" / "core"
        agg_dir.mkdir(parents=True)
        (agg_dir / "pooled_metrics.csv").write_text("metric,value\nAUROC,0.85\n")

        status = check_run_completion(runs[0])
        assert status["already_aggregated"] is True

    def test_reads_metadata_for_expected_splits(self, mock_incomplete_results):
        """Should use metadata for expected split count."""
        runs = discover_runs(mock_incomplete_results)
        status = check_run_completion(runs[0])

        # Metadata says 3 splits expected
        assert status["expected_splits"] == 3


class TestRunAggregateAll:
    """Tests for run_aggregate_all function."""

    def test_dry_run_no_aggregation(self, mock_results_tree):
        """Dry run should not call aggregation."""
        with patch("ced_ml.cli.aggregate_all.run_aggregate_splits") as mock_agg:
            result = run_aggregate_all(
                results_root=str(mock_results_tree),
                dry_run=True,
            )

            mock_agg.assert_not_called()
            assert result["aggregated"] == 2
            assert result["scanned"] == 2

    def test_aggregates_complete_runs(self, mock_results_tree):
        """Should aggregate complete runs."""
        with patch("ced_ml.cli.aggregate_all.run_aggregate_splits") as mock_agg:
            result = run_aggregate_all(
                results_root=str(mock_results_tree),
                dry_run=False,
            )

            assert mock_agg.call_count == 2
            assert result["aggregated"] == 2

    def test_skips_already_aggregated(self, mock_results_tree):
        """Should skip already-aggregated runs unless force=True."""
        # Mark both as aggregated
        runs = discover_runs(mock_results_tree)
        for run in runs:
            agg_dir = run["run_dir"] / "aggregated" / "core"
            agg_dir.mkdir(parents=True)
            (agg_dir / "pooled_metrics.csv").write_text("metric,value\nAUROC,0.85\n")

        with patch("ced_ml.cli.aggregate_all.run_aggregate_splits") as mock_agg:
            result = run_aggregate_all(
                results_root=str(mock_results_tree),
                dry_run=False,
            )

            mock_agg.assert_not_called()
            assert result["skipped_already_done"] == 2

    def test_force_reaggregates(self, mock_results_tree):
        """Should re-aggregate with force=True."""
        # Mark both as aggregated
        runs = discover_runs(mock_results_tree)
        for run in runs:
            agg_dir = run["run_dir"] / "aggregated" / "core"
            agg_dir.mkdir(parents=True)
            (agg_dir / "pooled_metrics.csv").write_text("metric,value\nAUROC,0.85\n")

        with patch("ced_ml.cli.aggregate_all.run_aggregate_splits") as mock_agg:
            result = run_aggregate_all(
                results_root=str(mock_results_tree),
                force=True,
                dry_run=False,
            )

            assert mock_agg.call_count == 2
            assert result["aggregated"] == 2

    def test_skips_incomplete_runs(self, mock_incomplete_results):
        """Should skip incomplete runs."""
        with patch("ced_ml.cli.aggregate_all.run_aggregate_splits") as mock_agg:
            result = run_aggregate_all(
                results_root=str(mock_incomplete_results),
                dry_run=False,
            )

            mock_agg.assert_not_called()
            assert result["skipped_incomplete"] == 1

    def test_handles_missing_results_root(self, tmp_path):
        """Should raise error for missing results root."""
        with pytest.raises(FileNotFoundError):
            run_aggregate_all(
                results_root=str(tmp_path / "nonexistent"),
            )

    def test_handles_aggregation_failure(self, mock_results_tree):
        """Should continue on aggregation failure and report it."""
        with patch("ced_ml.cli.aggregate_all.run_aggregate_splits") as mock_agg:
            mock_agg.side_effect = [Exception("Test error"), None]

            result = run_aggregate_all(
                results_root=str(mock_results_tree),
                dry_run=False,
            )

            assert result["aggregated"] == 1
            assert result["failed"] == 1
            assert "Test error" in result["details"]["failed"][0]["error"]
