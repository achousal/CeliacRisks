"""Tests for risk distribution plotting.

Tests cover:
- Distribution statistics computation
- Single-panel plots (histogram, binary, KDE)
- Multi-panel plots (incident/prevalent subplots)
- Threshold line rendering
- Metadata handling
- Edge cases (empty data, NaN handling)
"""

from pathlib import Path

import matplotlib
import numpy as np
import pytest

from ced_ml.plotting.risk_dist import (
    compute_distribution_stats,
    plot_risk_distribution,
)

matplotlib.use("Agg")


class TestComputeDistributionStats:
    """Tests for distribution statistics computation."""

    def test_basic_stats(self):
        """Test statistics computation for normal distribution."""
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        stats = compute_distribution_stats(scores)

        assert stats["mean"] == pytest.approx(0.5, abs=0.01)
        assert stats["median"] == pytest.approx(0.5, abs=0.01)
        assert stats["iqr"] == pytest.approx(0.4, abs=0.01)  # Q3(0.7) - Q1(0.3)
        assert stats["sd"] > 0

    def test_empty_array(self):
        """Test handling of empty array."""
        scores = np.array([])
        stats = compute_distribution_stats(scores)

        assert np.isnan(stats["mean"])
        assert np.isnan(stats["median"])
        assert np.isnan(stats["iqr"])
        assert np.isnan(stats["sd"])

    def test_nan_filtering(self):
        """Test NaN values are filtered before computation."""
        scores = np.array([0.1, np.nan, 0.3, np.inf, 0.5])
        stats = compute_distribution_stats(scores)

        # Should compute on [0.1, 0.3, 0.5] only
        assert not np.isnan(stats["mean"])
        assert stats["mean"] == pytest.approx(0.3, abs=0.01)

    def test_all_nans(self):
        """Test array with only NaN/inf values."""
        scores = np.array([np.nan, np.inf, -np.inf])
        stats = compute_distribution_stats(scores)

        assert np.isnan(stats["mean"])
        assert np.isnan(stats["median"])
        assert np.isnan(stats["iqr"])
        assert np.isnan(stats["sd"])

    def test_single_value(self):
        """Test statistics with single value (edge case)."""
        scores = np.array([0.42])
        stats = compute_distribution_stats(scores)

        assert stats["mean"] == pytest.approx(0.42)
        assert stats["median"] == pytest.approx(0.42)
        assert stats["iqr"] == pytest.approx(0.0)
        assert stats["sd"] == pytest.approx(0.0)


class TestPlotRiskDistribution:
    """Tests for risk distribution plotting."""

    def test_simple_histogram(self, tmp_path):
        """Test simple histogram without labels."""
        scores = np.random.uniform(0, 1, 100)
        out_path = tmp_path / "simple.png"

        plot_risk_distribution(
            y_true=None,
            scores=scores,
            out_path=out_path,
            title="Simple Distribution",
        )

        assert out_path.exists()

    def test_binary_histogram(self, tmp_path):
        """Test binary histogram (cases vs controls)."""
        scores = np.random.uniform(0, 1, 100)
        y_true = np.random.binomial(1, 0.3, 100)
        out_path = tmp_path / "binary.png"

        plot_risk_distribution(
            y_true=y_true,
            scores=scores,
            out_path=out_path,
            title="Binary Distribution",
            pos_label="CeD",
        )

        assert out_path.exists()

    def test_three_category_kde(self, tmp_path):
        """Test KDE plot with three categories."""
        n = 300
        scores = np.random.uniform(0, 1, n)
        category_col = np.random.choice(
            ["Controls", "Incident", "Prevalent"], size=n, p=[0.7, 0.2, 0.1]
        )
        out_path = tmp_path / "kde.png"

        plot_risk_distribution(
            y_true=None,
            scores=scores,
            out_path=out_path,
            title="Three-Category KDE",
            category_col=category_col,
        )

        assert out_path.exists()

    def test_incident_subplot(self, tmp_path):
        """Test incident subplot rendering."""
        n = 300
        scores = np.random.uniform(0, 1, n)
        category_col = np.random.choice(
            ["Controls", "Incident"], size=n, p=[0.8, 0.2]
        )
        out_path = tmp_path / "incident_subplot.png"

        plot_risk_distribution(
            y_true=None,
            scores=scores,
            out_path=out_path,
            title="Incident Subplot",
            category_col=category_col,
        )

        assert out_path.exists()

    def test_prevalent_subplot(self, tmp_path):
        """Test prevalent subplot rendering."""
        n = 300
        scores = np.random.uniform(0, 1, n)
        category_col = np.random.choice(
            ["Controls", "Prevalent"], size=n, p=[0.8, 0.2]
        )
        out_path = tmp_path / "prevalent_subplot.png"

        plot_risk_distribution(
            y_true=None,
            scores=scores,
            out_path=out_path,
            title="Prevalent Subplot",
            category_col=category_col,
        )

        assert out_path.exists()

    def test_all_subplots(self, tmp_path):
        """Test all three subplots (main + incident + prevalent)."""
        n = 300
        scores = np.random.uniform(0, 1, n)
        category_col = np.random.choice(
            ["Controls", "Incident", "Prevalent"], size=n, p=[0.7, 0.2, 0.1]
        )
        out_path = tmp_path / "all_subplots.png"

        plot_risk_distribution(
            y_true=None,
            scores=scores,
            out_path=out_path,
            title="All Subplots",
            category_col=category_col,
        )

        assert out_path.exists()

    def test_threshold_lines(self, tmp_path):
        """Test threshold line rendering."""
        scores = np.random.uniform(0, 1, 100)
        y_true = np.random.binomial(1, 0.3, 100)
        out_path = tmp_path / "thresholds.png"

        plot_risk_distribution(
            y_true=y_true,
            scores=scores,
            out_path=out_path,
            title="Thresholds",
            dca_threshold=0.15,
            spec95_threshold=0.25,
            youden_threshold=0.35,
        )

        assert out_path.exists()

    def test_threshold_metrics(self, tmp_path):
        """Test threshold metrics in legend."""
        scores = np.random.uniform(0, 1, 100)
        y_true = np.random.binomial(1, 0.3, 100)
        out_path = tmp_path / "threshold_metrics.png"

        metrics_at_thresholds = {
            "spec95": {"sensitivity": 0.82, "precision": 0.45, "fp": 12},
            "youden": {"sensitivity": 0.75, "precision": 0.55, "fp": 8},
            "dca": {"sensitivity": 0.88, "precision": 0.38, "fp": 15},
        }

        plot_risk_distribution(
            y_true=y_true,
            scores=scores,
            out_path=out_path,
            title="Threshold Metrics",
            dca_threshold=0.15,
            spec95_threshold=0.25,
            youden_threshold=0.35,
            metrics_at_thresholds=metrics_at_thresholds,
        )

        assert out_path.exists()

    def test_metadata_lines(self, tmp_path):
        """Test metadata rendering at bottom."""
        scores = np.random.uniform(0, 1, 100)
        out_path = tmp_path / "metadata.png"

        meta_lines = [
            "Model: RandomForest",
            "Seed: 42",
            "CV: 5-fold × 10 repeats",
        ]

        plot_risk_distribution(
            y_true=None,
            scores=scores,
            out_path=out_path,
            title="With Metadata",
            meta_lines=meta_lines,
        )

        assert out_path.exists()

    def test_subtitle(self, tmp_path):
        """Test subtitle rendering."""
        scores = np.random.uniform(0, 1, 100)
        out_path = tmp_path / "subtitle.png"

        plot_risk_distribution(
            y_true=None,
            scores=scores,
            out_path=out_path,
            title="Main Title",
            subtitle="Validation Set (n=222)",
        )

        assert out_path.exists()

    def test_custom_xlabel(self, tmp_path):
        """Test custom x-axis label."""
        scores = np.random.uniform(0, 1, 100)
        out_path = tmp_path / "xlabel.png"

        plot_risk_distribution(
            y_true=None,
            scores=scores,
            out_path=out_path,
            title="Custom X-Label",
            xlabel="Risk Score (0-1)",
        )

        assert out_path.exists()

    def test_x_limits(self, tmp_path):
        """Test custom x-axis limits."""
        scores = np.random.uniform(0, 1, 100)
        out_path = tmp_path / "xlimits.png"

        plot_risk_distribution(
            y_true=None,
            scores=scores,
            out_path=out_path,
            title="Custom X-Limits",
            x_limits=(0.0, 0.5),
        )

        assert out_path.exists()

    def test_target_spec_label(self, tmp_path):
        """Test custom target specificity label."""
        scores = np.random.uniform(0, 1, 100)
        y_true = np.random.binomial(1, 0.3, 100)
        out_path = tmp_path / "target_spec.png"

        plot_risk_distribution(
            y_true=y_true,
            scores=scores,
            out_path=out_path,
            title="Custom Target Spec",
            spec95_threshold=0.25,
            target_spec=0.99,
        )

        assert out_path.exists()

    def test_empty_scores(self, tmp_path):
        """Test handling of empty scores array."""
        scores = np.array([])
        out_path = tmp_path / "empty.png"

        plot_risk_distribution(
            y_true=None, scores=scores, out_path=out_path, title="Empty"
        )

        # Should not create file (early return on empty data)
        assert not out_path.exists()

    def test_all_nan_scores(self, tmp_path):
        """Test handling of all-NaN scores."""
        scores = np.array([np.nan, np.nan, np.nan])
        out_path = tmp_path / "all_nan.png"

        plot_risk_distribution(
            y_true=None, scores=scores, out_path=out_path, title="All NaN"
        )

        # Should not create file (early return after NaN filtering)
        assert not out_path.exists()

    def test_partial_nan_scores(self, tmp_path):
        """Test handling of partial NaN scores."""
        scores = np.array([0.1, np.nan, 0.3, np.inf, 0.5])
        out_path = tmp_path / "partial_nan.png"

        plot_risk_distribution(
            y_true=None, scores=scores, out_path=out_path, title="Partial NaN"
        )

        assert out_path.exists()

    def test_invalid_threshold_ignored(self, tmp_path):
        """Test invalid thresholds are ignored."""
        scores = np.random.uniform(0, 1, 100)
        y_true = np.random.binomial(1, 0.3, 100)
        out_path = tmp_path / "invalid_threshold.png"

        # Thresholds outside [0, 1] should be ignored
        plot_risk_distribution(
            y_true=y_true,
            scores=scores,
            out_path=out_path,
            title="Invalid Thresholds",
            dca_threshold=1.5,  # Invalid
            spec95_threshold=-0.1,  # Invalid
            youden_threshold=0.35,  # Valid
        )

        assert out_path.exists()

    def test_kde_fallback_to_histogram(self, tmp_path):
        """Test KDE fallback to histogram with few points."""
        # Very few incident points might cause KDE to fail
        n = 50
        scores = np.random.uniform(0, 1, n)
        category_col = np.array(["Controls"] * 48 + ["Incident"] * 2)
        out_path = tmp_path / "kde_fallback.png"

        plot_risk_distribution(
            y_true=None,
            scores=scores,
            out_path=out_path,
            title="KDE Fallback",
            category_col=category_col,
        )

        assert out_path.exists()

    def test_missing_category_filtered(self, tmp_path):
        """Test categories with zero samples are filtered."""
        n = 100
        scores = np.random.uniform(0, 1, n)
        # Only Controls and Incident, no Prevalent
        category_col = np.random.choice(["Controls", "Incident"], size=n, p=[0.8, 0.2])
        out_path = tmp_path / "missing_category.png"

        plot_risk_distribution(
            y_true=None,
            scores=scores,
            out_path=out_path,
            title="Missing Category",
            category_col=category_col,
        )

        assert out_path.exists()

    def test_pathlib_path(self, tmp_path):
        """Test Path object as out_path."""
        scores = np.random.uniform(0, 1, 100)
        out_path = Path(tmp_path) / "pathlib.png"

        plot_risk_distribution(
            y_true=None, scores=scores, out_path=out_path, title="Path Object"
        )

        assert out_path.exists()

    def test_string_path(self, tmp_path):
        """Test string as out_path."""
        scores = np.random.uniform(0, 1, 100)
        out_path = str(tmp_path / "string.png")

        plot_risk_distribution(
            y_true=None, scores=scores, out_path=out_path, title="String Path"
        )

        assert Path(out_path).exists()
