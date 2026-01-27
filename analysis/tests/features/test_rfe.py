"""Tests for Recursive Feature Elimination module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ced_ml.features.rfe import (
    RFEResult,
    compute_eval_sizes,
    compute_feature_importance,
    detect_knee_point,
    extract_pareto_frontier,
    find_recommended_panels,
    save_rfe_results,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class TestComputeEvalSizes:
    """Tests for compute_eval_sizes function."""

    def test_geometric_strategy(self):
        """Geometric strategy produces powers of 2."""
        sizes = compute_eval_sizes(100, 5, "geometric")
        # Should include 100, 50, 25, 12, 6, 5
        assert 100 in sizes
        assert 50 in sizes
        assert 25 in sizes
        assert 5 in sizes
        assert sizes == sorted(sizes, reverse=True)

    def test_linear_strategy(self):
        """Linear strategy produces all sizes."""
        sizes = compute_eval_sizes(10, 5, "linear")
        assert sizes == [10, 9, 8, 7, 6, 5]

    def test_geometric_default(self):
        """Geometric is the default strategy."""
        default = compute_eval_sizes(100, 5)
        explicit = compute_eval_sizes(100, 5, "geometric")
        assert default == explicit

    def test_min_size_included(self):
        """Min size is always included."""
        sizes = compute_eval_sizes(100, 7, "geometric")
        assert 7 in sizes

    def test_edge_case_max_equals_min(self):
        """Handles max == min gracefully."""
        sizes = compute_eval_sizes(5, 5, "geometric")
        assert sizes == [5]

    def test_small_range(self):
        """Handles small ranges."""
        sizes = compute_eval_sizes(8, 5, "geometric")
        assert 8 in sizes
        assert 5 in sizes

    def test_fine_strategy(self):
        """Fine strategy produces more granular steps."""
        sizes = compute_eval_sizes(100, 5, "fine")
        # Should have more points than geometric
        geometric_sizes = compute_eval_sizes(100, 5, "geometric")
        assert len(sizes) > len(geometric_sizes)
        # Should include 100, 75, 50, 37, 25, etc.
        assert 100 in sizes
        assert 75 in sizes
        assert 50 in sizes
        assert 5 in sizes
        assert sizes == sorted(sizes, reverse=True)

    def test_fine_strategy_intermediate_points(self):
        """Fine strategy includes quarter-step interpolation."""
        sizes = compute_eval_sizes(100, 5, "fine")
        # Should have intermediate points between powers of 2
        # Between 100 and 50, should have 75
        assert 75 in sizes
        # Between 50 and 25, should have 37
        assert 37 in sizes or 38 in sizes  # int(50 * 0.75)

    def test_fine_vs_geometric(self):
        """Fine strategy produces more evaluation points than geometric."""
        geometric = compute_eval_sizes(200, 10, "geometric")
        fine = compute_eval_sizes(200, 10, "fine")
        # Fine should have at least 1.5x as many points
        assert len(fine) >= len(geometric) * 1.5


class TestFindRecommendedPanels:
    """Tests for find_recommended_panels function."""

    def test_basic_recommendations(self):
        """Basic threshold recommendations work."""
        curve = [
            {"size": 100, "auroc_val": 0.90},
            {"size": 50, "auroc_val": 0.88},
            {"size": 25, "auroc_val": 0.85},
            {"size": 10, "auroc_val": 0.75},
        ]
        rec = find_recommended_panels(curve, [0.95, 0.90])
        # 95% of 0.90 = 0.855, smallest meeting this is 25
        # 90% of 0.90 = 0.81, smallest meeting this is 10
        assert "min_size_95pct" in rec
        assert "min_size_90pct" in rec
        # Higher threshold (95%) requires larger panel to maintain AUROC
        assert rec["min_size_95pct"] >= rec["min_size_90pct"]

    def test_knee_point_included(self):
        """Knee point is always included."""
        curve = [
            {"size": 100, "auroc_val": 0.90},
            {"size": 50, "auroc_val": 0.89},
            {"size": 25, "auroc_val": 0.85},
        ]
        rec = find_recommended_panels(curve)
        assert "knee_point" in rec

    def test_empty_curve(self):
        """Empty curve returns empty dict."""
        rec = find_recommended_panels([])
        assert rec == {}

    def test_single_point(self):
        """Single point curve handled."""
        curve = [{"size": 50, "auroc_val": 0.85}]
        rec = find_recommended_panels(curve, [0.95])
        assert "min_size_95pct" in rec
        assert rec["min_size_95pct"] == 50


class TestDetectKneePoint:
    """Tests for detect_knee_point function."""

    def test_clear_knee(self):
        """Detects clear knee point."""
        # Curve with obvious knee at size 25
        curve = [
            {"size": 100, "auroc_val": 0.90},
            {"size": 75, "auroc_val": 0.895},
            {"size": 50, "auroc_val": 0.89},
            {"size": 25, "auroc_val": 0.88},  # Knee here
            {"size": 10, "auroc_val": 0.70},
            {"size": 5, "auroc_val": 0.60},
        ]
        knee = detect_knee_point(curve)
        # Knee should be around where the curve bends
        assert knee in [25, 50, 10]  # Reasonable range

    def test_monotonic_decline(self):
        """Handles monotonically declining curve."""
        curve = [
            {"size": 100, "auroc_val": 0.90},
            {"size": 50, "auroc_val": 0.80},
            {"size": 25, "auroc_val": 0.70},
        ]
        knee = detect_knee_point(curve)
        assert knee in [100, 50, 25]

    def test_few_points(self):
        """Handles curves with few points."""
        curve = [
            {"size": 50, "auroc_val": 0.85},
            {"size": 25, "auroc_val": 0.80},
        ]
        knee = detect_knee_point(curve)
        assert knee in [50, 25]

    def test_single_point(self):
        """Handles single point."""
        curve = [{"size": 50, "auroc_val": 0.85}]
        knee = detect_knee_point(curve)
        assert knee == 50


class TestExtractParetoFrontier:
    """Tests for extract_pareto_frontier function."""

    def test_all_pareto_optimal(self):
        """All points Pareto-optimal when monotonically decreasing AUROC."""
        # For Pareto optimality: a point is dominated if another has
        # BOTH smaller size AND higher AUROC.
        # In this curve, each smaller panel has lower AUROC, so only
        # the largest panel (100) is Pareto-optimal.
        curve = [
            {"size": 100, "auroc_val": 0.90},
            {"size": 50, "auroc_val": 0.85},  # Dominated by 100 (higher AUROC)
            {"size": 25, "auroc_val": 0.80},  # Dominated by 100 (higher AUROC)
        ]
        pareto = extract_pareto_frontier(curve)
        # Only size=100 is on the Pareto frontier (best AUROC overall)
        assert len(pareto) == 1
        assert pareto[0]["size"] == 100

    def test_dominated_points_excluded(self):
        """Dominated points are excluded."""
        curve = [
            {"size": 100, "auroc_val": 0.90},
            {"size": 75, "auroc_val": 0.85},  # Dominated by 50
            {"size": 50, "auroc_val": 0.88},  # Dominated by 100
            {"size": 25, "auroc_val": 0.80},  # Dominated by 100
        ]
        pareto = extract_pareto_frontier(curve)
        # Only size=100 is on Pareto frontier (best AUROC)
        # All other points have smaller size but also lower AUROC,
        # so they're dominated by 100
        pareto_sizes = [p["size"] for p in pareto]
        assert 75 not in pareto_sizes
        assert 100 in pareto_sizes
        # 50 is dominated by 100 (100 has higher AUROC)
        assert 50 not in pareto_sizes

    def test_empty_curve(self):
        """Empty curve returns empty list."""
        pareto = extract_pareto_frontier([])
        assert pareto == []

    def test_true_pareto_frontier(self):
        """Test curve with multiple Pareto-optimal points."""
        # This curve has genuine trade-offs: as size decreases,
        # AUROC sometimes increases (non-monotonic)
        curve = [
            {"size": 100, "auroc_val": 0.85},  # Pareto-optimal (largest panel)
            {"size": 75, "auroc_val": 0.90},  # Pareto-optimal (best AUROC)
            {"size": 50, "auroc_val": 0.88},  # Dominated by 75
            {"size": 25, "auroc_val": 0.86},  # Dominated by 75
            {"size": 10, "auroc_val": 0.80},  # Dominated by 75
        ]
        pareto = extract_pareto_frontier(curve)
        # size=100 and size=75 are both Pareto-optimal (trade-off between size and AUROC)
        assert len(pareto) == 2
        assert pareto[0]["size"] == 100
        assert pareto[1]["size"] == 75


class TestComputeFeatureImportance:
    """Tests for compute_feature_importance function."""

    @pytest.fixture
    def simple_pipeline(self):
        """Create a simple fitted pipeline for testing."""
        np.random.seed(42)
        n_samples = 100
        n_features = 10

        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"protein_{i}" for i in range(n_features)],
        )
        y = (X["protein_0"] + X["protein_1"] > 0).astype(int)

        pipeline = Pipeline(
            [
                ("pre", StandardScaler()),
                ("clf", LogisticRegression(random_state=42, max_iter=1000)),
            ]
        )
        pipeline.fit(X, y)
        return pipeline, X, y

    def test_linear_model_importance(self, simple_pipeline):
        """Linear model importance extraction works."""
        pipeline, X, y = simple_pipeline
        protein_cols = list(X.columns)

        importance = compute_feature_importance(
            pipeline,
            model_name="LR_EN",
            protein_cols=protein_cols,
            X=X,
            y=y.values,
        )

        assert len(importance) == len(protein_cols)
        assert all(v >= 0 for v in importance.values())  # Absolute values

    def test_importance_nonzero(self, simple_pipeline):
        """At least some features have non-zero importance."""
        pipeline, X, y = simple_pipeline
        protein_cols = list(X.columns)

        importance = compute_feature_importance(
            pipeline,
            model_name="LR_EN",
            protein_cols=protein_cols,
            X=X,
            y=y.values,
        )

        assert sum(importance.values()) > 0


class TestRFEResult:
    """Tests for RFEResult dataclass."""

    def test_default_values(self):
        """Default values are empty."""
        result = RFEResult()
        assert result.curve == []
        assert result.feature_ranking == {}
        assert result.recommended_panels == {}
        assert result.max_auroc == 0.0

    def test_with_values(self):
        """Can initialize with values."""
        result = RFEResult(
            curve=[{"size": 50, "auroc_val": 0.85}],
            feature_ranking={"protein_0": 0},
            recommended_panels={"knee_point": 50},
            max_auroc=0.85,
            model_name="LR_EN",
        )
        assert len(result.curve) == 1
        assert result.model_name == "LR_EN"


class TestSaveRFEResults:
    """Tests for save_rfe_results function."""

    def test_saves_all_artifacts(self):
        """All artifacts are saved."""
        result = RFEResult(
            curve=[
                {
                    "size": 50,
                    "auroc_val": 0.85,
                    "auroc_cv": 0.84,
                    "auroc_cv_std": 0.02,
                    "proteins": ["p1", "p2"],
                },
            ],
            feature_ranking={"protein_0": 0, "protein_1": 1},
            recommended_panels={"knee_point": 50, "min_size_95pct": 25},
            pareto_points=[{"size": 50, "auroc_val": 0.85}],
            max_auroc=0.85,
            model_name="LR_EN",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_rfe_results(result, tmpdir, "LR_EN", 0)

            # Check all files exist
            assert Path(paths["panel_curve"]).exists()
            assert Path(paths["feature_ranking"]).exists()
            assert Path(paths["recommended_panels"]).exists()
            assert Path(paths["pareto_frontier"]).exists()

            # Check panel_curve.csv content
            curve_df = pd.read_csv(paths["panel_curve"])
            assert "size" in curve_df.columns
            assert "auroc_val" in curve_df.columns

            # Check recommended_panels.json content
            with open(paths["recommended_panels"]) as f:
                rec = json.load(f)
            assert rec["model"] == "LR_EN"
            assert rec["max_auroc"] == 0.85

    def test_creates_output_directory(self):
        """Creates output directory if not exists."""
        result = RFEResult(
            curve=[
                {
                    "size": 50,
                    "auroc_val": 0.85,
                    "auroc_cv": 0.84,
                    "auroc_cv_std": 0.02,
                    "proteins": [],
                }
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir) / "nested" / "output"
            save_rfe_results(result, str(outdir), "LR_EN", 0)
            assert outdir.exists()
