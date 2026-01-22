"""
Basic tests for Optuna hyperparameter search module.

Tests cover:
- Optuna availability check
- OptunaSearchCV basic instantiation
- Basic fitting workflow (if optuna available)

Note: Comprehensive Optuna tests should be added in future iterations.
This provides basic smoke testing to prevent regressions.
"""

import pytest
from ced_ml.models.optuna_search import OptunaSearchCV, optuna_available
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# Skip all tests if optuna not available
pytestmark = pytest.mark.skipif(not optuna_available(), reason="Optuna not installed")


class TestOptunaAvailability:
    """Test optuna availability detection."""

    def test_optuna_available_returns_bool(self):
        """optuna_available returns a boolean."""
        result = optuna_available()
        assert isinstance(result, bool)


@pytest.mark.skipif(not optuna_available(), reason="Optuna not installed")
class TestOptunaSearchCVBasic:
    """Basic smoke tests for OptunaSearchCV."""

    def test_instantiation(self):
        """OptunaSearchCV can be instantiated."""
        estimator = LogisticRegression()
        param_distributions = {
            "C": {"type": "float", "low": 0.01, "high": 10.0, "log": True},
        }

        search = OptunaSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            n_trials=5,
            random_state=42,
        )

        assert search.estimator is estimator
        assert search.n_trials == 5
        assert search.random_state == 42

    def test_fit_basic(self):
        """OptunaSearchCV can fit on toy data."""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=42,
        )

        estimator = LogisticRegression(max_iter=100)
        param_distributions = {
            "C": {"type": "float", "low": 0.01, "high": 10.0, "log": True},
        }

        search = OptunaSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            n_trials=3,
            cv=2,
            random_state=42,
        )

        search.fit(X, y)

        # Check basic sklearn interface attributes exist
        assert hasattr(search, "best_estimator_")
        assert hasattr(search, "best_params_")
        assert hasattr(search, "best_score_")
        assert search.best_score_ >= 0.0

    def test_different_samplers(self):
        """OptunaSearchCV supports different sampler types."""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        estimator = LogisticRegression(max_iter=100)
        param_distributions = {
            "C": {"type": "float", "low": 0.1, "high": 1.0},
        }

        for sampler in ["tpe", "random"]:
            search = OptunaSearchCV(
                estimator=estimator,
                param_distributions=param_distributions,
                n_trials=2,
                cv=2,
                sampler=sampler,
                random_state=42,
            )
            search.fit(X, y)
            assert search.best_score_ >= 0.0

    def test_pruning(self):
        """OptunaSearchCV supports pruning."""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        estimator = LogisticRegression(max_iter=100)
        param_distributions = {
            "C": {"type": "float", "low": 0.1, "high": 1.0},
        }

        search = OptunaSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            n_trials=5,
            cv=3,
            pruner="median",
            pruner_n_startup_trials=2,
            random_state=42,
        )
        search.fit(X, y)
        assert search.best_score_ >= 0.0
