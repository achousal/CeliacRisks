"""Tests for models.registry module.

Tests cover:
- Model instantiation (LR, SVM, RF, XGBoost)
- Hyperparameter grid generation
- sklearn version compatibility
- Parameter parsing utilities
"""

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ced_ml.config.schema import TrainingConfig
from ced_ml.models.registry import (
    XGBOOST_AVAILABLE,
    _coerce_int_or_none_list,
    _coerce_min_samples_leaf_list,
    _parse_float_list,
    _parse_int_list,
    _randomize_numeric_list,
    _require_int_list,
    build_linear_svm_calibrated,
    build_logistic_regression,
    build_models,
    build_random_forest,
    build_xgboost,
    compute_scale_pos_weight_from_y,
    get_param_distributions,
    make_logspace,
    parse_class_weight_options,
)


# ----------------------------
# Model builders
# ----------------------------
def test_build_logistic_regression():
    """Test LogisticRegression instantiation."""
    lr = build_logistic_regression(
        solver="saga", C=0.1, max_iter=1000, tol=1e-3, random_state=123, l1_ratio=0.5
    )
    assert isinstance(lr, LogisticRegression)
    assert lr.solver == "saga"
    assert lr.C == 0.1
    assert lr.max_iter == 1000
    assert lr.tol == 1e-3
    assert lr.random_state == 123


def test_build_linear_svm_calibrated():
    """Test CalibratedClassifierCV with LinearSVC."""
    svm = build_linear_svm_calibrated(
        C=1.0, max_iter=2000, calibration_method="sigmoid", calibration_cv=3, random_state=456
    )
    assert isinstance(svm, CalibratedClassifierCV)
    assert svm.method == "sigmoid"
    assert svm.cv == 3


def test_build_random_forest():
    """Test RandomForestClassifier instantiation."""
    rf = build_random_forest(
        n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=789, n_jobs=2
    )
    assert isinstance(rf, RandomForestClassifier)
    assert rf.n_estimators == 100
    assert rf.max_depth == 10
    assert rf.min_samples_leaf == 5
    assert rf.random_state == 789
    assert rf.n_jobs == 2


def test_build_random_forest_with_max_samples():
    """Test RF with max_samples parameter."""
    rf = build_random_forest(n_estimators=100, max_samples=0.8, random_state=42)
    assert rf.max_samples == 0.8


@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not installed")
def test_build_xgboost():
    """Test XGBClassifier instantiation."""
    xgb = build_xgboost(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=2.0,
        random_state=111,
        n_jobs=1,
    )
    assert xgb.n_estimators == 500
    assert xgb.max_depth == 5
    assert xgb.learning_rate == 0.05
    assert xgb.scale_pos_weight == 2.0
    assert xgb.random_state == 111


@pytest.mark.skipif(XGBOOST_AVAILABLE, reason="Test XGBoost ImportError")
def test_build_xgboost_not_available():
    """Test XGBoost raises ImportError when not available."""
    with pytest.raises(ImportError, match="XGBoost not available"):
        build_xgboost()


def test_build_models_lr_en(training_config):
    """Test build_models for LR_EN."""
    model = build_models("LR_EN", training_config, random_state=42, n_jobs=1)
    assert isinstance(model, LogisticRegression)


def test_build_models_lr_l1(training_config):
    """Test build_models for LR_L1."""
    model = build_models("LR_L1", training_config, random_state=42, n_jobs=1)
    assert isinstance(model, LogisticRegression)


def test_build_models_linsvm_cal(training_config):
    """Test build_models for LinSVM_cal."""
    model = build_models("LinSVM_cal", training_config, random_state=42, n_jobs=1)
    assert isinstance(model, CalibratedClassifierCV)


def test_build_models_rf(training_config):
    """Test build_models for RF."""
    model = build_models("RF", training_config, random_state=42, n_jobs=2)
    assert isinstance(model, RandomForestClassifier)
    assert model.n_jobs == 2


@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not installed")
def test_build_models_xgboost(training_config):
    """Test build_models for XGBoost."""
    model = build_models("XGBoost", training_config, random_state=42, n_jobs=1)
    assert model.n_estimators == training_config.hyperparameters.xgb_n_estimators


def test_build_models_unknown():
    """Test build_models raises ValueError for unknown model."""
    config = TrainingConfig(infile="dummy.csv")
    with pytest.raises(ValueError, match="Unknown model"):
        build_models("InvalidModel", config, random_state=42)


# ----------------------------
# Parameter parsing
# ----------------------------
def test_parse_float_list():
    """Test float list parsing."""
    assert _parse_float_list("1.0,2.5,3.7") == [1.0, 2.5, 3.7]
    assert _parse_float_list("") == []
    assert _parse_float_list("1.0, , 2.0") == [1.0, 2.0]
    assert _parse_float_list("1.0,invalid,2.0") == [1.0, 2.0]


def test_parse_int_list():
    """Test integer list parsing."""
    assert _parse_int_list("10,20,30") == [10, 20, 30]
    assert _parse_int_list("") == []
    assert _parse_int_list("10, , 20") == [10, 20]
    assert _parse_int_list("10,invalid,20") == [10, 20]


def test_require_int_list():
    """Test required integer list parsing."""
    assert _require_int_list("100,200", "test") == [100, 200]
    with pytest.raises(ValueError, match="must be a non-empty"):
        _require_int_list("", "test")


def test_coerce_int_or_none_list():
    """Test int-or-None coercion."""
    assert _coerce_int_or_none_list([10, None, 20.0], name="test") == [10, None, 20]
    assert _coerce_int_or_none_list(["10", "none", "20"], name="test") == [10, None, 20]
    with pytest.raises(ValueError, match="non-integer float"):
        _coerce_int_or_none_list([10.5], name="test")
    with pytest.raises(ValueError, match="invalid boolean"):
        _coerce_int_or_none_list([True], name="test")


def test_coerce_min_samples_leaf_list():
    """Test min_samples_leaf coercion."""
    assert _coerce_min_samples_leaf_list([5, 10.0, 0.2]) == [5, 10, 0.2]
    assert _coerce_min_samples_leaf_list(["5", "10", "0.2"]) == [5, 10, 0.2]
    with pytest.raises(ValueError, match="must be >= 1"):
        _coerce_min_samples_leaf_list([0])
    with pytest.raises(ValueError, match="must be in \\(0,1\\)"):
        _coerce_min_samples_leaf_list([10.5])


def test_parse_class_weight_options():
    """Test class_weight parsing."""
    assert parse_class_weight_options("none,balanced") == [None, "balanced"]
    assert parse_class_weight_options("balanced") == ["balanced"]
    assert parse_class_weight_options("") == [None, "balanced"]
    assert parse_class_weight_options("none,none,balanced") == [None, "balanced"]  # dedupe


def test_make_logspace():
    """Test log-spaced grid generation."""
    grid = make_logspace(1e-3, 1e3, 5)
    assert len(grid) == 5
    assert np.isclose(grid[0], 1e-3)
    assert np.isclose(grid[-1], 1e3)


def test_make_logspace_randomized():
    """Test randomized log-spaced grids."""
    rng = np.random.RandomState(42)
    grid = make_logspace(1e-3, 1e3, 5, rng=rng)
    assert len(grid) == 5
    assert grid[0] >= 1e-3 and grid[-1] <= 1e3


def test_make_logspace_invalid():
    """Test log-space with invalid inputs."""
    grid = make_logspace(0, 1e3, 5)  # invalid minv
    assert len(grid) == 13  # fallback


def test_compute_scale_pos_weight_from_y():
    """Test XGBoost scale_pos_weight computation."""
    y = np.array([0, 0, 0, 1])  # 3 neg, 1 pos
    spw = compute_scale_pos_weight_from_y(y)
    assert spw == 3.0


def test_compute_scale_pos_weight_edge_case():
    """Test scale_pos_weight with all positives."""
    y = np.array([1, 1, 1])
    spw = compute_scale_pos_weight_from_y(y)
    # All positive: neg=0 -> max(1, 0) = 1, pos=3 -> spw = 1/3
    assert np.isclose(spw, 1 / 3)


# ----------------------------
# Grid randomization
# ----------------------------
def test_randomize_numeric_list_preserve_non_numeric():
    """Test that non-numeric values are preserved."""
    rng = np.random.RandomState(42)
    values = ["sqrt", 10, 20]
    result = _randomize_numeric_list(values, rng, as_int=True, perturb_mode=True)
    assert result[0] == "sqrt"
    assert isinstance(result[1], int)
    assert isinstance(result[2], int)


def test_randomize_numeric_list_perturb_mode():
    """Test perturbation mode for floats."""
    rng = np.random.RandomState(42)
    values = [1.0, 2.0, 3.0]
    result = _randomize_numeric_list(values, rng, perturb_mode=True)
    assert len(result) == 3
    # Values should be perturbed but close to originals
    assert 0.5 < result[0] < 1.5
    assert 1.5 < result[1] < 2.5
    assert 2.5 < result[2] < 3.5


def test_randomize_numeric_list_int_unique():
    """Test unique integer randomization."""
    rng = np.random.RandomState(42)
    values = [10, 20, 30]
    result = _randomize_numeric_list(values, rng, as_int=True, unique_int=True, perturb_mode=True)
    assert len(result) == 3
    assert len(set(result)) == 3  # all unique
    assert all(isinstance(x, int) for x in result)


def test_randomize_numeric_list_with_bounds():
    """Test randomization with min/max bounds."""
    rng = np.random.RandomState(42)
    values = [1.0, 2.0, 3.0]
    result = _randomize_numeric_list(values, rng, min_float=0.5, max_float=3.5, perturb_mode=True)
    assert all(0.5 <= x <= 3.5 for x in result)


def test_randomize_numeric_list_no_rng():
    """Test that None rng returns unchanged list."""
    values = [1.0, 2.0, 3.0]
    result = _randomize_numeric_list(values, None, perturb_mode=True)
    assert result == values


# ----------------------------
# Hyperparameter grids
# ----------------------------
def test_get_param_distributions_lr_en(training_config):
    """Test parameter grid for LR_EN."""
    grid = get_param_distributions(
        "LR_EN",
        training_config,
        feature_select="none",
        k_grid=[],
        kbest_scope="protein",
        randomize_grids=False,
    )
    assert "clf__C" in grid
    assert "clf__l1_ratio" in grid
    assert "clf__class_weight" in grid
    assert len(grid["clf__C"]) > 0


def test_get_param_distributions_lr_l1(training_config):
    """Test parameter grid for LR_L1."""
    grid = get_param_distributions(
        "LR_L1",
        training_config,
        feature_select="none",
        k_grid=[],
        kbest_scope="protein",
        randomize_grids=False,
    )
    assert "clf__C" in grid
    assert "clf__class_weight" in grid
    assert "clf__l1_ratio" not in grid  # L1 only


def test_get_param_distributions_linsvm_cal(training_config):
    """Test parameter grid for LinSVM_cal."""
    grid = get_param_distributions(
        "LinSVM_cal",
        training_config,
        feature_select="none",
        k_grid=[],
        kbest_scope="protein",
        randomize_grids=False,
    )
    # Check for estimator or base_estimator key
    assert any("__C" in k for k in grid.keys())
    assert any("__class_weight" in k for k in grid.keys())


def test_get_param_distributions_rf(training_config):
    """Test parameter grid for RF."""
    grid = get_param_distributions(
        "RF",
        training_config,
        feature_select="none",
        k_grid=[],
        kbest_scope="protein",
        randomize_grids=False,
    )
    assert "clf__n_estimators" in grid
    assert "clf__max_depth" in grid
    assert "clf__min_samples_leaf" in grid
    assert "clf__min_samples_split" in grid
    assert "clf__max_features" in grid
    assert "clf__class_weight" in grid
    # Note: new config doesn't have bootstrap field


@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not installed")
def test_get_param_distributions_xgboost(training_config):
    """Test parameter grid for XGBoost."""
    grid = get_param_distributions(
        "XGBoost",
        training_config,
        feature_select="none",
        k_grid=[],
        kbest_scope="protein",
        xgb_scale_pos_weight=2.0,
        randomize_grids=False,
    )
    assert "clf__n_estimators" in grid
    assert "clf__max_depth" in grid
    assert "clf__learning_rate" in grid
    assert "clf__subsample" in grid
    assert "clf__colsample_bytree" in grid
    assert "clf__scale_pos_weight" in grid


def test_get_param_distributions_with_kbest_protein(training_config):
    """Test grid with protein-scope kbest."""
    grid = get_param_distributions(
        "LR_EN",
        training_config,
        feature_select="kbest",
        k_grid=[50, 100, 200],
        kbest_scope="protein",
        randomize_grids=False,
    )
    assert "prot_sel__k" in grid
    assert grid["prot_sel__k"] == [50, 100, 200]


def test_get_param_distributions_with_kbest_transformed(training_config):
    """Test grid with transformed-scope kbest."""
    grid = get_param_distributions(
        "LR_EN",
        training_config,
        feature_select="kbest",
        k_grid=[50, 100, 200],
        kbest_scope="transformed",
        randomize_grids=False,
    )
    assert "sel__k" in grid
    assert grid["sel__k"] == [50, 100, 200]


def test_get_param_distributions_kbest_missing_grid():
    """Test that kbest without k_grid raises error."""
    config = TrainingConfig(infile="dummy.csv")
    with pytest.raises(ValueError, match="requires k_grid"):
        get_param_distributions(
            "LR_EN",
            config,
            feature_select="kbest",
            k_grid=[],
            kbest_scope="protein",
            randomize_grids=False,
        )


def test_get_param_distributions_randomized(training_config):
    """Test randomized grid generation."""
    rng = np.random.RandomState(42)
    grid1 = get_param_distributions(
        "LR_EN",
        training_config,
        feature_select="none",
        k_grid=[],
        kbest_scope="protein",
        grid_rng=rng,
        randomize_grids=True,
    )
    rng2 = np.random.RandomState(123)
    grid2 = get_param_distributions(
        "LR_EN",
        training_config,
        feature_select="none",
        k_grid=[],
        kbest_scope="protein",
        grid_rng=rng2,
        randomize_grids=True,
    )
    # With new config, grids come from config not randomization
    # Randomization only applies to l1_ratio
    if len(grid1["clf__l1_ratio"]) > 1 and len(grid2["clf__l1_ratio"]) > 1:
        assert not np.allclose(grid1["clf__l1_ratio"], grid2["clf__l1_ratio"])


def test_get_param_distributions_empty_model():
    """Test unknown model returns empty dict."""
    config = TrainingConfig(infile="dummy.csv")
    grid = get_param_distributions(
        "UnknownModel",
        config,
        feature_select="none",
        k_grid=[],
        kbest_scope="protein",
        randomize_grids=False,
    )
    assert grid == {}


# ----------------------------
# Fixtures
# ----------------------------
@pytest.fixture
def training_config():
    """Create default TrainingConfig for testing."""
    return TrainingConfig(infile="dummy.csv")
