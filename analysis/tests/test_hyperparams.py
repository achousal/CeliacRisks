"""
Tests for models.hyperparams module (hyperparameter grids).

Tests cover:
- Parameter distributions for all models
- Grid randomization for sensitivity analysis
- Class weight parsing
- Log-space grid generation
- K-best parameter integration
"""

import numpy as np
import pytest
from ced_ml.models.hyperparams import (
    _make_logspace,
    _parse_class_weight_options,
    _randomize_float_list,
    _randomize_int_list,
    get_param_distributions,
)
from conftest import make_mock_config


@pytest.fixture
def minimal_config():
    """Minimal config for testing."""
    from types import SimpleNamespace

    return make_mock_config(
        models=SimpleNamespace(
            lr=SimpleNamespace(
                C_min=0.01,
                C_max=100.0,
                C_points=10,
                class_weight_options="None,balanced",
            ),
            svm=SimpleNamespace(
                C_min=0.01, C_max=100.0, C_points=10, class_weight_options="balanced"
            ),
            rf=SimpleNamespace(
                n_estimators_grid=[100, 200, 500],
                max_depth_grid=[5, 10, 20],
                min_samples_split_grid=[2, 5, 10],
                min_samples_leaf_grid=[1, 2, 4],
                max_features_grid=[0.3, 0.5, 0.7],
                class_weight_options="None,balanced",
            ),
            xgboost=SimpleNamespace(
                n_estimators_grid=[100, 200],
                max_depth_grid=[3, 5, 7],
                learning_rate_grid=[0.01, 0.1, 0.3],
                subsample_grid=[0.7, 0.8, 1.0],
                colsample_bytree_grid=[0.7, 0.8, 1.0],
                scale_pos_weight=None,
                scale_pos_weight_grid=[1.0, 5.0, 10.0],
            ),
        )
    )


# ==================== Parameter Distribution Tests ====================


def test_get_param_distributions_lr(minimal_config):
    """Test LR parameter distributions."""
    params = get_param_distributions("LR_EN", minimal_config)

    assert "clf__C" in params
    assert len(params["clf__C"]) == 10

    assert "clf__class_weight" in params
    assert None in params["clf__class_weight"]
    assert "balanced" in params["clf__class_weight"]


def test_get_param_distributions_svm(minimal_config):
    """Test LinSVM parameter distributions."""
    params = get_param_distributions("LinSVM_cal", minimal_config)

    # SVM wrapped in CalibratedClassifierCV uses estimator__ prefix
    assert "clf__estimator__C" in params
    assert len(params["clf__estimator__C"]) == 10

    assert "clf__estimator__class_weight" in params
    assert "balanced" in params["clf__estimator__class_weight"]


def test_get_param_distributions_rf(minimal_config):
    """Test RF parameter distributions."""
    params = get_param_distributions("RF", minimal_config)

    assert "clf__n_estimators" in params
    assert params["clf__n_estimators"] == [100, 200, 500]

    assert "clf__max_depth" in params
    assert params["clf__max_depth"] == [5, 10, 20]

    assert "clf__min_samples_split" in params
    assert "clf__min_samples_leaf" in params
    assert "clf__max_features" in params


def test_get_param_distributions_xgboost(minimal_config):
    """Test XGBoost parameter distributions."""
    params = get_param_distributions("XGBoost", minimal_config)

    assert "clf__n_estimators" in params
    assert "clf__max_depth" in params
    assert "clf__learning_rate" in params
    assert "clf__subsample" in params
    assert "clf__colsample_bytree" in params
    assert "clf__scale_pos_weight" in params


def test_get_param_distributions_xgboost_custom_spw(minimal_config):
    """Test XGBoost with custom scale_pos_weight."""
    params = get_param_distributions("XGBoost", minimal_config, xgb_spw=7.5)

    spw_grid = params["clf__scale_pos_weight"]
    # Should use custom spw +/- 20%
    assert 7.5 in spw_grid
    assert 7.5 * 0.8 in spw_grid
    assert 7.5 * 1.2 in spw_grid


def test_get_param_distributions_with_kbest(minimal_config):
    """Test parameter distributions with K-best selection."""
    minimal_config.features.selection.method = "kbest"
    minimal_config.features.selection.k_grid = [10, 25, 50, 100]
    minimal_config.features.selection.kbest_scope = "protein"

    params = get_param_distributions("LR_EN", minimal_config)

    assert "prot_sel__k" in params
    assert params["prot_sel__k"] == [10, 25, 50, 100]


def test_get_param_distributions_with_kbest_transformed(minimal_config):
    """Test parameter distributions with K-best in transformed space."""
    minimal_config.features.selection.method = "kbest"
    minimal_config.features.selection.k_grid = [10, 25, 50]
    minimal_config.features.selection.kbest_scope = "transformed"

    params = get_param_distributions("LR_EN", minimal_config)

    assert "sel__k" in params
    assert params["sel__k"] == [10, 25, 50]


def test_get_param_distributions_no_k_grid_raises(minimal_config):
    """Test that kbest without k_grid raises ValueError."""
    minimal_config.features.selection.method = "kbest"
    minimal_config.features.selection.k_grid = []

    with pytest.raises(ValueError, match="k_grid"):
        get_param_distributions("LR_EN", minimal_config)


def test_get_param_distributions_unknown_model(minimal_config):
    """Test that unknown model returns empty dict."""
    params = get_param_distributions("UnknownModel", minimal_config)
    assert params == {}


# ==================== Grid Randomization Tests ====================


def test_randomize_int_list():
    """Test integer list randomization."""
    rng = np.random.RandomState(42)
    values = [10, 50, 100, 200]

    randomized = _randomize_int_list(values, rng, min_val=1)

    # Should have same length
    assert len(randomized) == len(values)

    # Values should be different (with high probability)
    assert randomized != values

    # All values should be >= min_val
    assert all(v >= 1 for v in randomized)


def test_randomize_int_list_unique():
    """Test integer list randomization with uniqueness."""
    rng = np.random.RandomState(42)
    values = [10, 50, 100]

    randomized = _randomize_int_list(values, rng, min_val=1, unique=True)

    # Should have unique values
    assert len(randomized) == len(set(randomized))


def test_randomize_float_list():
    """Test float list randomization."""
    rng = np.random.RandomState(42)
    values = [0.1, 0.5, 1.0]

    randomized = _randomize_float_list(values, rng, min_val=0.0, max_val=1.0)

    # Should have same length
    assert len(randomized) == len(values)

    # Values should be in range
    assert all(0.0 <= v <= 1.0 for v in randomized)


def test_randomize_float_list_log_scale():
    """Test float list randomization in log scale."""
    rng = np.random.RandomState(42)
    values = [0.001, 0.01, 0.1, 1.0]

    randomized = _randomize_float_list(values, rng, min_val=1e-6, log_scale=True)

    # Should have same length
    assert len(randomized) == len(values)

    # All values should be positive
    assert all(v > 0 for v in randomized)


def test_get_param_distributions_with_randomization(minimal_config):
    """Test that grid randomization works end-to-end."""
    rng = np.random.RandomState(42)

    params1 = get_param_distributions("RF", minimal_config, grid_rng=rng)
    rng2 = np.random.RandomState(42)  # Same seed
    params2 = get_param_distributions("RF", minimal_config, grid_rng=rng2)

    # Same seed should give same results
    assert params1["clf__n_estimators"] == params2["clf__n_estimators"]

    # Different from original config
    assert params1["clf__n_estimators"] != minimal_config.models.rf.n_estimators_grid


# ==================== Utility Function Tests ====================


def test_make_logspace():
    """Test log-spaced grid generation."""
    grid = _make_logspace(0.01, 100.0, 5)

    assert len(grid) == 5
    assert grid[0] == pytest.approx(0.01, rel=1e-6)
    assert grid[-1] == pytest.approx(100.0, rel=1e-6)

    # Check log spacing
    log_grid = np.log10(grid)
    diffs = np.diff(log_grid)
    assert np.allclose(diffs, diffs[0])


def test_make_logspace_single_point():
    """Test log-spaced grid with n=1."""
    grid = _make_logspace(0.01, 100.0, 1)

    assert len(grid) == 1
    # Should be geometric mean
    assert grid[0] == pytest.approx(1.0, rel=1e-6)


def test_make_logspace_empty():
    """Test log-spaced grid with n=0."""
    grid = _make_logspace(0.01, 100.0, 0)
    assert grid == []


def test_make_logspace_with_randomization():
    """Test log-spaced grid with perturbation."""
    rng = np.random.RandomState(42)
    grid1 = _make_logspace(0.01, 100.0, 5)
    grid2 = _make_logspace(0.01, 100.0, 5, rng=rng)

    # Should be different
    assert grid1 != grid2

    # Should be in same range
    assert all(0.01 <= v <= 100.0 for v in grid2)


def test_parse_class_weight_options_none():
    """Test parsing 'None' class weight."""
    options = _parse_class_weight_options("None")
    assert options == [None]


def test_parse_class_weight_options_balanced():
    """Test parsing 'balanced' class weight."""
    options = _parse_class_weight_options("balanced")
    assert options == ["balanced"]


def test_parse_class_weight_options_dict():
    """Test parsing dictionary class weights."""
    options = _parse_class_weight_options("{0:1,1:5}")

    assert len(options) == 1
    assert options[0] == {0: 1, 1: 5}


def test_parse_class_weight_options_multiple():
    """Test parsing multiple class weight options."""
    options = _parse_class_weight_options("None,balanced,{0:1,1:10}")

    assert len(options) == 3
    assert None in options
    assert "balanced" in options
    assert {0: 1, 1: 10} in options


def test_parse_class_weight_options_empty():
    """Test parsing empty string."""
    options = _parse_class_weight_options("")
    assert options == [None]


def test_parse_class_weight_options_whitespace():
    """Test parsing with extra whitespace."""
    options = _parse_class_weight_options("  None , balanced  ")
    assert len(options) == 2
    assert None in options
    assert "balanced" in options


# ==================== Integration Tests ====================


def test_all_models_have_params(minimal_config):
    """Test that all standard models return parameters."""
    models = ["LR_EN", "LR_L1", "LinSVM_cal", "RF", "XGBoost"]

    for model in models:
        params = get_param_distributions(model, minimal_config)
        # All models should have at least one parameter to tune
        assert len(params) > 0, f"Model {model} has no parameters"


def test_params_are_json_serializable(minimal_config):
    """Test that all parameter values are JSON-serializable."""
    import json

    models = ["LR_EN", "LinSVM_cal", "RF", "XGBoost"]

    for model in models:
        params = get_param_distributions(model, minimal_config)

        # Convert to JSON-compatible format
        for key, values in params.items():
            # Lists of numbers/strings/dicts should be serializable
            try:
                json.dumps(values)
            except (TypeError, ValueError):
                pytest.fail(f"Parameter {key} for {model} is not JSON-serializable")
