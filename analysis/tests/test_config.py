"""
Tests for configuration system.
"""

import pytest
from ced_ml.config.defaults import DEFAULT_SPLITS_CONFIG
from ced_ml.config.loader import apply_overrides
from ced_ml.config.schema import CVConfig, SplitsConfig


def test_splits_config_defaults():
    """Test that SplitsConfig uses correct defaults."""
    config = SplitsConfig(**DEFAULT_SPLITS_CONFIG)

    assert config.mode == "development"
    assert config.scenarios == ["IncidentOnly"]
    assert config.n_splits == 1
    assert config.val_size == 0.0
    assert config.test_size == 0.30
    assert config.holdout_size == 0.30
    assert config.seed_start == 0
    assert config.prevalent_train_only is False
    assert config.prevalent_train_frac == 1.0
    assert config.train_control_per_case is None


def test_apply_overrides_simple():
    """Test applying simple CLI overrides."""
    config_dict = {"n_splits": 1, "val_size": 0.0}
    overrides = ["n_splits=10", "val_size=0.25"]

    result = apply_overrides(config_dict, overrides)

    assert result["n_splits"] == 10
    assert result["val_size"] == 0.25


def test_apply_overrides_nested():
    """Test applying nested CLI overrides."""
    config_dict = {"cv": {"folds": 5, "repeats": 3}, "features": {"screen_top_n": 0}}
    overrides = ["cv.folds=10", "features.screen_top_n=1000"]

    result = apply_overrides(config_dict, overrides)

    assert result["cv"]["folds"] == 10
    assert result["cv"]["repeats"] == 3  # Unchanged
    assert result["features"]["screen_top_n"] == 1000


def test_apply_overrides_boolean():
    """Test boolean parsing in overrides."""
    config_dict = {"prevalent_train_only": False}

    # Test True variants
    for val in ["true", "True", "yes", "1"]:
        result = apply_overrides(config_dict.copy(), [f"prevalent_train_only={val}"])
        assert result["prevalent_train_only"] is True

    # Test False variants
    for val in ["false", "False", "no", "0"]:
        result = apply_overrides(config_dict.copy(), [f"prevalent_train_only={val}"])
        assert result["prevalent_train_only"] is False


def test_apply_overrides_list():
    """Test list parsing in overrides."""
    config_dict = {"scenarios": []}
    overrides = ["scenarios=IncidentOnly,IncidentPlusPrevalent"]

    result = apply_overrides(config_dict, overrides)

    assert result["scenarios"] == ["IncidentOnly", "IncidentPlusPrevalent"]


def test_config_validation_invalid_split_sizes():
    """Test that invalid split sizes raise validation error."""
    with pytest.raises(ValueError, match="val_size.*test_size"):
        SplitsConfig(
            mode="development",
            val_size=0.6,
            test_size=0.5,  # Total > 1.0
        )


def test_cv_config_validation():
    """Test CVConfig validation."""
    # Valid config
    config = CVConfig(folds=5, repeats=10)
    assert config.folds == 5
    assert config.repeats == 10

    # Invalid: folds < 2
    with pytest.raises(ValueError):
        CVConfig(folds=1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
