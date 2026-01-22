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


def test_apply_overrides_run_id_string():
    """Test that run_id is treated as string, not parsed as int."""
    config_dict = {"run_id": None, "run_name": "test"}

    # Numeric-looking run_id should stay as string
    overrides = ["run_id=20260121125646"]
    result = apply_overrides(config_dict, overrides)

    assert result["run_id"] == "20260121125646"
    assert isinstance(result["run_id"], str)

    # run_name should also stay as string
    overrides = ["run_name=12345"]
    result = apply_overrides(config_dict, overrides)

    assert result["run_name"] == "12345"
    assert isinstance(result["run_name"], str)


def test_resolve_paths_relative_to_config(tmp_path):
    """Test path resolution relative to config file directory."""
    from pathlib import Path

    from ced_ml.config.loader import resolve_paths_relative_to_config

    # Create test directory structure
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create test data file
    test_file = data_dir / "input.csv"
    test_file.write_text("dummy")

    # Config file path
    config_file = config_dir / "test_config.yaml"

    # Config dict with relative path
    config_dict = {
        "infile": "../data/input.csv",  # Relative to config file
        "outdir": "../results",
    }

    # Resolve paths
    resolved = resolve_paths_relative_to_config(config_dict, config_file)

    # Check that infile was resolved (exists)
    assert Path(resolved["infile"]).exists()
    assert Path(resolved["infile"]).is_absolute()

    # Check that outdir was resolved (doesn't exist but path contains separator)
    assert Path(resolved["outdir"]).is_absolute()


def test_resolve_paths_nested_config(tmp_path):
    """Test path resolution in nested config dicts."""
    from pathlib import Path

    from ced_ml.config.loader import resolve_paths_relative_to_config

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    test_file = data_dir / "model.joblib"
    test_file.write_text("dummy")

    config_file = config_dir / "test.yaml"

    # Nested config
    config_dict = {
        "evaluation": {
            "model_artifact": "../data/model.joblib",
        },
        "output": {
            "outdir": "../results",
        },
    }

    resolved = resolve_paths_relative_to_config(config_dict, config_file)

    # Check nested path resolution
    assert Path(resolved["evaluation"]["model_artifact"]).exists()
    assert Path(resolved["evaluation"]["model_artifact"]).is_absolute()


def test_resolve_paths_absolute_unchanged(tmp_path):
    """Test that absolute paths are unchanged."""
    from ced_ml.config.loader import resolve_paths_relative_to_config

    config_file = tmp_path / "config.yaml"

    # Absolute path should not be changed
    abs_path = str(tmp_path / "data" / "input.csv")
    config_dict = {"infile": abs_path}

    resolved = resolve_paths_relative_to_config(config_dict, config_file)

    # Absolute path unchanged
    assert resolved["infile"] == abs_path


def test_resolve_paths_non_path_values_unchanged(tmp_path):
    """Test that non-path values are unchanged."""
    from ced_ml.config.loader import resolve_paths_relative_to_config

    config_file = tmp_path / "config.yaml"

    # Non-path values
    config_dict = {
        "n_splits": 10,
        "model": "LR_EN",
        "val_size": 0.25,
        "enabled": True,
    }

    resolved = resolve_paths_relative_to_config(config_dict, config_file)

    # All values unchanged
    assert resolved == config_dict


def test_unwired_feature_selection_config_defaults():
    """Test that default feature selection config produces no validation issues (H2 fix)."""
    from ced_ml.config.schema import TrainingConfig
    from ced_ml.config.validation import _validate_unwired_feature_selection_config

    config = TrainingConfig(infile="/tmp/test.parquet")
    issues = []
    _validate_unwired_feature_selection_config(config, issues)

    assert len(issues) == 0, "Default config should not produce validation issues"


def test_unwired_feature_selection_config_screen_top_n():
    """Test that non-default screen_top_n produces validation issue (H2 fix)."""
    from ced_ml.config.schema import FeatureConfig, TrainingConfig
    from ced_ml.config.validation import _validate_unwired_feature_selection_config

    config = TrainingConfig(infile="/tmp/test.parquet", features=FeatureConfig(screen_top_n=500))
    issues = []
    _validate_unwired_feature_selection_config(config, issues)

    assert len(issues) == 1, "Non-default screen_top_n should produce validation issue"
    assert "screen_top_n=500" in issues[0]


def test_unwired_feature_selection_config_advanced_mode():
    """Test that advanced feature_select modes produce validation issue (H2 fix)."""
    from ced_ml.config.schema import FeatureConfig, TrainingConfig
    from ced_ml.config.validation import _validate_unwired_feature_selection_config

    for mode in ["hybrid", "l1_stability"]:
        config = TrainingConfig(
            infile="/tmp/test.parquet", features=FeatureConfig(feature_select=mode)
        )
        issues = []
        _validate_unwired_feature_selection_config(config, issues)

        assert len(issues) == 1, f"Advanced mode '{mode}' should produce validation issue"
        assert mode in issues[0]


def test_unwired_feature_selection_config_multiple_non_defaults():
    """Test that multiple non-default settings are combined in one issue (H2 fix)."""
    from ced_ml.config.schema import FeatureConfig, TrainingConfig
    from ced_ml.config.validation import _validate_unwired_feature_selection_config

    config = TrainingConfig(
        infile="/tmp/test.parquet",
        features=FeatureConfig(
            screen_top_n=1000, stability_thresh=0.90, feature_select="l1_stability"
        ),
    )
    issues = []
    _validate_unwired_feature_selection_config(config, issues)

    assert len(issues) == 1, "Multiple non-defaults should be combined in one issue"
    assert "screen_top_n=1000" in issues[0]
    assert "stability_thresh=0.9" in issues[0]
    assert "l1_stability" in issues[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
