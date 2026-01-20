"""
Tests for Phase C config management tools.

Tests:
- Config migration (legacy args -> YAML)
- Config validation
- Config diff
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from ced_ml.cli.config_tools import (
    _map_to_nested_key,
    _parse_legacy_args,
    _parse_value_smart,
    diff_configs,
    migrate_legacy_args_to_yaml,
    validate_config_file,
)


class TestParseLegacyArgs:
    """Test legacy CLI argument parsing."""

    def test_parse_simple_args(self):
        """Test parsing simple flag and value args."""
        args = ["--n-splits", "10", "--val-size", "0.25", "--prevalent-train-only"]
        result = _parse_legacy_args(args, "save-splits")

        assert result["n_splits"] == 10
        assert result["val_size"] == 0.25
        assert result["prevalent_train_only"] is True

    def test_parse_list_args(self):
        """Test parsing comma-separated list values."""
        args = ["--scenarios", "IncidentOnly,IncidentPlusPrevalent"]
        result = _parse_legacy_args(args, "save-splits")

        assert result["scenarios"] == ["IncidentOnly", "IncidentPlusPrevalent"]

    def test_parse_nested_training_args(self):
        """Test mapping training args to nested structure."""
        args = ["--folds", "10", "--screen-top-n", "2000", "--threshold-objective", "youden"]
        result = _parse_legacy_args(args, "train")

        # Should be mapped to nested structure by _map_to_nested_key
        # This test verifies the structure after parsing
        assert "folds" in result or "cv" in result
        assert "screen_top_n" in result or "features" in result


class TestParseValueSmart:
    """Test value parsing logic."""

    def test_parse_boolean(self):
        """Test boolean parsing."""
        assert _parse_value_smart("true") is True
        assert _parse_value_smart("True") is True
        assert _parse_value_smart("yes") is True
        assert _parse_value_smart("false") is False
        assert _parse_value_smart("False") is False
        assert _parse_value_smart("no") is False

    def test_parse_none(self):
        """Test None parsing."""
        assert _parse_value_smart("none") is None
        assert _parse_value_smart("None") is None
        assert _parse_value_smart("null") is None

    def test_parse_int(self):
        """Test integer parsing."""
        assert _parse_value_smart("10") == 10
        assert _parse_value_smart("0") == 0
        assert _parse_value_smart("-5") == -5

    def test_parse_float(self):
        """Test float parsing."""
        assert _parse_value_smart("0.25") == 0.25
        assert _parse_value_smart("1.5") == 1.5
        assert _parse_value_smart("-2.3") == -2.3

    def test_parse_string(self):
        """Test string parsing."""
        assert _parse_value_smart("hello") == "hello"
        assert _parse_value_smart("IncidentOnly") == "IncidentOnly"

    def test_parse_list(self):
        """Test list parsing."""
        result = _parse_value_smart("1,2,3")
        assert result == [1, 2, 3]

        result = _parse_value_smart("0.1,0.2,0.3")
        assert result == [0.1, 0.2, 0.3]

        result = _parse_value_smart("a,b,c")
        assert result == ["a", "b", "c"]


class TestMapToNestedKey:
    """Test mapping flat keys to nested config structure."""

    def test_cv_keys(self):
        """Test CV parameter mapping."""
        assert _map_to_nested_key("folds") == "cv.folds"
        assert _map_to_nested_key("repeats") == "cv.repeats"
        assert _map_to_nested_key("scoring") == "cv.scoring"

    def test_feature_keys(self):
        """Test feature parameter mapping."""
        assert _map_to_nested_key("feature_select") == "features.feature_select"
        assert _map_to_nested_key("screen_top_n") == "features.screen_top_n"
        assert _map_to_nested_key("stability_thresh") == "features.stability_thresh"

    def test_threshold_keys(self):
        """Test threshold parameter mapping."""
        assert _map_to_nested_key("threshold_objective") == "thresholds.objective"
        assert _map_to_nested_key("threshold_source") == "thresholds.threshold_source"

    def test_panel_keys(self):
        """Test panel parameter mapping."""
        assert _map_to_nested_key("build_panels") == "panels.build_panels"
        assert _map_to_nested_key("panel_sizes") == "panels.panel_sizes"

    def test_evaluation_keys(self):
        """Test evaluation parameter mapping."""
        assert _map_to_nested_key("n_boot") == "evaluation.n_boot"
        assert _map_to_nested_key("test_ci_bootstrap") == "evaluation.test_ci_bootstrap"

    def test_top_level_keys(self):
        """Test top-level keys remain unchanged."""
        assert _map_to_nested_key("model") == "model"
        assert _map_to_nested_key("scenario") == "scenario"
        assert _map_to_nested_key("infile") == "infile"


class TestMigrateLegacyArgs:
    """Test full migration pipeline."""

    def test_migrate_splits_args(self):
        """Test migrating splits arguments to YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "splits_config.yaml"

            legacy_args = [
                "--n-splits",
                "10",
                "--val-size",
                "0.25",
                "--test-size",
                "0.25",
                "--prevalent-train-only",
                "--prevalent-train-frac",
                "0.5",
                "--train-control-per-case",
                "5",
            ]

            result_path = migrate_legacy_args_to_yaml(
                legacy_args=legacy_args,
                command="save-splits",
                output_file=output_file,
                verbose=0,
            )

            # Check file was created
            assert result_path.exists()

            # Load and verify YAML
            with open(result_path) as f:
                config = yaml.safe_load(f)

            assert config["n_splits"] == 10
            assert config["val_size"] == 0.25
            assert config["test_size"] == 0.25
            assert config["prevalent_train_only"] is True
            assert config["prevalent_train_frac"] == 0.5
            assert config["train_control_per_case"] == 5.0

    def test_migrate_training_args(self):
        """Test migrating training arguments to YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "training_config.yaml"

            legacy_args = [
                "--infile",
                "/path/to/data.csv",  # Required field
                "--model",
                "RF",
                "--folds",
                "10",
                "--scoring",
                "neg_brier_score",
                "--screen-top-n",
                "1000",
                "--threshold-objective",
                "youden",
                "--n-boot",
                "500",
            ]

            result_path = migrate_legacy_args_to_yaml(
                legacy_args=legacy_args,
                command="train",
                output_file=output_file,
                verbose=0,
            )

            # Check file was created
            assert result_path.exists()

            # Load and verify YAML
            with open(result_path) as f:
                config = yaml.safe_load(f)

            assert str(config["infile"]) == "/path/to/data.csv"
            assert config["model"] == "RF"
            assert config["cv"]["folds"] == 10
            assert config["cv"]["scoring"] == "neg_brier_score"
            assert config["features"]["screen_top_n"] == 1000
            assert config["thresholds"]["objective"] == "youden"
            assert config["evaluation"]["n_boot"] == 500


class TestValidateConfigFile:
    """Test config validation."""

    def test_validate_valid_splits_config(self):
        """Test validating a valid splits config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "splits_config.yaml"

            # Create valid config
            config_dict = {
                "mode": "development",
                "scenarios": ["IncidentOnly"],
                "n_splits": 10,
                "val_size": 0.25,
                "test_size": 0.25,
            }

            with open(config_file, "w") as f:
                yaml.dump(config_dict, f)

            is_valid, errors, warnings = validate_config_file(
                config_file=config_file,
                command="save-splits",
                strict=False,
                verbose=0,
            )

            assert is_valid is True
            assert len(errors) == 0

    def test_validate_invalid_splits_config(self):
        """Test validating an invalid splits config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "splits_config.yaml"

            # Create invalid config (val + test >= 1.0)
            config_dict = {
                "mode": "development",
                "val_size": 0.6,
                "test_size": 0.6,
            }

            with open(config_file, "w") as f:
                yaml.dump(config_dict, f)

            is_valid, errors, warnings = validate_config_file(
                config_file=config_file,
                command="save-splits",
                strict=False,
                verbose=0,
            )

            assert is_valid is False
            assert len(errors) > 0


class TestDiffConfigs:
    """Test config diff functionality."""

    def test_diff_identical_configs(self):
        """Test diffing identical configs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config1 = Path(tmpdir) / "config1.yaml"
            config2 = Path(tmpdir) / "config2.yaml"

            config_dict = {
                "n_splits": 10,
                "val_size": 0.25,
                "test_size": 0.25,
            }

            with open(config1, "w") as f:
                yaml.dump(config_dict, f)

            with open(config2, "w") as f:
                yaml.dump(config_dict, f)

            diff_result = diff_configs(
                config_file1=config1,
                config_file2=config2,
                verbose=0,
            )

            assert len(diff_result["only_in_first"]) == 0
            assert len(diff_result["only_in_second"]) == 0
            assert len(diff_result["different_values"]) == 0

    def test_diff_different_configs(self):
        """Test diffing different configs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config1 = Path(tmpdir) / "config1.yaml"
            config2 = Path(tmpdir) / "config2.yaml"

            config_dict1 = {
                "n_splits": 10,
                "val_size": 0.25,
                "test_size": 0.25,
            }

            config_dict2 = {
                "n_splits": 20,
                "val_size": 0.25,
                "test_size": 0.30,
                "new_param": "value",
            }

            with open(config1, "w") as f:
                yaml.dump(config_dict1, f)

            with open(config2, "w") as f:
                yaml.dump(config_dict2, f)

            diff_result = diff_configs(
                config_file1=config1,
                config_file2=config2,
                verbose=0,
            )

            # Different values
            assert "n_splits" in diff_result["different_values"]
            assert "test_size" in diff_result["different_values"]

            # Only in second
            assert "new_param" in diff_result["only_in_second"]

    def test_diff_nested_configs(self):
        """Test diffing nested config structures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config1 = Path(tmpdir) / "config1.yaml"
            config2 = Path(tmpdir) / "config2.yaml"

            config_dict1 = {
                "cv": {
                    "folds": 5,
                    "repeats": 10,
                },
                "features": {
                    "screen_top_n": 1000,
                },
            }

            config_dict2 = {
                "cv": {
                    "folds": 10,
                    "repeats": 10,
                },
                "features": {
                    "screen_top_n": 2000,
                },
            }

            with open(config1, "w") as f:
                yaml.dump(config_dict1, f)

            with open(config2, "w") as f:
                yaml.dump(config_dict2, f)

            diff_result = diff_configs(
                config_file1=config1,
                config_file2=config2,
                verbose=0,
            )

            # Check nested differences
            assert "cv.folds" in diff_result["different_values"]
            assert "features.screen_top_n" in diff_result["different_values"]
            assert "cv.repeats" in diff_result["same"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
