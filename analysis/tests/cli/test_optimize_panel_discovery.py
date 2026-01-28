"""
Tests for optimize_panel auto-discovery by run_id.
"""

import json

import pytest
from ced_ml.cli.optimize_panel import discover_models_by_run_id


@pytest.fixture
def mock_results_structure(tmp_path):
    """
    Create a mock results directory structure:
        results/
            LR_EN/
                run_20260127_115115/
                    aggregated/
                        aggregation_metadata.json
                run_20260127_120000/
                    (no aggregated dir)
            RF/
                run_20260127_115115/
                    aggregated/
                        aggregation_metadata.json
            XGBoost/
                run_20260127_115115/
                    (no aggregated dir)
            ENSEMBLE/
                run_20260127_115115/
                    aggregated/
                        aggregation_metadata.json
    """
    results_root = tmp_path / "results"

    # LR_EN with two runs, only one has aggregated
    lr_en_run1 = results_root / "LR_EN" / "run_20260127_115115" / "aggregated"
    lr_en_run1.mkdir(parents=True)
    (lr_en_run1 / "aggregation_metadata.json").write_text(json.dumps({"split_seeds": [0, 1]}))

    lr_en_run2 = results_root / "LR_EN" / "run_20260127_120000"
    lr_en_run2.mkdir(parents=True)

    # RF with aggregated
    rf_run1 = results_root / "RF" / "run_20260127_115115" / "aggregated"
    rf_run1.mkdir(parents=True)
    (rf_run1 / "aggregation_metadata.json").write_text(json.dumps({"split_seeds": [0, 1]}))

    # XGBoost without aggregated
    xgb_run1 = results_root / "XGBoost" / "run_20260127_115115"
    xgb_run1.mkdir(parents=True)

    # ENSEMBLE with aggregated
    ens_run1 = results_root / "ENSEMBLE" / "run_20260127_115115" / "aggregated"
    ens_run1.mkdir(parents=True)
    (ens_run1 / "aggregation_metadata.json").write_text(json.dumps({"split_seeds": [0, 1]}))

    return results_root


def test_discover_all_models(mock_results_structure):
    """Test discovering all models with aggregated results for a run_id."""
    discovered = discover_models_by_run_id(
        run_id="20260127_115115",
        results_root=mock_results_structure,
    )

    # Should find LR_EN and RF, but exclude ENSEMBLE and XGBoost (no aggregated)
    assert len(discovered) == 2
    assert "LR_EN" in discovered
    assert "RF" in discovered
    assert "ENSEMBLE" not in discovered  # Excluded by default
    assert "XGBoost" not in discovered  # No aggregated dir

    # Verify paths are correct
    assert discovered["LR_EN"] == mock_results_structure / "LR_EN" / "run_20260127_115115"
    assert discovered["RF"] == mock_results_structure / "RF" / "run_20260127_115115"


def test_discover_with_model_filter(mock_results_structure):
    """Test discovering models with a filter."""
    discovered = discover_models_by_run_id(
        run_id="20260127_115115",
        results_root=mock_results_structure,
        model_filter="LR_EN",
    )

    assert len(discovered) == 1
    assert "LR_EN" in discovered
    assert "RF" not in discovered
    assert "ENSEMBLE" not in discovered


def test_discover_nonexistent_run_id(mock_results_structure):
    """Test discovering with a run_id that doesn't exist."""
    discovered = discover_models_by_run_id(
        run_id="99999999_999999",
        results_root=mock_results_structure,
    )

    assert len(discovered) == 0


def test_discover_run_id_without_aggregation(mock_results_structure):
    """Test discovering run_id where no models have aggregated results."""
    discovered = discover_models_by_run_id(
        run_id="20260127_120000",
        results_root=mock_results_structure,
    )

    # Only LR_EN has this run_id, but it has no aggregated dir
    assert len(discovered) == 0


def test_discover_invalid_results_root():
    """Test error handling for nonexistent results root."""
    with pytest.raises(FileNotFoundError, match="Results root not found"):
        discover_models_by_run_id(
            run_id="20260127_115115",
            results_root="/nonexistent/path",
        )


def test_discover_empty_results_root(tmp_path):
    """Test discovering in an empty results directory."""
    empty_results = tmp_path / "results"
    empty_results.mkdir()

    discovered = discover_models_by_run_id(
        run_id="20260127_115115",
        results_root=empty_results,
    )

    assert len(discovered) == 0


def test_discover_with_partial_structure(tmp_path):
    """Test discovering when some models have incomplete structure."""
    results_root = tmp_path / "results"

    # Model with run dir but no aggregated subdir
    (results_root / "Model1" / "run_20260127_115115").mkdir(parents=True)

    # Model with aggregated dir but empty
    (results_root / "Model2" / "run_20260127_115115" / "aggregated").mkdir(parents=True)

    discovered = discover_models_by_run_id(
        run_id="20260127_115115",
        results_root=results_root,
    )

    # Model2 should be discovered (aggregated dir exists)
    assert len(discovered) == 1
    assert "Model2" in discovered


def test_discover_case_sensitive_model_filter(mock_results_structure):
    """Test that model filter is case-sensitive."""
    discovered = discover_models_by_run_id(
        run_id="20260127_115115",
        results_root=mock_results_structure,
        model_filter="lr_en",  # lowercase
    )

    # Should not match "LR_EN"
    assert len(discovered) == 0
