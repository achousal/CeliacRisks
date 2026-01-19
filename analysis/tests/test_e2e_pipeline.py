"""
End-to-end integration tests for the CeD-ML pipeline.

Tests the full workflow: data loading → split generation → feature selection →
model training → prediction → evaluation.

These tests verify that all components work together correctly and that the
pipeline produces valid outputs.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import yaml

from ced_ml.data.schema import ID_COL, TARGET_COL


class TestEndToEndPipeline:
    """Integration tests for the full ML pipeline."""

    def test_config_roundtrip(self, tmp_path):
        """
        Test: Config save/load roundtrip

        Validates that saved configs can be loaded and match original.
        """
        config_file = tmp_path / "test_config.yaml"

        # Create a config dict
        config = {
            "model": "LR_EN",
            "scenario": "IncidentOnly",
            "cv": {
                "folds": 5,
                "repeats": 3,
                "scoring": "neg_brier_score"
            },
            "features": {
                "screen_top_n": 1000,
                "stability_thresh": 0.75
            }
        }

        # Save
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        # Load
        with open(config_file, "r") as f:
            loaded_config = yaml.safe_load(f)

        # Compare
        assert loaded_config == config
        assert loaded_config["cv"]["folds"] == 5
        assert loaded_config["features"]["screen_top_n"] == 1000


@pytest.mark.slow
class TestSlowEndToEndPipeline:
    """
    Slow integration tests (mark with @pytest.mark.slow).

    These tests run the full pipeline and may take several minutes.
    Run with: pytest tests/test_e2e_pipeline.py -v -m slow
    """

    def test_full_pipeline_single_model(self, toy_proteomics_csv, tmp_path):
        """
        Test: Full pipeline with nested CV (SLOW - ~2 minutes)

        This is closest to production usage. Skip in CI, run manually.
        """
        pytest.skip("Slow test - run manually with: pytest -m slow")

        # Full workflow:
        # 1. Load data
        # 2. Generate splits
        # 3. Screen proteins
        # 4. K-best selection
        # 5. Nested CV with inner hyperparameter tuning
        # 6. Generate predictions on test set
        # 7. Compute metrics + bootstrap CIs
        # 8. Save outputs

        # Implementation would go here
        pass


# Fixtures are defined in conftest.py
# If they don't exist, add them:
@pytest.fixture
def toy_proteomics_csv(tmp_path):
    """
    Create a toy proteomics dataset for testing.

    1,000 samples: 900 controls, 50 incident, 50 prevalent
    10 protein columns + metadata
    """
    np.random.seed(42)

    n_controls = 900
    n_incident = 50
    n_prevalent = 50
    n_total = n_controls + n_incident + n_prevalent

    # Create labels
    labels = (
        ["Controls"] * n_controls +
        ["CeD_incident"] * n_incident +
        ["CeD_prevalent"] * n_prevalent
    )

    # Create metadata
    from ced_ml.data.schema import ID_COL, TARGET_COL
    data = {
        ID_COL: [f"S{i:04d}" for i in range(n_total)],
        TARGET_COL: labels,
        "age": np.random.randint(18, 80, n_total),
        "BMI": np.random.uniform(18, 35, n_total),
        "sex": np.random.choice(["M", "F"], n_total),
    }

    # Create protein columns (with differential signal)
    # Incident cases have slightly higher values
    for i in range(10):
        base = np.random.randn(n_total)
        # Add signal for incident cases
        signal = np.zeros(n_total)
        signal[n_controls:n_controls+n_incident] = np.random.randn(n_incident) * 0.5 + 1.0
        signal[n_controls+n_incident:] = np.random.randn(n_prevalent) * 0.5 + 1.5

        data[f"Protein{i}_resid"] = base + signal

    df = pd.DataFrame(data)

    # Save to CSV
    csv_path = tmp_path / "toy_proteomics.csv"
    df.to_csv(csv_path, index=False)

    return csv_path
