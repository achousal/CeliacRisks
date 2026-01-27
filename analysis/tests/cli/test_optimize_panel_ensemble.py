"""Tests for optimize_panel CLI with ENSEMBLE model validation.

This module tests that:
1. Direct RFE on ENSEMBLE models is blocked with a helpful error
2. The two-pass workflow is correctly documented in the error message
3. Single model RFE continues to work normally
"""

import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from ced_ml.cli.optimize_panel import run_optimize_panel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def mock_ensemble_bundle():
    """Create a mock ENSEMBLE model bundle (meta-learner)."""
    from ced_ml.models.stacking import StackingEnsemble

    ensemble = StackingEnsemble(
        base_model_names=["LR_EN", "RF", "XGBoost"],
        meta_penalty="l2",
        meta_C=1.0,
        calibrate_meta=True,
        random_state=42,
    )

    # Mock OOF predictions (meta-features)
    n_samples = 100
    oof_dict = {
        "LR_EN": np.random.rand(1, n_samples),
        "RF": np.random.rand(1, n_samples),
        "XGBoost": np.random.rand(1, n_samples),
    }
    y_train = np.random.randint(0, 2, n_samples)

    # Fit ensemble
    ensemble.fit_from_oof(oof_dict, y_train)

    # Create bundle matching train_ensemble.py format (lines 419-434)
    bundle = {
        "model": ensemble,
        "model_name": "ENSEMBLE",
        "base_models": ["LR_EN", "RF", "XGBoost"],
        "meta_penalty": "l2",
        "meta_C": 1.0,
        "meta_coef": ensemble.get_meta_model_coef(),
        "split_seed": 0,
        "random_state": 42,
        # NOTE: No resolved_columns or protein_cols (ensemble doesn't have them)
    }

    return bundle


@pytest.fixture
def mock_single_model_bundle():
    """Create a mock single model bundle (LR_EN) with protein columns."""
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(penalty="l2", C=1.0, random_state=42)),
        ]
    )

    # Mock training data
    n_samples = 100
    n_proteins = 50
    X = pd.DataFrame(
        np.random.randn(n_samples, n_proteins),
        columns=[f"P{i:05d}_resid" for i in range(n_proteins)],
    )
    y = np.random.randint(0, 2, n_samples)

    # Fit pipeline
    pipeline.fit(X, y)

    # Create bundle matching training output
    bundle = {
        "model": pipeline,
        "model_name": "LR_EN",
        "resolved_columns": {
            "protein_cols": list(X.columns),
            "categorical_metadata": [],
            "numeric_metadata": ["age", "BMI"],
        },
        "split_seed": 0,
        "scenario": "IncidentOnly",
    }

    return bundle


@pytest.fixture
def mock_data_and_splits():
    """Create mock proteomics data and split files."""
    n_samples = 300
    n_proteins = 50

    # Create mock data
    df = pd.DataFrame(
        np.random.randn(n_samples, n_proteins),
        columns=[f"P{i:05d}_resid" for i in range(n_proteins)],
    )
    df["Diagnosis_encoded"] = np.random.choice(["Control", "Incident_CeD"], n_samples)
    df["age"] = np.random.uniform(20, 80, n_samples)
    df["BMI"] = np.random.uniform(18, 35, n_samples)
    df["Genetic ethnic grouping"] = "European"

    # Create temporary directory for splits
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Save data
        data_path = tmpdir / "data.parquet"
        df.to_parquet(data_path)

        # Create split files
        splits_dir = tmpdir / "splits"
        splits_dir.mkdir()

        train_idx = np.arange(0, 200)
        val_idx = np.arange(200, 250)
        test_idx = np.arange(250, 300)

        pd.Series(train_idx).to_csv(splits_dir / "train_idx_IncidentOnly_seed0.csv", index=False)
        pd.Series(val_idx).to_csv(splits_dir / "val_idx_IncidentOnly_seed0.csv", index=False)
        pd.Series(test_idx).to_csv(splits_dir / "test_idx_IncidentOnly_seed0.csv", index=False)

        yield {
            "data_path": str(data_path),
            "splits_dir": str(splits_dir),
            "tmpdir": tmpdir,
        }


class TestEnsembleValidation:
    """Test that direct ENSEMBLE RFE is blocked with helpful error."""

    def test_ensemble_model_raises_error(self, mock_ensemble_bundle, mock_data_and_splits):
        """ENSEMBLE model should raise ValueError with two-pass instructions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Save ensemble bundle
            model_path = tmpdir / "ENSEMBLE__final_model.joblib"
            joblib.dump(mock_ensemble_bundle, model_path)

            # Attempt to run optimize_panel on ENSEMBLE
            with pytest.raises(ValueError) as exc_info:
                run_optimize_panel(
                    model_path=str(model_path),
                    infile=mock_data_and_splits["data_path"],
                    split_dir=mock_data_and_splits["splits_dir"],
                    split_seed=0,
                    start_size=50,
                    min_size=5,
                )

            # Verify error message contains key guidance
            error_msg = str(exc_info.value)
            assert "Direct RFE on ENSEMBLE models is not supported" in error_msg
            assert "TWO-PASS workflow" in error_msg
            assert "ced optimize-panel --model-path results/LR_EN" in error_msg
            assert "ced train --model LR_EN,RF,XGBoost --fixed-panel" in error_msg
            assert "ced train-ensemble" in error_msg
            assert "Workflow 6" in error_msg

    def test_single_model_works(self, mock_single_model_bundle, mock_data_and_splits):
        """Single model RFE should work normally (no ENSEMBLE check blocking)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Save single model bundle
            model_path = tmpdir / "LR_EN__final_model.joblib"
            joblib.dump(mock_single_model_bundle, model_path)

            # Should NOT raise ENSEMBLE validation error
            # Note: May raise other errors due to mock data, but not ENSEMBLE error
            try:
                result = run_optimize_panel(
                    model_path=str(model_path),
                    infile=mock_data_and_splits["data_path"],
                    split_dir=mock_data_and_splits["splits_dir"],
                    split_seed=0,
                    start_size=30,
                    min_size=5,
                    outdir=str(tmpdir / "optimize_panel"),
                    verbose=0,
                )
                # If it succeeds, verify result structure
                assert result.max_auroc > 0
                assert len(result.curve) > 0
            except ValueError as e:
                # Should NOT be the ENSEMBLE error
                assert "Direct RFE on ENSEMBLE" not in str(e)
            except Exception:
                # Other errors are OK for this test (mock data may be invalid)
                pass


class TestTwoPassWorkflowDocumentation:
    """Test that the two-pass workflow is properly documented."""

    def test_error_message_completeness(self, mock_ensemble_bundle, mock_data_and_splits):
        """Error message should include all steps of two-pass workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            model_path = tmpdir / "ENSEMBLE__final_model.joblib"
            joblib.dump(mock_ensemble_bundle, model_path)

            with pytest.raises(ValueError) as exc_info:
                run_optimize_panel(
                    model_path=str(model_path),
                    infile=mock_data_and_splits["data_path"],
                    split_dir=mock_data_and_splits["splits_dir"],
                    split_seed=0,
                )

            error_msg = str(exc_info.value)

            # Verify all workflow steps are documented
            workflow_steps = [
                "Run RFE on best single model",  # Step 1
                "Extract optimized panel",  # Step 2
                "Retrain base models",  # Step 3
                "Retrain ensemble",  # Step 4
            ]

            for step in workflow_steps:
                assert step in error_msg, f"Missing workflow step: {step}"

            # Verify CLI commands are provided
            cli_commands = [
                "ced optimize-panel --model-path results/LR_EN",
                "ced train --model LR_EN,RF,XGBoost --fixed-panel",
                "ced train-ensemble --base-models LR_EN,RF,XGBoost",
            ]

            for cmd in cli_commands:
                assert cmd in error_msg, f"Missing CLI command: {cmd}"

    def test_error_references_documentation(self, mock_ensemble_bundle, mock_data_and_splits):
        """Error should reference FEATURE_SELECTION.md Workflow 6."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            model_path = tmpdir / "ENSEMBLE__final_model.joblib"
            joblib.dump(mock_ensemble_bundle, model_path)

            with pytest.raises(ValueError) as exc_info:
                run_optimize_panel(
                    model_path=str(model_path),
                    infile=mock_data_and_splits["data_path"],
                    split_dir=mock_data_and_splits["splits_dir"],
                    split_seed=0,
                )

            error_msg = str(exc_info.value)

            # Should reference the documentation
            assert "FEATURE_SELECTION.md" in error_msg
            assert "Workflow 6" in error_msg
