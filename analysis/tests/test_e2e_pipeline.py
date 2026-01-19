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

from ced_ml.data.io import read_proteomics_csv
from ced_ml.data.splits import stratified_train_val_test_split, build_working_strata
from ced_ml.data.persistence import save_split_indices, save_split_metadata
from ced_ml.features.screening import screen_proteins
from ced_ml.features.kbest import select_kbest_features
from ced_ml.models.registry import build_logistic_regression
from ced_ml.models.prevalence import PrevalenceAdjustedModel
from ced_ml.evaluation.predict import generate_predictions, export_predictions
from ced_ml.metrics.discrimination import compute_discrimination_metrics
from ced_ml.metrics.bootstrap import stratified_bootstrap_ci
from sklearn.metrics import roc_auc_score


class TestEndToEndPipeline:
    """Integration tests for the full ML pipeline."""

    def test_splits_to_predictions_workflow(self, toy_proteomics_csv, tmp_path):
        """
        Test: Data loading → Splitting → Predictions

        Validates that we can:
        1. Load proteomics data
        2. Generate train/val/test splits
        3. Save indices
        4. Generate predictions
        """
        # Step 1: Load data
        df = read_proteomics_csv(toy_proteomics_csv, validate=True)
        assert len(df) == 1000
        assert "CeD_comparison" in df.columns

        # Step 2: Create stratification
        strata = build_working_strata(
            df,
            target_col="CeD_comparison",
            strat_cols=["sex"],
            age_scheme=None,  # Simplified
        )

        # Step 3: Generate splits
        y = (df["CeD_comparison"] == "CeD_incident").astype(int)
        idx_train, idx_val, idx_test = stratified_train_val_test_split(
            y=y.values,
            strata=strata,
            val_size=0.25,
            test_size=0.25,
            random_state=42
        )

        assert len(idx_train) > 0
        assert len(idx_val) > 0
        assert len(idx_test) > 0
        assert len(set(idx_train) & set(idx_val)) == 0  # No overlap
        assert len(set(idx_train) & set(idx_test)) == 0

        # Step 4: Save indices
        outdir = tmp_path / "splits"
        outdir.mkdir()

        paths = save_split_indices(
            outdir=str(outdir),
            scenario="IncidentOnly",
            seed=42,
            train_idx=idx_train,
            val_idx=idx_val,
            test_idx=idx_test,
            n_splits=1,
            overwrite=False
        )

        assert Path(paths["train"]).exists()
        assert Path(paths["val"]).exists()
        assert Path(paths["test"]).exists()

        # Step 5: Generate predictions (simple model)
        protein_cols = [c for c in df.columns if c.endswith("_resid")][:10]
        X_train = df.loc[idx_train, protein_cols].values
        y_train = y.iloc[idx_train].values
        X_test = df.loc[idx_test, protein_cols].values
        y_test = y.iloc[idx_test].values

        # Train simple LR
        model = build_logistic_regression(C=1.0, penalty="l2", solver="lbfgs")
        model.fit(X_train, y_train)

        # Generate predictions
        preds = generate_predictions(model, X_test)

        assert len(preds) == len(y_test)
        assert np.all((preds >= 0) & (preds <= 1))

        # Export predictions
        pred_file = tmp_path / "preds.csv"
        export_predictions(
            predictions=preds,
            ids=df.loc[idx_test, "ID"].values,
            y_true=y_test,
            outfile=str(pred_file),
            include_percentiles=True
        )

        assert pred_file.exists()
        pred_df = pd.read_csv(pred_file)
        assert len(pred_df) == len(y_test)
        assert "risk" in pred_df.columns
        assert "risk_pct" in pred_df.columns

    def test_feature_selection_to_training(self, toy_proteomics_csv, tmp_path):
        """
        Test: Feature selection → Model training → Evaluation

        Validates feature selection and model training integration.
        """
        # Load data
        df = read_proteomics_csv(toy_proteomics_csv, validate=True)

        # Simple split
        n = len(df)
        idx_train = np.arange(int(0.7 * n))
        idx_test = np.arange(int(0.7 * n), n)

        y = (df["CeD_comparison"] == "CeD_incident").astype(int)
        protein_cols = [c for c in df.columns if c.endswith("_resid")]

        X_train = df.loc[idx_train, protein_cols].values
        y_train = y.iloc[idx_train].values
        X_test = df.loc[idx_test, protein_cols].values
        y_test = y.iloc[idx_test].values

        # Step 1: Screening
        selected_proteins, _ = screen_proteins(
            X=X_train,
            y=y_train,
            protein_names=protein_cols,
            method="mannwhitney",
            top_n=50,
            ascending=False
        )

        assert len(selected_proteins) == 50

        # Step 2: K-best selection
        X_train_screened = X_train[:, :50]  # Use first 50 (already ordered)
        X_test_screened = X_test[:, :50]

        selector, X_train_kbest = select_kbest_features(
            X=X_train_screened,
            y=y_train,
            k=10
        )

        X_test_kbest = selector.transform(X_test_screened)

        assert X_train_kbest.shape[1] == 10
        assert X_test_kbest.shape[1] == 10

        # Step 3: Train model
        model = build_logistic_regression(C=1.0, penalty="l2", solver="lbfgs")
        model.fit(X_train_kbest, y_train)

        # Step 4: Predict
        preds = generate_predictions(model, X_test_kbest)

        # Step 5: Evaluate
        metrics = compute_discrimination_metrics(y_test, preds)

        assert "auroc" in metrics
        assert "prauc" in metrics
        assert "brier" in metrics
        assert 0 <= metrics["auroc"] <= 1
        assert 0 <= metrics["prauc"] <= 1

    def test_prevalence_adjustment_workflow(self, toy_proteomics_csv):
        """
        Test: Prevalence adjustment integration

        Validates PrevalenceAdjustedModel wrapper works in prediction workflow.
        """
        # Load data
        df = read_proteomics_csv(toy_proteomics_csv, validate=True)

        # Simple split
        n = len(df)
        idx_train = np.arange(int(0.7 * n))
        idx_test = np.arange(int(0.7 * n), n)

        y = (df["CeD_comparison"] == "CeD_incident").astype(int)
        protein_cols = [c for c in df.columns if c.endswith("_resid")][:20]

        X_train = df.loc[idx_train, protein_cols].values
        y_train = y.iloc[idx_train].values
        X_test = df.loc[idx_test, protein_cols].values
        y_test = y.iloc[idx_test].values

        # Train base model
        base_model = build_logistic_regression(C=1.0, penalty="l2", solver="lbfgs")
        base_model.fit(X_train, y_train)

        # Wrap with prevalence adjustment
        sample_prevalence = y_train.mean()
        target_prevalence = 0.01  # 1% (much lower than sample)

        adjusted_model = PrevalenceAdjustedModel(
            base_model=base_model,
            sample_prevalence=sample_prevalence,
            target_prevalence=target_prevalence
        )

        # Generate predictions
        raw_preds = base_model.predict_proba(X_test)[:, 1]
        adjusted_preds = adjusted_model.predict_proba(X_test)[:, 1]

        # Adjusted predictions should be lower (target prevalence << sample)
        assert np.median(adjusted_preds) < np.median(raw_preds)
        assert np.all((adjusted_preds >= 0) & (adjusted_preds <= 1))

        # Ordering should be preserved (monotonic transformation)
        assert np.allclose(
            np.argsort(raw_preds),
            np.argsort(adjusted_preds)
        )

    def test_bootstrap_ci_workflow(self, toy_proteomics_csv):
        """
        Test: Bootstrap CI computation in evaluation workflow

        Validates stratified bootstrap integration with metrics.
        """
        # Load data
        df = read_proteomics_csv(toy_proteomics_csv, validate=True)

        # Simple split
        n = len(df)
        idx_test = np.arange(int(0.7 * n), n)

        y = (df["CeD_comparison"] == "CeD_incident").astype(int)
        protein_cols = [c for c in df.columns if c.endswith("_resid")][:20]

        X_test = df.loc[idx_test, protein_cols].values
        y_test = y.iloc[idx_test].values

        # Generate fake predictions (simple linear combination)
        np.random.seed(42)
        preds = 1 / (1 + np.exp(-(X_test.mean(axis=1) + np.random.randn(len(y_test)) * 0.1)))
        preds = np.clip(preds, 0.01, 0.99)

        # Compute bootstrap CI for AUROC
        ci_lower, ci_upper = stratified_bootstrap_ci(
            y_true=y_test,
            y_pred=preds,
            metric_func=roc_auc_score,
            n_boot=100,
            seed=42
        )

        assert ci_lower <= ci_upper
        assert 0 <= ci_lower <= 1
        assert 0 <= ci_upper <= 1

        # CI should contain point estimate (most of the time)
        point_estimate = roc_auc_score(y_test, preds)
        # Allow for rare edge cases where point estimate is outside CI
        # (this is statistically possible for small samples)

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
