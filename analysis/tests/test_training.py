"""
Tests for models.training module (nested CV orchestration).

Tests cover:
- Out-of-fold prediction generation
- Nested CV structure (outer + inner folds)
- Hyperparameter search integration
- Feature selection tracking
- Protein extraction from fitted models
- Calibration wrapper
- Edge cases (n_splits < 2, no hyperparams to tune)
"""

import json

import numpy as np
import pandas as pd
import pytest
from ced_ml.models.training import (
    _compute_xgb_scale_pos_weight,
    _extract_from_kbest_transformed,
    _extract_from_model_coefficients,
    _get_search_n_jobs,
    _maybe_apply_calibration,
    oof_predictions_with_nested_cv,
)
from conftest import make_mock_config
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def toy_data():
    """Small dataset for quick training tests."""
    rng = np.random.default_rng(42)
    n_samples = 100
    n_proteins = 20

    X = pd.DataFrame(
        rng.standard_normal((n_samples, n_proteins)),
        columns=[f"prot_{i}_resid" for i in range(n_proteins)],
    )
    # Add demographics
    X["age"] = rng.uniform(20, 80, n_samples)
    X["sex"] = rng.choice(["M", "F"], n_samples)

    # Imbalanced labels (10% positive)
    y = rng.binomial(1, 0.1, n_samples)

    protein_cols = [c for c in X.columns if c.endswith("_resid")]

    return X, y, protein_cols


@pytest.fixture
def simple_pipeline():
    """Minimal pipeline for testing."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), [f"prot_{i}_resid" for i in range(20)]),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(penalty="elasticnet", solver="saga", random_state=42, max_iter=100)

    return Pipeline([("pre", preprocessor), ("clf", clf)])


@pytest.fixture
def minimal_config():
    """Minimal config for testing."""
    return make_mock_config()


# ==================== OOF Predictions Tests ====================


def test_oof_predictions_basic(toy_data, simple_pipeline, minimal_config):
    """Test basic OOF prediction generation."""
    X, y, protein_cols = toy_data

    preds, elapsed, best_params_df, selected_proteins_df, _ = oof_predictions_with_nested_cv(
        simple_pipeline,
        "LR_EN",
        X,
        y,
        protein_cols,
        minimal_config,
        random_state=42,
    )

    # Check shape
    assert preds.shape == (minimal_config.cv.repeats, len(y))

    # Check no missing predictions
    assert not np.isnan(preds).any()

    # Check predictions in [0, 1]
    assert np.all((preds >= 0) & (preds <= 1))

    # Check elapsed time is positive
    assert elapsed > 0

    # Check metadata DataFrames
    assert len(best_params_df) == minimal_config.cv.folds * minimal_config.cv.repeats
    assert "model" in best_params_df.columns
    assert "repeat" in best_params_df.columns
    assert "outer_split" in best_params_df.columns


def test_oof_predictions_folds_less_than_2_raises(toy_data, simple_pipeline, minimal_config):
    """Test that cv.folds < 2 raises ValueError (not just a warning)."""
    X, y, protein_cols = toy_data
    minimal_config.cv.folds = 1

    with pytest.raises(ValueError, match="cv.folds must be >= 2"):
        oof_predictions_with_nested_cv(
            simple_pipeline,
            "LR_EN",
            X,
            y,
            protein_cols,
            minimal_config,
            random_state=42,
        )


def test_oof_predictions_invalid_repeats(toy_data, simple_pipeline, minimal_config):
    """Test that repeats < 1 raises ValueError."""
    X, y, protein_cols = toy_data
    minimal_config.cv.repeats = 0

    with pytest.raises(ValueError, match="cv.repeats must be >= 1"):
        oof_predictions_with_nested_cv(
            simple_pipeline,
            "LR_EN",
            X,
            y,
            protein_cols,
            minimal_config,
            random_state=42,
        )


def test_oof_predictions_no_nan_after_training(toy_data, simple_pipeline, minimal_config):
    """Test that OOF predictions contain no NaN values after training completes.

    This is a critical validation to ensure that the clipping and handling in
    training.py:189 properly prevents NaN propagation to downstream metrics.
    """
    X, y, protein_cols = toy_data

    preds, elapsed, best_params_df, selected_proteins_df, _ = oof_predictions_with_nested_cv(
        simple_pipeline,
        "LR_EN",
        X,
        y,
        protein_cols,
        minimal_config,
        random_state=42,
    )

    # Validate no NaN in predictions
    assert np.isfinite(preds).all(), (
        f"OOF predictions contain {np.sum(~np.isfinite(preds))} non-finite values. "
        "All predictions must be finite for metric computation."
    )

    # Validate predictions are in valid probability range
    assert np.all((preds >= 0) & (preds <= 1)), (
        f"OOF predictions outside [0,1] range. " f"Min: {preds.min():.6f}, Max: {preds.max():.6f}"
    )


# ==================== XGBoost Scale Pos Weight Tests ====================


def test_compute_xgb_scale_pos_weight_auto(minimal_config):
    """Test automatic computation of scale_pos_weight."""
    y_train = np.array([0, 0, 0, 0, 1])  # 4 neg, 1 pos
    spw = _compute_xgb_scale_pos_weight(y_train, minimal_config)
    assert spw == 4.0


def test_compute_xgb_scale_pos_weight_manual(minimal_config):
    """Test manual override of scale_pos_weight via single-element grid."""
    from types import SimpleNamespace

    # Single-element grid forces that value to be used
    minimal_config.xgboost = SimpleNamespace(scale_pos_weight_grid=[10.0])
    y_train = np.array([0, 0, 0, 0, 1])
    spw = _compute_xgb_scale_pos_weight(y_train, minimal_config)
    assert spw == 10.0


def test_compute_xgb_scale_pos_weight_no_positives(minimal_config):
    """Test handling of no positive samples."""
    y_train = np.array([0, 0, 0, 0, 0])
    spw = _compute_xgb_scale_pos_weight(y_train, minimal_config)
    assert spw == 1.0


# ==================== Search N Jobs Tests ====================


def test_get_search_n_jobs_auto_lr(minimal_config):
    """Test auto n_jobs for LR (parallelize search)."""
    n_jobs = _get_search_n_jobs("LR_EN", minimal_config)
    assert n_jobs == 2  # cpus=2


def test_get_search_n_jobs_auto_rf(minimal_config):
    """Test auto n_jobs for RF (sequential search)."""
    n_jobs = _get_search_n_jobs("RF", minimal_config)
    assert n_jobs == 1  # Keep search single-threaded


def test_get_search_n_jobs_manual(minimal_config):
    """Test manual override of n_jobs."""
    minimal_config.compute.tune_n_jobs = 4
    n_jobs = _get_search_n_jobs("LR_EN", minimal_config)
    assert n_jobs == 2  # Capped at cpus=2


# ==================== Calibration Tests ====================


def test_maybe_apply_calibration_disabled(simple_pipeline, minimal_config, toy_data):
    """Test that calibration is skipped when disabled."""
    X, y, _ = toy_data
    minimal_config.calibration.enabled = False

    simple_pipeline.fit(X, y)
    calibrated = _maybe_apply_calibration(
        simple_pipeline, "LR_EN", minimal_config, X, y, random_state=42
    )

    assert calibrated is simple_pipeline
    assert not isinstance(calibrated, CalibratedClassifierCV)


def test_maybe_apply_calibration_enabled(simple_pipeline, minimal_config, toy_data):
    """Test that calibration is applied when enabled with per_fold strategy."""
    X, y, _ = toy_data
    minimal_config.calibration.enabled = True
    minimal_config.calibration.strategy = "per_fold"  # per_fold wraps in Calibrated

    simple_pipeline.fit(X, y)
    calibrated = _maybe_apply_calibration(
        simple_pipeline, "LR_EN", minimal_config, X, y, random_state=42
    )

    assert isinstance(calibrated, CalibratedClassifierCV)


def test_maybe_apply_calibration_skip_svm(simple_pipeline, minimal_config, toy_data):
    """Test that LinSVM_cal is not double-calibrated."""
    X, y, _ = toy_data
    minimal_config.calibration.enabled = True

    simple_pipeline.fit(X, y)
    calibrated = _maybe_apply_calibration(
        simple_pipeline, "LinSVM_cal", minimal_config, X, y, random_state=42
    )

    assert not isinstance(calibrated, CalibratedClassifierCV)


# ==================== Protein Extraction Tests ====================


def test_extract_from_kbest_transformed_no_sel_step():
    """Test extraction when no 'sel' step exists."""
    pipeline = Pipeline([("pre", StandardScaler()), ("clf", LogisticRegression())])
    proteins = _extract_from_kbest_transformed(pipeline, ["prot_0_resid"])
    assert proteins == set()


def test_extract_from_kbest_transformed_with_sel(toy_data):
    """Test extraction from SelectKBest in transformed space."""
    from sklearn.feature_selection import SelectKBest, f_classif

    X, y, protein_cols = toy_data

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), protein_cols),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        [
            ("pre", preprocessor),
            ("sel", SelectKBest(score_func=f_classif, k=5)),
            ("clf", LogisticRegression(random_state=42)),
        ]
    )

    pipeline.fit(X, y)

    proteins = _extract_from_kbest_transformed(pipeline, protein_cols)

    # Should extract 5 proteins
    assert len(proteins) == 5
    assert all(p in protein_cols for p in proteins)


def test_extract_from_kbest_transformed_plain_names():
    """Test extraction with plain feature names (no prefix/suffix).

    This tests the fix for verbose_feature_names_out=False, where feature
    names don't have the 'num__' prefix or '_resid' suffix.
    """
    from sklearn.feature_selection import SelectKBest, f_classif

    rng = np.random.default_rng(42)
    # Plain protein names (no prefix/suffix) - this is what verbose_feature_names_out=False produces
    protein_cols = ["PROT_A", "PROT_B", "PROT_C", "PROT_D", "PROT_E"]
    X = pd.DataFrame(rng.standard_normal((50, 5)), columns=protein_cols)
    y = (X["PROT_A"] + X["PROT_B"] > 0).astype(int)  # Make first two predictive

    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), protein_cols)],
        verbose_feature_names_out=False,  # Key setting that produces plain names
    )

    pipeline = Pipeline(
        [
            ("pre", preprocessor),
            ("sel", SelectKBest(score_func=f_classif, k=3)),
            ("clf", LogisticRegression(random_state=42)),
        ]
    )

    pipeline.fit(X, y)

    proteins = _extract_from_kbest_transformed(pipeline, protein_cols)

    # Should extract 3 proteins
    assert len(proteins) == 3
    assert all(p in protein_cols for p in proteins)
    # The predictive proteins should be captured
    assert "PROT_A" in proteins or "PROT_B" in proteins


def test_extract_from_model_coefficients(toy_data, minimal_config):
    """Test extraction from linear model coefficients."""
    X, y, protein_cols = toy_data

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), protein_cols),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        [
            ("pre", preprocessor),
            ("clf", LogisticRegression(random_state=42, C=0.01, max_iter=100)),
        ]
    )

    pipeline.fit(X, y)

    proteins = _extract_from_model_coefficients(pipeline, "LR_EN", protein_cols, minimal_config)

    # Should extract some proteins with non-zero coefficients
    assert isinstance(proteins, set)
    assert all(p in protein_cols for p in proteins)


def test_extract_from_model_coefficients_high_threshold(toy_data, minimal_config):
    """Test that high coef threshold returns fewer proteins."""
    X, y, protein_cols = toy_data

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), protein_cols),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        [
            ("pre", preprocessor),
            ("clf", LogisticRegression(random_state=42, C=0.01, max_iter=100)),
        ]
    )

    pipeline.fit(X, y)

    # Low threshold
    minimal_config.features.coef_threshold = 1e-12
    proteins_low = _extract_from_model_coefficients(pipeline, "LR_EN", protein_cols, minimal_config)

    # High threshold
    minimal_config.features.coef_threshold = 10.0
    proteins_high = _extract_from_model_coefficients(
        pipeline, "LR_EN", protein_cols, minimal_config
    )

    # High threshold should return fewer proteins
    assert len(proteins_high) <= len(proteins_low)


# ==================== Integration Tests ====================


def test_oof_predictions_with_kbest(toy_data, minimal_config):
    """Test OOF predictions with K-best feature selection."""
    from sklearn.feature_selection import SelectKBest, f_classif

    X, y, protein_cols = toy_data

    minimal_config.features.feature_select = "kbest"
    minimal_config.features.k_grid = [5, 10]
    minimal_config.features.kbest_scope = "transformed"

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), protein_cols),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        [
            ("pre", preprocessor),
            ("sel", SelectKBest(score_func=f_classif, k=10)),
            (
                "clf",
                LogisticRegression(
                    penalty="elasticnet", solver="saga", random_state=42, max_iter=100
                ),
            ),
        ]
    )

    preds, elapsed, best_params_df, selected_proteins_df, _ = oof_predictions_with_nested_cv(
        pipeline, "LR_EN", X, y, protein_cols, minimal_config, random_state=42
    )

    # Check that k was tuned
    assert len(best_params_df) > 0
    best_params_sample = json.loads(best_params_df.iloc[0]["best_params"])
    assert "sel__k" in best_params_sample


def test_oof_predictions_tracks_selected_proteins(toy_data, minimal_config):
    """Test that selected proteins are tracked across CV folds."""
    X, y, protein_cols = toy_data

    minimal_config.features.feature_select = "l1_stability"

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), protein_cols),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        [
            ("pre", preprocessor),
            (
                "clf",
                LogisticRegression(
                    penalty="elasticnet", solver="saga", C=0.1, random_state=42, max_iter=100
                ),
            ),
        ]
    )

    preds, elapsed, best_params_df, selected_proteins_df, _ = oof_predictions_with_nested_cv(
        pipeline, "LR_EN", X, y, protein_cols, minimal_config, random_state=42
    )

    # Check that selected_proteins_df is returned (may be empty if no features selected)
    assert isinstance(selected_proteins_df, pd.DataFrame)

    # If proteins were selected, verify structure
    if len(selected_proteins_df) > 0:
        assert "n_selected_proteins" in selected_proteins_df.columns
        assert "selected_proteins" in selected_proteins_df.columns

        # Parse selected proteins
        sample_proteins = json.loads(selected_proteins_df.iloc[0]["selected_proteins"])
        assert isinstance(sample_proteins, list)
        assert all(p in protein_cols for p in sample_proteins)


# ==================== Edge Cases ====================


def test_oof_predictions_single_repeat(toy_data, simple_pipeline, minimal_config):
    """Test OOF with single repeat (edge case)."""
    X, y, protein_cols = toy_data
    minimal_config.cv.repeats = 1

    preds, elapsed, best_params_df, selected_proteins_df, _ = oof_predictions_with_nested_cv(
        simple_pipeline,
        "LR_EN",
        X,
        y,
        protein_cols,
        minimal_config,
        random_state=42,
    )

    assert preds.shape == (1, len(y))
    assert not np.isnan(preds).any()


def test_oof_predictions_no_inner_tuning(toy_data, simple_pipeline, minimal_config):
    """Test OOF with no inner CV (no hyperparameter tuning)."""
    X, y, protein_cols = toy_data
    minimal_config.cv.inner_folds = 1  # Disable tuning

    preds, elapsed, best_params_df, selected_proteins_df, _ = oof_predictions_with_nested_cv(
        simple_pipeline,
        "LR_EN",
        X,
        y,
        protein_cols,
        minimal_config,
        random_state=42,
    )

    # Should still work (no tuning, just CV)
    assert preds.shape == (minimal_config.cv.repeats, len(y))
    assert not np.isnan(preds).any()

    # Best params should be empty
    best_params_sample = json.loads(best_params_df.iloc[0]["best_params"])
    assert best_params_sample == {}


def test_fixed_panel_bypasses_feature_selection(tmp_path):
    """Test that fixed panel mode disables feature selection."""
    from ced_ml.cli.train import build_training_pipeline
    from ced_ml.models.registry import build_models
    from conftest import make_mock_config

    # Create minimal config
    config = make_mock_config()
    config.features.feature_selection_strategy = "hybrid_stability"
    config.features.screen_top_n = 100
    config.features.k_grid = [10, 20]

    # Create mock fixed panel CSV
    fixed_panel_path = tmp_path / "test_panel.csv"
    panel_proteins = [f"prot_{i}_resid" for i in range(5)]
    pd.DataFrame({"protein": panel_proteins}).to_csv(fixed_panel_path, index=False)

    # Simulate fixed panel override (as done in run_train)
    config.features.feature_selection_strategy = "none"
    config.features.screen_top_n = 0

    # Build pipeline
    classifier = build_models("LR_EN", config, random_state=42, n_jobs=1)
    pipeline = build_training_pipeline(
        config,
        classifier,
        protein_cols=panel_proteins,
        cat_cols=["sex"],
        meta_num_cols=["age"],
    )

    # Verify no feature selection steps
    step_names = [name for name, _ in pipeline.steps]
    assert "screen" not in step_names, "Screening should be disabled"
    assert "sel" not in step_names, "SelectKBest should be disabled"
    assert "pre" in step_names, "Preprocessing should still exist"
    assert "clf" in step_names, "Classifier should still exist"


def test_fixed_panel_protein_validation(tmp_path):
    """Test that fixed panel validates proteins exist in dataset."""

    # Create fixed panel with non-existent proteins
    bad_panel_path = tmp_path / "bad_panel.csv"
    pd.DataFrame({"protein": ["FAKE_PROT_1", "FAKE_PROT_2"]}).to_csv(bad_panel_path, index=False)

    # Actual protein columns in dataset
    actual_proteins = [f"prot_{i}_resid" for i in range(10)]

    # This validation should happen in run_train before pipeline building
    # Simulating the validation logic
    fixed_panel_df = pd.read_csv(bad_panel_path)
    fixed_panel_proteins = fixed_panel_df["protein"].tolist()

    missing_proteins = set(fixed_panel_proteins) - set(actual_proteins)
    assert len(missing_proteins) == 2, "Should detect missing proteins"
    assert "FAKE_PROT_1" in missing_proteins
    assert "FAKE_PROT_2" in missing_proteins


def test_rfecv_kbest_prefilter_config_defaults():
    """Test that k-best pre-filter config parameters have correct defaults."""
    from ced_ml.config.schema import FeatureConfig

    config = FeatureConfig()

    # Verify defaults
    assert config.rfe_kbest_prefilter is True, "K-best pre-filter should be enabled by default"
    assert config.rfe_kbest_k == 100, "Default k should be 100"


def test_rfecv_kbest_prefilter_config_validation():
    """Test that k-best pre-filter config validates correctly."""
    from ced_ml.config.schema import FeatureConfig
    from pydantic import ValidationError

    # Valid config
    config = FeatureConfig(rfe_kbest_k=50)
    assert config.rfe_kbest_k == 50

    # Invalid k (too small)
    with pytest.raises(ValidationError):  # Pydantic validation error
        FeatureConfig(rfe_kbest_k=5)  # Less than ge=10 constraint
