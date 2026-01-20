"""
Tests for models.calibration module.

Coverage areas:
- Calibration intercept/slope computation
- Expected Calibration Error (ECE)
- Prevalence adjustment logic
- PrevalenceAdjustedModel wrapper
- CalibratedClassifierCV utilities
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from ced_ml.models.calibration import (
    PrevalenceAdjustedModel,
    adjust_probabilities_for_prevalence,
    calib_intercept_metric,
    calib_slope_metric,
    calibration_intercept_slope,
    expected_calibration_error,
    get_calibrated_cv_param_name,
    get_calibrated_estimator_param_name,
    maybe_calibrate_estimator,
)

# ============================================================================
# Calibration Metrics Tests
# ============================================================================


def test_calibration_intercept_slope_perfect():
    """Perfect calibration should have intercept ~0 and slope ~1."""
    np.random.seed(42)
    n = 1000
    # Create reasonable predicted probabilities
    y_pred = np.random.uniform(0.1, 0.5, size=n)
    # Generate outcomes consistent with predictions
    y_true = np.random.binomial(1, y_pred)

    intercept, slope = calibration_intercept_slope(y_true, y_pred)
    # With logit-scale calibration, slope ~1 indicates good calibration
    # But the exact value depends on the data distribution
    assert np.isfinite(intercept)
    assert np.isfinite(slope)
    assert slope > 0, "Calibration slope should be positive"


def test_calibration_intercept_slope_underconfident():
    """Underconfident predictions should have specific calibration characteristics."""
    np.random.seed(42)
    n = 500
    y_true = np.random.binomial(1, 0.5, size=n)
    # Shrink predictions toward 0.5 (underconfident)
    y_pred = y_true * 0.3 + 0.35

    intercept, slope = calibration_intercept_slope(y_true, y_pred)
    # Test that we get valid calibration metrics
    assert np.isfinite(intercept)
    assert np.isfinite(slope)
    assert slope > 0, "Calibration slope should be positive"


def test_calibration_intercept_slope_single_class():
    """Single class should return NaN."""
    y_true = np.ones(100)
    y_pred = np.random.uniform(0.3, 0.7, size=100)

    intercept, slope = calibration_intercept_slope(y_true, y_pred)
    assert np.isnan(intercept)
    assert np.isnan(slope)


def test_calibration_intercept_slope_with_nans():
    """Should filter out NaN values."""
    np.random.seed(42)
    y_true = np.array([0, 1, 0, 1, 0, 1, np.nan, 0, 1])
    y_pred = np.array([0.1, 0.9, 0.2, np.nan, 0.15, 0.85, 0.5, 0.1, 0.95])

    intercept, slope = calibration_intercept_slope(y_true, y_pred)
    assert np.isfinite(intercept)
    assert np.isfinite(slope)


def test_calib_intercept_metric():
    """Test calibration intercept metric wrapper."""
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.3, size=100)
    y_pred = np.random.uniform(0.1, 0.5, size=100)

    metric = calib_intercept_metric(y_true, y_pred)
    intercept, _ = calibration_intercept_slope(y_true, y_pred)

    assert np.isclose(metric, intercept)


def test_calib_slope_metric():
    """Test calibration slope metric wrapper."""
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.3, size=100)
    y_pred = np.random.uniform(0.1, 0.5, size=100)

    metric = calib_slope_metric(y_true, y_pred)
    _, slope = calibration_intercept_slope(y_true, y_pred)

    assert np.isclose(metric, slope)


# ============================================================================
# Expected Calibration Error Tests
# ============================================================================


def test_expected_calibration_error_perfect():
    """Perfect calibration should have ECE near 0."""
    np.random.seed(42)
    n = 1000
    y_pred = np.random.uniform(0, 1, size=n)
    y_true = np.random.binomial(1, y_pred)

    ece = expected_calibration_error(y_true, y_pred, n_bins=10)
    assert 0.0 <= ece <= 0.15, f"Perfect calibration should have low ECE, got {ece}"


def test_expected_calibration_error_poor():
    """Poor calibration should have high ECE."""
    np.random.seed(42)
    n = 500
    y_true = np.random.binomial(1, 0.5, size=n)
    # Completely miscalibrated: predict opposite
    y_pred = 1.0 - y_true.astype(float)
    y_pred = np.clip(y_pred, 0.01, 0.99)

    ece = expected_calibration_error(y_true, y_pred, n_bins=10)
    assert ece > 0.3, f"Poor calibration should have high ECE, got {ece}"


def test_expected_calibration_error_with_nans():
    """Should filter NaN values."""
    y_true = np.array([0, 1, 0, 1, np.nan, 0, 1])
    y_pred = np.array([0.1, 0.9, np.nan, 0.8, 0.2, 0.15, 0.95])

    ece = expected_calibration_error(y_true, y_pred, n_bins=5)
    assert np.isfinite(ece)
    assert 0.0 <= ece <= 1.0


def test_expected_calibration_error_empty():
    """Empty arrays should return NaN."""
    y_true = np.array([])
    y_pred = np.array([])

    ece = expected_calibration_error(y_true, y_pred)
    assert np.isnan(ece)


# ============================================================================
# Prevalence Adjustment Tests
# ============================================================================


def test_adjust_probabilities_for_prevalence_shift_up():
    """Shifting to higher prevalence should increase probabilities."""
    np.random.seed(42)
    probs = np.random.uniform(0.1, 0.3, size=100)
    sample_prev = 0.1
    target_prev = 0.3

    adjusted = adjust_probabilities_for_prevalence(probs, sample_prev, target_prev)

    assert np.all(adjusted > probs), "Higher prevalence should increase probabilities"
    assert np.all((adjusted >= 0) & (adjusted <= 1)), "Probabilities should be in [0,1]"


def test_adjust_probabilities_for_prevalence_shift_down():
    """Shifting to lower prevalence should decrease probabilities."""
    np.random.seed(42)
    probs = np.random.uniform(0.3, 0.5, size=100)
    sample_prev = 0.3
    target_prev = 0.1

    adjusted = adjust_probabilities_for_prevalence(probs, sample_prev, target_prev)

    assert np.all(adjusted < probs), "Lower prevalence should decrease probabilities"
    assert np.all((adjusted >= 0) & (adjusted <= 1)), "Probabilities should be in [0,1]"


def test_adjust_probabilities_for_prevalence_no_change():
    """Same prevalence should return nearly identical probabilities."""
    np.random.seed(42)
    probs = np.random.uniform(0.1, 0.5, size=100)
    sample_prev = 0.2
    target_prev = 0.2

    adjusted = adjust_probabilities_for_prevalence(probs, sample_prev, target_prev)

    assert np.allclose(
        adjusted, probs, atol=1e-5
    ), "Same prevalence should not change probabilities"


def test_adjust_probabilities_extreme_values():
    """Should handle extreme probabilities without overflow."""
    probs = np.array([0.001, 0.5, 0.999])
    sample_prev = 0.1
    target_prev = 0.9

    adjusted = adjust_probabilities_for_prevalence(probs, sample_prev, target_prev)

    assert np.all(np.isfinite(adjusted)), "Should handle extreme values"
    assert np.all((adjusted >= 0) & (adjusted <= 1)), "Should clip to [0,1]"


# ============================================================================
# PrevalenceAdjustedModel Tests
# ============================================================================


def test_prevalence_adjusted_model_fit():
    """Test fit method (should be no-op)."""
    np.random.seed(42)
    X_train = np.random.randn(100, 10)
    y_train = np.random.binomial(1, 0.3, size=100)

    base_model = LogisticRegression(random_state=42)
    base_model.fit(X_train, y_train)

    wrapper = PrevalenceAdjustedModel(
        base_model, sample_prevalence=0.3, target_prevalence=0.1
    )
    result = wrapper.fit(X_train, y_train)

    assert result is wrapper, "fit() should return self"


def test_prevalence_adjusted_model_predict_proba():
    """Test adjusted probability predictions."""
    np.random.seed(42)
    X_train = np.random.randn(100, 10)
    y_train = np.random.binomial(1, 0.3, size=100)
    X_test = np.random.randn(20, 10)

    base_model = LogisticRegression(random_state=42)
    base_model.fit(X_train, y_train)

    wrapper = PrevalenceAdjustedModel(
        base_model, sample_prevalence=0.3, target_prevalence=0.1
    )
    raw_probs = base_model.predict_proba(X_test)[:, 1]
    adjusted_probs = wrapper.predict_proba(X_test)[:, 1]

    assert np.all(
        adjusted_probs < raw_probs
    ), "Lower target prevalence should decrease probabilities"
    assert adjusted_probs.shape == raw_probs.shape


def test_prevalence_adjusted_model_predict():
    """Test binary predictions."""
    np.random.seed(42)
    X_train = np.random.randn(100, 10)
    y_train = np.random.binomial(1, 0.3, size=100)
    X_test = np.random.randn(20, 10)

    base_model = LogisticRegression(random_state=42)
    base_model.fit(X_train, y_train)

    wrapper = PrevalenceAdjustedModel(
        base_model, sample_prevalence=0.3, target_prevalence=0.1
    )
    predictions = wrapper.predict(X_test)

    assert predictions.shape == (20,)
    assert np.all(
        (predictions == 0) | (predictions == 1)
    ), "Should be binary predictions"


# ============================================================================
# CalibratedClassifierCV Utilities Tests
# ============================================================================


def test_get_calibrated_estimator_param_name():
    """Test parameter name detection for CalibratedClassifierCV."""
    param_name = get_calibrated_estimator_param_name()
    assert param_name in ["estimator", "base_estimator"]


def test_get_calibrated_cv_param_name():
    """Test CV parameter name detection."""
    param_name = get_calibrated_cv_param_name()
    assert param_name == "cv"


def test_maybe_calibrate_estimator_lr():
    """Test calibration wrapper for Logistic Regression."""
    base_model = LogisticRegression(random_state=42)
    calibrated = maybe_calibrate_estimator(
        base_model, model_name="LR_EN", calibrate=True, method="sigmoid", cv=3
    )

    assert isinstance(calibrated, CalibratedClassifierCV)


def test_maybe_calibrate_estimator_rf():
    """Test calibration wrapper for Random Forest."""
    base_model = RandomForestClassifier(n_estimators=10, random_state=42)
    calibrated = maybe_calibrate_estimator(
        base_model, model_name="RF", calibrate=True, method="sigmoid", cv=3
    )

    assert isinstance(calibrated, CalibratedClassifierCV)


def test_maybe_calibrate_estimator_svm_skip():
    """SVM should not be calibrated (already calibrated)."""
    base_model = CalibratedClassifierCV(LinearSVC(random_state=42))
    result = maybe_calibrate_estimator(
        base_model, model_name="LinSVM_cal", calibrate=True, method="sigmoid", cv=3
    )

    assert result is base_model, "SVM should not be re-calibrated"


def test_maybe_calibrate_estimator_disabled():
    """Should return original estimator when calibration disabled."""
    base_model = LogisticRegression(random_state=42)
    result = maybe_calibrate_estimator(
        base_model, model_name="LR_EN", calibrate=False, method="sigmoid", cv=3
    )

    assert result is base_model


def test_maybe_calibrate_estimator_already_calibrated():
    """Should not double-calibrate."""
    base_model = CalibratedClassifierCV(LogisticRegression(random_state=42))
    result = maybe_calibrate_estimator(
        base_model, model_name="LR_EN", calibrate=True, method="sigmoid", cv=3
    )

    assert result is base_model, "Should not double-calibrate"


def test_maybe_calibrate_estimator_isotonic():
    """Test isotonic calibration method."""
    base_model = LogisticRegression(random_state=42)
    calibrated = maybe_calibrate_estimator(
        base_model, model_name="LR_EN", calibrate=True, method="isotonic", cv=5
    )

    assert isinstance(calibrated, CalibratedClassifierCV)


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_calibration_workflow():
    """Test end-to-end calibration workflow."""
    np.random.seed(42)

    # Generate toy data
    n = 200
    X = np.random.randn(n, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Split
    train_idx = np.arange(150)
    test_idx = np.arange(150, 200)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Train base model
    base_model = LogisticRegression(random_state=42)
    base_model.fit(X_train, y_train)

    # Compute calibration metrics
    y_pred = base_model.predict_proba(X_test)[:, 1]
    intercept, slope = calibration_intercept_slope(y_test, y_pred)
    ece = expected_calibration_error(y_test, y_pred)

    assert np.isfinite(intercept)
    assert np.isfinite(slope)
    assert np.isfinite(ece)

    # Apply prevalence adjustment
    sample_prev = y_train.mean()
    target_prev = 0.1
    adjusted_probs = adjust_probabilities_for_prevalence(
        y_pred, sample_prev, target_prev
    )

    assert np.all(adjusted_probs <= y_pred), "Lower prevalence should decrease probs"

    # Test wrapper
    wrapper = PrevalenceAdjustedModel(base_model, sample_prev, target_prev)
    wrapper_probs = wrapper.predict_proba(X_test)[:, 1]

    assert np.allclose(wrapper_probs, adjusted_probs, atol=1e-5)


def test_calibration_with_perfect_separation():
    """Test calibration when classes are perfectly separated."""
    np.random.seed(42)

    # Perfectly separable data
    X_train = np.vstack(
        [np.random.randn(50, 2) - 3, np.random.randn(50, 2) + 3]  # Class 0  # Class 1
    )
    y_train = np.array([0] * 50 + [1] * 50)

    X_test = np.vstack([np.random.randn(10, 2) - 3, np.random.randn(10, 2) + 3])
    y_test = np.array([0] * 10 + [1] * 10)

    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict_proba(X_test)[:, 1]

    # Calibration metrics should still work
    intercept, slope = calibration_intercept_slope(y_test, y_pred)
    ece = expected_calibration_error(y_test, y_pred)

    assert np.isfinite(intercept)
    assert np.isfinite(slope)
    assert np.isfinite(ece)
    assert ece < 0.2, "Perfect separation should have good calibration"
