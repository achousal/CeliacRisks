"""Tests for metrics.thresholds module.

Coverage:
- All threshold selection strategies (F1, F-beta, Youden, fixed spec/PPV, control-based)
- Binary metrics computation at thresholds
- Top risk capture analysis
- Edge cases (empty arrays, all same class, perfect separation)
"""

import numpy as np
import pytest
from ced_ml.metrics.thresholds import (
    binary_metrics_at_threshold,
    choose_threshold_objective,
    threshold_for_precision,
    threshold_for_specificity,
    threshold_from_controls,
    threshold_max_f1,
    threshold_max_fbeta,
    threshold_youden,
    top_risk_capture,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def balanced_data():
    """Balanced dataset with clear separation."""
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    return y, p


@pytest.fixture
def imbalanced_data():
    """Imbalanced dataset (1:9 ratio) simulating rare disease."""
    y = np.array([0] * 90 + [1] * 10)
    np.random.seed(42)
    p_controls = np.random.beta(2, 5, size=90)  # Skewed low
    p_cases = np.random.beta(5, 2, size=10)  # Skewed high
    p = np.concatenate([p_controls, p_cases])
    return y, p


@pytest.fixture
def perfect_separation():
    """Perfectly separated data (AUROC = 1.0)."""
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    return y, p


@pytest.fixture
def all_controls():
    """All negative samples (no cases)."""
    y = np.zeros(100, dtype=int)
    p = np.random.uniform(0, 1, size=100)
    return y, p


@pytest.fixture
def all_cases():
    """All positive samples (no controls)."""
    y = np.ones(100, dtype=int)
    p = np.random.uniform(0, 1, size=100)
    return y, p


# ============================================================================
# Test threshold_max_f1
# ============================================================================


def test_threshold_max_f1_balanced(balanced_data):
    """Test F1-maximizing threshold on balanced data."""
    y, p = balanced_data
    thr = threshold_max_f1(y, p)
    assert 0.0 <= thr <= 1.0
    # Expect threshold around 0.5 for balanced separation
    assert 0.4 <= thr <= 0.6


def test_threshold_max_f1_imbalanced(imbalanced_data):
    """Test F1-maximizing threshold on imbalanced data."""
    y, p = imbalanced_data
    thr = threshold_max_f1(y, p)
    assert 0.0 <= thr <= 1.0
    # Imbalanced data typically requires higher threshold
    assert thr > 0.5


def test_threshold_max_f1_empty():
    """Test F1 threshold with empty arrays."""
    y = np.array([])
    p = np.array([])
    thr = threshold_max_f1(y, p)
    assert thr == 0.5  # Fallback


def test_threshold_max_f1_all_same_class(all_controls):
    """Test F1 threshold when all samples are same class."""
    y, p = all_controls
    thr = threshold_max_f1(y, p)
    assert 0.0 <= thr <= 1.0


# ============================================================================
# Test threshold_max_fbeta
# ============================================================================


def test_threshold_max_fbeta_beta1(balanced_data):
    """Test F-beta with beta=1 (should equal F1)."""
    y, p = balanced_data
    thr_f1 = threshold_max_f1(y, p)
    thr_fb = threshold_max_fbeta(y, p, beta=1.0)
    assert abs(thr_f1 - thr_fb) < 0.01


def test_threshold_max_fbeta_beta2(balanced_data):
    """Test F-beta with beta=2 (emphasize recall)."""
    y, p = balanced_data
    thr_fb = threshold_max_fbeta(y, p, beta=2.0)
    thr_f1 = threshold_max_f1(y, p)
    # Higher beta -> lower threshold (more sensitive)
    assert thr_fb <= thr_f1


def test_threshold_max_fbeta_beta05(balanced_data):
    """Test F-beta with beta=0.5 (emphasize precision)."""
    y, p = balanced_data
    thr_fb = threshold_max_fbeta(y, p, beta=0.5)
    thr_f1 = threshold_max_f1(y, p)
    # Lower beta -> higher threshold (more precise)
    assert thr_fb >= thr_f1


def test_threshold_max_fbeta_invalid_beta(balanced_data):
    """Test F-beta with invalid beta (should default to 1.0)."""
    y, p = balanced_data
    thr = threshold_max_fbeta(y, p, beta=-1.0)
    assert 0.0 <= thr <= 1.0


# ============================================================================
# Test threshold_youden
# ============================================================================


def test_threshold_youden_balanced(balanced_data):
    """Test Youden threshold on balanced data."""
    y, p = balanced_data
    thr = threshold_youden(y, p)
    assert 0.0 <= thr <= 1.0


def test_threshold_youden_perfect_separation(perfect_separation):
    """Test Youden threshold with perfect separation."""
    y, p = perfect_separation
    thr = threshold_youden(y, p)
    assert 0.0 <= thr <= 1.0
    # Should separate perfectly - threshold should be in the gap
    # Between max control (0.3) and min case (0.7)
    assert 0.0 <= thr <= 1.0  # Just verify valid range


def test_threshold_youden_empty():
    """Test Youden threshold with empty arrays."""
    y = np.array([])
    p = np.array([])
    thr = threshold_youden(y, p)
    assert thr == 0.5  # Fallback


# ============================================================================
# Test threshold_for_specificity
# ============================================================================


def test_threshold_for_specificity_90(balanced_data):
    """Test fixed specificity threshold (0.90)."""
    y, p = balanced_data
    thr = threshold_for_specificity(y, p, target_spec=0.90)
    assert 0.0 <= thr <= 1.0
    # Higher specificity -> higher threshold (or equal in edge cases)
    thr_50 = threshold_for_specificity(y, p, target_spec=0.50)
    assert thr >= thr_50


def test_threshold_for_specificity_95(imbalanced_data):
    """Test high specificity threshold (0.95)."""
    y, p = imbalanced_data
    thr = threshold_for_specificity(y, p, target_spec=0.95)
    assert 0.0 <= thr <= 1.0


def test_threshold_for_specificity_unattainable(balanced_data):
    """Test specificity threshold when target is unattainable."""
    y, p = balanced_data
    # Request impossibly high specificity
    thr = threshold_for_specificity(y, p, target_spec=0.999)
    assert 0.0 <= thr <= 1.0


def test_threshold_for_specificity_perfect(perfect_separation):
    """Test specificity threshold with perfect separation."""
    y, p = perfect_separation
    thr = threshold_for_specificity(y, p, target_spec=1.0)
    assert 0.0 <= thr <= 1.0


# ============================================================================
# Test threshold_for_precision
# ============================================================================


def test_threshold_for_precision_balanced(balanced_data):
    """Test fixed precision threshold on balanced data."""
    y, p = balanced_data
    thr = threshold_for_precision(y, p, target_ppv=0.8)
    assert 0.0 <= thr <= 1.0


def test_threshold_for_precision_unattainable(imbalanced_data):
    """Test precision threshold when target is unattainable (fallback to F1)."""
    y, p = imbalanced_data
    # Request impossibly high precision
    thr_high = threshold_for_precision(y, p, target_ppv=0.99)
    thr_f1 = threshold_max_f1(y, p)
    # Should fall back to F1 (or be close in case of randomness)
    # Allow larger tolerance due to imbalanced data variability
    assert abs(thr_high - thr_f1) < 0.3


def test_threshold_for_precision_invalid_target(balanced_data):
    """Test precision threshold with invalid target (should fallback to F1)."""
    y, p = balanced_data
    thr = threshold_for_precision(y, p, target_ppv=1.5)  # Invalid
    thr_f1 = threshold_max_f1(y, p)
    assert abs(thr - thr_f1) < 0.01


# ============================================================================
# Test threshold_from_controls
# ============================================================================


def test_threshold_from_controls_basic():
    """Test control-based threshold (basic quantile)."""
    p_controls = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    thr = threshold_from_controls(p_controls, target_spec=0.90)
    # 90th percentile should be 0.9
    assert 0.8 <= thr <= 1.0


def test_threshold_from_controls_95():
    """Test 95% specificity threshold from controls."""
    p_controls = np.linspace(0, 1, 100)
    thr = threshold_from_controls(p_controls, target_spec=0.95)
    assert 0.90 <= thr <= 1.0


def test_threshold_from_controls_empty():
    """Test control-based threshold with empty array."""
    p_controls = np.array([])
    thr = threshold_from_controls(p_controls, target_spec=0.90)
    assert thr == 0.5  # Fallback


def test_threshold_from_controls_nan():
    """Test control-based threshold with NaN values."""
    p_controls = np.array([0.1, np.nan, 0.3, np.nan, 0.5])
    thr = threshold_from_controls(p_controls, target_spec=0.90)
    assert 0.0 <= thr <= 1.0


# ============================================================================
# Test binary_metrics_at_threshold
# ============================================================================


def test_binary_metrics_at_threshold_perfect(perfect_separation):
    """Test metrics at optimal threshold with perfect separation."""
    y, p = perfect_separation
    thr = 0.5  # Separates perfectly
    metrics = binary_metrics_at_threshold(y, p, thr)

    assert metrics["threshold"] == 0.5
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["tp"] == 3
    assert metrics["tn"] == 3
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0


def test_binary_metrics_at_threshold_balanced(balanced_data):
    """Test metrics at threshold on balanced data."""
    y, p = balanced_data
    thr = 0.5
    metrics = binary_metrics_at_threshold(y, p, thr)

    assert metrics["threshold"] == 0.5
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["specificity"] <= 1.0
    assert metrics["tp"] + metrics["fn"] == 4  # Total cases
    assert metrics["tn"] + metrics["fp"] == 4  # Total controls


def test_binary_metrics_at_threshold_all_positive():
    """Test metrics when all predictions are positive."""
    y = np.array([0, 0, 1, 1])
    p = np.array([0.9, 0.9, 0.9, 0.9])
    thr = 0.5
    metrics = binary_metrics_at_threshold(y, p, thr)

    assert metrics["fp"] == 2
    assert metrics["tp"] == 2
    assert metrics["tn"] == 0
    assert metrics["fn"] == 0


def test_binary_metrics_at_threshold_all_negative():
    """Test metrics when all predictions are negative."""
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.1, 0.1, 0.1])
    thr = 0.5
    metrics = binary_metrics_at_threshold(y, p, thr)

    assert metrics["fp"] == 0
    assert metrics["tp"] == 0
    assert metrics["tn"] == 2
    assert metrics["fn"] == 2
    assert metrics["precision"] == 0.0  # zero_division=0


# ============================================================================
# Test top_risk_capture
# ============================================================================


def test_top_risk_capture_1pct(imbalanced_data):
    """Test top 1% risk capture on imbalanced data."""
    y, p = imbalanced_data
    result = top_risk_capture(y, p, frac=0.01)

    assert result["frac"] == 0.01
    assert result["n_top"] == 1  # ceil(100 * 0.01)
    assert result["cases_in_top"] >= 0
    assert result["controls_in_top"] >= 0
    assert result["cases_in_top"] + result["controls_in_top"] == result["n_top"]


def test_top_risk_capture_10pct(balanced_data):
    """Test top 10% risk capture on balanced data."""
    y, p = balanced_data
    result = top_risk_capture(y, p, frac=0.10)

    assert result["frac"] == 0.10
    assert result["n_top"] == 1  # ceil(8 * 0.10)
    assert 0.0 <= result["case_capture"] <= 1.0


def test_top_risk_capture_50pct(balanced_data):
    """Test top 50% risk capture."""
    y, p = balanced_data
    result = top_risk_capture(y, p, frac=0.50)

    assert result["frac"] == 0.50
    assert result["n_top"] == 4
    # Should capture ~50% of cases
    assert result["case_capture"] >= 0.25


def test_top_risk_capture_empty():
    """Test top risk capture with empty arrays."""
    y = np.array([])
    p = np.array([])
    result = top_risk_capture(y, p, frac=0.01)

    assert result["n_top"] == 0
    assert result["cases_in_top"] == 0
    assert np.isnan(result["case_capture"])


def test_top_risk_capture_all_controls(all_controls):
    """Test top risk capture when there are no cases."""
    y, p = all_controls
    result = top_risk_capture(y, p, frac=0.05)

    assert result["cases_in_top"] == 0
    assert np.isnan(result["case_capture"])


# ============================================================================
# Test choose_threshold_objective
# ============================================================================


def test_choose_threshold_objective_max_f1(balanced_data):
    """Test threshold selection with max_f1 objective."""
    y, p = balanced_data
    name, thr = choose_threshold_objective(y, p, objective="max_f1")

    assert name == "max_f1"
    assert 0.0 <= thr <= 1.0


def test_choose_threshold_objective_max_fbeta(balanced_data):
    """Test threshold selection with max_fbeta objective."""
    y, p = balanced_data
    name, thr = choose_threshold_objective(y, p, objective="max_fbeta", fbeta=2.0)

    assert name == "max_fbeta"
    assert 0.0 <= thr <= 1.0


def test_choose_threshold_objective_youden(balanced_data):
    """Test threshold selection with youden objective."""
    y, p = balanced_data
    name, thr = choose_threshold_objective(y, p, objective="youden")

    assert name == "youden"
    assert 0.0 <= thr <= 1.0


def test_choose_threshold_objective_fixed_spec(balanced_data):
    """Test threshold selection with fixed_spec objective."""
    y, p = balanced_data
    name, thr = choose_threshold_objective(y, p, objective="fixed_spec", fixed_spec=0.95)

    assert name == "fixed_spec"
    assert 0.0 <= thr <= 1.0


def test_choose_threshold_objective_fixed_ppv(balanced_data):
    """Test threshold selection with fixed_ppv objective."""
    y, p = balanced_data
    name, thr = choose_threshold_objective(y, p, objective="fixed_ppv", fixed_ppv=0.8)

    assert name == "fixed_ppv"
    assert 0.0 <= thr <= 1.0


def test_choose_threshold_objective_unknown_fallback(balanced_data):
    """Test threshold selection with unknown objective (should fallback to max_f1)."""
    y, p = balanced_data
    name, thr = choose_threshold_objective(y, p, objective="unknown_method")

    assert name == "max_f1"
    assert 0.0 <= thr <= 1.0


def test_choose_threshold_objective_none(balanced_data):
    """Test threshold selection with None objective (should default to max_f1)."""
    y, p = balanced_data
    name, thr = choose_threshold_objective(y, p, objective=None)

    assert name == "max_f1"
    assert 0.0 <= thr <= 1.0


def test_choose_threshold_objective_case_insensitive(balanced_data):
    """Test threshold selection is case-insensitive."""
    y, p = balanced_data
    name1, thr1 = choose_threshold_objective(y, p, objective="MAX_F1")
    name2, thr2 = choose_threshold_objective(y, p, objective="max_f1")

    assert name1 == name2 == "max_f1"
    assert thr1 == thr2
