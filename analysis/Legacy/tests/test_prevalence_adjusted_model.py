"""
Unit tests for PrevalenceAdjustedModel wrapper.

Ensures the exported artifact reproduces the prevalence-corrected
probabilities used during evaluation time.
"""

import sys
from pathlib import Path

import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))

from celiacML_faith import PrevalenceAdjustedModel, adjust_probabilities_for_prevalence


class _ConstantProbModel:
    """Simple binary classifier stub returning a constant probability."""

    def __init__(self, prob: float):
        self.prob = float(prob)
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        n = len(X)
        pos = np.full(n, self.prob, dtype=float)
        neg = 1.0 - pos
        return np.column_stack([neg, pos])


def test_prevalence_adjusted_model_matches_adjust_function():
    base = _ConstantProbModel(prob=0.40)
    wrapper = PrevalenceAdjustedModel(base, sample_prevalence=0.40, target_prevalence=0.05)

    X = np.zeros((5, 1))
    adjusted = wrapper.predict_proba(X)

    expected = adjust_probabilities_for_prevalence(
        np.full(len(X), 0.40, dtype=float),
        sample_prev=0.40,
        target_prev=0.05,
    )

    assert np.allclose(adjusted[:, 1], expected)
    assert np.allclose(adjusted[:, 0], 1.0 - expected)


def test_predict_uses_adjusted_probabilities():
    base = _ConstantProbModel(prob=0.90)
    wrapper = PrevalenceAdjustedModel(base, sample_prevalence=0.90, target_prevalence=0.10)
    preds = wrapper.predict(np.zeros((4, 1)))

    # After downshifting prevalence, predicted positives should flip to 0
    assert np.all(preds == 0)


def test_joblib_roundtrip_preserves_behavior(tmp_path):
    base = _ConstantProbModel(prob=0.65)
    wrapper = PrevalenceAdjustedModel(base, sample_prevalence=0.65, target_prevalence=0.15)

    path = tmp_path / "wrapper.joblib"
    joblib.dump(wrapper, path)

    loaded = joblib.load(path)
    X = np.zeros((3, 1))

    assert np.allclose(loaded.predict_proba(X), wrapper.predict_proba(X))
