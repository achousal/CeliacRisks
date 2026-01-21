"""
Test suite for data quality - Outcome Label Validation

CRITICAL: Prevent reverse causality bias from mixing incident/prevalent cases.

Tests cover:
1. No prevalent cases in modeling dataset (IncidentOnly)
2. Outcome binary encoding (Controls=0, Incident=1)
3. CeD_date temporal consistency
4. Uncertain controls removal
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


class TestOutcomeLabels:
    """Test outcome label integrity."""

    def test_incident_only_excludes_prevalent(self, synthetic_celiac_data):
        """IncidentOnly scenario should exclude ALL prevalent cases."""
        df, _ = synthetic_celiac_data

        # Simulate IncidentOnly filtering
        mask_incident = df["CeD_comparison"].isin(["Controls", "Incident"])
        df_incident = df[mask_incident].copy()

        # Verify no prevalent cases
        n_prevalent = (df_incident["CeD_comparison"] == "Prevalent").sum()

        assert n_prevalent == 0, \
            f"IncidentOnly scenario contains {n_prevalent} prevalent cases (reverse causality risk!)"

    def test_outcome_values_valid(self, synthetic_celiac_data):
        """CeD_comparison should only contain valid values."""
        df, _ = synthetic_celiac_data

        valid_values = {"Controls", "Incident", "Prevalent"}
        unique_values = set(df["CeD_comparison"].unique())

        invalid = unique_values - valid_values

        assert len(invalid) == 0, \
            f"Invalid outcome values detected: {invalid}"

    def test_binary_encoding_correct(self, synthetic_celiac_data):
        """Binary encoding should be: Controls=0, Incident=1."""
        df, _ = synthetic_celiac_data

        mask_incident = df["CeD_comparison"].isin(["Controls", "Incident"])
        df_incident = df[mask_incident].copy()

        y = (df_incident["CeD_comparison"] == "Incident").astype(int)

        # Verify only 0 and 1
        unique_y = set(y.unique())
        assert unique_y == {0, 1}, \
            f"Binary outcome should be {{0, 1}}, got {unique_y}"

        # Verify Controls -> 0
        controls_mask = df_incident["CeD_comparison"] == "Controls"
        assert (y[controls_mask] == 0).all(), \
            "Controls should be encoded as 0"

        # Verify Incident -> 1
        incident_mask = df_incident["CeD_comparison"] == "Incident"
        assert (y[incident_mask] == 1).all(), \
            "Incident should be encoded as 1"

    def test_prevalence_realistic(self, synthetic_celiac_data):
        """
        Incident prevalence should be realistic (~0.34% in real data).

        For synthetic data, we allow wider range but check it's rare.
        """
        df, _ = synthetic_celiac_data

        mask_incident = df["CeD_comparison"].isin(["Controls", "Incident"])
        df_incident = df[mask_incident].copy()

        y = (df_incident["CeD_comparison"] == "Incident").astype(int)
        prevalence = y.mean()

        # Synthetic data should still be rare disease
        assert 0.001 <= prevalence <= 0.05, \
            f"Prevalence {prevalence:.4f} outside realistic range [0.1%, 5%]"


class TestCeDDateConsistency:
    """Test CeD diagnosis date consistency."""

    def test_controls_mostly_no_date(self, synthetic_celiac_data):
        """
        Most Controls should NOT have CeD_date.

        Exception: "uncertain controls" (Controls with CeD_date) should be filtered.
        """
        df, _ = synthetic_celiac_data

        controls = df[df["CeD_comparison"] == "Controls"]
        n_controls = len(controls)
        n_with_date = controls["CeD_date"].notna().sum()

        frac_with_date = n_with_date / n_controls if n_controls > 0 else 0

        # In synthetic data, we added 3 uncertain controls (should be <5%)
        assert frac_with_date < 0.05, \
            f"{frac_with_date*100:.1f}% of controls have CeD_date (expected <5%)"

    def test_incident_cases_have_date(self, synthetic_celiac_data):
        """Incident cases should have CeD_date (diagnosis date)."""
        df, _ = synthetic_celiac_data

        incident = df[df["CeD_comparison"] == "Incident"]
        n_incident = len(incident)

        if n_incident > 0:
            n_with_date = incident["CeD_date"].notna().sum()

            assert n_with_date == n_incident, \
                f"Only {n_with_date}/{n_incident} incident cases have CeD_date"

    def test_prevalent_cases_have_date(self, synthetic_celiac_data):
        """Prevalent cases should have CeD_date (diagnosis date)."""
        df, _ = synthetic_celiac_data

        prevalent = df[df["CeD_comparison"] == "Prevalent"]
        n_prevalent = len(prevalent)

        if n_prevalent > 0:
            n_with_date = prevalent["CeD_date"].notna().sum()

            assert n_with_date == n_prevalent, \
                f"Only {n_with_date}/{n_prevalent} prevalent cases have CeD_date"


class TestUncertainControlsFiltering:
    """Test uncertain controls (Controls with CeD_date) are filtered."""

    def test_uncertain_controls_identified(self, synthetic_celiac_data):
        """Should correctly identify Controls with CeD_date."""
        df, _ = synthetic_celiac_data

        mask_uncertain = (
            (df["CeD_comparison"] == "Controls") &
            df["CeD_date"].notna()
        )

        n_uncertain = mask_uncertain.sum()

        # Our synthetic data has 3 uncertain controls
        assert n_uncertain == 3, \
            f"Expected 3 uncertain controls, found {n_uncertain}"

    def test_uncertain_controls_removed_after_filtering(self, sample_splits):
        """
        After applying row filters, no uncertain controls should remain.

        This uses the sample_splits fixture which applies row filters.
        """
        df = sample_splits["df"]

        mask_uncertain = (
            (df["CeD_comparison"] == "Controls") &
            df["CeD_date"].notna()
        )

        n_uncertain = mask_uncertain.sum()

        assert n_uncertain == 0, \
            f"After filtering, {n_uncertain} uncertain controls remain (should be 0)"


class TestOutcomeCounts:
    """Test outcome counts are as expected."""

    def test_controls_dominate(self, synthetic_celiac_data):
        """Controls should be vast majority (realistic class imbalance)."""
        df, _ = synthetic_celiac_data

        n_total = len(df)
        n_controls = (df["CeD_comparison"] == "Controls").sum()

        frac_controls = n_controls / n_total

        assert frac_controls > 0.90, \
            f"Controls are {frac_controls*100:.1f}% (expected >90% for rare disease)"

    def test_incident_prevalent_balanced(self, synthetic_celiac_data):
        """Incident and prevalent counts should be similar (in real data)."""
        df, _ = synthetic_celiac_data

        n_incident = (df["CeD_comparison"] == "Incident").sum()
        n_prevalent = (df["CeD_comparison"] == "Prevalent").sum()

        if n_incident > 0 and n_prevalent > 0:
            ratio = max(n_incident, n_prevalent) / min(n_incident, n_prevalent)

            # In real data: 148 incident, 150 prevalent (nearly equal)
            # Allow up to 2:1 ratio
            assert ratio < 2.0, \
                f"Incident ({n_incident}) vs Prevalent ({n_prevalent}) ratio {ratio:.2f} seems off"

    def test_no_empty_outcomes(self, synthetic_celiac_data):
        """All outcome categories should have at least 1 sample."""
        df, _ = synthetic_celiac_data

        for outcome in ["Controls", "Incident", "Prevalent"]:
            n = (df["CeD_comparison"] == outcome).sum()
            assert n > 0, f"No samples with outcome={outcome}"


class TestIncidentOnlyScenario:
    """Test IncidentOnly scenario (primary analysis)."""

    def test_incident_only_composition(self, sample_splits):
        """IncidentOnly should be Controls + Incident only."""
        df = sample_splits["df"]

        unique_outcomes = set(df["CeD_comparison"].unique())

        assert unique_outcomes == {"Controls", "Incident"}, \
            f"IncidentOnly should only have Controls/Incident, got {unique_outcomes}"

    def test_incident_only_sufficient_cases(self, sample_splits):
        """IncidentOnly should have enough incident cases for modeling."""
        y = sample_splits["y"]
        n_cases = y.sum()

        # With repeated CV (5x10), each outer fold has ~80% of cases
        # Need at least 2 cases per outer fold -> 2 / 0.8 = 2.5, round up to 5 minimum
        assert n_cases >= 5, \
            f"Only {n_cases} incident cases (need ≥5 for 5x10 CV)"

    def test_incident_only_row_count(self, synthetic_celiac_data, sample_splits):
        """
        IncidentOnly row count should match expected after filtering.

        Expected = Controls + Incident - uncertain controls - missing metadata
        """
        df_raw, _ = synthetic_celiac_data
        df_filtered = sample_splits["df"]

        # Count expected
        mask_incident = df_raw["CeD_comparison"].isin(["Controls", "Incident"])
        df_expected = df_raw[mask_incident].copy()

        # Apply same filters
        mask_uncertain = (
            (df_expected["CeD_comparison"] == "Controls") &
            df_expected["CeD_date"].notna()
        )
        df_expected = df_expected[~mask_uncertain]
        df_expected = df_expected.dropna(subset=["age", "BMI"])

        n_expected = len(df_expected)
        n_actual = len(df_filtered)

        assert n_actual == n_expected, \
            f"IncidentOnly has {n_actual} rows, expected {n_expected}"
