"""
Test suite for missing data assumptions.

CRITICAL: CLAUDE.MD claims "ZERO missing values" but code has extensive
imputation logic. These tests verify the claim and identify unnecessary code.

Tests cover:
1. Zero missing in protein features
2. Metadata completeness after filtering
3. Imputation code should be unreachable
4. Missing data handling configuration
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


class TestProteinMissingness:
    """Test missingness in protein features (*_resid columns)."""

    def test_zero_missing_in_proteins_synthetic(self, synthetic_celiac_data):
        """Synthetic data should have zero missing in proteins."""
        df, _ = synthetic_celiac_data

        protein_cols = [c for c in df.columns if c.endswith("_resid")]

        assert len(protein_cols) > 0, "No protein columns found"

        n_missing = df[protein_cols].isna().sum().sum()

        assert n_missing == 0, \
            f"Found {n_missing} missing values in protein columns (expected 0)"

    def test_protein_data_types(self, synthetic_celiac_data):
        """Protein columns should be numeric."""
        df, _ = synthetic_celiac_data

        protein_cols = [c for c in df.columns if c.endswith("_resid")]

        for col in protein_cols:
            assert pd.api.types.is_numeric_dtype(df[col]), \
                f"Protein column {col} is not numeric: {df[col].dtype}"

    def test_protein_zscored(self, synthetic_celiac_data):
        """
        Proteins with _resid suffix should be z-scored (mean~0, std~1).

        Note: This is approximate due to case enrichment, but controls
        should be centered.
        """
        df, _ = synthetic_celiac_data

        # Check controls only (should be N(0,1) for proteins)
        controls = df[df["CeD_comparison"] == "Controls"]
        protein_cols = [c for c in controls.columns if c.endswith("_resid")]

        for col in protein_cols[:10]:  # Check first 10
            mean = controls[col].mean()
            std = controls[col].std()

            # Allow some deviation due to sampling
            assert -0.2 < mean < 0.2, \
                f"{col} mean {mean:.3f} not centered (expected ~0)"
            assert 0.8 < std < 1.2, \
                f"{col} std {std:.3f} not unit variance (expected ~1)"


class TestMetadataMissingness:
    """Test missingness in demographic metadata."""

    def test_metadata_complete_after_filtering(self, sample_splits):
        """
        After row filters, age/BMI should have no missing values.

        This is enforced by dropna_meta_num=True in save_splits.py.
        """
        df = sample_splits["df"]

        n_missing_age = df["age"].isna().sum()
        n_missing_bmi = df["BMI"].isna().sum()

        assert n_missing_age == 0, \
            f"After filtering, {n_missing_age} rows missing age (should be 0)"
        assert n_missing_bmi == 0, \
            f"After filtering, {n_missing_bmi} rows missing BMI (should be 0)"

    def test_categorical_metadata_complete(self, sample_splits):
        """Categorical metadata (sex, genetics) should be complete."""
        df = sample_splits["df"]

        n_missing_sex = df["sex"].isna().sum()
        n_missing_genetics = df["Genetic ethnic grouping"].isna().sum()

        assert n_missing_sex == 0, \
            f"{n_missing_sex} rows missing sex"
        assert n_missing_genetics == 0, \
            f"{n_missing_genetics} rows missing genetics"

    def test_eid_unique_and_complete(self, sample_splits):
        """Participant IDs should be unique and complete."""
        df = sample_splits["df"]

        n_missing_eid = df["eid"].isna().sum()
        n_unique_eid = df["eid"].nunique()
        n_total = len(df)

        assert n_missing_eid == 0, \
            f"{n_missing_eid} rows missing eid"
        assert n_unique_eid == n_total, \
            f"eid not unique: {n_unique_eid} unique vs {n_total} rows"


class TestMissingDataWithMissing:
    """Test datasets WITH missing values (for imputation logic testing)."""

    def test_missing_introduced_correctly(self, synthetic_celiac_data_with_missing):
        """Fixture should introduce missing values as expected."""
        df, _ = synthetic_celiac_data_with_missing

        n_missing_age = df["age"].isna().sum()
        n_missing_bmi = df["BMI"].isna().sum()

        # Should have ~5% missing in each
        n_total = len(df)
        expected_missing = n_total * 0.05

        assert n_missing_age > 0, "Missing values not introduced in age"
        assert n_missing_bmi > 0, "Missing values not introduced in BMI"

        assert 0.03 < n_missing_age / n_total < 0.07, \
            f"Age missing rate {n_missing_age/n_total:.2%} not near 5%"

    def test_dropna_removes_missing(self, synthetic_celiac_data_with_missing):
        """Dropna should remove all rows with missing age/BMI."""
        df, _ = synthetic_celiac_data_with_missing

        n_before = len(df)
        n_missing = df[["age", "BMI"]].isna().any(axis=1).sum()

        df_clean = df.dropna(subset=["age", "BMI"])
        n_after = len(df_clean)

        assert n_after == n_before - n_missing, \
            "Dropna didn't remove expected number of rows"

        assert df_clean[["age", "BMI"]].isna().sum().sum() == 0, \
            "Missing values remain after dropna"


class TestImputationCodeReachability:
    """
    Test whether imputation code is actually reached during execution.

    If CLAUDE.MD claim is true (zero missing), imputation should never run.
    """

    def test_imputation_config_documented(self):
        """
        Imputation configuration should be documented.

        This is a placeholder - real test would check celiacML_faith.py
        for MISSING_IMPUTER setting.
        """
        # In celiacML_faith.py, there should be:
        # - MISSING_IMPUTER = "median" (or similar)
        # - MIN_NONMISSING = 0.50 (or similar)

        # This test just documents the requirement
        # Actual implementation would parse celiacML_faith.py
        pass

    def test_variance_prefilter_enabled_flag(self):
        """
        Variance/missingness prefilter should be DISABLED if zero missing.

        From CLAUDE.MD: "Remove variance_missingness_prefilter() function"
        This test documents that the function exists but shouldn't be used.
        """
        # Check that prefilter can be disabled
        # Real implementation would verify:
        # 1. Function exists in code
        # 2. Flag to disable it exists
        # 3. Flag is set to disabled by default
        pass


class TestMissingnessStatistics:
    """Compute missingness statistics for reporting."""

    def test_overall_completeness_rate(self, sample_splits):
        """Calculate overall data completeness for modeling columns."""
        df = sample_splits["df"]

        # Exclude CeD_date from completeness check - it's expected to be NaT for controls
        # The key columns for modeling are: proteins, age, BMI, sex, ethnicity
        modeling_cols = [c for c in df.columns if c not in ["CeD_date"]]
        df_modeling = df[modeling_cols]

        total_cells = df_modeling.shape[0] * df_modeling.shape[1]
        missing_cells = df_modeling.isna().sum().sum()

        completeness = 1 - (missing_cells / total_cells)

        # Should be 100% complete after filtering (excluding CeD_date)
        assert completeness == 1.0, \
            f"Dataset is {completeness*100:.2f}% complete (expected 100%)"

    def test_per_column_completeness(self, sample_splits):
        """All modeling columns should be 100% complete."""
        df = sample_splits["df"]

        # Exclude CeD_date - it's expected to be NaT for controls
        modeling_cols = [c for c in df.columns if c not in ["CeD_date"]]

        missing_per_col = df[modeling_cols].isna().sum()
        incomplete_cols = missing_per_col[missing_per_col > 0]

        assert len(incomplete_cols) == 0, \
            f"Columns with missing values: {incomplete_cols.to_dict()}"

    def test_missing_value_patterns(self, synthetic_celiac_data_with_missing):
        """
        Analyze missing value patterns in dataset WITH missing.

        This is diagnostic - helps understand if missing is MCAR, MAR, or MNAR.
        """
        df, _ = synthetic_celiac_data_with_missing

        # Check if missingness correlates with outcome
        has_missing_age = df["age"].isna()

        if has_missing_age.sum() > 0:
            # Compare prevalence in missing vs. non-missing
            y = (df["CeD_comparison"] == "Incident").astype(int)

            prev_with_missing = y[has_missing_age].mean()
            prev_without_missing = y[~has_missing_age].mean()

            # If these differ substantially, missing is MAR/MNAR (informative)
            # For random missing (MCAR), they should be similar

            # Just document the pattern (no assertion)
            print(f"Prevalence with missing age: {prev_with_missing:.4f}")
            print(f"Prevalence without missing age: {prev_without_missing:.4f}")


@pytest.mark.skipif(True, reason="Requires real data - run manually")
class TestRealDataMissingness:
    """
    Test missingness in REAL Celiac dataset.

    These tests are skipped by default (require actual data).
    Run manually with: pytest tests/test_missing_data.py::TestRealDataMissingness -v
    """

    def test_real_data_zero_missing_claim(self, data_dir):
        """
        Verify CLAUDE.MD claim: ZERO missing values in real data.

        This is the CRITICAL test that validates removing imputation code.
        """
        from conftest import real_data_exists

        if not real_data_exists(data_dir):
            pytest.skip("Real data not available")

        data_path = data_dir / "Celiac_dataset_proteomics.csv"

        # Load real data (may be large)
        df = pd.read_csv(data_path, low_memory=False)

        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isna().sum().sum()

        if missing_cells > 0:
            # Print diagnostic info
            print(f"\nMissingness found in real data:")
            print(f"  Total cells: {total_cells:,}")
            print(f"  Missing cells: {missing_cells:,}")
            print(f"  Completeness: {(1 - missing_cells/total_cells)*100:.4f}%")

            missing_per_col = df.isna().sum()
            incomplete_cols = missing_per_col[missing_per_col > 0].sort_values(ascending=False)
            print(f"\n  Columns with missing (top 10):")
            for col, n_miss in incomplete_cols.head(10).items():
                print(f"    {col}: {n_miss:,} ({n_miss/len(df)*100:.2f}%)")

        # ASSERT zero missing to validate CLAUDE.MD claim
        assert missing_cells == 0, \
            f"CLAUDE.MD claims zero missing, but found {missing_cells:,} missing values"
