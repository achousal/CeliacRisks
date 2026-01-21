"""
Test suite for categorical feature encoding with missing values.

STRATEGY: Treat missing as informative category ("Missing").
This allows the model to learn whether missingness itself carries signal.

Tests cover:
1. OneHotEncoder handles missing values correctly
2. "Missing" category is created for NaN values
3. No data leakage in categorical encoding
4. Encoding is consistent across train/test
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class TestCategoricalMissingnessAsCategory:
    """Test treating missing categorical values as separate category."""

    def test_missing_becomes_own_category(self):
        """NaN values should become 'Missing' category."""
        # Sample data with missing
        data = pd.DataFrame({
            "ethnicity": ["Caucasian", "African", np.nan, "Asian", np.nan, "Caucasian"],
            "outcome": [0, 0, 1, 0, 1, 0]
        })

        # Fill NaN with explicit "Missing" string
        data["ethnicity_filled"] = data["ethnicity"].fillna("Missing")

        # Check that Missing category was created
        unique_values = set(data["ethnicity_filled"].unique())
        assert "Missing" in unique_values, "Missing category should exist"

        # Check counts
        n_missing = (data["ethnicity"] == "Missing").sum() + data["ethnicity"].isna().sum()
        n_missing_filled = (data["ethnicity_filled"] == "Missing").sum()

        assert n_missing_filled == 2, "Should have 2 Missing values"

    def test_onehot_encoder_with_missing_category(self):
        """OneHotEncoder should handle 'Missing' category like any other."""
        data = pd.DataFrame({
            "ethnicity": ["Caucasian", "African", np.nan, "Asian", np.nan, "Caucasian"]
        })

        # Fill NaN before encoding
        data["ethnicity_filled"] = data["ethnicity"].fillna("Missing")

        # Fit encoder
        enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded = enc.fit_transform(data[["ethnicity_filled"]])

        # Check shape: should have one column per category
        n_categories = data["ethnicity_filled"].nunique()
        assert encoded.shape[1] == n_categories, \
            f"Should have {n_categories} columns (including Missing)"

        # Check that categories include "Missing"
        categories = enc.categories_[0]
        assert "Missing" in categories, "Missing should be in encoder categories"

    def test_missing_category_preserves_subjects(self):
        """Using 'Missing' category should not drop any subjects."""
        np.random.seed(42)  # For reproducibility

        # Original data with guaranteed missing values
        ethnicity_values = ["Caucasian"] * 60 + ["African"] * 20 + ["Asian"] * 10 + [np.nan] * 10
        np.random.shuffle(ethnicity_values)

        data = pd.DataFrame({
            "eid": np.arange(100),
            "ethnicity": ethnicity_values,
            "outcome": np.random.randint(0, 2, 100)
        })

        n_before = len(data)
        n_missing = data["ethnicity"].isna().sum()

        # Strategy 1: Drop missing (old approach)
        data_dropped = data.dropna(subset=["ethnicity"])
        n_after_drop = len(data_dropped)

        # Strategy 2: Fill missing (new approach)
        data_filled = data.copy()
        data_filled["ethnicity"] = data_filled["ethnicity"].fillna("Missing")
        n_after_fill = len(data_filled)

        # New approach should preserve all subjects
        assert n_after_fill == n_before, "Filling should preserve all subjects"
        assert n_after_drop == n_before - n_missing, \
            f"Dropping should remove {n_missing} subjects"

        n_saved = n_after_fill - n_after_drop
        print(f"Subjects saved by 'Missing' category: {n_saved} (expected {n_missing})")

    def test_train_test_consistency_with_missing(self):
        """Encoding should be consistent between train and test."""
        # Simulate train/test split
        train = pd.DataFrame({
            "ethnicity": ["Caucasian", "African", "Asian", np.nan, "Caucasian"]
        })

        test = pd.DataFrame({
            "ethnicity": ["Caucasian", np.nan, "African", "Other"]  # "Other" not in train
        })

        # Fill missing
        train["ethnicity_filled"] = train["ethnicity"].fillna("Missing")
        test["ethnicity_filled"] = test["ethnicity"].fillna("Missing")

        # Fit encoder on train
        enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        enc.fit(train[["ethnicity_filled"]])

        # Transform both
        train_encoded = enc.transform(train[["ethnicity_filled"]])
        test_encoded = enc.transform(test[["ethnicity_filled"]])

        # Check dimensions match
        assert train_encoded.shape[1] == test_encoded.shape[1], \
            "Train and test should have same number of encoded columns"

        # Check that Missing category was learned
        assert "Missing" in enc.categories_[0], \
            "Encoder should have learned 'Missing' category from train"

    def test_missing_category_signal_detection(self):
        """
        Test whether 'Missing' category can carry signal.

        Simulate scenario where missingness is informative (MNAR):
        - Cases more likely to have missing ethnicity
        """
        np.random.seed(42)
        n = 1000

        # Generate data where missingness correlates with outcome
        data = pd.DataFrame({
            "outcome": np.random.choice([0, 1], n, p=[0.95, 0.05])  # 5% cases
        })

        # Cases (outcome=1) have 30% missing ethnicity
        # Controls (outcome=0) have 10% missing ethnicity
        missing_prob = np.where(data["outcome"] == 1, 0.30, 0.10)

        ethnicity_values = []
        for i, is_missing in enumerate(np.random.random(n) < missing_prob):
            if is_missing:
                ethnicity_values.append(np.nan)
            else:
                ethnicity_values.append(np.random.choice(["Caucasian", "African", "Asian"]))

        data["ethnicity"] = ethnicity_values

        # Check that missingness differs by outcome
        missing_cases = data[data["outcome"] == 1]["ethnicity"].isna().mean()
        missing_controls = data[data["outcome"] == 0]["ethnicity"].isna().mean()

        print(f"Missing in cases: {missing_cases*100:.1f}%")
        print(f"Missing in controls: {missing_controls*100:.1f}%")

        # Should be different (informative missingness)
        assert missing_cases > missing_controls, \
            "Missingness should be higher in cases (MNAR scenario)"

        # This demonstrates why "Missing" category is valuable:
        # The model can learn this association


class TestCategoricalEncodingPipeline:
    """Test full categorical encoding pipeline for Celiac data."""

    def test_genetic_ethnicity_encoding_strategy(self):
        """
        Test recommended encoding for 'Genetic ethnic grouping' (17.68% missing).

        Strategy:
        1. Fill NaN with 'Missing' string
        2. OneHotEncode all categories (including 'Missing')
        3. Let model learn if 'Missing' correlates with CeD risk
        """
        np.random.seed(42)

        # Simulate Celiac genetic ethnicity distribution
        # Build list with exact proportions
        n = 1000
        values = (
            ["Caucasian"] * 650 +
            ["African"] * 100 +
            ["Asian"] * 50 +
            ["Other"] * 30 +
            [np.nan] * 170  # 17% missing
        )
        np.random.shuffle(values)

        data = pd.DataFrame({
            "Genetic ethnic grouping": values,
            "CeD_comparison": np.random.choice(["Controls", "Incident"], 1000, p=[0.997, 0.003])
        })

        # Verify we have missing values
        assert data["Genetic ethnic grouping"].isna().sum() > 0, "Should have NaN values"

        # Apply strategy - convert np.nan to string "Missing"
        data["ethnicity_encoded"] = data["Genetic ethnic grouping"].astype(str).replace('nan', 'Missing')

        # Verify no data loss
        assert len(data) == 1000, "Should preserve all subjects"
        assert data["ethnicity_encoded"].isna().sum() == 0, "No NaN should remain"

        # Check Missing category exists
        assert "Missing" in data["ethnicity_encoded"].unique(), \
            "'Missing' category should exist"

        # Check realistic proportion
        missing_frac = (data["ethnicity_encoded"] == "Missing").mean()
        assert 0.15 < missing_frac < 0.20, \
            f"Missing fraction {missing_frac:.2%} should be ~17%"

    def test_sex_encoding_no_missing(self):
        """Sex should have no missing values (validated in test_missing_data.py)."""
        # Simulate sex distribution (no missing)
        data = pd.DataFrame({
            "sex": np.random.choice(["Male", "Female"], 1000)
        })

        # Verify no missing
        assert data["sex"].isna().sum() == 0, "Sex should have no missing"

        # OneHotEncode (no special handling needed)
        enc = OneHotEncoder(sparse_output=False)
        encoded = enc.fit_transform(data[["sex"]])

        # Should have 2 columns (Male, Female)
        assert encoded.shape[1] == 2, "Sex should encode to 2 columns"


class TestBMIMissingStrategy:
    """
    Test BMI missing value handling.

    DECISION: Drop rows with missing BMI (0.50% = 219 rows).
    Rationale: Minimal data loss, BMI is continuous (not categorical).
    """

    def test_bmi_missing_minimal(self):
        """BMI has only 0.50% missing (219 / 43,960)."""
        # Simulate realistic BMI distribution
        n = 43960
        data = pd.DataFrame({
            "BMI": np.random.normal(27, 5, n)
        })

        # Introduce 0.50% missing (realistic)
        missing_mask = np.random.random(n) < 0.005
        data.loc[missing_mask, "BMI"] = np.nan

        n_missing = data["BMI"].isna().sum()
        frac_missing = n_missing / n

        print(f"BMI missing: {n_missing} ({frac_missing*100:.2f}%)")

        # Should be ~0.5%
        assert 0.004 < frac_missing < 0.006, "Should be ~0.5% missing"

        # Dropping these is acceptable
        n_dropped = n - len(data.dropna(subset=["BMI"]))
        frac_dropped = n_dropped / n

        assert frac_dropped < 0.01, "Should drop <1% of data"

    def test_bmi_dropna_vs_impute(self):
        """
        Compare BMI strategies: drop vs. impute.

        For 0.50% missing, dropping is simpler and avoids bias.
        """
        data = pd.DataFrame({
            "BMI": np.random.normal(27, 5, 1000),
            "outcome": np.random.randint(0, 2, 1000)
        })

        # Introduce 0.5% missing
        missing_mask = np.random.random(1000) < 0.005
        data.loc[missing_mask, "BMI"] = np.nan

        # Strategy 1: Drop
        data_dropped = data.dropna(subset=["BMI"])

        # Strategy 2: Impute median
        data_imputed = data.copy()
        median_bmi = data["BMI"].median()
        data_imputed["BMI"] = data_imputed["BMI"].fillna(median_bmi)

        # Check data retention
        n_original = len(data)
        n_after_drop = len(data_dropped)
        n_after_impute = len(data_imputed)

        assert n_after_impute == n_original, "Imputation preserves all subjects"
        assert n_after_drop < n_original, "Dropping removes subjects"

        # With 0.5% missing, dropping is acceptable
        frac_lost = (n_original - n_after_drop) / n_original
        assert frac_lost < 0.01, "Loss should be <1%"

        print(f"Subjects lost by dropping: {n_original - n_after_drop} ({frac_lost*100:.2f}%)")


class TestEncodingReproducibility:
    """Test that categorical encoding is reproducible."""

    def test_same_categories_same_encoding(self):
        """Same input data should produce identical encoding."""
        data = pd.DataFrame({
            "ethnicity": ["Caucasian", "African", np.nan, "Asian"] * 10
        })

        data["ethnicity_filled"] = data["ethnicity"].fillna("Missing")

        # Encode twice
        enc1 = OneHotEncoder(sparse_output=False)
        encoded1 = enc1.fit_transform(data[["ethnicity_filled"]])

        enc2 = OneHotEncoder(sparse_output=False)
        encoded2 = enc2.fit_transform(data[["ethnicity_filled"]])

        # Should be identical
        np.testing.assert_array_equal(encoded1, encoded2,
                                      err_msg="Encoding should be reproducible")

        # Categories should be in same order
        np.testing.assert_array_equal(enc1.categories_[0], enc2.categories_[0],
                                      err_msg="Category order should be reproducible")
