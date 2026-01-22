"""
Tests for data splitting module.
"""

import numpy as np
import pandas as pd
import pytest
from ced_ml.data.schema import TARGET_COL
from ced_ml.data.splits import (
    add_prevalent_to_train,
    age_bins,
    build_working_strata,
    collapse_rare_strata,
    compute_split_id,
    downsample_controls,
    make_strata,
    stratified_train_val_test_split,
    summarize_split,
    temporal_order_indices,
    temporal_train_val_test_split,
    validate_strata,
)

# ============================================================================
# Stratification Tests
# ============================================================================


class TestAgeBins:
    """Test age binning for stratification."""

    def test_age3_binning(self):
        """Should bin age into 3 groups."""
        age = pd.Series([25, 45, 65])
        result = age_bins(age, "age3")
        assert result.tolist() == ["young", "middle", "old"]

    def test_age2_binning(self):
        """Should bin age into 2 groups."""
        age = pd.Series([25, 65])
        result = age_bins(age, "age2")
        assert result.tolist() == ["lt60", "ge60"]

    def test_fill_missing_with_median(self):
        """Should fill missing age values with median."""
        age = pd.Series([25, None, 65])
        result = age_bins(age, "age2")
        # Median is 45, which should be in "lt60"
        assert result[1] == "lt60"

    def test_invalid_scheme_raises(self):
        """Should raise for unknown binning scheme."""
        age = pd.Series([25, 30])
        with pytest.raises(ValueError, match="Unknown age scheme"):
            age_bins(age, "invalid")


class TestMakeStrata:
    """Test stratification label creation."""

    def setup_method(self):
        """Create sample dataframe for tests."""
        self.df = pd.DataFrame(
            {
                TARGET_COL: ["Controls", "Incident", "Controls"],
                "sex": ["Male", "Female", "Male"],
                "age": [25, 45, 65],
            }
        )

    def test_outcome_only(self):
        """Should stratify by outcome only."""
        strata = make_strata(self.df, "outcome")
        assert strata.tolist() == ["Controls", "Incident", "Controls"]

    def test_outcome_sex(self):
        """Should stratify by outcome and sex."""
        strata = make_strata(self.df, "outcome+sex")
        expected = ["Controls_Male", "Incident_Female", "Controls_Male"]
        assert strata.tolist() == expected

    def test_outcome_age3(self):
        """Should stratify by outcome and age3."""
        strata = make_strata(self.df, "outcome+age3")
        expected = ["Controls_young", "Incident_middle", "Controls_old"]
        assert strata.tolist() == expected

    def test_outcome_sex_age3(self):
        """Should stratify by outcome, sex, and age3."""
        strata = make_strata(self.df, "outcome+sex+age3")
        expected = [
            "Controls_Male_young",
            "Incident_Female_middle",
            "Controls_Male_old",
        ]
        assert strata.tolist() == expected

    def test_missing_values_filled(self):
        """Should handle missing sex/outcome values."""
        df = pd.DataFrame(
            {
                TARGET_COL: [None, "Incident"],
                "sex": ["Male", None],
                "age": [25, 30],
            }
        )
        strata = make_strata(df, "outcome+sex")
        # Pandas converts None to "None" string when using .astype(str)
        assert "None" in strata[0] or "UnknownOutcome" in strata[0]
        assert "None" in strata[1] or "UnknownSex" in strata[1]

    def test_invalid_scheme_raises(self):
        """Should raise for unknown scheme."""
        with pytest.raises(ValueError, match="Unknown stratification scheme"):
            make_strata(self.df, "invalid")


class TestCollapseRareStrata:
    """Test rare strata collapsing."""

    def test_collapse_rare_strata(self):
        """Should collapse strata with < min_count samples."""
        df = pd.DataFrame(
            {
                TARGET_COL: ["Controls", "Controls", "Incident", "Controls"],
            }
        )
        strata = pd.Series(["A", "B", "C", "A"])  # B and C are rare (count=1)
        result = collapse_rare_strata(df, strata, min_count=2)
        assert result[0] == "A"  # Not rare
        assert result[1] == "Controls_RARE"  # Collapsed
        assert result[2] == "Incident_RARE"  # Collapsed
        assert result[3] == "A"  # Not rare

    def test_no_rare_strata(self):
        """Should return unchanged if no rare strata."""
        df = pd.DataFrame({TARGET_COL: ["Controls", "Controls"]})
        strata = pd.Series(["A", "A"])
        result = collapse_rare_strata(df, strata, min_count=2)
        assert result.tolist() == ["A", "A"]


class TestValidateStrata:
    """Test strata validation."""

    def test_valid_strata(self):
        """Should pass validation with sufficient samples."""
        strata = pd.Series(["A", "A", "B", "B"])
        is_valid, reason = validate_strata(strata)
        assert is_valid
        assert reason == "ok"

    def test_invalid_strata(self):
        """Should fail validation with singleton stratum."""
        strata = pd.Series(["A", "A", "B"])  # B has only 1 sample
        is_valid, reason = validate_strata(strata)
        assert not is_valid
        assert "min stratum count is 1" in reason


class TestBuildWorkingStrata:
    """Test robust stratification builder."""

    def test_selects_most_granular_valid_scheme(self):
        """Should select most granular scheme that passes validation."""
        # Create data where outcome+sex+age3 works
        df = pd.DataFrame(
            {
                TARGET_COL: ["Controls"] * 10 + ["Incident"] * 10,
                "sex": ["Male"] * 10 + ["Female"] * 10,
                "age": [25] * 10 + [65] * 10,
            }
        )
        strata, scheme = build_working_strata(df, min_count=2)
        # Should use outcome+sex+age3 (most granular)
        assert "outcome+sex+age3" in scheme or "outcome+sex+age2" in scheme

    def test_fallback_to_outcome_only(self):
        """Should fallback to simpler scheme when complex ones have rare strata."""
        # Create data where complex schemes produce rare strata
        df = pd.DataFrame(
            {
                TARGET_COL: ["Controls", "Controls", "Incident", "Incident"],
                "sex": [
                    "Male",
                    "Female",
                    "Male",
                    "Female",
                ],  # Each outcome+sex combo appears once
                "age": [25, 30, 35, 40],  # Different ages
            }
        )
        strata, scheme = build_working_strata(df, min_count=2)
        # Should use a simpler scheme (outcome-only or outcome+sex+age with collapsing)
        # The key test is that it doesn't crash and produces valid strata
        is_valid, _ = validate_strata(strata)
        assert is_valid


# ============================================================================
# Control Downsampling Tests
# ============================================================================


class TestDownsampleControls:
    """Test control downsampling."""

    def setup_method(self):
        """Create sample dataframe."""
        self.df = pd.DataFrame(
            {
                TARGET_COL: ["Controls"] * 10 + ["Incident"] * 2,
            }
        )
        self.rng = np.random.RandomState(42)

    def test_downsample_to_target_ratio(self):
        """Should downsample controls to target ratio."""
        idx = np.arange(12)
        result = downsample_controls(idx, self.df, ["Incident"], 2.0, self.rng)
        # 2 cases + 4 controls (2:1 ratio)
        assert len(result) == 6

    def test_keep_all_cases(self):
        """Should keep all case samples."""
        idx = np.arange(12)
        result = downsample_controls(idx, self.df, ["Incident"], 1.0, self.rng)
        cases = self.df.loc[result, TARGET_COL] == "Incident"
        assert cases.sum() == 2  # All cases preserved

    def test_no_downsampling_if_none(self):
        """Should skip downsampling if controls_per_case is None."""
        idx = np.arange(12)
        result = downsample_controls(idx, self.df, ["Incident"], None, self.rng)
        assert len(result) == 12  # All samples kept

    def test_no_downsampling_if_below_target(self):
        """Should keep all controls if already below target."""
        idx = np.arange(12)
        result = downsample_controls(idx, self.df, ["Incident"], 10.0, self.rng)
        assert len(result) == 12  # Can't reach 10:1 ratio, keep all

    def test_reproducible_with_seed(self):
        """Should produce same results with same seed."""
        idx = np.arange(12)
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)
        result1 = downsample_controls(idx, self.df, ["Incident"], 2.0, rng1)
        result2 = downsample_controls(idx, self.df, ["Incident"], 2.0, rng2)
        assert (result1 == result2).all()


# ============================================================================
# Temporal Ordering Tests
# ============================================================================


class TestTemporalOrderIndices:
    """Test temporal index ordering."""

    def test_datetime_column(self):
        """Should sort by datetime column."""
        dates = pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-02"])
        df = pd.DataFrame({"date": dates})
        indices = temporal_order_indices(df, "date")
        assert indices.tolist() == [1, 2, 0]  # Sorted by date

    def test_numeric_column(self):
        """Should sort by numeric column."""
        df = pd.DataFrame({"value": [30, 10, 20]})
        indices = temporal_order_indices(df, "value")
        assert indices.tolist() == [1, 2, 0]  # Sorted numerically

    def test_missing_values_first(self):
        """Should put missing values first (filled with min-1)."""
        df = pd.DataFrame({"value": [30, None, 20]})
        indices = temporal_order_indices(df, "value")
        assert indices[0] == 1  # Missing value first

    def test_missing_column_raises(self):
        """Should raise if column not found."""
        df = pd.DataFrame({"age": [25, 30]})
        with pytest.raises(ValueError, match="not found"):
            temporal_order_indices(df, "nonexistent")


# ============================================================================
# Prevalent Enrichment Tests
# ============================================================================


class TestAddPrevalentToTrain:
    """Test prevalent case enrichment."""

    def setup_method(self):
        """Create sample dataframe."""
        self.df = pd.DataFrame(
            {
                TARGET_COL: ["Controls"] * 5 + ["Incident"] * 2 + ["Prevalent"] * 4,
            }
        )
        self.rng = np.random.RandomState(42)

    def test_add_all_prevalent(self):
        """Should add all prevalent if frac=1.0."""
        train_idx = np.array([0, 1, 5, 6])  # Controls + Incident
        result = add_prevalent_to_train(train_idx, self.df, 1.0, self.rng)
        # Should add all 4 prevalent (indices 7-10)
        assert len(result) == 8

    def test_add_fraction_of_prevalent(self):
        """Should add fraction of prevalent cases."""
        train_idx = np.array([0, 1, 5, 6])
        result = add_prevalent_to_train(train_idx, self.df, 0.5, self.rng)
        # Should add ~2 prevalent
        assert len(result) == 6

    def test_add_no_prevalent(self):
        """Should add no prevalent if frac=0.0."""
        train_idx = np.array([0, 1, 5, 6])
        result = add_prevalent_to_train(train_idx, self.df, 0.0, self.rng)
        assert (result == train_idx).all()

    def test_reproducible_with_seed(self):
        """Should produce same results with same seed."""
        train_idx = np.array([0, 1, 5, 6])
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)
        result1 = add_prevalent_to_train(train_idx, self.df, 0.5, rng1)
        result2 = add_prevalent_to_train(train_idx, self.df, 0.5, rng2)
        assert (result1 == result2).all()


# ============================================================================
# Three-Way Split Tests
# ============================================================================


class TestStratifiedTrainValTestSplit:
    """Test stratified three-way splitting."""

    def setup_method(self):
        """Create sample data."""
        self.indices = np.arange(100)
        self.y = np.array([0] * 80 + [1] * 20)  # 20% positive - ensures min 2 cases per split
        self.strata = pd.Series(["A"] * 50 + ["B"] * 50)

    def test_three_way_split_sizes(self):
        """Should produce correct split sizes."""
        idx_tr, idx_val, idx_te, y_tr, y_val, y_te = stratified_train_val_test_split(
            self.indices,
            self.y,
            self.strata,
            val_size=0.25,
            test_size=0.25,
            random_state=42,
        )
        # 50% train, 25% val, 25% test
        assert len(idx_tr) == 50
        assert len(idx_val) == 25
        assert len(idx_te) == 25

    def test_two_way_split_no_val(self):
        """Should handle val_size=0."""
        idx_tr, idx_val, idx_te, y_tr, y_val, y_te = stratified_train_val_test_split(
            self.indices,
            self.y,
            self.strata,
            val_size=0.0,
            test_size=0.30,
            random_state=42,
        )
        assert len(idx_tr) == 70
        assert len(idx_val) == 0
        assert len(idx_te) == 30

    def test_reproducible_with_seed(self):
        """Should produce same splits with same seed."""
        result1 = stratified_train_val_test_split(
            self.indices,
            self.y,
            self.strata,
            val_size=0.25,
            test_size=0.25,
            random_state=42,
        )
        result2 = stratified_train_val_test_split(
            self.indices,
            self.y,
            self.strata,
            val_size=0.25,
            test_size=0.25,
            random_state=42,
        )
        assert (result1[0] == result2[0]).all()  # Same train indices


class TestTemporalTrainValTestSplit:
    """Test temporal three-way splitting."""

    def setup_method(self):
        """Create sample data."""
        self.indices = np.arange(100)  # Already temporally ordered
        # Distribute 20 cases evenly across time to ensure each temporal split gets enough
        # Pattern: 4 controls, 1 case repeated (ensures train/val/test all get cases)
        self.y = np.array([0, 0, 0, 0, 1] * 20)

    def test_three_way_temporal_split(self):
        """Should split chronologically."""
        idx_tr, idx_val, idx_te, y_tr, y_val, y_te = temporal_train_val_test_split(
            self.indices, self.y, val_size=0.25, test_size=0.25
        )
        # Train = earliest (0-49), Val = middle (50-74), Test = latest (75-99)
        assert idx_tr[0] < idx_val[0] < idx_te[0]
        assert len(idx_tr) == 50
        assert len(idx_val) == 25
        assert len(idx_te) == 25

    def test_two_way_temporal_split(self):
        """Should handle val_size=0."""
        idx_tr, idx_val, idx_te, y_tr, y_val, y_te = temporal_train_val_test_split(
            self.indices, self.y, val_size=0.0, test_size=0.30
        )
        assert len(idx_tr) == 70
        assert len(idx_val) == 0
        assert len(idx_te) == 30

    def test_clamping_prevents_empty_splits(self):
        """Should clamp values to prevent empty splits."""
        # With 50 samples, val=0.35 + test=0.45 would normally exceed 100%
        # But clamping should adjust to ensure reasonable splits
        # Distribute cases evenly across time to ensure each split gets min 2 cases
        small_idx = np.arange(50)
        small_y = np.array([0, 0, 1, 1] * 12 + [0, 0])  # 24 cases distributed evenly
        idx_tr, idx_val, idx_te, y_tr, y_val, y_te = temporal_train_val_test_split(
            small_idx, small_y, val_size=0.35, test_size=0.45
        )
        # Should clamp to ensure all splits have at least some samples
        assert len(idx_tr) >= 1
        assert len(idx_te) >= 1
        # Verify total = original
        assert len(idx_tr) + len(idx_val) + len(idx_te) == 50
        # Verify each split has min 2 cases (validation requirement)
        assert y_tr.sum() >= 2
        assert y_te.sum() >= 2

    def test_too_few_samples_raises(self):
        """Should raise if < 2 samples."""
        with pytest.raises(ValueError, match="at least 2 samples"):
            temporal_train_val_test_split(
                np.array([0]), np.array([0]), val_size=0.25, test_size=0.25
            )


# ============================================================================
# Utility Tests
# ============================================================================


class TestComputeSplitId:
    """Test split ID generation."""

    def test_deterministic_hash(self):
        """Should produce same hash for same indices."""
        idx = np.array([0, 1, 2, 3])
        hash1 = compute_split_id(idx)
        hash2 = compute_split_id(idx)
        assert hash1 == hash2

    def test_order_independent(self):
        """Should produce same hash regardless of input order."""
        idx1 = np.array([0, 1, 2, 3])
        idx2 = np.array([3, 2, 1, 0])
        assert compute_split_id(idx1) == compute_split_id(idx2)

    def test_different_indices_different_hash(self):
        """Should produce different hashes for different indices."""
        idx1 = np.array([0, 1, 2, 3])
        idx2 = np.array([0, 1, 2, 4])
        assert compute_split_id(idx1) != compute_split_id(idx2)

    def test_hash_length(self):
        """Should produce 12-character hash."""
        idx = np.array([0, 1, 2])
        hash_id = compute_split_id(idx)
        assert len(hash_id) == 12


class TestSummarizeSplit:
    """Test split summary generation."""

    def test_basic_summary(self):
        """Should compute basic split statistics."""
        idx_tr = np.array([0, 1, 2, 3])
        idx_val = np.array([4, 5])
        idx_te = np.array([6, 7])
        y_tr = np.array([0, 0, 1, 1])
        y_val = np.array([0, 1])
        y_te = np.array([0, 1])

        summary = summarize_split(idx_tr, idx_val, idx_te, y_tr, y_val, y_te)

        assert summary["n_train"] == 4
        assert summary["n_val"] == 2
        assert summary["n_test"] == 2
        assert summary["n_train_pos"] == 2
        assert summary["prevalence_train"] == 0.5

    def test_no_validation_set(self):
        """Should handle missing validation set."""
        idx_tr = np.array([0, 1])
        idx_val = np.array([])
        idx_te = np.array([2, 3])
        y_tr = np.array([0, 1])
        y_val = np.array([])
        y_te = np.array([0, 1])

        summary = summarize_split(idx_tr, idx_val, idx_te, y_tr, y_val, y_te)

        assert "n_val" not in summary
        assert "prevalence_val" not in summary
        assert summary["n_train"] == 2
