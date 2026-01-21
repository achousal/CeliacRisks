"""
Shared pytest fixtures for Celiac ML pipeline tests.

Provides:
- Synthetic dataset matching real schema
- Pre-generated splits for validation
- Temporary directories for output
"""

import os
import pytest
import numpy as np
import pandas as pd
from pathlib import Path


@pytest.fixture
def project_root():
    """Return path to project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def data_dir(project_root):
    """Return path to data directory."""
    return project_root / "data"


@pytest.fixture
def analysis_dir(project_root):
    """Return path to analysis scripts directory."""
    return project_root / "analysis"


@pytest.fixture
def synthetic_celiac_data(tmp_path):
    """
    Generate minimal realistic dataset matching Celiac schema.

    Returns:
        tuple: (DataFrame, output_path)

    Schema matches real data:
    - eid: participant ID
    - CeD_comparison: outcome (Controls, Incident, Prevalent)
    - age, BMI: continuous demographics
    - sex, Genetic ethnic grouping: categorical demographics
    - CeD_date: diagnosis date (for incident/prevalent)
    - *_resid: protein z-scores (100 proteins for speed)
    """
    np.random.seed(42)

    # Sample sizes (mimic real prevalence ~0.34%)
    n_controls = 1000
    n_incident = 5
    n_prevalent = 5
    n_total = n_controls + n_incident + n_prevalent

    # Generate base data
    data = {
        "eid": np.arange(1, n_total + 1),
        "CeD_comparison": (
            ["Controls"] * n_controls +
            ["Incident"] * n_incident +
            ["Prevalent"] * n_prevalent
        ),
        "age": np.random.normal(55, 10, n_total).clip(20, 80),
        "BMI": np.random.normal(27, 5, n_total).clip(18, 45),
        "sex": np.random.choice(["Male", "Female"], n_total),
        "Genetic ethnic grouping": np.random.choice(
            ["Caucasian", "African", "Asian", "Other"],
            n_total,
            p=[0.7, 0.15, 0.1, 0.05]
        ),
    }

    # Add CeD_date (only for cases, some controls as "uncertain")
    ced_dates = [pd.NaT] * n_controls
    # Add dates to incident cases
    ced_dates.extend([pd.Timestamp("2020-01-01")] * n_incident)
    # Add dates to prevalent cases
    ced_dates.extend([pd.Timestamp("2015-01-01")] * n_prevalent)

    # Add 3 "uncertain controls" (Controls with CeD_date - should be filtered)
    for i in [10, 50, 100]:
        ced_dates[i] = pd.Timestamp("2021-01-01")

    data["CeD_date"] = ced_dates

    df = pd.DataFrame(data)

    # Add 100 synthetic protein features (*_resid suffix)
    # Cases have elevated proteins (simulate real signal)
    n_proteins = 100
    for i in range(n_proteins):
        protein_name = f"protein_{i:03d}_resid"

        # Controls: N(0, 1)
        control_values = np.random.normal(0, 1, n_controls)

        # Incident cases: elevated (Cohen's d ~ 1.5 for top 10 proteins)
        if i < 10:  # Top 10 proteins are informative
            incident_values = np.random.normal(1.5, 1, n_incident)
            prevalent_values = np.random.normal(2.0, 1, n_prevalent)  # Even higher
        else:  # Rest are noise
            incident_values = np.random.normal(0, 1, n_incident)
            prevalent_values = np.random.normal(0, 1, n_prevalent)

        df[protein_name] = np.concatenate([
            control_values,
            incident_values,
            prevalent_values
        ])

    # Save to temp directory
    output_path = tmp_path / "synthetic_celiac_data.csv"
    df.to_csv(output_path, index=False)

    return df, output_path


@pytest.fixture
def synthetic_celiac_data_with_missing(tmp_path, synthetic_celiac_data):
    """
    Generate dataset WITH missing values for missing data tests.

    Returns:
        tuple: (DataFrame, output_path)
    """
    df_orig, _ = synthetic_celiac_data
    df = df_orig.copy()

    # Set seed for reproducibility
    np.random.seed(123)

    # Introduce missing values (5% random)
    n_rows = len(df)
    missing_mask_age = np.random.random(n_rows) < 0.05
    missing_mask_bmi = np.random.random(n_rows) < 0.05

    df.loc[missing_mask_age, "age"] = np.nan
    df.loc[missing_mask_bmi, "BMI"] = np.nan

    # Missing in some proteins
    protein_cols = [c for c in df.columns if c.endswith("_resid")]
    for col in protein_cols[:10]:  # First 10 proteins
        missing_mask = np.random.random(n_rows) < 0.02
        df.loc[missing_mask, col] = np.nan

    output_path = tmp_path / "synthetic_celiac_data_with_missing.csv"
    df.to_csv(output_path, index=False)

    return df, output_path


@pytest.fixture
def sample_splits(tmp_path, synthetic_celiac_data):
    """
    Generate sample train/test splits using save_splits.py logic.

    Returns:
        dict: {
            "train_idx": np.ndarray,
            "test_idx": np.ndarray,
            "holdout_idx": np.ndarray,
            "paths": dict of file paths
        }
    """
    df, _ = synthetic_celiac_data

    # Simulate IncidentOnly filtering
    mask_incident = df["CeD_comparison"].isin(["Controls", "Incident"])
    df_incident = df[mask_incident].copy()

    # Apply row filters (match save_splits.py logic)
    # Drop uncertain controls
    mask_uncertain = (
        (df_incident["CeD_comparison"] == "Controls") &
        df_incident["CeD_date"].notna()
    )
    df_incident = df_incident[~mask_uncertain].copy()

    # Drop rows with missing age/BMI
    df_incident = df_incident.dropna(subset=["age", "BMI"]).copy()
    df_incident = df_incident.reset_index(drop=True)

    n_total = len(df_incident)

    # Create simple splits
    from sklearn.model_selection import train_test_split

    y = (df_incident["CeD_comparison"] == "Incident").astype(int).values
    indices = np.arange(n_total)

    # Holdout split (30%)
    dev_idx, holdout_idx = train_test_split(
        indices, test_size=0.30, random_state=42, stratify=y
    )

    # Dev split (70% of dev -> train, 30% -> test)
    y_dev = y[dev_idx]
    train_idx, test_idx = train_test_split(
        dev_idx, test_size=0.30, random_state=0, stratify=y_dev
    )

    # Save to files
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir(exist_ok=True)

    paths = {
        "train": splits_dir / "IncidentOnly_train_idx.csv",
        "test": splits_dir / "IncidentOnly_test_idx.csv",
        "holdout": splits_dir / "IncidentOnly_HOLDOUT_idx.csv",
    }

    pd.DataFrame({"idx": train_idx}).to_csv(paths["train"], index=False)
    pd.DataFrame({"idx": test_idx}).to_csv(paths["test"], index=False)
    pd.DataFrame({"idx": holdout_idx}).to_csv(paths["holdout"], index=False)

    return {
        "train_idx": train_idx,
        "test_idx": test_idx,
        "holdout_idx": holdout_idx,
        "dev_idx": dev_idx,
        "full_idx": indices,
        "y": y,
        "paths": paths,
        "df": df_incident,
    }


@pytest.fixture
def real_data_path(data_dir):
    """
    Return path to real Celiac dataset if it exists.

    Tests that need real data should use:
        @pytest.mark.skipif(not real_data_exists(), reason="Real data not found")
    """
    path = data_dir / "Celiac_dataset_proteomics.csv"
    return path if path.exists() else None


def real_data_exists(data_dir=None):
    """Helper to check if real data exists."""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"
    path = data_dir / "Celiac_dataset_proteomics.csv"
    return path.exists()
