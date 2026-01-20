"""
Data I/O utilities for CeliacRiskML pipeline.

This module handles reading proteomics CSV files with schema validation,
dtype coercion, and quality checks.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from ced_ml.data.schema import (
    CAT_COLS,
    CED_DATE_COL,
    ID_COL,
    META_NUM_COLS,
    TARGET_COL,
)
from ced_ml.utils.logging import get_logger

logger = get_logger(__name__)


def usecols_for_proteomics(
    numeric_metadata: Optional[List[str]] = None,
    categorical_metadata: Optional[List[str]] = None,
) -> Callable[[str], bool]:
    """
    Create column filter function for pd.read_csv(usecols=...).

    Returns columns needed for modeling:
    - ID column (eid)
    - Target column (CeD_comparison)
    - Date column (CeD_date)
    - Numeric metadata (defaults to schema META_NUM_COLS if not specified)
    - Categorical metadata (defaults to schema CAT_COLS if not specified)
    - Protein features (*_resid columns)

    Args:
        numeric_metadata: Numeric metadata columns to include (default: META_NUM_COLS)
        categorical_metadata: Categorical metadata columns to include (default: CAT_COLS)

    Returns:
        Function that takes column name and returns True if column should be loaded
    """
    num_cols = numeric_metadata if numeric_metadata is not None else META_NUM_COLS
    cat_cols = categorical_metadata if categorical_metadata is not None else CAT_COLS

    def _filter(col: str) -> bool:
        if col in (ID_COL, TARGET_COL, CED_DATE_COL):
            return True
        if col in num_cols:
            return True
        if col in cat_cols:
            return True
        if isinstance(col, str) and col.endswith("_resid"):
            return True
        return False

    return _filter


def read_proteomics_csv(
    filepath: str,
    *,
    usecols: Optional[Callable[[str], bool]] = None,
    low_memory: bool = False,
    validate: bool = True,
) -> pd.DataFrame:
    """
    Read proteomics CSV file with schema-aware column filtering.

    Args:
        filepath: Path to CSV file
        usecols: Optional column filter function (default: usecols_for_proteomics())
        low_memory: Whether to use low_memory mode for pd.read_csv (default: False)
        validate: Whether to validate required columns after loading (default: True)

    Returns:
        DataFrame with selected columns

    Raises:
        FileNotFoundError: If filepath does not exist
        ValueError: If validate=True and required columns are missing

    Example:
        >>> df = read_proteomics_csv("data/celiac_proteomics.csv")
        >>> assert "eid" in df.columns
        >>> assert "CeD_comparison" in df.columns
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if usecols is None:
        usecols = usecols_for_proteomics()

    logger.info(f"Reading CSV: {filepath}")
    df = pd.read_csv(filepath, usecols=usecols, low_memory=low_memory)
    logger.info(f"Loaded {len(df):,} rows × {len(df.columns):,} columns")

    if validate:
        validate_required_columns(df)

    return df


def validate_required_columns(df: pd.DataFrame) -> None:
    """
    Validate that required columns are present in DataFrame.

    Required columns:
    - ID_COL (eid)
    - TARGET_COL (CeD_comparison)

    Args:
        df: DataFrame to validate

    Raises:
        ValueError: If required columns are missing
    """
    required = [ID_COL, TARGET_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Required columns missing: {missing}. " f"Available columns: {list(df.columns)}"
        )
    logger.debug(f"Validated required columns: {required}")


def coerce_numeric_columns(
    df: pd.DataFrame, columns: list[str], inplace: bool = False
) -> pd.DataFrame:
    """
    Coerce columns to numeric dtype, converting errors to NaN.

    Note: pd.to_numeric() will return int64 if all values are integers,
    and float64 if there are floats or NaN values. This matches pandas behavior.

    Args:
        df: DataFrame to process
        columns: List of column names to coerce
        inplace: Whether to modify df in place (default: False)

    Returns:
        DataFrame with coerced columns (int64 or float64 depending on values)

    Example:
        >>> df = pd.DataFrame({"age": ["25", "30", "invalid"]})
        >>> result = coerce_numeric_columns(df, ["age"])
        >>> assert pd.api.types.is_numeric_dtype(result["age"])
        >>> assert pd.isna(result.loc[2, "age"])
    """
    if not inplace:
        df = df.copy()

    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found, skipping coercion")
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        logger.debug(f"Coerced column '{col}' to numeric")

    return df


def fill_missing_categorical(
    df: pd.DataFrame, columns: list[str], fill_value: str = "Missing", inplace: bool = False
) -> pd.DataFrame:
    """
    Fill missing values in categorical columns with explicit category.

    Strategy: Convert to string and replace NaN/None representations with fill_value.
    This is more robust than .fillna() for pandas string dtypes.

    Args:
        df: DataFrame to process
        columns: List of categorical column names
        fill_value: Value to use for missing data (default: "Missing")
        inplace: Whether to modify df in place (default: False)

    Returns:
        DataFrame with missing values filled

    Example:
        >>> df = pd.DataFrame({"sex": ["Male", None, "Female"]})
        >>> result = fill_missing_categorical(df, ["sex"])
        >>> assert result.loc[1, "sex"] == "Missing"
    """
    if not inplace:
        df = df.copy()

    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found, skipping missing fill")
            continue
        # Convert to string and replace nan/None representations
        df[col] = df[col].astype(str).replace(["nan", "None"], fill_value)
        logger.debug(f"Filled missing values in '{col}' with '{fill_value}'")

    return df


def identify_protein_columns(df: pd.DataFrame) -> list[str]:
    """
    Identify protein feature columns (*_resid suffix).

    Args:
        df: DataFrame to scan

    Returns:
        List of protein column names (sorted)

    Raises:
        ValueError: If no protein columns found

    Example:
        >>> df = pd.DataFrame({"age": [25], "APOE_resid": [0.5], "IL6_resid": [1.2]})
        >>> proteins = identify_protein_columns(df)
        >>> assert proteins == ["APOE_resid", "IL6_resid"]
    """
    protein_cols = sorted([c for c in df.columns if isinstance(c, str) and c.endswith("_resid")])
    if not protein_cols:
        raise ValueError(
            "No protein columns (*_resid) found. " "Check column naming or usecols filter."
        )
    logger.info(f"Identified {len(protein_cols):,} protein columns")
    return protein_cols


def get_data_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute summary statistics for loaded data.

    Args:
        df: DataFrame to summarize

    Returns:
        Dictionary with:
        - n_rows: Number of rows
        - n_cols: Number of columns
        - n_proteins: Number of protein columns
        - outcome_counts: Counts by TARGET_COL
        - missing_metadata: Missing value counts for metadata columns

    Example:
        >>> df = pd.DataFrame({"eid": [1, 2], "CeD_comparison": ["Controls", "Incident"]})
        >>> stats = get_data_stats(df)
        >>> assert stats["n_rows"] == 2
    """
    stats = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
    }

    # Protein count
    try:
        protein_cols = identify_protein_columns(df)
        stats["n_proteins"] = len(protein_cols)
    except ValueError:
        stats["n_proteins"] = 0

    # Outcome distribution
    if TARGET_COL in df.columns:
        stats["outcome_counts"] = df[TARGET_COL].value_counts().to_dict()

    # Missing metadata
    metadata_cols = META_NUM_COLS + CAT_COLS
    missing = {}
    for col in metadata_cols:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                missing[col] = int(n_missing)
    if missing:
        stats["missing_metadata"] = missing

    return stats


def log_data_summary(df: pd.DataFrame) -> None:
    """
    Log summary of loaded data to logger.

    Args:
        df: DataFrame to summarize
    """
    stats = get_data_stats(df)
    logger.info("Data summary:")
    logger.info(f"  Rows: {stats['n_rows']:,}")
    logger.info(f"  Columns: {stats['n_cols']:,}")
    logger.info(f"  Proteins: {stats.get('n_proteins', 0):,}")

    if "outcome_counts" in stats:
        logger.info("  Outcome distribution:")
        for label, count in sorted(stats["outcome_counts"].items()):
            logger.info(f"    {label}: {count:,}")

    if "missing_metadata" in stats:
        logger.info("  Missing metadata:")
        for col, count in stats["missing_metadata"].items():
            logger.info(f"    {col}: {count:,} ({100*count/stats['n_rows']:.1f}%)")


# Convenience function matching legacy behavior
def load_data(infile: str) -> pd.DataFrame:
    """
    Load proteomics CSV with default settings (backward compatibility).

    This function matches the behavior of legacy code:
    - Uses usecols filter for proteomics schema
    - Sets low_memory=False for consistent dtype inference

    Args:
        infile: Path to CSV file

    Returns:
        DataFrame with selected columns

    Example:
        >>> df = load_data("data/celiac_proteomics.csv")
    """
    return read_proteomics_csv(infile, low_memory=False)
