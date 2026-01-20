"""
Column resolution utilities for flexible metadata handling.

Supports auto-detection of metadata columns from data or explicit specification.
"""

import logging
from dataclasses import dataclass
from typing import List, Union

import pandas as pd

from ..config.schema import ColumnsConfig
from .schema import CAT_COLS, META_NUM_COLS, get_protein_columns

logger = logging.getLogger(__name__)


@dataclass
class ResolvedColumns:
    """Container for resolved column sets."""

    protein_cols: List[str]
    numeric_metadata: List[str]
    categorical_metadata: List[str]

    @property
    def all_feature_cols(self) -> List[str]:
        """All feature columns (proteins + metadata)."""
        return self.protein_cols + self.numeric_metadata + self.categorical_metadata

    @property
    def all_metadata(self) -> List[str]:
        """All metadata columns."""
        return self.numeric_metadata + self.categorical_metadata


def resolve_columns(
    df_or_columns: Union[pd.DataFrame, List[str]],
    config: ColumnsConfig,
) -> ResolvedColumns:
    """
    Resolve column sets based on configuration and available data.

    Parameters
    ----------
    df_or_columns : pd.DataFrame or List[str]
        DataFrame or list of column names from the data.
    config : ColumnsConfig
        Column configuration specifying mode and explicit columns.

    Returns
    -------
    ResolvedColumns
        Resolved protein and metadata column sets.

    Raises
    ------
    ValueError
        If explicit mode is used but columns are not specified.
        If no protein columns are found.
    """
    # Extract column list
    if isinstance(df_or_columns, pd.DataFrame):
        available_cols = df_or_columns.columns.tolist()
    else:
        available_cols = df_or_columns

    available_set = set(available_cols)

    # Resolve protein columns (always auto-detected)
    protein_cols = get_protein_columns(available_cols)
    if not protein_cols:
        raise ValueError(
            "No protein columns found (expected columns ending with '_resid'). "
            f"Available columns: {available_cols[:10]}..."
        )

    # Resolve metadata columns
    if config.mode == "explicit":
        # Explicit mode: use specified columns
        if config.numeric_metadata is None and config.categorical_metadata is None:
            raise ValueError(
                "mode='explicit' requires at least one of numeric_metadata or "
                "categorical_metadata to be specified"
            )

        numeric_metadata = config.numeric_metadata or []
        categorical_metadata = config.categorical_metadata or []

        # Validate that specified columns exist
        specified = set(numeric_metadata + categorical_metadata)
        missing = specified - available_set
        if missing:
            raise ValueError(f"Specified metadata columns not found in data: {sorted(missing)}")

        logger.info(
            f"Explicit mode: using {len(numeric_metadata)} numeric + "
            f"{len(categorical_metadata)} categorical metadata columns"
        )

    else:
        # Auto mode: detect from schema defaults
        default_numeric = set(META_NUM_COLS)
        default_categorical = set(CAT_COLS)

        # Find which defaults exist in data
        numeric_metadata = sorted(default_numeric & available_set)
        categorical_metadata = sorted(default_categorical & available_set)

        # Warn about missing defaults if configured
        if config.warn_missing_defaults:
            missing_numeric = default_numeric - available_set
            missing_categorical = default_categorical - available_set

            if missing_numeric:
                logger.warning(
                    f"Auto mode: default numeric metadata not in data: {sorted(missing_numeric)}"
                )
            if missing_categorical:
                logger.warning(
                    f"Auto mode: default categorical metadata not in data: {sorted(missing_categorical)}"
                )

        logger.info(
            f"Auto mode: detected {len(numeric_metadata)} numeric + "
            f"{len(categorical_metadata)} categorical metadata columns from defaults"
        )

    return ResolvedColumns(
        protein_cols=protein_cols,
        numeric_metadata=numeric_metadata,
        categorical_metadata=categorical_metadata,
    )


def get_available_columns_from_file(filepath: str, nrows: int = 1) -> List[str]:
    """
    Read just the column names from a Parquet file without loading full data.

    Parameters
    ----------
    filepath : str
        Path to Parquet file.
    nrows : int
        Number of rows to read (default: 1, minimal overhead).

    Returns
    -------
    List[str]
        Column names from the file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is corrupted, incomplete, or not a valid Parquet file.
    """
    from pathlib import Path

    file_path = Path(filepath)

    # Check file exists
    if not file_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {filepath}\n"
            f"Ensure the file path is correct and accessible from the current environment."
        )

    # Check file size
    file_size = file_path.stat().st_size
    if file_size == 0:
        raise ValueError(
            f"Input file is empty (0 bytes): {filepath}\n"
            f"The file may not have been created properly or transfer may have failed."
        )

    # Parquet files have magic bytes 'PAR1' at start and end
    # Minimum valid Parquet file is ~200 bytes
    if file_size < 200:
        raise ValueError(
            f"Input file is suspiciously small ({file_size} bytes): {filepath}\n"
            f"Parquet files are typically at least 200 bytes. The file may be corrupted or incomplete."
        )

    try:
        # Try using PyArrow's lower-level API for better error messages
        import pyarrow.parquet as pq

        # This will validate the file structure
        parquet_file = pq.ParquetFile(filepath)
        schema = parquet_file.schema_arrow
        return schema.names

    except Exception as e:
        # Provide actionable diagnostics
        error_msg = (
            f"Failed to read Parquet file: {filepath}\n"
            f"File size: {file_size:,} bytes\n"
            f"Error: {str(e)}\n\n"
            f"Common causes on HPC:\n"
            f"  1. File transfer incomplete - verify full copy with: ls -lh {filepath}\n"
            f"  2. Network filesystem caching - try: cat {filepath} > /dev/null\n"
            f"  3. File corruption during write - regenerate the file\n"
            f"  4. Not a Parquet file - verify format with: file {filepath}\n\n"
            f"Diagnostic steps:\n"
            f"  1. Check file integrity: python -c \"import pyarrow.parquet as pq; pq.read_metadata('{filepath}')\"\n"
            f"  2. Check file size on source vs destination\n"
            f"  3. Try reading with: parquet-tools inspect {filepath}\n"
        )
        raise ValueError(error_msg) from e
