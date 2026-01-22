"""
Row filtering logic for data preprocessing.

Provides consistent row filtering across split generation and training to ensure
index alignment. Filters are applied before stratified splitting.
"""

from typing import Any

import pandas as pd

from .schema import CED_DATE_COL, CONTROL_LABEL, META_NUM_COLS, TARGET_COL


def apply_row_filters(
    df: pd.DataFrame,
    drop_uncertain_controls: bool = True,
    dropna_meta_num: bool = True,
    meta_num_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Apply row filters consistently across split generation and training.

    CRITICAL: This function must be used by BOTH save_splits.py and training scripts
    to ensure index alignment. If you modify filtering logic here, the same changes
    apply everywhere automatically.

    Filters applied:
    1. drop_uncertain_controls: Remove Controls that have a CeD_date present
       (these are ambiguous - possibly undiagnosed cases)
    2. dropna_meta_num: Remove rows missing age or BMI
       (required for modeling)

    Args:
        df: DataFrame with CeD_comparison, CeD_date, and metadata columns
        drop_uncertain_controls: If True, drop Controls with CeD_date present
        dropna_meta_num: If True, drop rows missing numeric metadata
        meta_num_cols: Numeric metadata columns to check (defaults to schema META_NUM_COLS)

    Returns:
        (filtered_df, stats_dict) where stats_dict contains filtering statistics:
            - n_in: Input row count
            - drop_uncertain_controls: Whether filter was applied
            - dropna_meta_num: Whether filter was applied
            - meta_num_cols_used: List of numeric metadata columns checked for NaN
            - n_removed_uncertain_controls: Rows removed by uncertain control filter
            - n_removed_dropna_meta_num: Rows removed by missing metadata filter
            - n_out: Output row count

    Example:
        >>> df_filtered, stats = apply_row_filters(df)
        >>> print(f"Removed {stats['n_removed_uncertain_controls']} uncertain controls")
        >>> print(f"Removed {stats['n_removed_dropna_meta_num']} rows with missing metadata")
    """
    # Determine which columns will be checked for NaN filtering
    cols_to_check = meta_num_cols if meta_num_cols is not None else META_NUM_COLS

    stats: dict[str, Any] = {
        "n_in": len(df),
        "drop_uncertain_controls": drop_uncertain_controls,
        "dropna_meta_num": dropna_meta_num,
        "meta_num_cols_used": list(cols_to_check),  # Record which columns were used
        "n_removed_uncertain_controls": 0,
        "n_removed_dropna_meta_num": 0,
        "n_out": 0,
    }

    df2 = df.copy()

    # Filter 1: Drop "uncertain controls" (Controls with CeD_date present)
    if drop_uncertain_controls and (CED_DATE_COL in df2.columns):
        mask_uncertain = (df2[TARGET_COL] == CONTROL_LABEL) & df2[CED_DATE_COL].notna()
        n_uncertain = int(mask_uncertain.sum())
        if n_uncertain > 0:
            df2 = df2.loc[~mask_uncertain].copy()
        stats["n_removed_uncertain_controls"] = n_uncertain

    # Filter 2: Drop rows missing required numeric metadata
    if dropna_meta_num:
        meta_present = [c for c in cols_to_check if c in df2.columns]
        if meta_present:
            n_before = len(df2)
            df2 = df2.dropna(subset=meta_present).copy()
            stats["n_removed_dropna_meta_num"] = n_before - len(df2)

    df2 = df2.reset_index(drop=True)
    stats["n_out"] = len(df2)

    return df2, stats
