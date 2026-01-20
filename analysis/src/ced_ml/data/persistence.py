"""Split validation and persistence utilities.

This module handles saving split indices to CSV files, generating split metadata
(JSON), and validating split integrity before persistence.

Design:
    - save_split_indices(): Save TRAIN/VAL/TEST indices to CSV
    - save_split_metadata(): Generate and save JSON metadata
    - save_holdout_indices(): Save holdout set indices
    - check_split_files_exist(): Check for existing split files
    - validate_split_indices(): Validate split integrity (no overlap, coverage)

Behavioral equivalence:
    - Matches save_splits.py CSV/JSON output format exactly
    - Preserves index sorting behavior (ascending)
    - Maintains metadata schema compatibility with celiacML_faith.py
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .splits import compute_split_id

# ============================================================================
# Split Validation
# ============================================================================


def validate_split_indices(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    val_idx: Optional[np.ndarray] = None,
    total_samples: Optional[int] = None,
) -> Tuple[bool, str]:
    """Validate split indices for integrity.

    Args:
        train_idx: Training set indices
        test_idx: Test set indices
        val_idx: Validation set indices (optional)
        total_samples: Total number of samples in dataset (for coverage check)

    Returns:
        Tuple of (is_valid, error_message). error_message is empty string if valid.

    Validation checks:
        1. No overlap between TRAIN/VAL/TEST
        2. All indices are non-negative integers
        3. Optional: All indices < total_samples
        4. Optional: Full coverage (union covers all samples)
    """
    # Check for empty arrays
    if len(train_idx) == 0:
        return False, "TRAIN set is empty"
    if len(test_idx) == 0:
        return False, "TEST set is empty"

    # Collect all splits
    splits = {"train": train_idx, "test": test_idx}
    if val_idx is not None and len(val_idx) > 0:
        splits["val"] = val_idx

    # Check data types and non-negative
    for name, idx in splits.items():
        if not np.issubdtype(idx.dtype, np.integer):
            return False, f"{name.upper()} indices must be integers, got {idx.dtype}"
        if np.any(idx < 0):
            return False, f"{name.upper()} contains negative indices"

    # Check for overlaps
    if val_idx is not None and len(val_idx) > 0:
        train_set = set(train_idx)
        val_set = set(val_idx)
        test_set = set(test_idx)

        train_val_overlap = train_set & val_set
        train_test_overlap = train_set & test_set
        val_test_overlap = val_set & test_set

        if train_val_overlap:
            return False, f"TRAIN/VAL overlap: {len(train_val_overlap)} samples"
        if train_test_overlap:
            return False, f"TRAIN/TEST overlap: {len(train_test_overlap)} samples"
        if val_test_overlap:
            return False, f"VAL/TEST overlap: {len(val_test_overlap)} samples"
    else:
        # Two-way split
        train_set = set(train_idx)
        test_set = set(test_idx)
        overlap = train_set & test_set
        if overlap:
            return False, f"TRAIN/TEST overlap: {len(overlap)} samples"

    # Check bounds if total_samples provided
    if total_samples is not None:
        for name, idx in splits.items():
            if np.any(idx >= total_samples):
                bad_count = np.sum(idx >= total_samples)
                return (
                    False,
                    f"{name.upper()} contains {bad_count} indices >= {total_samples}",
                )

    return True, ""


def check_split_files_exist(
    outdir: str,
    scenario: str,
    seed: int,
    has_val: bool = False,
    n_splits: int = 1,
) -> Tuple[bool, List[str]]:
    """Check if split files already exist in output directory.

    Args:
        outdir: Output directory path
        scenario: Scenario name (e.g., "IncidentOnly")
        seed: Random seed
        has_val: Whether validation set is expected
        n_splits: Total number of splits (affects suffix)

    Returns:
        Tuple of (files_exist, existing_paths)
    """
    suffix = f"_seed{seed if seed is not None else 0}"

    expected_files = [
        os.path.join(outdir, f"{scenario}_train_idx{suffix}.csv"),
        os.path.join(outdir, f"{scenario}_test_idx{suffix}.csv"),
    ]
    if has_val:
        expected_files.append(os.path.join(outdir, f"{scenario}_val_idx{suffix}.csv"))

    # Also check metadata
    expected_files.append(
        os.path.join(outdir, f"{scenario}_split_meta_seed{seed}.json")
    )

    existing = [f for f in expected_files if os.path.exists(f)]

    return len(existing) > 0, existing


# ============================================================================
# Index Persistence
# ============================================================================


def save_split_indices(
    outdir: str,
    scenario: str,
    seed: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    val_idx: Optional[np.ndarray] = None,
    n_splits: int = 1,
    overwrite: bool = False,
) -> Dict[str, str]:
    """Save train/val/test indices to CSV files.

    Args:
        outdir: Output directory path
        scenario: Scenario name (e.g., "IncidentOnly")
        seed: Random seed used for split
        train_idx: Training set indices (sorted ascending)
        test_idx: Test set indices (sorted ascending)
        val_idx: Validation set indices (sorted ascending, optional)
        n_splits: Total number of splits (affects filename suffix)
        overwrite: Whether to overwrite existing files

    Returns:
        Dictionary mapping split name to saved file path

    Raises:
        FileExistsError: If files exist and overwrite=False
        ValueError: If indices are invalid

    Output format:
        - CSV with single column "idx" containing indices
        - Filenames: {scenario}_train_idx_seed{seed}.csv (if n_splits > 1)
                     {scenario}_train_idx.csv (if n_splits == 1)
    """
    # Validate indices
    is_valid, error_msg = validate_split_indices(train_idx, test_idx, val_idx)
    if not is_valid:
        raise ValueError(f"Invalid split indices: {error_msg}")

    # Check for existing files
    has_val = val_idx is not None and len(val_idx) > 0
    files_exist, existing = check_split_files_exist(
        outdir, scenario, seed, has_val, n_splits
    )
    if files_exist and not overwrite:
        raise FileExistsError(
            "Split files already exist. Use overwrite=True to replace:\n"
            + "\n".join(f"  {p}" for p in existing)
        )

    # Build file paths
    suffix = f"_seed{seed if seed is not None else 0}"
    paths = {
        "train": os.path.join(outdir, f"{scenario}_train_idx{suffix}.csv"),
        "test": os.path.join(outdir, f"{scenario}_test_idx{suffix}.csv"),
    }
    if has_val:
        paths["val"] = os.path.join(outdir, f"{scenario}_val_idx{suffix}.csv")

    # Sort indices (match legacy behavior)
    train_idx = np.sort(train_idx.astype(int))
    test_idx = np.sort(test_idx.astype(int))
    if has_val:
        val_idx = np.sort(val_idx.astype(int))

    # Save CSVs
    os.makedirs(outdir, exist_ok=True)

    pd.DataFrame({"idx": train_idx}).to_csv(paths["train"], index=False)
    pd.DataFrame({"idx": test_idx}).to_csv(paths["test"], index=False)
    if has_val:
        pd.DataFrame({"idx": val_idx}).to_csv(paths["val"], index=False)

    return paths


def save_holdout_indices(
    outdir: str,
    scenario: str,
    holdout_idx: np.ndarray,
    overwrite: bool = False,
) -> str:
    """Save holdout set indices to CSV.

    Args:
        outdir: Output directory path
        scenario: Scenario name
        holdout_idx: Holdout set indices (sorted ascending)
        overwrite: Whether to overwrite existing file

    Returns:
        Path to saved CSV file

    Raises:
        FileExistsError: If file exists and overwrite=False
    """
    holdout_path = os.path.join(outdir, f"{scenario}_HOLDOUT_idx.csv")

    if os.path.exists(holdout_path) and not overwrite:
        raise FileExistsError(
            f"Holdout file already exists: {holdout_path}\n"
            f"Use overwrite=True to replace."
        )

    os.makedirs(outdir, exist_ok=True)

    # Sort indices (match legacy behavior)
    holdout_idx = np.sort(holdout_idx.astype(int))

    pd.DataFrame({"idx": holdout_idx}).to_csv(holdout_path, index=False)

    return holdout_path


# ============================================================================
# Metadata Persistence
# ============================================================================


def save_split_metadata(
    outdir: str,
    scenario: str,
    seed: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    val_idx: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    split_type: str = "development",
    strat_scheme: Optional[str] = None,
    row_filter_stats: Optional[Dict[str, Any]] = None,
    index_space: str = "full",
    temporal_split: bool = False,
    temporal_col: Optional[str] = None,
    temporal_train_end: Optional[str] = None,
    temporal_test_start: Optional[str] = None,
    temporal_test_end: Optional[str] = None,
) -> str:
    """Save split metadata to JSON file.

    Args:
        outdir: Output directory path
        scenario: Scenario name
        seed: Random seed used for split
        train_idx: Training set indices
        test_idx: Test set indices
        y_train: Training set labels
        y_test: Test set labels
        val_idx: Validation set indices (optional)
        y_val: Validation set labels (optional)
        split_type: "development" or "holdout"
        strat_scheme: Stratification scheme used (e.g., "outcome+sex+age3")
        row_filter_stats: Row filter statistics dict
        index_space: "full" or "dev" (index space reference)
        temporal_split: Whether temporal splitting was used
        temporal_col: Column used for temporal ordering
        temporal_train_end: Last temporal value in TRAIN
        temporal_test_start: First temporal value in TEST
        temporal_test_end: Last temporal value in TEST

    Returns:
        Path to saved JSON metadata file

    Metadata schema:
        {
            "scenario": str,
            "seed": int,
            "split_type": str,
            "index_space": str,
            "n_train": int,
            "n_test": int,
            "n_train_pos": int,
            "n_test_pos": int,
            "prevalence_train": float,
            "prevalence_test": float,
            "split_id_train": str,
            "split_id_test": str,
            "n_val": int (optional),
            "n_val_pos": int (optional),
            "prevalence_val": float (optional),
            "split_id_val": str (optional),
            "stratification_scheme": str (optional),
            "row_filters": dict (optional),
            "temporal_split": bool (optional),
            "temporal_col": str (optional),
            "temporal_train_end_value": str (optional),
            "temporal_test_start_value": str (optional),
            "temporal_test_end_value": str (optional)
        }
    """
    meta: Dict[str, Any] = {
        "scenario": scenario,
        "seed": int(seed),
        "split_type": split_type,
        "index_space": index_space,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_train_pos": int(y_train.sum()),
        "n_test_pos": int(y_test.sum()),
        "prevalence_train": float(y_train.mean()),
        "prevalence_test": float(y_test.mean()),
        "split_id_train": compute_split_id(train_idx),
        "split_id_test": compute_split_id(test_idx),
    }

    # Add validation set metadata if present
    if val_idx is not None and y_val is not None and len(val_idx) > 0:
        meta.update(
            {
                "n_val": int(len(val_idx)),
                "n_val_pos": int(y_val.sum()),
                "prevalence_val": float(y_val.mean()),
                "split_id_val": compute_split_id(val_idx),
            }
        )

    # Add stratification scheme if provided
    if strat_scheme is not None:
        meta["stratification_scheme"] = strat_scheme

    # Add row filter stats if provided
    if row_filter_stats is not None:
        meta["row_filters"] = row_filter_stats

    # Add temporal metadata if applicable
    if temporal_split:
        meta["temporal_split"] = True
        if temporal_col is not None:
            meta["temporal_col"] = temporal_col
        if temporal_train_end is not None:
            meta["temporal_train_end_value"] = temporal_train_end
        if temporal_test_start is not None:
            meta["temporal_test_start_value"] = temporal_test_start
        if temporal_test_end is not None:
            meta["temporal_test_end_value"] = temporal_test_end

    # Save to file
    meta_path = os.path.join(outdir, f"{scenario}_split_meta_seed{seed}.json")
    os.makedirs(outdir, exist_ok=True)

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return meta_path


def save_holdout_metadata(
    outdir: str,
    scenario: str,
    holdout_idx: np.ndarray,
    y_holdout: np.ndarray,
    strat_scheme: Optional[str] = None,
    row_filter_stats: Optional[Dict[str, Any]] = None,
    warning: Optional[str] = None,
    temporal_split: bool = False,
    temporal_col: Optional[str] = None,
    temporal_start: Optional[str] = None,
    temporal_end: Optional[str] = None,
) -> str:
    """Save holdout set metadata to JSON file.

    Args:
        outdir: Output directory path
        scenario: Scenario name
        holdout_idx: Holdout set indices
        y_holdout: Holdout set labels
        strat_scheme: Stratification scheme used
        row_filter_stats: Row filter statistics dict
        warning: Optional warning message (e.g., reverse causality)
        temporal_split: Whether temporal splitting was used
        temporal_col: Column used for temporal ordering
        temporal_start: First temporal value in HOLDOUT
        temporal_end: Last temporal value in HOLDOUT

    Returns:
        Path to saved JSON metadata file
    """
    meta: Dict[str, Any] = {
        "scenario": scenario,
        "split_type": "holdout",
        "seed": 42,  # Fixed seed for holdout
        "n_holdout": int(len(holdout_idx)),
        "n_holdout_pos": int(y_holdout.sum()),
        "prevalence_holdout": float(y_holdout.mean()),
        "split_id_holdout": compute_split_id(holdout_idx),
        "index_space": "full",
        "note": "NEVER use this set during development. Final evaluation only.",
    }

    if strat_scheme is not None:
        meta["stratification_scheme"] = strat_scheme

    if row_filter_stats is not None:
        meta["row_filters"] = row_filter_stats

    if warning is not None:
        meta["reverse_causality_warning"] = warning

    if temporal_split:
        meta["temporal_split"] = True
        if temporal_col is not None:
            meta["temporal_col"] = temporal_col
        if temporal_start is not None:
            meta["temporal_start_value"] = temporal_start
        if temporal_end is not None:
            meta["temporal_end_value"] = temporal_end

    meta_path = os.path.join(outdir, f"{scenario}_HOLDOUT_meta.json")
    os.makedirs(outdir, exist_ok=True)

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return meta_path
