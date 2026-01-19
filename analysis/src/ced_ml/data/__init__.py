"""Data handling and schema definitions."""

from ced_ml.data.schema import (
    ID_COL,
    TARGET_COL,
    CED_DATE_COL,
    META_NUM_COLS,
    CAT_COLS,
    CONTROL_LABEL,
    INCIDENT_LABEL,
    PREVALENT_LABEL,
    SCENARIO_DEFINITIONS,
    get_protein_columns,
    get_scenario_labels,
    get_positive_label,
)

from ced_ml.data.persistence import (
    validate_split_indices,
    check_split_files_exist,
    save_split_indices,
    save_holdout_indices,
    save_split_metadata,
    save_holdout_metadata,
)

from ced_ml.data.filters import apply_row_filters

__all__ = [
    # Schema
    "ID_COL",
    "TARGET_COL",
    "CED_DATE_COL",
    "META_NUM_COLS",
    "CAT_COLS",
    "CONTROL_LABEL",
    "INCIDENT_LABEL",
    "PREVALENT_LABEL",
    "SCENARIO_DEFINITIONS",
    "get_protein_columns",
    "get_scenario_labels",
    "get_positive_label",
    # Persistence
    "validate_split_indices",
    "check_split_files_exist",
    "save_split_indices",
    "save_holdout_indices",
    "save_split_metadata",
    "save_holdout_metadata",
    # Filters
    "apply_row_filters",
]
