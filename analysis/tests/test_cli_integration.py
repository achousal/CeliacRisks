"""
Integration tests for CLI save-splits command.

Ensures CLI properly uses data layer modules (io, splits, persistence).
"""

import pytest
from pathlib import Path
import pandas as pd
import numpy as np


def test_cli_imports_data_modules():
    """Test that CLI imports from data layer modules."""
    from ced_ml.cli import save_splits

    # Verify imports from data.splits
    assert hasattr(save_splits, 'build_working_strata')
    assert hasattr(save_splits, 'stratified_train_val_test_split')
    assert hasattr(save_splits, 'downsample_controls')

    # Verify imports from data.persistence
    assert hasattr(save_splits, 'save_split_indices')
    assert hasattr(save_splits, 'save_holdout_indices')
    assert hasattr(save_splits, 'save_split_metadata')
    assert hasattr(save_splits, 'save_holdout_metadata')


def test_no_duplicate_save_metadata_function():
    """Test that CLI does not define its own save_split_metadata."""
    from ced_ml.cli import save_splits
    import inspect

    # Get all functions defined in the CLI module
    cli_functions = [
        name for name, obj in inspect.getmembers(save_splits, inspect.isfunction)
        if obj.__module__ == 'ced_ml.cli.save_splits'
    ]

    # Should NOT have save_split_metadata (imported from data.persistence)
    assert 'save_split_metadata' not in cli_functions, \
        "CLI should import save_split_metadata, not define it"

    # Should NOT have save_holdout_metadata (imported from data.persistence)
    assert 'save_holdout_metadata' not in cli_functions, \
        "CLI should import save_holdout_metadata, not define it"

    # Should NOT have save_split_indices (imported from data.persistence)
    assert 'save_split_indices' not in cli_functions, \
        "CLI should import save_split_indices, not define it"


def test_no_inline_csv_writing():
    """Test that CLI source code does not contain inline CSV writes."""
    from ced_ml.cli import save_splits
    import inspect

    # Get CLI source code
    source = inspect.getsource(save_splits)

    # Should NOT contain inline pd.DataFrame().to_csv() calls
    assert 'pd.DataFrame({"idx":' not in source, \
        "CLI should use save_split_indices(), not inline DataFrame.to_csv()"


def test_cli_imports_complete():
    """Test that all required persistence functions are imported."""
    from ced_ml.cli.save_splits import (
        save_split_indices,
        save_holdout_indices,
        save_split_metadata,
        save_holdout_metadata,
    )

    # Verify they are the correct functions from data.persistence
    from ced_ml.data.persistence import (
        save_split_indices as ref_split_indices,
        save_holdout_indices as ref_holdout_indices,
        save_split_metadata as ref_split_metadata,
        save_holdout_metadata as ref_holdout_metadata,
    )

    assert save_split_indices is ref_split_indices
    assert save_holdout_indices is ref_holdout_indices
    assert save_split_metadata is ref_split_metadata
    assert save_holdout_metadata is ref_holdout_metadata
