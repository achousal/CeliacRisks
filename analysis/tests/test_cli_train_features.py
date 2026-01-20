"""
Integration tests for CLI train command with features modules.

Ensures CLI properly imports and uses feature layer modules without duplication.
"""

import inspect


def test_cli_train_imports_screening_modules():
    """Test that CLI train imports from features.screening."""
    from ced_ml.cli import train

    # Verify imports from features.screening
    assert hasattr(train, "mann_whitney_screen")
    assert hasattr(train, "f_statistic_screen")
    assert hasattr(train, "variance_missingness_prefilter")
    assert hasattr(train, "screen_proteins")


def test_cli_train_imports_kbest_modules():
    """Test that CLI train imports from features.kbest."""
    from ced_ml.cli import train

    # Verify imports from features.kbest
    assert hasattr(train, "select_kbest_features")
    assert hasattr(train, "compute_f_classif_scores")
    assert hasattr(train, "extract_selected_proteins_from_kbest")
    assert hasattr(train, "rank_features_by_score")


def test_cli_train_imports_stability_modules():
    """Test that CLI train imports from features.stability."""
    from ced_ml.cli import train

    # Verify imports from features.stability
    assert hasattr(train, "compute_selection_frequencies")
    assert hasattr(train, "extract_stable_panel")
    assert hasattr(train, "build_frequency_panel")
    assert hasattr(train, "rank_proteins_by_frequency")


def test_no_duplicate_mann_whitney_function():
    """Test that CLI does not define its own mann_whitney_screen."""
    from ced_ml.cli import train

    # Get all functions defined in the CLI module
    cli_functions = [
        name
        for name, obj in inspect.getmembers(train, inspect.isfunction)
        if obj.__module__ == "ced_ml.cli.train"
    ]

    # Should NOT have mann_whitney_screen (imported from features.screening)
    assert (
        "mann_whitney_screen" not in cli_functions
    ), "CLI should import mann_whitney_screen, not define it"

    # Should NOT have f_statistic_screen (imported from features.screening)
    assert (
        "f_statistic_screen" not in cli_functions
    ), "CLI should import f_statistic_screen, not define it"


def test_no_duplicate_kbest_function():
    """Test that CLI does not define its own select_kbest_features."""
    from ced_ml.cli import train

    # Get all functions defined in the CLI module
    cli_functions = [
        name
        for name, obj in inspect.getmembers(train, inspect.isfunction)
        if obj.__module__ == "ced_ml.cli.train"
    ]

    # Should NOT have select_kbest_features (imported from features.kbest)
    assert (
        "select_kbest_features" not in cli_functions
    ), "CLI should import select_kbest_features, not define it"

    # Should NOT have extract_selected_proteins_from_kbest
    assert (
        "extract_selected_proteins_from_kbest" not in cli_functions
    ), "CLI should import extract_selected_proteins_from_kbest, not define it"


def test_no_duplicate_stability_function():
    """Test that CLI does not define its own stability functions."""
    from ced_ml.cli import train

    # Get all functions defined in the CLI module
    cli_functions = [
        name
        for name, obj in inspect.getmembers(train, inspect.isfunction)
        if obj.__module__ == "ced_ml.cli.train"
    ]

    # Should NOT have compute_selection_frequencies (imported from features.stability)
    assert (
        "compute_selection_frequencies" not in cli_functions
    ), "CLI should import compute_selection_frequencies, not define it"

    # Should NOT have extract_stable_panel
    assert (
        "extract_stable_panel" not in cli_functions
    ), "CLI should import extract_stable_panel, not define it"


def test_no_inline_feature_selection():
    """Test that CLI source code does not contain inline feature selection."""
    from ced_ml.cli import train

    # Get CLI source code
    source = inspect.getsource(train)

    # Should NOT contain inline SelectKBest instantiation
    assert (
        "SelectKBest(" not in source
    ), "CLI should use select_kbest_features(), not inline SelectKBest"

    # Should NOT contain inline Mann-Whitney U test
    assert (
        "mannwhitneyu(" not in source
    ), "CLI should use mann_whitney_screen(), not inline mannwhitneyu"

    # Should NOT contain inline f_classif
    assert (
        "f_classif(" not in source
    ), "CLI should use compute_f_classif_scores(), not inline f_classif"


def test_cli_imports_complete():
    """Test that all required feature functions are imported correctly."""
    from ced_ml.cli.train import (
        compute_selection_frequencies,
        extract_selected_proteins_from_kbest,
        extract_stable_panel,
        f_statistic_screen,
        mann_whitney_screen,
        screen_proteins,
        select_kbest_features,
    )
    from ced_ml.features.kbest import (
        extract_selected_proteins_from_kbest as ref_extract,
    )
    from ced_ml.features.kbest import (
        select_kbest_features as ref_kbest,
    )
    from ced_ml.features.screening import (
        f_statistic_screen as ref_f_stat,
    )

    # Verify they are the correct functions from features modules
    from ced_ml.features.screening import (
        mann_whitney_screen as ref_mann_whitney,
    )
    from ced_ml.features.screening import (
        screen_proteins as ref_screen,
    )
    from ced_ml.features.stability import (
        compute_selection_frequencies as ref_freq,
    )
    from ced_ml.features.stability import (
        extract_stable_panel as ref_stable,
    )

    assert mann_whitney_screen is ref_mann_whitney
    assert f_statistic_screen is ref_f_stat
    assert screen_proteins is ref_screen
    assert select_kbest_features is ref_kbest
    assert extract_selected_proteins_from_kbest is ref_extract
    assert compute_selection_frequencies is ref_freq
    assert extract_stable_panel is ref_stable


def test_screening_module_identity():
    """Test that screening functions are from features.screening, not duplicated."""
    from ced_ml.cli.train import f_statistic_screen, mann_whitney_screen
    from ced_ml.features.screening import (
        f_statistic_screen as orig_f_stat,
    )
    from ced_ml.features.screening import (
        mann_whitney_screen as orig_mann_whitney,
    )

    # Should be the exact same function object (not a copy)
    assert (
        mann_whitney_screen is orig_mann_whitney
    ), "mann_whitney_screen should be imported, not duplicated"
    assert (
        f_statistic_screen is orig_f_stat
    ), "f_statistic_screen should be imported, not duplicated"


def test_kbest_module_identity():
    """Test that kbest functions are from features.kbest, not duplicated."""
    from ced_ml.cli.train import compute_f_classif_scores, select_kbest_features
    from ced_ml.features.kbest import (
        compute_f_classif_scores as orig_scores,
    )
    from ced_ml.features.kbest import (
        select_kbest_features as orig_kbest,
    )

    # Should be the exact same function object (not a copy)
    assert (
        select_kbest_features is orig_kbest
    ), "select_kbest_features should be imported, not duplicated"
    assert (
        compute_f_classif_scores is orig_scores
    ), "compute_f_classif_scores should be imported, not duplicated"


def test_stability_module_identity():
    """Test that stability functions are from features.stability, not duplicated."""
    from ced_ml.cli.train import (
        build_frequency_panel,
        compute_selection_frequencies,
        extract_stable_panel,
    )
    from ced_ml.features.stability import (
        build_frequency_panel as orig_panel,
    )
    from ced_ml.features.stability import (
        compute_selection_frequencies as orig_freq,
    )
    from ced_ml.features.stability import (
        extract_stable_panel as orig_stable,
    )

    # Should be the exact same function object (not a copy)
    assert (
        compute_selection_frequencies is orig_freq
    ), "compute_selection_frequencies should be imported, not duplicated"
    assert (
        extract_stable_panel is orig_stable
    ), "extract_stable_panel should be imported, not duplicated"
    assert (
        build_frequency_panel is orig_panel
    ), "build_frequency_panel should be imported, not duplicated"


def test_no_redundant_screening_code():
    """Test that CLI does not contain redundant screening logic."""
    from ced_ml.cli import train

    source = inspect.getsource(train)

    # Should not contain manual U-statistic computation
    redundant_patterns = [
        "ranksums(",
        "mannwhitneyu(",
        "u_statistic =",
        "rank_biserial",
    ]

    for pattern in redundant_patterns:
        assert (
            pattern not in source
        ), f"CLI should not contain inline '{pattern}' - use features.screening instead"


def test_no_redundant_kbest_code():
    """Test that CLI does not contain redundant K-best logic."""
    from ced_ml.cli import train

    source = inspect.getsource(train)

    # Should not contain manual K-best selection
    redundant_patterns = [
        "SelectKBest(",
        "f_classif(",
        ".get_support(",
        "scores_ =",
    ]

    for pattern in redundant_patterns:
        assert (
            pattern not in source
        ), f"CLI should not contain inline '{pattern}' - use features.kbest instead"


def test_no_redundant_stability_code():
    """Test that CLI does not contain redundant stability logic."""
    from ced_ml.cli import train

    source = inspect.getsource(train)

    # Should not contain manual frequency computation
    redundant_patterns = [
        "selection_count =",
        "value_counts(",
        "stability_threshold",
        "/ n_repeats",
    ]

    for pattern in redundant_patterns:
        assert (
            pattern not in source
        ), f"CLI should not contain inline '{pattern}' - use features.stability instead"


def test_features_modules_available():
    """Test that all features submodules are available via CLI."""
    from ced_ml.cli import train

    # Check all three feature submodules are imported
    screening_funcs = [
        "mann_whitney_screen",
        "f_statistic_screen",
        "variance_missingness_prefilter",
        "screen_proteins",
    ]
    kbest_funcs = [
        "select_kbest_features",
        "compute_f_classif_scores",
        "extract_selected_proteins_from_kbest",
        "rank_features_by_score",
    ]
    stability_funcs = [
        "compute_selection_frequencies",
        "extract_stable_panel",
        "build_frequency_panel",
        "rank_proteins_by_frequency",
    ]

    for func_name in screening_funcs + kbest_funcs + stability_funcs:
        assert hasattr(
            train, func_name
        ), f"CLI train should import {func_name} from features module"
