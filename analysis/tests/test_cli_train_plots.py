"""Integration tests for plotting module imports in CLI.

Verifies:
1. All plotting functions are imported from library modules (not duplicated)
2. No inline matplotlib/plotting code in CLI
3. Function identity checks (imports are references, not copies)
4. Complete coverage of all plotting submodules
"""

import ast
import inspect
from pathlib import Path


def test_train_imports_plotting_functions():
    """CLI train.py imports from ced_ml.plotting module."""
    from ced_ml.cli import train

    # Check that train module has access to key plotting functions
    assert hasattr(train, "plot_roc_curve"), "plot_roc_curve not imported"
    assert hasattr(train, "plot_calibration_curve"), "plot_calibration_curve not imported"
    assert hasattr(train, "plot_risk_distribution"), "plot_risk_distribution not imported"
    assert hasattr(train, "plot_dca"), "plot_dca not imported"
    assert hasattr(train, "plot_learning_curve"), "plot_learning_curve not imported"


def test_all_plotting_modules_accessible_from_cli():
    """All plotting submodules are accessible via imports."""
    from ced_ml.cli import train

    # ROC/PR module functions
    assert hasattr(train, "plot_roc_curve")
    assert hasattr(train, "plot_pr_curve")

    # Calibration module functions
    assert hasattr(train, "plot_calibration_curve")

    # DCA module functions
    assert hasattr(train, "plot_dca")
    assert hasattr(train, "plot_dca_curve")
    assert hasattr(train, "apply_plot_metadata")

    # Risk distribution module functions
    assert hasattr(train, "plot_risk_distribution")
    assert hasattr(train, "compute_distribution_stats")

    # Learning curve module functions
    assert hasattr(train, "plot_learning_curve")
    assert hasattr(train, "plot_learning_curve_summary")
    assert hasattr(train, "compute_learning_curve")
    assert hasattr(train, "save_learning_curve_csv")
    assert hasattr(train, "aggregate_learning_curve_runs")


def test_no_duplicate_plot_roc_curve():
    """CLI does not define its own plot_roc_curve function."""
    cli_path = Path(__file__).parent.parent / "src" / "ced_ml" / "cli" / "train.py"
    with open(cli_path) as f:
        tree = ast.parse(f.read())

    # Check for any function definitions with "plot_roc" in the name
    func_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    assert "plot_roc_curve" not in func_names, "CLI should not define plot_roc_curve"
    assert "plot_roc" not in func_names, "CLI should not define plot_roc variant"


def test_no_duplicate_plot_calibration():
    """CLI does not define its own plot_calibration_curve function."""
    cli_path = Path(__file__).parent.parent / "src" / "ced_ml" / "cli" / "train.py"
    with open(cli_path) as f:
        tree = ast.parse(f.read())

    func_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    assert (
        "plot_calibration_curve" not in func_names
    ), "CLI should not define plot_calibration_curve"
    assert "plot_calibration" not in func_names, "CLI should not define plot_calibration variant"


def test_no_duplicate_plot_dca():
    """CLI does not define its own DCA plotting functions."""
    cli_path = Path(__file__).parent.parent / "src" / "ced_ml" / "cli" / "train.py"
    with open(cli_path) as f:
        tree = ast.parse(f.read())

    func_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    assert "plot_dca" not in func_names, "CLI should not define plot_dca"
    assert "plot_dca_curve" not in func_names, "CLI should not define plot_dca_curve"


def test_no_duplicate_plot_risk_distribution():
    """CLI does not define its own plot_risk_distribution function."""
    cli_path = Path(__file__).parent.parent / "src" / "ced_ml" / "cli" / "train.py"
    with open(cli_path) as f:
        tree = ast.parse(f.read())

    func_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    assert (
        "plot_risk_distribution" not in func_names
    ), "CLI should not define plot_risk_distribution"


def test_no_duplicate_plot_learning_curve():
    """CLI does not define its own learning curve plotting functions."""
    cli_path = Path(__file__).parent.parent / "src" / "ced_ml" / "cli" / "train.py"
    with open(cli_path) as f:
        tree = ast.parse(f.read())

    func_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    assert "plot_learning_curve" not in func_names, "CLI should not define plot_learning_curve"
    assert "plot_learning_curve_summary" not in func_names


def test_no_inline_matplotlib_imports():
    """CLI does not import matplotlib directly (uses plotting module)."""
    cli_path = Path(__file__).parent.parent / "src" / "ced_ml" / "cli" / "train.py"
    with open(cli_path) as f:
        content = f.read()

    # Check for matplotlib imports
    assert "import matplotlib" not in content, "CLI should not import matplotlib directly"
    assert "from matplotlib" not in content, "CLI should not import from matplotlib"

    # Exception: matplotlib.pyplot imports are allowed if they come from plotting module
    # But train.py should not have direct pyplot imports
    lines = content.split("\n")
    direct_pyplot_imports = [
        line
        for line in lines
        if "import matplotlib.pyplot" in line and not line.strip().startswith("#")
    ]
    assert len(direct_pyplot_imports) == 0, f"Found direct pyplot imports: {direct_pyplot_imports}"


def test_plotting_functions_are_imported_not_copied():
    """Verify CLI imports reference actual plotting functions (not copies)."""
    from ced_ml.cli import train
    from ced_ml.plotting import (
        plot_calibration_curve,
        plot_dca,
        plot_learning_curve,
        plot_risk_distribution,
        plot_roc_curve,
    )

    # Check identity (same object, not copy)
    assert train.plot_roc_curve is plot_roc_curve, "plot_roc_curve should be imported, not copied"
    assert (
        train.plot_calibration_curve is plot_calibration_curve
    ), "plot_calibration_curve should be imported"
    assert (
        train.plot_risk_distribution is plot_risk_distribution
    ), "plot_risk_distribution should be imported"
    assert train.plot_dca is plot_dca, "plot_dca should be imported"
    assert (
        train.plot_learning_curve is plot_learning_curve
    ), "plot_learning_curve should be imported"


def test_plotting_module_has_all_functions():
    """Verify plotting module exports all expected functions."""
    from ced_ml import plotting

    expected_functions = [
        # ROC/PR
        "plot_roc_curve",
        "plot_pr_curve",
        # Calibration
        "plot_calibration_curve",
        # DCA
        "plot_dca",
        "plot_dca_curve",
        "apply_plot_metadata",
        # Risk distribution
        "plot_risk_distribution",
        "compute_distribution_stats",
        # Learning curves
        "plot_learning_curve",
        "plot_learning_curve_summary",
        "compute_learning_curve",
        "save_learning_curve_csv",
        "aggregate_learning_curve_runs",
    ]

    for func_name in expected_functions:
        assert hasattr(plotting, func_name), f"plotting module missing {func_name}"
        assert callable(getattr(plotting, func_name)), f"{func_name} is not callable"


def test_no_plotting_function_duplication_in_cli():
    """Comprehensive check: no plotting function defined in CLI."""
    cli_path = Path(__file__).parent.parent / "src" / "ced_ml" / "cli" / "train.py"
    with open(cli_path) as f:
        tree = ast.parse(f.read())

    # Get all function definitions in CLI
    func_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

    # Known plotting function prefixes
    plotting_prefixes = [
        "plot_",
        "compute_learning_curve",
        "save_learning_curve",
        "aggregate_learning",
    ]

    # Check that no function in CLI starts with plotting prefixes (except run_* orchestration)
    for func_name in func_names:
        if func_name.startswith("run_"):
            continue  # run_* functions are orchestration, not plotting logic

        for prefix in plotting_prefixes:
            assert not func_name.startswith(
                prefix
            ), f"CLI defines plotting function {func_name} - should import from plotting module"


def test_roc_pr_module_functions_count():
    """Verify roc_pr module has expected number of public functions."""
    from ced_ml.plotting import roc_pr

    public_funcs = [
        name for name in dir(roc_pr) if not name.startswith("_") and callable(getattr(roc_pr, name))
    ]

    # roc_pr has 2 public plot functions
    expected = ["plot_roc_curve", "plot_pr_curve"]
    for func in expected:
        assert func in public_funcs, f"roc_pr module missing {func}"


def test_calibration_module_functions_count():
    """Verify calibration module has expected number of public functions."""
    from ced_ml.plotting import calibration

    public_funcs = [
        name
        for name in dir(calibration)
        if not name.startswith("_") and callable(getattr(calibration, name))
    ]

    # calibration has 1 main plot function
    assert "plot_calibration_curve" in public_funcs


def test_dca_module_functions_count():
    """Verify DCA module has expected number of public functions."""
    from ced_ml.plotting import dca

    public_funcs = [
        name for name in dir(dca) if not name.startswith("_") and callable(getattr(dca, name))
    ]

    # DCA has 3 public functions
    expected = ["plot_dca", "plot_dca_curve", "apply_plot_metadata"]
    for func in expected:
        assert func in public_funcs, f"dca module missing {func}"


def test_risk_dist_module_functions_count():
    """Verify risk_dist module has expected number of public functions."""
    from ced_ml.plotting import risk_dist

    public_funcs = [
        name
        for name in dir(risk_dist)
        if not name.startswith("_") and callable(getattr(risk_dist, name))
    ]

    # risk_dist has 2 public functions
    expected = ["plot_risk_distribution", "compute_distribution_stats"]
    for func in expected:
        assert func in public_funcs, f"risk_dist module missing {func}"


def test_learning_curve_module_functions_count():
    """Verify learning_curve module has expected number of public functions."""
    from ced_ml.plotting import learning_curve

    public_funcs = [
        name
        for name in dir(learning_curve)
        if not name.startswith("_") and callable(getattr(learning_curve, name))
    ]

    # learning_curve has 5 public functions
    expected = [
        "plot_learning_curve",
        "plot_learning_curve_summary",
        "compute_learning_curve",
        "save_learning_curve_csv",
        "aggregate_learning_curve_runs",
    ]
    for func in expected:
        assert func in public_funcs, f"learning_curve module missing {func}"


def test_cli_imports_cover_all_plotting_modules():
    """Verify CLI imports from all 5 plotting submodules."""
    cli_path = Path(__file__).parent.parent / "src" / "ced_ml" / "cli" / "train.py"
    with open(cli_path) as f:
        content = f.read()

    # Check that CLI has imports from ced_ml.plotting
    assert "from ced_ml.plotting import" in content, "CLI should import from ced_ml.plotting"

    # Verify specific functions from each submodule are imported
    # ROC/PR
    assert "plot_roc_curve" in content
    assert "plot_pr_curve" in content

    # Calibration
    assert "plot_calibration_curve" in content

    # DCA
    assert "plot_dca" in content
    assert "plot_dca_curve" in content

    # Risk distribution
    assert "plot_risk_distribution" in content
    assert "compute_distribution_stats" in content

    # Learning curves
    assert "plot_learning_curve" in content
    assert "compute_learning_curve" in content


def test_no_fig_savefig_in_cli():
    """CLI should not have matplotlib fig.savefig() calls (uses plotting functions)."""
    cli_path = Path(__file__).parent.parent / "src" / "ced_ml" / "cli" / "train.py"
    with open(cli_path) as f:
        content = f.read()

    # Check for savefig calls (indicates inline plotting)
    lines = content.split("\n")
    savefig_lines = [
        line for line in lines if ".savefig(" in line and not line.strip().startswith("#")
    ]

    # Should be zero (all plotting delegated to plotting module)
    assert len(savefig_lines) == 0, f"Found savefig calls in CLI: {savefig_lines}"


def test_cli_train_signature_unchanged():
    """run_train signature has not changed (stability check)."""
    from ced_ml.cli.train import run_train

    sig = inspect.signature(run_train)
    params = list(sig.parameters.keys())

    # Expected parameters
    expected = ["config_file", "cli_args", "overrides", "verbose"]
    for param in expected:
        assert param in params, f"run_train missing parameter: {param}"


def test_plotting_init_exports_all_functions():
    """Verify plotting/__init__.py exports all public functions."""
    from ced_ml import plotting

    # Check __all__ list exists and is complete
    assert hasattr(plotting, "__all__"), "plotting module should define __all__"

    all_list = plotting.__all__

    # Verify count (should have 13 functions)
    assert len(all_list) == 13, f"Expected 13 exports, got {len(all_list)}: {all_list}"

    # Verify all functions are actually accessible
    for func_name in all_list:
        assert hasattr(
            plotting, func_name
        ), f"plotting.__all__ lists {func_name} but it's not accessible"


def test_no_duplicate_compute_distribution_stats():
    """CLI does not define compute_distribution_stats."""
    cli_path = Path(__file__).parent.parent / "src" / "ced_ml" / "cli" / "train.py"
    with open(cli_path) as f:
        tree = ast.parse(f.read())

    func_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    assert "compute_distribution_stats" not in func_names


def test_no_duplicate_apply_plot_metadata():
    """CLI does not define apply_plot_metadata."""
    cli_path = Path(__file__).parent.parent / "src" / "ced_ml" / "cli" / "train.py"
    with open(cli_path) as f:
        tree = ast.parse(f.read())

    func_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    assert "apply_plot_metadata" not in func_names
