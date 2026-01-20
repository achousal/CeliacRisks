"""
Integration tests for CLI train command with metrics modules.

Ensures CLI properly imports and uses metrics layer modules without duplication.
"""

import inspect


def test_cli_train_imports_discrimination_modules():
    """Test that CLI train imports from metrics.discrimination."""
    from ced_ml.cli import train

    # Verify imports from metrics.discrimination
    assert hasattr(train, "auroc")
    assert hasattr(train, "prauc")
    assert hasattr(train, "youden_j")
    assert hasattr(train, "alpha_sensitivity_at_specificity")
    assert hasattr(train, "compute_discrimination_metrics")
    assert hasattr(train, "compute_brier_score")
    assert hasattr(train, "compute_log_loss")


def test_cli_train_imports_threshold_modules():
    """Test that CLI train imports from metrics.thresholds."""
    from ced_ml.cli import train

    # Verify imports from metrics.thresholds
    assert hasattr(train, "threshold_max_f1")
    assert hasattr(train, "threshold_max_fbeta")
    assert hasattr(train, "threshold_youden")
    assert hasattr(train, "threshold_for_specificity")
    assert hasattr(train, "threshold_for_precision")
    assert hasattr(train, "threshold_from_controls")
    assert hasattr(train, "binary_metrics_at_threshold")
    assert hasattr(train, "top_risk_capture")
    assert hasattr(train, "choose_threshold_objective")


def test_cli_train_imports_dca_modules():
    """Test that CLI train imports from metrics.dca."""
    from ced_ml.cli import train

    # Verify imports from metrics.dca
    assert hasattr(train, "decision_curve_analysis")
    assert hasattr(train, "decision_curve_table")
    assert hasattr(train, "net_benefit")
    assert hasattr(train, "net_benefit_treat_all")
    assert hasattr(train, "compute_dca_summary")
    assert hasattr(train, "save_dca_results")
    assert hasattr(train, "find_dca_zero_crossing")
    assert hasattr(train, "generate_dca_thresholds")
    assert hasattr(train, "parse_dca_report_points")


def test_cli_train_imports_bootstrap_modules():
    """Test that CLI train imports from metrics.bootstrap."""
    from ced_ml.cli import train

    # Verify imports from metrics.bootstrap
    assert hasattr(train, "stratified_bootstrap_ci")
    assert hasattr(train, "stratified_bootstrap_diff_ci")


def test_no_duplicate_discrimination_functions():
    """Test that CLI does not define its own discrimination metrics."""
    from ced_ml.cli import train

    # Get all functions defined in the CLI module
    cli_functions = [
        name
        for name, obj in inspect.getmembers(train, inspect.isfunction)
        if obj.__module__ == "ced_ml.cli.train"
    ]

    # Should NOT have discrimination functions (imported from metrics.discrimination)
    assert "auroc" not in cli_functions, "CLI should import auroc, not define it"
    assert "prauc" not in cli_functions, "CLI should import prauc, not define it"
    assert "youden_j" not in cli_functions, "CLI should import youden_j, not define it"
    assert (
        "compute_discrimination_metrics" not in cli_functions
    ), "CLI should import compute_discrimination_metrics, not define it"


def test_no_duplicate_threshold_functions():
    """Test that CLI does not define its own threshold selection functions."""
    from ced_ml.cli import train

    # Get all functions defined in the CLI module
    cli_functions = [
        name
        for name, obj in inspect.getmembers(train, inspect.isfunction)
        if obj.__module__ == "ced_ml.cli.train"
    ]

    # Should NOT have threshold functions (imported from metrics.thresholds)
    assert (
        "threshold_max_f1" not in cli_functions
    ), "CLI should import threshold_max_f1, not define it"
    assert (
        "threshold_youden" not in cli_functions
    ), "CLI should import threshold_youden, not define it"
    assert (
        "threshold_for_specificity" not in cli_functions
    ), "CLI should import threshold_for_specificity, not define it"
    assert (
        "choose_threshold_objective" not in cli_functions
    ), "CLI should import choose_threshold_objective, not define it"


def test_no_duplicate_dca_functions():
    """Test that CLI does not define its own DCA functions."""
    from ced_ml.cli import train

    # Get all functions defined in the CLI module
    cli_functions = [
        name
        for name, obj in inspect.getmembers(train, inspect.isfunction)
        if obj.__module__ == "ced_ml.cli.train"
    ]

    # Should NOT have DCA functions (imported from metrics.dca)
    assert (
        "decision_curve_analysis" not in cli_functions
    ), "CLI should import decision_curve_analysis, not define it"
    assert "net_benefit" not in cli_functions, "CLI should import net_benefit, not define it"
    assert (
        "compute_dca_summary" not in cli_functions
    ), "CLI should import compute_dca_summary, not define it"


def test_no_duplicate_bootstrap_functions():
    """Test that CLI does not define its own bootstrap functions."""
    from ced_ml.cli import train

    # Get all functions defined in the CLI module
    cli_functions = [
        name
        for name, obj in inspect.getmembers(train, inspect.isfunction)
        if obj.__module__ == "ced_ml.cli.train"
    ]

    # Should NOT have bootstrap functions (imported from metrics.bootstrap)
    assert (
        "stratified_bootstrap_ci" not in cli_functions
    ), "CLI should import stratified_bootstrap_ci, not define it"
    assert (
        "stratified_bootstrap_diff_ci" not in cli_functions
    ), "CLI should import stratified_bootstrap_diff_ci, not define it"


def test_no_inline_metric_computation():
    """Test that CLI does not contain inline metric computation code."""
    import inspect

    from ced_ml.cli import train

    # Read the CLI module source
    source = inspect.getsource(train)

    # Should not have inline sklearn.metrics imports
    assert (
        "from sklearn.metrics import roc_auc_score" not in source
    ), "CLI should use ced_ml.metrics.auroc instead of sklearn.metrics directly"
    assert (
        "from sklearn.metrics import average_precision_score" not in source
    ), "CLI should use ced_ml.metrics.prauc instead of sklearn.metrics directly"

    # Should not have inline ROC computation
    assert (
        "roc_curve(" not in source
    ), "CLI should use ced_ml.metrics functions, not inline roc_curve"
    assert (
        "roc_auc_score(" not in source
    ), "CLI should use ced_ml.metrics.auroc, not inline roc_auc_score"


def test_function_identity_discrimination():
    """Test that discrimination functions are imported (not copied)."""
    from ced_ml.cli import train
    from ced_ml.metrics import discrimination

    # Verify these are the SAME objects (not copies)
    assert train.auroc is discrimination.auroc
    assert train.prauc is discrimination.prauc
    assert train.youden_j is discrimination.youden_j
    assert train.compute_discrimination_metrics is discrimination.compute_discrimination_metrics


def test_function_identity_thresholds():
    """Test that threshold functions are imported (not copied)."""
    from ced_ml.cli import train
    from ced_ml.metrics import thresholds

    # Verify these are the SAME objects (not copies)
    assert train.threshold_max_f1 is thresholds.threshold_max_f1
    assert train.threshold_youden is thresholds.threshold_youden
    assert train.threshold_for_specificity is thresholds.threshold_for_specificity
    assert train.choose_threshold_objective is thresholds.choose_threshold_objective


def test_function_identity_dca():
    """Test that DCA functions are imported (not copied)."""
    from ced_ml.cli import train
    from ced_ml.metrics import dca

    # Verify these are the SAME objects (not copies)
    assert train.decision_curve_analysis is dca.decision_curve_analysis
    assert train.net_benefit is dca.net_benefit
    assert train.compute_dca_summary is dca.compute_dca_summary
    assert train.save_dca_results is dca.save_dca_results


def test_function_identity_bootstrap():
    """Test that bootstrap functions are imported (not copied)."""
    from ced_ml.cli import train
    from ced_ml.metrics import bootstrap

    # Verify these are the SAME objects (not copies)
    assert train.stratified_bootstrap_ci is bootstrap.stratified_bootstrap_ci
    assert train.stratified_bootstrap_diff_ci is bootstrap.stratified_bootstrap_diff_ci


def test_all_metrics_functions_accessible():
    """Test that all metrics functions are accessible from CLI."""
    from ced_ml.cli import train
    from ced_ml.metrics import bootstrap, dca, discrimination, thresholds

    # All discrimination functions
    for func_name in [
        "auroc",
        "prauc",
        "youden_j",
        "alpha_sensitivity_at_specificity",
        "compute_discrimination_metrics",
        "compute_brier_score",
        "compute_log_loss",
    ]:
        assert hasattr(train, func_name)
        assert getattr(train, func_name) is getattr(discrimination, func_name)

    # All threshold functions
    for func_name in [
        "threshold_max_f1",
        "threshold_max_fbeta",
        "threshold_youden",
        "threshold_for_specificity",
        "threshold_for_precision",
        "threshold_from_controls",
        "binary_metrics_at_threshold",
        "top_risk_capture",
        "choose_threshold_objective",
    ]:
        assert hasattr(train, func_name)
        assert getattr(train, func_name) is getattr(thresholds, func_name)

    # All DCA functions
    for func_name in [
        "decision_curve_analysis",
        "decision_curve_table",
        "net_benefit",
        "net_benefit_treat_all",
        "compute_dca_summary",
        "save_dca_results",
        "find_dca_zero_crossing",
        "generate_dca_thresholds",
        "parse_dca_report_points",
    ]:
        assert hasattr(train, func_name)
        assert getattr(train, func_name) is getattr(dca, func_name)

    # All bootstrap functions
    for func_name in ["stratified_bootstrap_ci", "stratified_bootstrap_diff_ci"]:
        assert hasattr(train, func_name)
        assert getattr(train, func_name) is getattr(bootstrap, func_name)


def test_metrics_module_count():
    """Test that we have all four metrics modules imported."""
    import inspect

    from ced_ml.cli import train

    # Read the CLI module source
    source = inspect.getsource(train)

    # Should have imports from all four metrics modules
    assert "from ced_ml.metrics.discrimination import" in source
    assert "from ced_ml.metrics.thresholds import" in source
    assert "from ced_ml.metrics.dca import" in source
    assert "from ced_ml.metrics.bootstrap import" in source


def test_no_duplicate_metric_classes():
    """Test that CLI does not define any metric-related classes."""
    from ced_ml.cli import train

    # Get all classes defined in the CLI module
    cli_classes = [
        name
        for name, obj in inspect.getmembers(train, inspect.isclass)
        if obj.__module__ == "ced_ml.cli.train"
    ]

    # Should not have any metric-related classes
    metric_class_names = [
        "DiscriminationMetrics",
        "ThresholdSelector",
        "DCAAnalyzer",
        "BootstrapCI",
        "CalibrationMetrics",
    ]
    for class_name in metric_class_names:
        assert class_name not in cli_classes, f"CLI should not define {class_name} class"
