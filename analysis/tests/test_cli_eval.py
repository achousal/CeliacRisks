"""
Integration tests for CLI eval-holdout command with evaluation modules.

Ensures CLI properly imports and uses evaluation layer modules without duplication.
"""

import pytest
import inspect
from pathlib import Path


def test_cli_eval_holdout_imports_evaluation_module():
    """Test that CLI eval_holdout imports from evaluation module."""
    from ced_ml.cli import eval_holdout

    # Verify imports from evaluation
    assert hasattr(eval_holdout, 'evaluate_holdout'), \
        "CLI should import evaluate_holdout from evaluation module"


def test_evaluation_module_exports_all_functions():
    """Test that evaluation module exports all required functions."""
    from ced_ml import evaluation

    # Core prediction functions
    assert hasattr(evaluation, 'generate_predictions')
    assert hasattr(evaluation, 'generate_predictions_with_adjustment')
    assert hasattr(evaluation, 'export_predictions')
    assert hasattr(evaluation, 'predict_on_validation')
    assert hasattr(evaluation, 'predict_on_test')
    assert hasattr(evaluation, 'predict_on_holdout')

    # Results management
    assert hasattr(evaluation, 'OutputDirectories')
    assert hasattr(evaluation, 'ResultsWriter')


def test_no_duplicate_evaluate_holdout_function():
    """Test that CLI does not define its own evaluate_holdout."""
    from ced_ml.cli import eval_holdout

    # Get all functions defined in the CLI module
    cli_functions = [
        name for name, obj in inspect.getmembers(eval_holdout, inspect.isfunction)
        if obj.__module__ == 'ced_ml.cli.eval_holdout'
    ]

    # Should NOT have evaluate_holdout (imported from evaluation.holdout)
    assert 'evaluate_holdout' not in cli_functions, \
        "CLI should import evaluate_holdout, not define it"


def test_no_duplicate_prediction_functions():
    """Test that CLI does not redefine prediction functions."""
    from ced_ml.cli import eval_holdout

    # Get all functions defined in the CLI module
    cli_functions = [
        name for name, obj in inspect.getmembers(eval_holdout, inspect.isfunction)
        if obj.__module__ == 'ced_ml.cli.eval_holdout'
    ]

    # Should NOT have prediction functions (imported from evaluation.predict)
    prediction_funcs = [
        'generate_predictions',
        'generate_predictions_with_adjustment',
        'export_predictions',
        'predict_on_validation',
        'predict_on_test',
        'predict_on_holdout',
    ]

    for func in prediction_funcs:
        assert func not in cli_functions, \
            f"CLI should not define {func} (should import from evaluation.predict)"


def test_no_duplicate_results_writer_class():
    """Test that CLI does not define its own ResultsWriter."""
    from ced_ml.cli import eval_holdout

    # Get all classes defined in the CLI module
    cli_classes = [
        name for name, obj in inspect.getmembers(eval_holdout, inspect.isclass)
        if obj.__module__ == 'ced_ml.cli.eval_holdout'
    ]

    # Should NOT have ResultsWriter (imported from evaluation.reports)
    assert 'ResultsWriter' not in cli_classes, \
        "CLI should import ResultsWriter, not define it"
    assert 'OutputDirectories' not in cli_classes, \
        "CLI should import OutputDirectories, not define it"


def test_evaluate_holdout_identity():
    """Test that CLI's evaluate_holdout is the same object as evaluation.holdout's."""
    from ced_ml.cli.eval_holdout import evaluate_holdout as cli_evaluate_holdout
    from ced_ml.evaluation.holdout import evaluate_holdout as lib_evaluate_holdout

    # Should be the exact same function object (not a copy)
    assert cli_evaluate_holdout is lib_evaluate_holdout, \
        "CLI should import the function directly, not copy it"


def test_cli_eval_holdout_no_inline_model_loading():
    """Test that CLI doesn't implement model loading logic inline."""
    from ced_ml.cli import eval_holdout
    import ast

    # Read the CLI module source
    source = inspect.getsource(eval_holdout)

    # Parse source code
    tree = ast.parse(source)

    # Check for inline joblib.load calls (should use load_model_artifact from library)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                # joblib.load() pattern
                if (hasattr(node.func, 'value') and
                    hasattr(node.func.value, 'id') and
                    node.func.value.id == 'joblib' and
                    node.func.attr == 'load'):
                    pytest.fail("CLI should not call joblib.load directly, use load_model_artifact")
            elif isinstance(node.func, ast.Name):
                # Direct load() call
                if node.func.id == 'load':
                    # Check if it's from joblib context
                    pytest.fail("CLI should use load_model_artifact from evaluation.holdout")


def test_cli_eval_holdout_no_inline_metrics_computation():
    """Test that CLI doesn't compute metrics inline."""
    from ced_ml.cli import eval_holdout
    import ast

    # Read the CLI module source
    source = inspect.getsource(eval_holdout)

    # Parse source code
    tree = ast.parse(source)

    # Check for inline sklearn.metrics imports (should use evaluation/metrics modules)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and 'sklearn.metrics' in node.module:
                pytest.fail("CLI should not import sklearn.metrics directly, use metrics module")


def test_evaluation_predict_functions_accessible():
    """Test that all predict functions are accessible from evaluation module."""
    from ced_ml import evaluation

    # All should be callable
    assert callable(evaluation.generate_predictions)
    assert callable(evaluation.generate_predictions_with_adjustment)
    assert callable(evaluation.export_predictions)
    assert callable(evaluation.predict_on_validation)
    assert callable(evaluation.predict_on_test)
    assert callable(evaluation.predict_on_holdout)


def test_evaluation_reports_classes_accessible():
    """Test that results management classes are accessible."""
    from ced_ml import evaluation

    # All should be classes
    assert inspect.isclass(evaluation.OutputDirectories)
    assert inspect.isclass(evaluation.ResultsWriter)


def test_cli_run_eval_holdout_signature():
    """Test that run_eval_holdout has the correct signature."""
    from ced_ml.cli.eval_holdout import run_eval_holdout

    sig = inspect.signature(run_eval_holdout)
    params = list(sig.parameters.keys())

    # Should have required parameters
    assert 'infile' in params
    assert 'holdout_idx' in params
    assert 'model_artifact' in params
    assert 'outdir' in params

    # Should have optional parameters
    assert 'scenario' in params
    assert 'compute_dca' in params
    assert 'save_preds' in params
    assert 'target_prevalence' in params

    # Should have kwargs for extensibility
    assert 'kwargs' in params


def test_no_duplicate_code_patterns():
    """Test that CLI doesn't duplicate common code patterns."""
    from ced_ml.cli import eval_holdout
    import ast

    # Read the CLI module source
    source = inspect.getsource(eval_holdout)
    tree = ast.parse(source)

    # Count function definitions (should be minimal in CLI)
    func_defs = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

    # Should only have run_eval_holdout and maybe helpers
    func_names = [f.name for f in func_defs]
    assert len(func_names) <= 3, \
        f"CLI should have minimal functions (found: {func_names})"


def test_evaluation_module_no_missing_imports():
    """Test that evaluation module imports don't have issues."""
    # This will fail if there are import errors
    from ced_ml import evaluation

    # All attributes should be importable without errors
    for attr in evaluation.__all__:
        assert hasattr(evaluation, attr), \
            f"evaluation.__all__ lists {attr} but it's not accessible"


def test_holdout_module_functions_exist():
    """Test that holdout module has all expected functions."""
    from ced_ml.evaluation import holdout

    # Core functions
    assert hasattr(holdout, 'evaluate_holdout')
    assert hasattr(holdout, 'load_holdout_indices')
    assert hasattr(holdout, 'load_model_artifact')
    assert hasattr(holdout, 'extract_holdout_data')
    assert hasattr(holdout, 'compute_holdout_metrics')
    assert hasattr(holdout, 'compute_top_risk_capture')
    assert hasattr(holdout, 'save_holdout_predictions')


def test_predict_module_functions_exist():
    """Test that predict module has all expected functions."""
    from ced_ml.evaluation import predict

    # Core functions
    assert hasattr(predict, 'generate_predictions')
    assert hasattr(predict, 'generate_predictions_with_adjustment')
    assert hasattr(predict, 'export_predictions')
    assert hasattr(predict, 'predict_on_validation')
    assert hasattr(predict, 'predict_on_test')
    assert hasattr(predict, 'predict_on_holdout')


def test_reports_module_classes_exist():
    """Test that reports module has all expected classes."""
    from ced_ml.evaluation import reports

    # Core classes
    assert hasattr(reports, 'OutputDirectories')
    assert hasattr(reports, 'ResultsWriter')


def test_all_evaluation_modules_importable():
    """Test that all evaluation submodules can be imported."""
    from ced_ml.evaluation import holdout, predict, reports

    # All should import without errors
    assert holdout is not None
    assert predict is not None
    assert reports is not None
