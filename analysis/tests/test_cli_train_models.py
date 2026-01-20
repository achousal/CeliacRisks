"""
Integration tests for CLI train command with models modules.

Ensures CLI properly imports and uses model layer modules without duplication.
"""

import inspect


def test_cli_train_imports_registry_modules():
    """Test that CLI train imports from models.registry."""
    from ced_ml.cli import train

    # Verify imports from models.registry
    assert hasattr(train, "build_models")
    assert hasattr(train, "build_logistic_regression")
    assert hasattr(train, "build_linear_svm_calibrated")
    assert hasattr(train, "build_random_forest")
    assert hasattr(train, "build_xgboost")
    assert hasattr(train, "parse_class_weight_options")
    assert hasattr(train, "compute_scale_pos_weight_from_y")


def test_cli_train_imports_hyperparams_modules():
    """Test that CLI train imports from models.hyperparams."""
    from ced_ml.cli import train

    # Verify imports from models.hyperparams
    assert hasattr(train, "get_param_distributions")


def test_cli_train_imports_training_modules():
    """Test that CLI train imports from models.training."""
    from ced_ml.cli import train

    # Verify imports from models.training
    assert hasattr(train, "oof_predictions_with_nested_cv")


def test_cli_train_imports_calibration_modules():
    """Test that CLI train imports from models.calibration."""
    from ced_ml.cli import train

    # Verify imports from models.calibration
    assert hasattr(train, "maybe_calibrate_estimator")
    assert hasattr(train, "calibration_intercept_slope")
    assert hasattr(train, "expected_calibration_error")


def test_cli_train_imports_prevalence_modules():
    """Test that CLI train imports from models.prevalence."""
    from ced_ml.cli import train

    # Verify imports from models.prevalence
    assert hasattr(train, "PrevalenceAdjustedModel")
    assert hasattr(train, "adjust_probabilities_for_prevalence")


def test_no_duplicate_build_models_function():
    """Test that CLI does not define its own build_models."""
    from ced_ml.cli import train

    # Get all functions defined in the CLI module
    cli_functions = [
        name
        for name, obj in inspect.getmembers(train, inspect.isfunction)
        if obj.__module__ == "ced_ml.cli.train"
    ]

    # Should NOT have build_models (imported from models.registry)
    assert "build_models" not in cli_functions, "CLI should import build_models, not define it"

    # Should NOT have get_param_distributions (imported from models.hyperparams)
    assert (
        "get_param_distributions" not in cli_functions
    ), "CLI should import get_param_distributions, not define it"

    # Should NOT have oof_predictions_with_nested_cv (imported from models.training)
    assert (
        "oof_predictions_with_nested_cv" not in cli_functions
    ), "CLI should import oof_predictions_with_nested_cv, not define it"


def test_no_duplicate_calibration_function():
    """Test that CLI does not define its own calibration functions."""
    from ced_ml.cli import train

    # Get all functions defined in the CLI module
    cli_functions = [
        name
        for name, obj in inspect.getmembers(train, inspect.isfunction)
        if obj.__module__ == "ced_ml.cli.train"
    ]

    # Should NOT have calibration functions (imported from models.calibration)
    assert (
        "maybe_calibrate_estimator" not in cli_functions
    ), "CLI should import maybe_calibrate_estimator, not define it"
    assert (
        "calibration_intercept_slope" not in cli_functions
    ), "CLI should import calibration_intercept_slope, not define it"


def test_no_duplicate_prevalence_function():
    """Test that CLI does not define its own PrevalenceAdjustedModel."""
    from ced_ml.cli import train

    # Get all classes defined in the CLI module
    cli_classes = [
        name
        for name, obj in inspect.getmembers(train, inspect.isclass)
        if obj.__module__ == "ced_ml.cli.train"
    ]

    # Should NOT have PrevalenceAdjustedModel (imported from models.prevalence)
    assert (
        "PrevalenceAdjustedModel" not in cli_classes
    ), "CLI should import PrevalenceAdjustedModel, not define it"


def test_no_inline_model_instantiation():
    """Test that CLI source code does not contain inline model instantiation."""
    from ced_ml.cli import train

    # Get CLI source code
    source = inspect.getsource(train)

    # Should NOT contain inline model instantiation
    assert (
        "LogisticRegression(" not in source
    ), "CLI should use build_logistic_regression(), not inline LogisticRegression"

    assert (
        "LinearSVC(" not in source
    ), "CLI should use build_linear_svm_calibrated(), not inline LinearSVC"

    assert (
        "RandomForestClassifier(" not in source
    ), "CLI should use build_random_forest(), not inline RandomForestClassifier"

    assert (
        "XGBClassifier(" not in source
    ), "CLI should use build_xgboost(), not inline XGBClassifier"


def test_no_inline_randomized_search():
    """Test that CLI does not contain inline RandomizedSearchCV setup."""
    from ced_ml.cli import train

    # Get CLI source code
    source = inspect.getsource(train)

    # Should NOT contain inline RandomizedSearchCV
    assert (
        "RandomizedSearchCV(" not in source
    ), "CLI should use oof_predictions_with_nested_cv(), not inline RandomizedSearchCV"


def test_cli_imports_complete():
    """Test that all required model functions are imported correctly."""
    from ced_ml.cli.train import (
        PrevalenceAdjustedModel,
        build_models,
        get_param_distributions,
        maybe_calibrate_estimator,
        oof_predictions_with_nested_cv,
    )
    from ced_ml.models.calibration import (
        maybe_calibrate_estimator as ref_calib,
    )
    from ced_ml.models.hyperparams import (
        get_param_distributions as ref_params,
    )
    from ced_ml.models.prevalence import (
        PrevalenceAdjustedModel as ref_prev,
    )

    # Verify they are the correct functions from models modules
    from ced_ml.models.registry import (
        build_models as ref_build,
    )
    from ced_ml.models.training import (
        oof_predictions_with_nested_cv as ref_oof,
    )

    assert build_models is ref_build
    assert get_param_distributions is ref_params
    assert oof_predictions_with_nested_cv is ref_oof
    assert maybe_calibrate_estimator is ref_calib
    assert PrevalenceAdjustedModel is ref_prev


def test_registry_module_identity():
    """Test that registry functions are from models.registry, not duplicated."""
    from ced_ml.cli.train import build_models, parse_class_weight_options
    from ced_ml.models.registry import (
        build_models as orig_build,
    )
    from ced_ml.models.registry import (
        parse_class_weight_options as orig_parse,
    )

    # Should be the exact same function object (not a copy)
    assert build_models is orig_build, "build_models should be imported, not duplicated"
    assert (
        parse_class_weight_options is orig_parse
    ), "parse_class_weight_options should be imported, not duplicated"


def test_hyperparams_module_identity():
    """Test that hyperparams functions are from models.hyperparams, not duplicated."""
    from ced_ml.cli.train import get_param_distributions
    from ced_ml.models.hyperparams import (
        get_param_distributions as orig_params,
    )

    # Should be the exact same function object (not a copy)
    assert (
        get_param_distributions is orig_params
    ), "get_param_distributions should be imported, not duplicated"


def test_training_module_identity():
    """Test that training functions are from models.training, not duplicated."""
    from ced_ml.cli.train import oof_predictions_with_nested_cv
    from ced_ml.models.training import (
        oof_predictions_with_nested_cv as orig_oof,
    )

    # Should be the exact same function object (not a copy)
    assert (
        oof_predictions_with_nested_cv is orig_oof
    ), "oof_predictions_with_nested_cv should be imported, not duplicated"


def test_calibration_module_identity():
    """Test that calibration functions are from models.calibration, not duplicated."""
    from ced_ml.cli.train import (
        calibration_intercept_slope,
        maybe_calibrate_estimator,
    )
    from ced_ml.models.calibration import (
        calibration_intercept_slope as orig_slope,
    )
    from ced_ml.models.calibration import (
        maybe_calibrate_estimator as orig_calib,
    )

    # Should be the exact same function object (not a copy)
    assert (
        maybe_calibrate_estimator is orig_calib
    ), "maybe_calibrate_estimator should be imported, not duplicated"
    assert (
        calibration_intercept_slope is orig_slope
    ), "calibration_intercept_slope should be imported, not duplicated"


def test_prevalence_module_identity():
    """Test that prevalence classes are from models.prevalence, not duplicated."""
    from ced_ml.cli.train import PrevalenceAdjustedModel
    from ced_ml.models.prevalence import (
        PrevalenceAdjustedModel as orig_prev,
    )

    # Should be the exact same class object (not a copy)
    assert (
        PrevalenceAdjustedModel is orig_prev
    ), "PrevalenceAdjustedModel should be imported, not duplicated"


def test_no_redundant_registry_code():
    """Test that CLI does not contain redundant model registry logic."""
    from ced_ml.cli import train

    source = inspect.getsource(train)

    # Should not contain manual model instantiation
    redundant_patterns = [
        "LogisticRegression(",
        "LinearSVC(",
        "RandomForestClassifier(",
        "XGBClassifier(",
        "penalty=",
        "l1_ratio=",
        "class_weight=",
    ]

    for pattern in redundant_patterns:
        assert (
            pattern not in source
        ), f"CLI should not contain inline '{pattern}' - use models.registry instead"


def test_no_redundant_hyperparams_code():
    """Test that CLI does not contain redundant hyperparameter logic."""
    from ced_ml.cli import train

    source = inspect.getsource(train)

    # Should not contain manual param distribution construction
    redundant_patterns = [
        "param_distributions =",
        "loguniform(",
        "uniform(",
        "randint(",
    ]

    for pattern in redundant_patterns:
        assert (
            pattern not in source
        ), f"CLI should not contain inline '{pattern}' - use models.hyperparams instead"


def test_no_redundant_training_code():
    """Test that CLI does not contain redundant nested CV logic."""
    from ced_ml.cli import train

    source = inspect.getsource(train)

    # Should not contain manual nested CV setup
    redundant_patterns = [
        "RandomizedSearchCV(",
        "StratifiedKFold(",
        "RepeatedStratifiedKFold(",
        "cross_val_predict(",
    ]

    for pattern in redundant_patterns:
        assert (
            pattern not in source
        ), f"CLI should not contain inline '{pattern}' - use models.training instead"


def test_models_modules_available():
    """Test that all models submodules are available via CLI."""
    from ced_ml.cli import train

    # Check all four model submodules are imported
    registry_funcs = [
        "build_models",
        "build_logistic_regression",
        "build_linear_svm_calibrated",
        "build_random_forest",
        "build_xgboost",
        "parse_class_weight_options",
        "compute_scale_pos_weight_from_y",
    ]
    hyperparams_funcs = [
        "get_param_distributions",
    ]
    training_funcs = [
        "oof_predictions_with_nested_cv",
    ]
    calibration_funcs = [
        "maybe_calibrate_estimator",
        "calibration_intercept_slope",
        "expected_calibration_error",
    ]
    prevalence_funcs = [
        "PrevalenceAdjustedModel",
        "adjust_probabilities_for_prevalence",
    ]

    for func_name in (
        registry_funcs + hyperparams_funcs + training_funcs + calibration_funcs + prevalence_funcs
    ):
        assert hasattr(train, func_name), f"CLI train should import {func_name} from models module"


def test_models_package_exports():
    """Test that models package __init__.py exports all public functions."""
    from ced_ml import models

    # Verify key functions are accessible via package import
    assert hasattr(models, "build_models")
    assert hasattr(models, "get_param_distributions")
    assert hasattr(models, "oof_predictions_with_nested_cv")
    assert hasattr(models, "PrevalenceAdjustedModel")
    assert hasattr(models, "calibration_intercept_slope")
