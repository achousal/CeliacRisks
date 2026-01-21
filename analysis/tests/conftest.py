"""
Shared pytest fixtures for CeD-ML tests.
"""

from types import SimpleNamespace


def make_mock_config(**overrides):
    """
    Create a mock config object for testing.

    This bypasses Pydantic validation and creates a simple namespace
    with reasonable defaults. Use this for unit tests of isolated modules.

    Args:
        **overrides: Override default values

    Returns:
        SimpleNamespace with config attributes
    """
    defaults = {
        "cv": SimpleNamespace(
            folds=3, repeats=2, inner_folds=2, n_iter=5, scoring="neg_brier_score"
        ),
        "features": SimpleNamespace(
            feature_select="none",
            k_grid=[],
            kbest_scope="protein",
            coef_threshold=1e-12,
            rf_use_permutation=False,
            rf_perm_top_n=50,
            rf_perm_repeats=3,
            rf_perm_min_importance=0.0,
        ),
        "calibration": SimpleNamespace(enabled=False, method="sigmoid", cv=3),
        "compute": SimpleNamespace(cpus=2, tune_n_jobs=None),
        # Model configs at top level (matching TrainingConfig schema)
        "lr": SimpleNamespace(
            C_min=0.01,
            C_max=100.0,
            C_points=5,
            class_weight_options="None,balanced",
        ),
        "svm": SimpleNamespace(
            C_min=0.01, C_max=100.0, C_points=5, class_weight_options="balanced"
        ),
        "rf": SimpleNamespace(
            n_estimators_grid=[100, 200],
            max_depth_grid=[5, 10],
            min_samples_split_grid=[2, 5],
            min_samples_leaf_grid=[1, 2],
            max_features_grid=[0.3, 0.5],
            class_weight_options="None,balanced",
        ),
        "xgboost": SimpleNamespace(
            n_estimators_grid=[100, 200],
            max_depth_grid=[3, 5],
            learning_rate_grid=[0.01, 0.1],
            subsample_grid=[0.8, 1.0],
            colsample_bytree_grid=[0.8, 1.0],
            scale_pos_weight=None,
            scale_pos_weight_grid=[1.0, 5.0],
        ),
        "optuna": SimpleNamespace(enabled=False),
    }

    # Deep merge overrides
    config_dict = defaults.copy()
    for key, value in overrides.items():
        if key in config_dict and isinstance(value, dict):
            # Merge nested dict
            for k2, v2 in value.items():
                setattr(config_dict[key], k2, v2)
        else:
            config_dict[key] = value

    return SimpleNamespace(**config_dict)
