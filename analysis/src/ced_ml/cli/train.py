"""
CLI implementation for train command.

Thin wrapper around existing celiacML_faith.py logic with new config system.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional

from ced_ml.config.loader import load_training_config, save_config
from ced_ml.config.validation import validate_training_config
from ced_ml.utils.logging import setup_logger, log_section

# Feature selection modules
from ced_ml.features.screening import (
    mann_whitney_screen,
    f_statistic_screen,
    variance_missingness_prefilter,
    screen_proteins,
)
from ced_ml.features.kbest import (
    select_kbest_features,
    compute_f_classif_scores,
    extract_selected_proteins_from_kbest,
    rank_features_by_score,
)
from ced_ml.features.stability import (
    compute_selection_frequencies,
    extract_stable_panel,
    build_frequency_panel,
    rank_proteins_by_frequency,
)

# Model modules
from ced_ml.models.registry import (
    build_models,
    build_logistic_regression,
    build_linear_svm_calibrated,
    build_random_forest,
    build_xgboost,
    parse_class_weight_options,
    compute_scale_pos_weight_from_y,
)
from ced_ml.models.hyperparams import (
    get_param_distributions,
)
from ced_ml.models.training import (
    oof_predictions_with_nested_cv,
)
from ced_ml.models.calibration import (
    maybe_calibrate_estimator,
    calibration_intercept_slope,
    expected_calibration_error,
)
from ced_ml.models.prevalence import (
    PrevalenceAdjustedModel,
    adjust_probabilities_for_prevalence,
)

# Metrics modules
from ced_ml.metrics.discrimination import (
    auroc,
    prauc,
    youden_j,
    alpha_sensitivity_at_specificity,
    compute_discrimination_metrics,
    compute_brier_score,
    compute_log_loss,
)
from ced_ml.metrics.thresholds import (
    threshold_max_f1,
    threshold_max_fbeta,
    threshold_youden,
    threshold_for_specificity,
    threshold_for_precision,
    threshold_from_controls,
    binary_metrics_at_threshold,
    top_risk_capture,
    choose_threshold_objective,
)
from ced_ml.metrics.dca import (
    decision_curve_analysis,
    decision_curve_table,
    net_benefit,
    net_benefit_treat_all,
    compute_dca_summary,
    save_dca_results,
    find_dca_zero_crossing,
    generate_dca_thresholds,
    parse_dca_report_points,
)
from ced_ml.metrics.bootstrap import (
    stratified_bootstrap_ci,
    stratified_bootstrap_diff_ci,
)

# Plotting modules
from ced_ml.plotting import (
    plot_roc_curve,
    plot_pr_curve,
    plot_calibration_curve,
    plot_risk_distribution,
    compute_distribution_stats,
    plot_dca,
    plot_dca_curve,
    apply_plot_metadata,
    plot_learning_curve,
    plot_learning_curve_summary,
    compute_learning_curve,
    save_learning_curve_csv,
    aggregate_learning_curve_runs,
)


def run_train(
    config_file: Optional[str] = None,
    cli_args: Optional[Dict[str, Any]] = None,
    overrides: Optional[List[str]] = None,
    verbose: int = 0,
):
    """
    Run model training with new config system.
    
    Args:
        config_file: Path to YAML config file (optional)
        cli_args: Dictionary of CLI arguments (optional)
        overrides: List of config overrides (optional)
        verbose: Verbosity level (0=INFO, 1=DEBUG)
    """
    # Setup logger
    log_level = 20 - (verbose * 10)
    logger = setup_logger("ced_ml.train", level=log_level)
    
    log_section(logger, "CeD-ML Model Training")
    
    # Build overrides list from CLI args
    all_overrides = list(overrides) if overrides else []
    if cli_args:
        for key, value in cli_args.items():
            if value is not None:
                all_overrides.append(f"{key}={value}")
    
    # Load and validate config
    logger.info("Loading configuration...")
    config = load_training_config(config_file=config_file, overrides=all_overrides)
    
    logger.info("Validating configuration...")
    validate_training_config(config)
    
    # Save resolved config
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    config_path = outdir / "training_config.yaml"
    save_config(config, config_path)
    logger.info(f"Saved resolved config to: {config_path}")
    
    # Log config summary
    logger.info(f"Model: {config.model}")
    logger.info(f"Scenario: {config.scenario}")
    logger.info(f"CV: {config.cv.folds} folds × {config.cv.repeats} repeats")
    logger.info(f"Scoring: {config.cv.scoring}")
    
    # Placeholder
    logger.warning("Model training not yet implemented in refactored code.")
    logger.warning("Use legacy celiacML_faith.py for now.")
    logger.info(f"Config saved to: {config_path}")
