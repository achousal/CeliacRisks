"""
Configuration schema for CeD-ML pipeline.

Defines Pydantic models for all pipeline configuration parameters (~200 total).
All defaults match the current implementation exactly for behavioral equivalence.
"""

from pathlib import Path
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

# ============================================================================
# Data and Split Configuration
# ============================================================================


class ColumnsConfig(BaseModel):
    """Configuration for metadata column selection."""

    mode: Literal["auto", "explicit"] = "auto"
    numeric_metadata: Optional[List[str]] = None
    categorical_metadata: Optional[List[str]] = None
    warn_missing_defaults: bool = True


class SplitsConfig(BaseModel):
    """Configuration for data split generation."""

    mode: Literal["development", "holdout"] = "development"
    scenarios: List[str] = Field(default_factory=lambda: ["IncidentOnly"])
    n_splits: int = Field(default=1, ge=1)
    val_size: float = Field(default=0.0, ge=0.0, le=1.0)
    test_size: float = Field(default=0.30, ge=0.0, le=1.0)
    holdout_size: float = Field(default=0.30, ge=0.0, le=1.0)
    seed_start: int = Field(default=0, ge=0)

    # Prevalent case handling
    prevalent_train_only: bool = False
    prevalent_train_frac: float = Field(default=1.0, ge=0.0, le=1.0)

    # Control downsampling
    train_control_per_case: Optional[float] = Field(default=None, ge=1.0)
    eval_control_per_case: Optional[float] = Field(default=None, ge=1.0)

    # Temporal split
    temporal_split: bool = False
    temporal_col: str = "CeD_date"
    temporal_cutoff: Optional[str] = None

    # Output
    outdir: Path = Field(default=Path("splits"))
    save_indices_only: bool = False
    overwrite: bool = False

    @model_validator(mode="after")
    def validate_split_sizes(self):
        """Validate that split sizes don't exceed 1.0."""
        if self.mode == "development":
            total = self.val_size + self.test_size
            if total >= 1.0:
                raise ValueError(
                    f"val_size ({self.val_size}) + test_size ({self.test_size}) >= 1.0. "
                    "No data left for training."
                )
        return self


# ============================================================================
# Cross-Validation Configuration
# ============================================================================


class CVConfig(BaseModel):
    """Configuration for cross-validation structure."""

    folds: int = Field(default=5, ge=2)
    repeats: int = Field(default=3, ge=1)
    inner_folds: int = Field(default=5, ge=2)
    scoring: str = "average_precision"
    n_iter: int = Field(default=30, ge=1)
    random_state: int = 0
    tune_n_jobs: Union[int, str] = "auto"
    error_score: str = "nan"
    grid_randomize: bool = False


# ============================================================================
# Feature Selection Configuration
# ============================================================================


class FeatureConfig(BaseModel):
    """Configuration for feature selection methods."""

    feature_select: Literal["none", "kbest", "l1_stability", "hybrid"] = "none"
    screen_method: Literal["mannwhitney", "f_classif"] = "mannwhitney"
    screen_top_n: int = Field(default=0, ge=0)

    # KBest selection
    kbest_scope: Literal["protein", "transformed"] = "protein"
    kbest_max: int = Field(default=500, ge=1)
    k_grid: List[int] = Field(default_factory=lambda: [50, 100, 200, 500])

    # Stability-based selection
    stability_thresh: float = Field(default=0.70, ge=0.0, le=1.0)
    stable_corr_thresh: float = Field(default=0.80, ge=0.0, le=1.0)

    # L1 stability
    l1_c_grid: List[float] = Field(default_factory=lambda: [0.001, 0.01, 0.1, 1.0])
    l1_stability_thresh: float = Field(default=0.70, ge=0.0, le=1.0)

    # Hybrid mode
    hybrid_kbest_first: bool = True
    hybrid_k_for_stability: int = 200


# ============================================================================
# Panel Building Configuration
# ============================================================================


class PanelConfig(BaseModel):
    """Configuration for biomarker panel building."""

    build_panels: bool = False
    panel_sizes: List[int] = Field(default_factory=lambda: [10, 25, 50, 100])
    panel_corr_thresh: float = Field(default=0.80, ge=0.0, le=1.0)
    panel_corr_method: Literal["pearson", "spearman"] = "pearson"
    panel_rep_tiebreak: Literal["first", "random"] = "first"
    panel_refit: bool = True
    panel_stability_mode: Literal["frequency", "rank"] = "frequency"


# ============================================================================
# Model-Specific Hyperparameter Configurations
# ============================================================================


class LRConfig(BaseModel):
    """Logistic Regression hyperparameters."""

    penalty: List[str] = Field(default_factory=lambda: ["l1", "l2", "elasticnet"])
    C: List[float] = Field(default_factory=lambda: [0.001, 0.01, 0.1, 1.0, 10.0])
    l1_ratio: List[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9])
    solver: str = "saga"
    max_iter: int = 1000
    class_weight: Optional[str] = "balanced"
    random_state: int = 0


class SVMConfig(BaseModel):
    """Support Vector Machine hyperparameters."""

    C: List[float] = Field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0])
    kernel: List[str] = Field(default_factory=lambda: ["linear", "rbf"])
    gamma: List[Union[str, float]] = Field(
        default_factory=lambda: ["scale", "auto", 0.001, 0.01]
    )
    class_weight: Optional[str] = "balanced"
    max_iter: int = 5000
    probability: bool = True
    random_state: int = 0


class RFConfig(BaseModel):
    """Random Forest hyperparameters."""

    n_estimators: List[int] = Field(default_factory=lambda: [100, 300, 500])
    max_depth: List[Optional[int]] = Field(default_factory=lambda: [None, 10, 20, 30])
    min_samples_split: List[int] = Field(default_factory=lambda: [2, 5, 10])
    min_samples_leaf: List[int] = Field(default_factory=lambda: [1, 2, 4])
    max_features: List[Union[str, float]] = Field(
        default_factory=lambda: ["sqrt", "log2", 0.5]
    )
    class_weight: Optional[str] = "balanced"
    n_jobs: int = -1
    random_state: int = 0


class XGBoostConfig(BaseModel):
    """XGBoost hyperparameters."""

    n_estimators: List[int] = Field(default_factory=lambda: [100, 300, 500])
    max_depth: List[int] = Field(default_factory=lambda: [3, 5, 7, 10])
    learning_rate: List[float] = Field(default_factory=lambda: [0.01, 0.05, 0.1, 0.3])
    min_child_weight: List[int] = Field(default_factory=lambda: [1, 3, 5])
    gamma: List[float] = Field(default_factory=lambda: [0.0, 0.1, 0.2])
    subsample: List[float] = Field(default_factory=lambda: [0.7, 0.8, 1.0])
    colsample_bytree: List[float] = Field(default_factory=lambda: [0.7, 0.8, 1.0])
    reg_alpha: List[float] = Field(default_factory=lambda: [0.0, 0.1, 1.0])
    reg_lambda: List[float] = Field(default_factory=lambda: [1.0, 5.0, 10.0])
    scale_pos_weight: str = "auto"
    tree_method: str = "hist"
    n_jobs: int = -1
    random_state: int = 0


class CalibrationConfig(BaseModel):
    """Calibration wrapper configuration."""

    method: Literal["sigmoid", "isotonic"] = "sigmoid"
    cv: int = 5
    ensemble: bool = False


# ============================================================================
# Threshold Selection Configuration
# ============================================================================


class ThresholdConfig(BaseModel):
    """Configuration for threshold selection."""

    objective: Literal["max_f1", "max_fbeta", "youden", "fixed_spec", "fixed_ppv"] = (
        "max_f1"
    )
    fbeta: float = Field(default=1.0, gt=0.0)
    fixed_spec: float = Field(default=0.90, ge=0.0, le=1.0)
    fixed_ppv: float = Field(default=0.10, ge=0.0, le=1.0)
    threshold_source: Literal["val", "test", "train_oof"] = "val"
    target_prevalence_source: Literal["val", "test", "train", "fixed"] = "test"
    target_prevalence_fixed: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    risk_prob_source: Literal["val", "test"] = "test"


# ============================================================================
# Evaluation and Reporting Configuration
# ============================================================================


class EvaluationConfig(BaseModel):
    """Configuration for evaluation metrics and reporting."""

    # Bootstrap confidence intervals
    test_ci_bootstrap: bool = True
    n_boot: int = Field(default=500, ge=100)
    boot_random_state: int = 0

    # Learning curves
    learning_curve: bool = False
    lc_train_sizes: List[float] = Field(
        default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 1.0]
    )

    # Feature importance
    feature_reports: bool = True
    feature_report_max: int = 100

    # Specificity/sensitivity targets
    control_spec_targets: List[float] = Field(
        default_factory=lambda: [0.90, 0.95, 0.99]
    )
    toprisk_fracs: List[float] = Field(default_factory=lambda: [0.01, 0.05, 0.10])


# ============================================================================
# Decision Curve Analysis Configuration
# ============================================================================


class DCAConfig(BaseModel):
    """Configuration for decision curve analysis."""

    compute_dca: bool = False
    dca_threshold_min: float = Field(default=0.0005, ge=0.0, le=1.0)
    dca_threshold_max: float = Field(default=1.0, ge=0.0, le=1.0)
    dca_threshold_step: float = Field(default=0.001, gt=0.0)
    dca_report_points: List[float] = Field(
        default_factory=lambda: [0.01, 0.05, 0.10, 0.20]
    )


# ============================================================================
# Output Control Configuration
# ============================================================================


class OutputConfig(BaseModel):
    """Configuration for output file generation."""

    save_train_preds: bool = False
    save_val_preds: bool = True
    save_test_preds: bool = True
    save_calibration: bool = True
    calib_bins: int = Field(default=10, ge=2)
    save_feature_importance: bool = True
    save_plots: bool = True
    plot_format: str = "png"
    plot_dpi: int = 300


# ============================================================================
# Strictness and Validation Configuration
# ============================================================================


class StrictnessConfig(BaseModel):
    """Configuration for validation strictness."""

    level: Literal["off", "warn", "error"] = "warn"
    check_split_overlap: bool = True
    check_prevalent_in_eval: bool = True
    check_threshold_source: bool = True
    check_prevalence_adjustment: bool = True
    check_feature_leakage: bool = True


# ============================================================================
# Master Training Configuration
# ============================================================================


class TrainingConfig(BaseModel):
    """Complete training configuration."""

    # Data
    infile: Path
    split_dir: Optional[Path] = None
    scenario: str = "IncidentOnly"
    split_seed: int = 0

    # Model selection
    model: str = "LR_EN"

    # Sub-configurations
    columns: ColumnsConfig = Field(default_factory=ColumnsConfig)
    cv: CVConfig = Field(default_factory=CVConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    panels: PanelConfig = Field(default_factory=PanelConfig)
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    dca: DCAConfig = Field(default_factory=DCAConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    strictness: StrictnessConfig = Field(default_factory=StrictnessConfig)

    # Model-specific hyperparameters
    lr: LRConfig = Field(default_factory=LRConfig)
    svm: SVMConfig = Field(default_factory=SVMConfig)
    rf: RFConfig = Field(default_factory=RFConfig)
    xgboost: XGBoostConfig = Field(default_factory=XGBoostConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)

    # Output
    outdir: Path = Field(default=Path("results"))
    run_name: Optional[str] = None

    # Resources
    n_jobs: int = -1
    verbose: int = 1

    @model_validator(mode="after")
    def validate_config(self):
        """Cross-field validation."""
        # Ensure threshold source is available
        if self.thresholds.threshold_source == "val" and self.cv.folds < 2:
            raise ValueError("threshold_source='val' requires val_size > 0")

        # Check prevalence source consistency
        if self.thresholds.target_prevalence_source == "fixed":
            if self.thresholds.target_prevalence_fixed is None:
                raise ValueError(
                    "target_prevalence_source='fixed' requires target_prevalence_fixed"
                )

        return self


# ============================================================================
# Master Configuration (All Subcommands)
# ============================================================================


class RootConfig(BaseModel):
    """Root configuration for all CeD-ML commands."""

    # Common
    random_state: int = 0
    verbose: int = 1

    # Sub-configs (populated based on command)
    splits: Optional[SplitsConfig] = None
    training: Optional[TrainingConfig] = None

    model_config = {"arbitrary_types_allowed": True}
