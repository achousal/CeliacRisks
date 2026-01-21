"""
ResultsWriter: Structured output directory management and results serialization.

Provides:
- OutputDirectories: Directory structure creation and path management
- ResultsWriter: High-level API for saving metrics, predictions, reports

Design:
- Centralized path management (no scattered os.path.join calls)
- Type-safe output directory structure
- Consistent file naming conventions
- Atomic writes where feasible
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OutputDirectories:
    """
    Structured output directory paths.

    Attributes:
        root: Base output directory
        core: Core metrics and settings (val_metrics.csv, test_metrics.csv, run_settings.json)
        cv: Cross-validation artifacts (best_params_per_split.csv, cv_repeat_metrics.csv)
        preds_test: Test set predictions
        preds_val: Validation set predictions
        preds_controls: Control subjects OOF predictions
        preds_train_oof: Out-of-fold train predictions
        preds_plots: Prediction-related plots (risk distributions)
        reports_features: Feature importance reports
        reports_stable: Stable panel reports
        reports_panels: Panel building reports (N=10, 25, 50, etc.)
        reports_subgroups: Subgroup analysis reports
        diag: Diagnostics root
        diag_splits: Split trace and diagnostics
        diag_calibration: Calibration curves
        diag_learning: Learning curves
        diag_dca: Decision curve analysis
        diag_tuning: Hyperparameter tuning diagnostics
        diag_plots: Diagnostic plots (ROC, PR, calibration, DCA)
        diag_screening: Feature screening results
        diag_test_ci: Bootstrap CI files for test metrics
    """

    root: str
    core: str
    cv: str
    preds_test: str
    preds_val: str
    preds_controls: str
    preds_train_oof: str
    preds_plots: str
    reports_features: str
    reports_stable: str
    reports_panels: str
    reports_subgroups: str
    diag: str
    diag_splits: str
    diag_calibration: str
    diag_learning: str
    diag_dca: str
    diag_tuning: str
    diag_plots: str
    diag_screening: str
    diag_test_ci: str

    @classmethod
    def create(
        cls,
        root: str,
        exist_ok: bool = True,
        split_seed: int | None = None,
    ) -> "OutputDirectories":
        """
        Create output directory structure.

        Args:
            root: Base output directory path
            exist_ok: If True, do not raise if directories exist
            split_seed: If provided, creates a split-specific subdirectory
                        (e.g., root/split_seed0/) for multi-split runs

        Returns:
            OutputDirectories instance with all paths created

        Raises:
            OSError: If directory creation fails
        """
        root_path = Path(root)

        # If split_seed is provided, nest under a split-specific subdirectory
        if split_seed is not None:
            root_path = root_path / f"split_seed{split_seed}"

        # Define structure (relative to root)
        structure = {
            "core": "core",
            "cv": "cv",
            "preds_test": "preds/test_preds",
            "preds_val": "preds/val_preds",
            "preds_controls": "preds/controls_oof",
            "preds_train_oof": "preds/train_oof",
            "preds_plots": "preds/plots",
            "reports_features": "reports/feature_reports",
            "reports_stable": "reports/stable_panel",
            "reports_panels": "reports/panels",
            "reports_subgroups": "reports/subgroups",
            "diag": "diagnostics",
            "diag_splits": "diagnostics/splits",
            "diag_calibration": "diagnostics/calibration",
            "diag_learning": "diagnostics/learning_curve",
            "diag_dca": "diagnostics/dca",
            "diag_tuning": "diagnostics/tuning",
            "diag_plots": "diagnostics/plots",
            "diag_screening": "diagnostics/screening",
            "diag_test_ci": "diagnostics/test_ci_files",
        }

        # Create all directories
        paths = {"root": str(root_path)}
        for key, rel_path in structure.items():
            abs_path = root_path / rel_path
            abs_path.mkdir(parents=True, exist_ok=exist_ok)
            paths[key] = str(abs_path)

        logger.debug(f"Created output structure at: {root}")
        return cls(**paths)

    def get_path(self, category: str, filename: str) -> str:
        """
        Construct full path for a file in a specific category.

        Args:
            category: Directory category (e.g., "core", "cv", "reports_features")
            filename: File name

        Returns:
            Full file path as string

        Raises:
            ValueError: If category is invalid
        """
        if not hasattr(self, category):
            raise ValueError(f"Unknown output category: {category}")
        return os.path.join(getattr(self, category), filename)


class ResultsWriter:
    """
    High-level API for writing training results.

    Handles:
    - Metrics (val_metrics.csv, test_metrics.csv, cv_repeat_metrics.csv)
    - Predictions (test/val/train_oof)
    - Cross-validation artifacts (best params, selected proteins)
    - Reports (features, panels, subgroups)
    - Model artifacts (joblib serialization)
    - Run settings (JSON configuration)

    Usage:
        writer = ResultsWriter(output_dirs)
        writer.save_test_metrics(metrics_dict, scenario="IncidentPlusPrevalent", model="LR_EN")
        writer.save_model_artifact(model, metadata, scenario, model_name)
    """

    def __init__(self, output_dirs: OutputDirectories):
        """
        Initialize ResultsWriter.

        Args:
            output_dirs: OutputDirectories instance
        """
        self.dirs = output_dirs

    # ========== Settings and Configuration ==========

    def save_run_settings(self, settings: Dict[str, Any]) -> str:
        """
        Save run configuration to core/run_settings.json.

        Args:
            settings: Configuration dictionary (typically from argparse.Namespace)

        Returns:
            Path to saved file
        """
        path = self.dirs.get_path("core", "run_settings.json")
        with open(path, "w") as f:
            json.dump(settings, f, indent=2, sort_keys=True)
        logger.info(f"Saved run settings: {path}")
        return str(path)

    # ========== Metrics ==========

    def save_val_metrics(self, metrics: Dict[str, Any], scenario: str, model: str) -> str:
        """
        Save validation metrics to core/val_metrics.csv (append mode).

        Args:
            metrics: Validation metrics dictionary
            scenario: Scenario name (e.g., "IncidentPlusPrevalent")
            model: Model name (e.g., "LR_EN")

        Returns:
            Path to saved file
        """
        df = pd.DataFrame([metrics])
        df.insert(0, "scenario", scenario)
        df.insert(1, "model", model)

        path = self.dirs.get_path("core", "val_metrics.csv")

        # Append mode: add to existing file or create new
        if os.path.exists(path):
            df.to_csv(path, mode="a", header=False, index=False)
        else:
            df.to_csv(path, index=False)

        logger.info(f"Saved validation metrics: {path}")
        return str(path)

    def save_test_metrics(self, metrics: Dict[str, Any], scenario: str, model: str) -> str:
        """
        Save test metrics to core/test_metrics.csv (append mode).

        Args:
            metrics: Test metrics dictionary
            scenario: Scenario name
            model: Model name

        Returns:
            Path to saved file
        """
        df = pd.DataFrame([metrics])
        df.insert(0, "scenario", scenario)
        df.insert(1, "model", model)

        path = self.dirs.get_path("core", "test_metrics.csv")

        # Append mode: add to existing file or create new
        if os.path.exists(path):
            df.to_csv(path, mode="a", header=False, index=False)
        else:
            df.to_csv(path, index=False)

        logger.info(f"Saved test metrics: {path}")
        return str(path)

    def save_cv_repeat_metrics(
        self, cv_results: List[Dict[str, Any]], scenario: str, model: str
    ) -> str:
        """
        Save cross-validation repeat metrics to cv/cv_repeat_metrics.csv (append mode).

        Args:
            cv_results: List of metrics dictionaries (one per repeat)
            scenario: Scenario name
            model: Model name

        Returns:
            Path to saved file
        """
        df = pd.DataFrame(cv_results)
        # Only insert columns if not already present (caller may have added them)
        if "scenario" not in df.columns:
            df.insert(0, "scenario", scenario)
        if "model" not in df.columns:
            df.insert(1, "model", model)

        path = self.dirs.get_path("cv", "cv_repeat_metrics.csv")

        # Append mode: add to existing file or create new
        if os.path.exists(path):
            df.to_csv(path, mode="a", header=False, index=False)
        else:
            df.to_csv(path, index=False)

        logger.info(f"Saved CV repeat metrics: {path}")
        return str(path)

    def save_bootstrap_ci_metrics(self, metrics: Dict[str, Any], scenario: str, model: str) -> str:
        """
        Save bootstrap CI metrics to core/test_bootstrap_ci.csv.

        Args:
            metrics: Bootstrap CI metrics dictionary
            scenario: Scenario name
            model: Model name

        Returns:
            Path to saved file
        """
        df = pd.DataFrame([metrics])
        df.insert(0, "scenario", scenario)
        df.insert(1, "model", model)

        path = self.dirs.get_path("core", "test_bootstrap_ci.csv")
        df.to_csv(path, index=False)
        logger.info(f"Saved bootstrap CI metrics: {path}")
        return str(path)

    # ========== Cross-Validation Artifacts ==========

    def save_best_params_per_split(self, best_params: List[Dict[str, Any]], scenario: str) -> str:
        """
        Save best hyperparameters per outer split to cv/best_params_per_split.csv.

        Args:
            best_params: List of best param dicts (one per outer fold)
            scenario: Scenario name

        Returns:
            Path to saved file
        """
        df = pd.DataFrame(best_params)
        df.insert(0, "scenario", scenario)

        path = self.dirs.get_path("cv", "best_params_per_split.csv")
        df.to_csv(path, index=False)
        logger.info(f"Saved best params per split: {path}")
        return str(path)

    def save_selected_proteins_per_split(
        self, selected_proteins: List[Dict[str, Any]], scenario: str
    ) -> str:
        """
        Save selected proteins per outer split to cv/selected_proteins_per_outer_split.csv.

        Args:
            selected_proteins: List of dicts with outer_split and selected_proteins_split
            scenario: Scenario name

        Returns:
            Path to saved file
        """
        df = pd.DataFrame(selected_proteins)
        df.insert(0, "scenario", scenario)

        path = self.dirs.get_path("cv", "selected_proteins_per_outer_split.csv")
        df.to_csv(path, index=False)
        logger.info(f"Saved selected proteins per split: {path}")
        return str(path)

    # ========== Predictions ==========

    def save_test_predictions(self, predictions_df: pd.DataFrame, scenario: str, model: str) -> str:
        """
        Save test predictions to preds/test_preds/test_preds__{model}.csv.

        Args:
            predictions_df: DataFrame with ID, y_true, predictions
            scenario: Scenario name
            model: Model name

        Returns:
            Path to saved file
        """
        filename = f"test_preds__{model}.csv"
        path = self.dirs.get_path("preds_test", filename)
        predictions_df.to_csv(path, index=False)
        logger.info(f"Saved test predictions: {path}")
        return str(path)

    def save_val_predictions(self, predictions_df: pd.DataFrame, scenario: str, model: str) -> str:
        """
        Save validation predictions to preds/val_preds/val_preds__{model}.csv.

        Args:
            predictions_df: DataFrame with ID, y_true, predictions
            scenario: Scenario name
            model: Model name

        Returns:
            Path to saved file
        """
        filename = f"val_preds__{model}.csv"
        path = self.dirs.get_path("preds_val", filename)
        predictions_df.to_csv(path, index=False)
        logger.info(f"Saved validation predictions: {path}")
        return str(path)

    def save_train_oof_predictions(
        self, predictions_df: pd.DataFrame, scenario: str, model: str
    ) -> str:
        """
        Save train out-of-fold predictions to preds/train_oof/train_oof__{model}.csv.

        Args:
            predictions_df: DataFrame with ID, y_true, OOF predictions
            scenario: Scenario name
            model: Model name

        Returns:
            Path to saved file
        """
        filename = f"train_oof__{model}.csv"
        path = self.dirs.get_path("preds_train_oof", filename)
        predictions_df.to_csv(path, index=False)
        logger.info(f"Saved train OOF predictions: {path}")
        return str(path)

    def save_controls_predictions(
        self, predictions_df: pd.DataFrame, scenario: str, model: str
    ) -> str:
        """
        Save control subjects predictions to preds/controls/controls_risk__{model}__oof_mean.csv.

        Args:
            predictions_df: DataFrame with control subject predictions
            scenario: Scenario name
            model: Model name

        Returns:
            Path to saved file
        """
        filename = f"controls_risk__{model}__oof_mean.csv"
        path = self.dirs.get_path("preds_controls", filename)
        predictions_df.to_csv(path, index=False)
        logger.info(f"Saved controls predictions: {path}")
        return str(path)

    # ========== Reports ==========

    def save_feature_report(self, report_df: pd.DataFrame, scenario: str, model: str) -> str:
        """
        Save feature importance report to reports/feature_reports/{model}__feature_report_train.csv.

        Args:
            report_df: DataFrame with protein, effect_size, p_value, selection_freq
            scenario: Scenario name
            model: Model name

        Returns:
            Path to saved file
        """
        filename = f"{model}__feature_report_train.csv"
        path = self.dirs.get_path("reports_features", filename)
        report_df.to_csv(path, index=False)
        logger.info(f"Saved feature report: {path}")
        return str(path)

    def save_stable_panel_report(
        self, panel_df: pd.DataFrame, scenario: str, panel_type: str = "KBest"
    ) -> str:
        """
        Save stable panel report to reports/stable_panel/stable_panel__{panel_type}.csv.

        Args:
            panel_df: DataFrame with stable panel proteins and statistics
            scenario: Scenario name
            panel_type: Panel type (e.g., "KBest", "screening")

        Returns:
            Path to saved file
        """
        filename = f"stable_panel__{panel_type}.csv"
        path = self.dirs.get_path("reports_stable", filename)
        panel_df.to_csv(path, index=False)
        logger.info(f"Saved stable panel report: {path}")
        return str(path)

    def save_panel_manifest(
        self, manifest: Dict[str, Any], scenario: str, model: str, panel_size: int
    ) -> str:
        """
        Save panel manifest to reports/panels/{model}__N{size}__panel_manifest.json.

        Args:
            manifest: Panel metadata dictionary
            scenario: Scenario name
            model: Model name
            panel_size: Panel size (N)

        Returns:
            Path to saved file
        """
        filename = f"{model}__N{panel_size}__panel_manifest.json"
        path = self.dirs.get_path("reports_panels", filename)
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        logger.info(f"Saved panel manifest: {path}")
        return str(path)

    def save_final_test_panel(
        self, panel_proteins: List[str], scenario: str, model: str, metadata: Dict[str, Any] = None
    ) -> str:
        """
        Save final test panel (proteins used in final model) to reports/panels/{model}__final_test_panel.json.

        Args:
            panel_proteins: List of protein names selected in final model
            scenario: Scenario name
            model: Model name
            metadata: Optional metadata dict (e.g., selection_method, n_train, timestamp)

        Returns:
            Path to saved file
        """
        manifest = {
            "scenario": scenario,
            "model": model,
            "panel_size": len(panel_proteins),
            "proteins": sorted(panel_proteins),
            "metadata": metadata or {},
        }
        filename = f"{model}__final_test_panel.json"
        path = self.dirs.get_path("reports_panels", filename)
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        logger.info(f"Saved final test panel: {path}")
        return str(path)

    def save_subgroup_metrics(self, subgroup_df: pd.DataFrame, scenario: str, model: str) -> str:
        """
        Save subgroup analysis to reports/subgroups/{model}__test_subgroup_metrics.csv.

        Args:
            subgroup_df: DataFrame with per-subgroup metrics
            scenario: Scenario name
            model: Model name

        Returns:
            Path to saved file
        """
        filename = f"{model}__test_subgroup_metrics.csv"
        path = self.dirs.get_path("reports_subgroups", filename)
        subgroup_df.to_csv(path, index=False)
        logger.info(f"Saved subgroup metrics: {path}")
        return str(path)

    # ========== Model Artifacts ==========

    def save_model_artifact(
        self, model: Any, metadata: Dict[str, Any], scenario: str, model_name: str
    ) -> str:
        """
        Save trained model artifact to core/{model}__final_model.joblib.

        Args:
            model: Trained model object (sklearn pipeline or estimator)
            metadata: Model metadata (hyperparams, thresholds, etc.)
            scenario: Scenario name
            model_name: Model name

        Returns:
            Path to saved file

        Raises:
            IOError: If serialization fails
        """
        bundle = {
            "model": model,
            "metadata": metadata,
            "scenario": scenario,
            "model_name": model_name,
        }

        filename = f"{model_name}__final_model.joblib"
        path = self.dirs.get_path("core", filename)

        try:
            joblib.dump(bundle, path, compress=3)
            logger.info(f"Saved model artifact: {path}")
            return str(path)
        except Exception as e:
            logger.error(f"Failed to save model artifact: {e}")
            raise OSError(f"Model serialization failed: {e}") from e

    def load_model_artifact(self, scenario: str, model_name: str) -> Dict[str, Any]:
        """
        Load trained model artifact from core/{model}__final_model.joblib.

        Args:
            scenario: Scenario name
            model_name: Model name

        Returns:
            Bundle dictionary with keys: model, metadata, scenario, model_name

        Raises:
            FileNotFoundError: If artifact does not exist
            IOError: If deserialization fails
        """
        filename = f"{model_name}__final_model.joblib"
        path = self.dirs.get_path("core", filename)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model artifact not found: {path}")

        try:
            bundle = joblib.load(path)
            logger.info(f"Loaded model artifact: {path}")
            return bundle
        except Exception as e:
            logger.error(f"Failed to load model artifact: {e}")
            raise OSError(f"Model deserialization failed: {e}") from e

    # ========== Diagnostics ==========

    def save_calibration_curve(
        self, calibration_df: pd.DataFrame, scenario: str, model: str
    ) -> str:
        """
        Save calibration curve data to diagnostics/calibration/{model}__calibration.csv.

        Args:
            calibration_df: DataFrame with prob_pred, prob_true columns
            scenario: Scenario name
            model: Model name

        Returns:
            Path to saved file
        """
        filename = f"{model}__calibration.csv"
        path = self.dirs.get_path("diag_calibration", filename)
        calibration_df.to_csv(path, index=False)
        logger.debug(f"Saved calibration curve: {path}")
        return str(path)

    def save_learning_curve(
        self, learning_curve_df: pd.DataFrame, scenario: str, model: str
    ) -> str:
        """
        Save learning curve data to diagnostics/learning/{model}__learning_curve.csv.

        Args:
            learning_curve_df: DataFrame with train_size, metric, train/test scores
            scenario: Scenario name
            model: Model name

        Returns:
            Path to saved file
        """
        filename = f"{model}__learning_curve.csv"
        path = self.dirs.get_path("diag_learning", filename)
        learning_curve_df.to_csv(path, index=False)
        logger.debug(f"Saved learning curve: {path}")
        return str(path)

    def save_split_trace(self, split_trace_df: pd.DataFrame, scenario: str) -> str:
        """
        Save split trace to diagnostics/splits/train_test_split_trace.csv.

        Args:
            split_trace_df: DataFrame with split assignments and labels
            scenario: Scenario name

        Returns:
            Path to saved file
        """
        filename = "train_test_split_trace.csv"
        path = self.dirs.get_path("diag_splits", filename)
        split_trace_df.to_csv(path, index=False)
        logger.debug(f"Saved split trace: {path}")
        return str(path)

    # ========== Utility Methods ==========

    def summarize_outputs(self) -> List[str]:
        """
        Return list of key output files that exist.

        Returns:
            List of relative paths (from root) for existing key files
        """
        key_files = [
            ("core", "val_metrics.csv"),
            ("core", "test_metrics.csv"),
            ("core", "run_settings.json"),
            ("cv", "cv_repeat_metrics.csv"),
            ("cv", "best_params_per_split.csv"),
            ("cv", "selected_proteins_per_outer_split.csv"),
        ]

        existing = []
        for category, filename in key_files:
            path = self.dirs.get_path(category, filename)
            if os.path.exists(path):
                rel_path = os.path.relpath(path, self.dirs.root)
                existing.append(rel_path)

        return existing
