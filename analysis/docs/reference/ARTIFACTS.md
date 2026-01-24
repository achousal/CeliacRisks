# CeliacRisks Output Artifacts

**Version:** 1.0
**Date:** 2026-01-24
**Status:** Reference documentation for output artifacts

---

## Table of Contents

1. [Directory Structure](#1-directory-structure)
2. [Core Artifacts](#2-core-artifacts)
3. [Cross-Validation Artifacts](#3-cross-validation-artifacts)
4. [Plots](#4-plots)
5. [File Formats](#5-file-formats)

---

## 1. Directory Structure

### 1.1 Single Split Output

```
results_hpc/{model}/split_seed{N}/
  core/
    final_model.pkl               # Trained sklearn model (pickled)
    oof_predictions.csv           # OOF predictions (TRAIN set)
    val_predictions.csv           # VAL predictions
    test_predictions.csv          # TEST predictions
    train_metrics.json            # TRAIN metrics
    val_metrics.json              # VAL metrics
    test_metrics.json             # TEST metrics
    run_settings.json             # Full config + metadata
    stable_features.txt           # Stability panel proteins
  cv/
    cv_repeat_metrics.csv         # Per-repeat OOF metrics
    best_params.csv               # Best hyperparameters per fold
    optuna/                       # (if Optuna enabled)
      optuna_config.json          # Optuna settings
      best_params_optuna.csv      # Best params with trial metadata
      study.pkl                   # Optuna study object
  plots/
    roc_pr.png                    # ROC + PR curves
    calibration.png               # Calibration plot
    risk_dist.png                 # Risk distribution
    dca.png                       # Decision curve analysis
    oof_roc.png                   # OOF ROC with confidence bands
    oof_pr.png                    # OOF PR with confidence bands
    oof_calibration.png           # OOF calibration plot
  diag_splits/
    train_test_split_trace.csv    # Split assignment trace
```

### 1.2 Aggregated Output (Multiple Splits)

```
results_hpc/{model}/aggregated/
  aggregate_metrics.json          # Mean ± SE across splits
  pooled_predictions.csv          # Predictions from all splits
  pooled_metrics.json             # Metrics on pooled predictions
  consensus_panel_*.txt           # Proteins selected across splits
  per_split_summary.csv           # Per-split performance
```

---

## 2. Core Artifacts

### 2.1 final_model.pkl

**Type:** Pickled Python object

**Contents:** Trained sklearn-compatible model (e.g., sklearn.linear_model.LogisticRegression, xgboost.XGBClassifier) with calibration wrapper if enabled.

**Metadata stored separately in:**
- `run_settings.json` - Full config, hyperparameters, feature names
- `stable_features.txt` - Selected feature panel

**Usage:**
```python
import pickle
with open('final_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Standard sklearn predictions at training prevalence (16.7%)
y_pred = model.predict_proba(X)[:, 1]
```

**Note on prevalence:** Predictions are calibrated at training prevalence (16.7%). For real-world deployment at different prevalence (e.g., 0.34%), see [ADR-010](../adr/ADR-010-prevalence-adjustment.md) for speculative adjustment strategies.

### 2.2 Prediction CSVs

**Format:** CSV with headers

**Columns:**
- `eid` - Sample identifier
- `y_true` - True label (0/1)
- `y_pred_proba` - Predicted probability
- `y_pred` - Binary prediction (using selected threshold)
- `fold` - CV fold index (for OOF predictions)
- `repeat` - CV repeat index (for OOF predictions)

**Example:**
```csv
eid,y_true,y_pred_proba,y_pred,fold,repeat
1001,0,0.012,0,0,0
1002,1,0.842,1,0,0
1003,0,0.098,0,1,0
```

**Files:**
- `oof_predictions.csv` - TRAIN set OOF predictions (5 folds × 10 repeats)
- `val_predictions.csv` - VAL set predictions
- `test_predictions.csv` - TEST set predictions

### 2.3 Metrics JSONs

**Format:** JSON with numeric values

**Common fields:**
- `auroc` - Area under ROC curve
- `prauc` - Area under Precision-Recall curve
- `brier` - Brier score (lower is better)
- `threshold` - Selected decision threshold
- `sensitivity` - True positive rate
- `specificity` - True negative rate
- `ppv` - Positive predictive value
- `npv` - Negative predictive value
- `accuracy` - Overall accuracy
- `f1` - F1 score
- `calibration_slope` - Calibration slope (1.0 = perfect)
- `calibration_intercept` - Calibration intercept (0.0 = perfect)

**Example:**
```json
{
  "auroc": 0.8534,
  "prauc": 0.4231,
  "brier": 0.0812,
  "threshold": 0.3521,
  "sensitivity": 0.7812,
  "specificity": 0.8241,
  "ppv": 0.1234,
  "npv": 0.9876,
  "accuracy": 0.8198,
  "f1": 0.2134,
  "calibration_slope": 0.987,
  "calibration_intercept": 0.012
}
```

**Files:**
- `train_metrics.json` - TRAIN set metrics
- `val_metrics.json` - VAL set metrics
- `test_metrics.json` - TEST set metrics

### 2.4 run_settings.json

**Format:** JSON with nested structure

**Contents:**
- Full resolved configuration (all parameters)
- Resolved metadata columns (auto-detected or explicit)
- Split seed, random state
- Model hyperparameters (selected via CV)
- Feature selection parameters
- Threshold selection settings
- Software versions (Python, numpy, sklearn, xgboost, etc.)
- Timestamp, runtime
- Git commit hash (if available)

**Purpose:** Complete provenance for reproducibility.

### 2.5 stable_features.txt

**Format:** Plain text, one feature per line

**Contents:** List of protein features selected in ≥75% of CV folds (stability panel).

**Example:**
```
APOE_resid
SERPINA1_resid
ALB_resid
TTR_resid
...
```

---

## 3. Cross-Validation Artifacts

### 3.1 cv_repeat_metrics.csv

**Format:** CSV with headers

**Columns:**
- `repeat` - CV repeat index (0-9)
- `fold` - CV fold index (0-4)
- `auroc` - OOF AUROC for this fold
- `prauc` - OOF PR-AUC for this fold
- `brier` - OOF Brier score for this fold
- Additional metrics...

**Purpose:** Per-fold performance for stability analysis.

### 3.2 best_params.csv

**Format:** CSV with headers

**Columns:**
- `repeat` - CV repeat index
- `fold` - CV fold index
- Model-specific hyperparameters (e.g., `C`, `max_depth`, `learning_rate`)

**Purpose:** Track hyperparameter selection across CV folds.

### 3.3 Optuna Artifacts (if enabled)

**optuna_config.json:**
- Optuna settings (n_trials, sampler, pruner)
- Study metadata

**best_params_optuna.csv:**
- Best hyperparameters per trial
- Trial metadata (value, duration, state)

**study.pkl:**
- Pickled Optuna study object for resume/analysis

---

## 4. Plots

### 4.1 roc_pr.png

**Type:** PNG image (matplotlib figure)

**Contents:** Dual-panel plot:
- Left: ROC curve with AUROC annotation
- Right: Precision-Recall curve with PR-AUC annotation

**Colors:** TRAIN (blue), VAL (orange), TEST (green)

### 4.2 calibration.png

**Type:** PNG image (matplotlib figure)

**Contents:** Calibration plot (predicted vs. observed probabilities) with:
- Perfect calibration line (diagonal)
- Observed calibration curve (binned)
- Calibration slope/intercept annotations

**Purpose:** Assess probability calibration quality.

### 4.3 risk_dist.png

**Type:** PNG image (matplotlib figure)

**Contents:** Histogram of predicted probabilities:
- Separate distributions for cases (red) and controls (blue)
- Selected threshold vertical line (dashed)

**Purpose:** Visualize risk score separation.

### 4.4 dca.png

**Type:** PNG image (matplotlib figure)

**Contents:** Decision Curve Analysis plot:
- Net benefit vs. threshold probability
- Model curve vs. "treat all" and "treat none" strategies
- Auto-ranged threshold axis based on prevalence

**Purpose:** Evaluate clinical utility at different decision thresholds.

### 4.5 OOF Plots (oof_roc.png, oof_pr.png, oof_calibration.png)

**Type:** PNG images (matplotlib figures)

**Contents:** Out-of-fold predictions across CV repeats:
- Mean curve across repeats
- 95% confidence bands (shaded region)
- Individual repeat curves (faint lines)

**Purpose:** Assess model stability across CV repeats.

### 4.6 Optuna Plots (if enabled)

Generated in `cv/optuna/` subdirectory:
- `optimization_history.png` - Trial objective values over time
- `param_importances.png` - Feature importance bar chart
- `parallel_coordinate.png` - Hyperparameter interactions
- `slice.png` - Marginal effects of each hyperparameter

---

## 5. File Formats

### 5.1 Split Index CSVs

**Location:** `splits/` directory (sibling to `results/`)

**Format:** Single-column CSV

**Filename pattern:** `{scenario}_{split}_idx_seed{N}.csv`

**Example:**
```
splits/
  IncidentPlusPrevalent_train_idx_seed42.csv
  IncidentPlusPrevalent_val_idx_seed42.csv
  IncidentPlusPrevalent_test_idx_seed42.csv
```

**Contents:**
```csv
row_idx
0
5
12
...
```

**Purpose:** Reproducible split indices (language-agnostic, version-controllable).

### 5.2 Aggregated Metrics JSON

**Location:** `results_hpc/{model}/aggregated/aggregate_metrics.json`

**Format:** JSON with nested structure

**Example:**
```json
{
  "test": {
    "auroc_mean": 0.8534,
    "auroc_se": 0.0234,
    "auroc_min": 0.8102,
    "auroc_max": 0.8876,
    "auroc_median": 0.8547,
    "auroc_q1": 0.8312,
    "auroc_q3": 0.8702,
    "prauc_mean": 0.4231,
    "prauc_se": 0.0512,
    ...
  },
  "val": { ... }
}
```

**Purpose:** Summary statistics across multiple split seeds.

### 5.3 Consensus Panels

**Location:** `results_hpc/{model}/aggregated/consensus_panel_*.txt`

**Format:** Plain text, one feature per line

**Example:**
```
consensus_panel_75pct.txt  # Proteins in ≥75% of splits
consensus_panel_50pct.txt  # Proteins in ≥50% of splits
```

**Purpose:** Identify features consistently selected across different data partitions.

---

## File Size Estimates

**Typical sizes for 43,960-sample dataset:**
- `final_model.pkl`: 1-10 MB (depends on model type)
- `oof_predictions.csv`: 5-10 MB (TRAIN set)
- `val_predictions.csv`: 2-5 MB
- `test_predictions.csv`: 2-5 MB
- `*_metrics.json`: <1 KB each
- `run_settings.json`: 5-20 KB
- Plots: 50-200 KB each
- `cv_repeat_metrics.csv`: 10-50 KB
- `best_params.csv`: 5-20 KB
- Split index CSVs: 500 KB - 2 MB each

**Total per split:** ~20-50 MB

**Total for 10 splits:** ~200-500 MB

---

## Artifact Retention Policy

**Development runs:**
- Keep: `*_metrics.json`, `run_settings.json`, plots
- Optional: predictions CSVs, CV artifacts
- Discard: `final_model.pkl` (unless best model)

**Production runs:**
- Keep all artifacts for reproducibility
- Archive to long-term storage after validation

---

**End of ARTIFACTS.md**
