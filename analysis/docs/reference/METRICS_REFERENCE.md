# Metrics Reference

## Overview

This document describes the behavior of all metrics used in the CeliacRisks ML pipeline, with special focus on edge cases and single-class handling.

## Metric Categories

### 1. Discrimination Metrics

Discrimination metrics evaluate a model's ability to rank positive cases higher than negative cases, independent of the chosen decision threshold.

#### AUROC (Area Under ROC Curve)

**Module**: `ced_ml.metrics.discrimination.auroc`

**Description**: Measures the probability that a randomly chosen positive case has a higher predicted score than a randomly chosen negative case.

**Range**: [0.0, 1.0]
- 1.0: Perfect discrimination
- 0.5: No discrimination (random classifier)
- <0.5: Worse than random (usually indicates label swap)

**Single-class behavior**: Returns `np.nan` with UserWarning if only one class present

**Example**:
```python
from ced_ml.metrics.discrimination import auroc
import numpy as np

y_true = np.array([0, 0, 1, 1])
y_pred = np.array([0.1, 0.4, 0.6, 0.9])
score = auroc(y_true, y_pred)  # Returns 1.0
```

**Single-class example**:
```python
y_true = np.array([1, 1, 1, 1])  # Only positives
y_pred = np.array([0.1, 0.4, 0.6, 0.9])
score = auroc(y_true, y_pred)  # Returns np.nan with warning
```

---

#### PR-AUC (Precision-Recall Area Under Curve)

**Module**: `ced_ml.metrics.discrimination.prauc`

**Description**: More informative than AUROC for imbalanced datasets where the positive class is rare. Focuses on positive class predictions and penalizes false positives more heavily.

**Range**: [0.0, 1.0]
- 1.0: Perfect precision and recall
- Baseline: prevalence (random classifier)

**Single-class behavior**: Returns `np.nan` with UserWarning if only one class present

**Example**:
```python
from ced_ml.metrics.discrimination import prauc
import numpy as np

y_true = np.array([0, 0, 1, 1])
y_pred = np.array([0.1, 0.4, 0.6, 0.9])
score = prauc(y_true, y_pred)  # Returns ~1.0
```

---

#### Youden's J Statistic

**Module**: `ced_ml.metrics.discrimination.youden_j`

**Description**: Maximum vertical distance between the ROC curve and the diagonal (random classifier line). Identifies the threshold that optimally balances sensitivity and specificity.

**Formula**: J = max(TPR - FPR) = max(Sensitivity + Specificity - 1)

**Range**: [0.0, 1.0]
- 1.0: Perfect separation (TPR=1, FPR=0)
- 0.0: No discrimination beyond chance

**Single-class behavior**: Returns `np.nan` with UserWarning if only one class present

**Example**:
```python
from ced_ml.metrics.discrimination import youden_j
import numpy as np

y_true = np.array([0, 0, 1, 1])
y_pred = np.array([0.1, 0.4, 0.6, 0.9])
j_stat = youden_j(y_true, y_pred)  # Returns 1.0
```

---

#### Alpha (Sensitivity at Target Specificity)

**Module**: `ced_ml.metrics.discrimination.alpha_sensitivity_at_specificity`

**Description**: Evaluates model performance at high-specificity operating points, useful for clinical screening where false positives must be minimized.

**Parameters**:
- `target_specificity`: Target specificity level (default: 0.95 for 95% specificity)

**Range**: [0.0, 1.0]
- Returns max sensitivity among thresholds meeting target specificity
- If target unachievable: returns sensitivity at closest achievable specificity

**Single-class behavior**: Not applicable (does not use `_validate_binary_labels` guard)

**Example**:
```python
from ced_ml.metrics.discrimination import alpha_sensitivity_at_specificity
import numpy as np

y_true = np.array([0, 0, 1, 1])
y_pred = np.array([0.1, 0.4, 0.6, 0.9])
alpha = alpha_sensitivity_at_specificity(y_true, y_pred, target_specificity=0.95)
```

---

### 2. Custom Scorers

#### TPR at FPR Score

**Module**: `ced_ml.metrics.scorers.tpr_at_fpr_score`

**Description**: Calculate TPR (sensitivity) at a target FPR (1 - specificity). Used for hyperparameter optimization at clinical operating points.

**Parameters**:
- `target_fpr`: Target false positive rate (e.g., 0.05 for 95% specificity)

**Returns**:
- Maximum TPR achievable at or below target_fpr
- Returns 0.0 if no threshold achieves target_fpr
- Returns `np.nan` if only one class present (single-class guard)

**Single-class behavior**: Returns `np.nan` for consistency with discrimination metrics

**Example**:
```python
from ced_ml.metrics.scorers import tpr_at_fpr_score
import numpy as np

y_true = np.array([0, 0, 0, 1, 1, 1])
y_score = np.array([0.1, 0.2, 0.3, 0.9, 0.95, 0.99])
tpr = tpr_at_fpr_score(y_true, y_score, target_fpr=0.05)  # Returns 1.0
```

**Single-class example**:
```python
y_true = np.array([0, 0, 0, 0])  # Only negatives
y_score = np.array([0.1, 0.2, 0.3, 0.4])
tpr = tpr_at_fpr_score(y_true, y_score, target_fpr=0.05)  # Returns np.nan
```

---

### 3. Calibration Metrics

#### Brier Score

**Module**: `ced_ml.metrics.discrimination.compute_brier_score`

**Description**: Mean squared error of predicted probabilities. Lower is better.

**Range**: [0.0, 1.0]
- 0.0: Perfect calibration and discrimination
- 0.25: Baseline for balanced dataset (constant 0.5 prediction)

**Single-class behavior**: No guard (computes normally, may return unexpected values)

**Example**:
```python
from ced_ml.metrics.discrimination import compute_brier_score
import numpy as np

y_true = np.array([0, 0, 1, 1])
y_pred = np.array([0.1, 0.1, 0.9, 0.9])
brier = compute_brier_score(y_true, y_pred)  # Returns 0.02
```

---

#### Log Loss

**Module**: `ced_ml.metrics.discrimination.compute_log_loss`

**Description**: Cross-entropy loss with numerical stability clipping. Heavily penalizes confident wrong predictions.

**Parameters**:
- `eps`: Clipping threshold to avoid log(0) (default: 1e-15)

**Range**: [0.0, ∞)
- 0.0: Perfect predictions
- log(2) ≈ 0.693: Random predictor (p=0.5 always)

**Single-class behavior**: No guard (computes normally)

**Example**:
```python
from ced_ml.metrics.discrimination import compute_log_loss
import numpy as np

y_true = np.array([0, 0, 1, 1])
y_pred = np.array([0.1, 0.1, 0.9, 0.9])
ll = compute_log_loss(y_true, y_pred)  # Returns ~0.105
```

---

## Single-Class Edge Cases in Cross-Validation

### Why Single-Class Cases Occur

In nested cross-validation with imbalanced datasets (e.g., 0.34% prevalence), some CV folds may contain only positive or only negative samples by chance. This is especially common when:
- Using small inner CV folds (e.g., 3-fold CV)
- Working with rare outcomes (e.g., incident disease)
- Applying downsampling to control samples

### Design Decision: Return NaN

**Rationale**: Metrics that cannot be meaningfully computed on single-class data return `np.nan` instead of:
- Raising an error (would break CV loops)
- Returning 0.0 or 1.0 (would bias metric aggregation)
- Passing through sklearn's behavior (inconsistent across metrics)

**Consistency**: All discrimination metrics (`auroc`, `prauc`, `youden_j`) and custom scorers (`tpr_at_fpr_score`) use the same `_validate_binary_labels` helper to ensure uniform behavior.

### Handling NaN in Downstream Code

When aggregating metrics across CV splits, use `np.nanmean` to ignore NaN values:

```python
import numpy as np

# CV scores may contain NaN for single-class splits
cv_scores = np.array([0.85, 0.88, np.nan, 0.87, 0.90])

# Use nanmean to aggregate (ignores NaN)
mean_score = np.nanmean(cv_scores)  # Returns 0.875
std_score = np.nanstd(cv_scores)    # Returns ~0.0204
```

### Warning Messages

Single-class cases trigger a `UserWarning`:

```
UserWarning: AUROC requires both classes (0 and 1) in y_true,
but only found [1]. Returning NaN.
```

These warnings are **expected** and **harmless** during cross-validation. To suppress them:

```python
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning,
                          message=".*requires both classes.*")
    # Run CV here
    scores = cross_val_score(clf, X, y, cv=3, scoring=scorer)
```

---

## Aggregated Metrics

### `compute_discrimination_metrics`

**Module**: `ced_ml.metrics.discrimination.compute_discrimination_metrics`

**Description**: Compute all discrimination metrics in one pass for efficiency. Reuses computed ROC curves.

**Parameters**:
- `y_true`: True binary labels (0/1)
- `y_pred`: Predicted probabilities
- `include_youden`: Whether to compute Youden's J (default: True)
- `include_alpha`: Whether to compute Alpha (default: True)
- `alpha_target_specificity`: Target specificity for Alpha (default: 0.95)

**Returns**: Dictionary with keys:
- `"AUROC"`: Area under ROC curve (NaN if single-class)
- `"PR_AUC"`: Precision-Recall AUC (NaN if single-class)
- `"Youden"`: Youden's J statistic (NaN if single-class)
- `"Alpha"`: Sensitivity at target specificity (NaN if single-class)

**Single-class behavior**: Returns all NaN values with one UserWarning

**Example**:
```python
from ced_ml.metrics.discrimination import compute_discrimination_metrics
import numpy as np

y_true = np.array([0, 0, 1, 1])
y_pred = np.array([0.1, 0.4, 0.6, 0.9])
metrics = compute_discrimination_metrics(y_true, y_pred)

print(metrics)
# {'AUROC': 1.0, 'PR_AUC': 1.0, 'Youden': 1.0, 'Alpha': 1.0}
```

**Single-class example**:
```python
y_true = np.array([1, 1, 1, 1])  # Only positives
y_pred = np.array([0.1, 0.4, 0.6, 0.9])
metrics = compute_discrimination_metrics(y_true, y_pred)

print(metrics)
# {'AUROC': nan, 'PR_AUC': nan, 'Youden': nan, 'Alpha': nan}
```

---

## Threshold-Dependent Metrics

### Sensitivity, Specificity, PPV, NPV

**Module**: `ced_ml.metrics.thresholds`

**Description**: Classification metrics that depend on a decision threshold.

**Single-class behavior**: No guards (computed normally, may return undefined values)

**Note**: These metrics are computed **after** threshold selection on validation/test sets, where single-class cases should not occur. If they do, results may be undefined.

---

## References

- [ADR-002](adr/ADR-002-prevalent-train-only.md): Prevalent cases in training only
- [ADR-004](adr/ADR-004-auroc-optimization.md): AUROC optimization
- [ADR-008](adr/ADR-008-nested-cv.md): Nested CV
- [ADR-010](adr/ADR-010-fixed-spec-95.md): Fixed specificity 0.95

---

## Testing

Comprehensive tests for single-class behavior:
- `tests/test_metrics_discrimination.py::TestSingleClassGuards`
- `tests/test_metrics_scorers.py::TestScorerEdgeCases`

Run tests:
```bash
pytest tests/test_metrics_discrimination.py -v
pytest tests/test_metrics_scorers.py -v
```

---

**Last Updated**: 2026-01-21
