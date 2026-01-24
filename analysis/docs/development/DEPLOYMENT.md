# Real-World Deployment: Best Practices (Speculative)

**Version:** 1.0
**Date:** 2026-01-24
**Purpose:** Design guide for deploying ML models to clinical settings (not currently integrated into training pipeline)

---

## Overview

This document outlines best practices for deploying proteomics-based ML models to real-world clinical settings. **These practices are NOT currently integrated into the training pipeline** — they represent a roadmap for future deployment workflows.

**Key Design Principles:**
- Probabilities must be calibrated for the target deployment population
- Clinical thresholds must be validated on deployment data
- All deployment logic should be external to training (separation of concerns)

---

## 1. Dataset Prevalence Practice

### Training Configuration
Models are trained using controlled case:control downsampling to balance computational efficiency with statistical power:

- **Training/Validation/Test Prevalence:** 16.7% (5:1 case:control ratio)
- **UK Biobank Dataset Prevalence:** 0.34% (148 incident cases / 43,960 samples)
- **Downsampling Ratio:** Controls are randomly sampled at 5:1 ratio to cases

This training strategy (see [ADR-003](../adr/ADR-003-control-downsampling.md)) provides:
- Faster training (740 controls vs. 43,662 controls)
- Better class balance for optimization
- Consistent prevalence across train/val/test splits

---

## 2. Real-World Deployment Prevalence

### Epidemiological Context

Celiac disease prevalence varies substantially by population:

**Global Prevalence (All Cases):** 0.7-1.4% [1,2]
- Serologic screening: ~1.4%
- Biopsy-confirmed: ~0.7%

**Regional Variation:** [2]
- Europe: 0.8%
- North America: 0.5%
- Asia: 0.6%
- South America: 0.4%

**Incidence (New Cases):** 12-21 per 100,000 person-years (~0.012-0.021% per year) [3,4]
- Women: 17.4 per 100,000 person-years
- Men: 7.8 per 100,000 person-years
- Children: 21.3 per 100,000 person-years
- Increasing 7.5% annually [3]

**Population-Specific Factors:**
- Higher in females and children [1,2]
- Risk factors: family history, HLA-DQ2/DQ8 genotype, symptoms
- Geographic and ethnic variation

### Deployment Prevalence Estimation

For clinical deployment, prevalence must be estimated from the **target screening population**, not assumed from the UK Biobank cohort (0.34%) or global statistics. Target prevalence depends on:

1. **Population Selection Criteria:**
   - Symptomatic vs. asymptomatic screening
   - High-risk (family history, HLA+) vs. general population
   - Age and sex distribution

2. **Geographic and Temporal Factors:**
   - Regional prevalence variation
   - Secular trends (increasing incidence)
   - Healthcare system and diagnostic practices

3. **Cohort Characteristics:**
   - Incident vs. prevalent cases
   - Time window for outcome assessment
   - Competing risks (mortality, loss to follow-up)

---

## 3. Prevalence Adjustment for Deployment

### Why Adjustment Is Needed

When deploying a model trained at one prevalence (16.7%) to a population with different prevalence (e.g., 0.34%-1.4%), predicted probabilities require adjustment to maintain calibration.

**Example:**
```
Model prediction (at training prevalence 16.7%): p = 0.30
Adjusted for deployment (at 0.34% prevalence):    p ≈ 0.006
Adjusted for deployment (at 1.0% prevalence):     p ≈ 0.018
```

Without adjustment, predicted probabilities will be scaled for the training prevalence, not the deployment population.

### Recommended Method: Logit-Scale Adjustment

### Formula

Steyerberg (2019) describes a mathematically principled approach using logit-space adjustment:

```
P_adjusted(Y=1|X) = sigmoid(logit(p) + logit(π_new) - logit(π_old))

where:
  p           = Model's predicted probability (at training prevalence 16.7%)
  π_old       = Training/test prevalence (0.167)
  π_new       = Target deployment prevalence (e.g., 0.0034 for UK Biobank cohort)
  sigmoid(x)  = 1 / (1 + exp(-x))
  logit(π)    = log(π / (1 - π))

**Important:** π_new must be estimated from the target deployment population, not assumed to be 0.0034. The UK Biobank prevalence (0.34%) is cohort-specific and may differ from clinical screening populations.
```

### Why This Works

1. **Mathematically sound:** Derived from Bayes' theorem; assumes threshold classification doesn't change
2. **Preserves discrimination:** AUROC, PR-AUC remain unchanged
3. **Adjusts calibration:** Probability calibration shifts to deployment prevalence
4. **Simple implementation:** One-line formula

### Implementation Pattern

```python
import numpy as np
from scipy.special import expit, logit

def adjust_probabilities_for_prevalence(p, prevalence_old, prevalence_new):
    """
    Adjust predicted probabilities from one prevalence to another.

    Args:
        p (array): Predicted probabilities at prevalence_old
        prevalence_old (float): Training prevalence (0.167)
        prevalence_new (float): Target deployment prevalence
                                (e.g., 0.0034 for UK Biobank;
                                 must be estimated for target population)

    Returns:
        array: Adjusted probabilities at prevalence_new
    """
    # Ensure valid inputs
    if not (0 < prevalence_old < 1):
        raise ValueError(f"prevalence_old must be in (0, 1), got {prevalence_old}")
    if not (0 < prevalence_new < 1):
        raise ValueError(f"prevalence_new must be in (0, 1), got {prevalence_new}")
    if not np.all((p >= 0) & (p <= 1)):
        raise ValueError("Predicted probabilities must be in [0, 1]")

    # Apply logit-scale adjustment
    logit_p = logit(np.clip(p, 1e-10, 1 - 1e-10))  # Avoid inf/nan
    adjustment = logit(prevalence_new) - logit(prevalence_old)
    adjusted_logit_p = logit_p + adjustment

    return expit(adjusted_logit_p)  # sigmoid
```

---

## 3. Deployment Architecture

### 3.1 Model Loading

```python
import joblib
from pathlib import Path

# Load trained model (no prevalence adjustment applied)
model_path = Path("results/LR_EN/split_seed0/core/final_model.pkl")
model_bundle = joblib.load(model_path)

# Extract components
base_model = model_bundle["model"]  # Sklearn pipeline
metadata = model_bundle["metadata"]  # Training/test metrics
thresholds = model_bundle["thresholds"]  # Threshold info

# Note: No deployment metadata in current implementation
# In future deployments, could store:
#   - training_prevalence (0.167)
#   - feature_names (for input validation)
#   - calibration_params (isotonic regression state)
```

### 3.2 Prediction Wrapper

```python
class DeploymentModel:
    """
    Wraps a trained sklearn model for clinical deployment.

    Handles:
    - Feature validation
    - Prevalence adjustment
    - Threshold-based decision making
    """

    def __init__(self, base_model, training_prevalence=0.167):
        self.base_model = base_model
        self.training_prevalence = training_prevalence

    def predict_proba(self, X, deployment_prevalence=0.0034):
        """Get prevalence-adjusted probabilities."""
        p_raw = self.base_model.predict_proba(X)[:, 1]
        p_adjusted = self._adjust_prevalence(
            p_raw,
            self.training_prevalence,
            deployment_prevalence
        )
        return p_adjusted

    @staticmethod
    def _adjust_prevalence(p, prev_old, prev_new):
        """Apply logit-space adjustment."""
        return adjust_probabilities_for_prevalence(p, prev_old, prev_new)

    def predict_risk_category(self, X, thresholds):
        """
        Categorize patients into risk groups.

        Args:
            X: Feature matrix
            thresholds: dict with keys like 'low', 'medium', 'high'
                       (e.g., {'low': 0.01, 'high': 0.05})

        Returns:
            dict with patient assignments
        """
        p_adjusted = self.predict_proba(X)

        risk_categories = []
        for score in p_adjusted:
            if score < thresholds['low']:
                risk_categories.append('low')
            elif score < thresholds['high']:
                risk_categories.append('medium')
            else:
                risk_categories.append('high')

        return risk_categories
```

---

## 4. Clinical Decision Thresholds

### Threshold Selection for Deployment

Thresholds for clinical decisions should be selected based on the deployment population's prevalence and clinical utility considerations.

**Performance Metrics Across Prevalence:**

The same probability threshold applied at different prevalences yields different positive predictive values (PPV):

```
Example: Threshold = 0.25 (after prevalence adjustment)

At training prevalence (16.7%):
  Sensitivity: 0.75, Specificity: 0.95, PPV: ~76%

At UK Biobank prevalence (0.34%):
  Sensitivity: 0.75, Specificity: 0.95, PPV: ~4.7%

At general population prevalence (1.0%):
  Sensitivity: 0.75, Specificity: 0.95, PPV: ~13%
```

**Key Considerations:**
- **Sensitivity and specificity** are preserved across prevalences (discrimination unchanged)
- **PPV and NPV** vary with prevalence (predictive values are prevalence-dependent)
- **Clinical utility** depends on prevalence, costs of false positives/negatives, and intervention feasibility

### Recommended Approach: Decision Curve Analysis (DCA)

Rather than relying on a single threshold, use Decision Curve Analysis (DCA) to evaluate clinical utility across a range of thresholds:

```python
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

def plot_clinical_utility_curves(y_true, y_proba, prevalence_deployed):
    """
    Plot clinical utility across threshold range.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)

    # Compute clinical utility at each threshold
    sensitivity = tpr  # True positive rate
    specificity = 1 - fpr  # True negative rate

    # Expected value of screening = sensitivity * π - (1 - specificity) * (1 - π) * (cost_ratio)
    # where π is prevalence and cost_ratio is cost of false positive / cost of true positive

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # ROC curve (discrimination)
    ax1.plot(fpr, tpr, label='Model')
    ax1.plot([0, 1], [0, 1], 'k--', label='Random')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.legend()
    ax1.set_title(f'ROC Curve (prevalence={prevalence_deployed:.4f})')

    # Clinical utility across thresholds
    for cost_ratio in [1, 5, 10, 100]:  # Cost of FP / cost of TP
        utility = sensitivity * prevalence_deployed - (1 - specificity) * (1 - prevalence_deployed) * cost_ratio
        ax2.plot(thresholds, utility, label=f'Cost ratio={cost_ratio}')

    ax2.set_xlabel('Decision Threshold')
    ax2.set_ylabel('Net Benefit')
    ax2.legend()
    ax2.set_title('Clinical Utility Across Thresholds')
    plt.tight_layout()
    plt.savefig('clinical_utility.png', dpi=150)

    return fig
```

---

## 5. Validation on Deployment Data

### Pre-Deployment Checklist

Before deploying a model to clinical use:

1. **Obtain labeled deployment cohort**
   - Representative of target deployment population (estimate true prevalence)
   - Same feature collection process as training (proteomics platform, QC)
   - Sufficient sample size (≥500 cases + controls for reliable calibration)

2. **Validate discrimination**
   ```python
   from sklearn.metrics import roc_auc_score

   y_deployed = deployment_cohort['incident_ced']
   X_deployed = deployment_cohort[feature_cols]

   # Estimate prevalence from deployment cohort
   deployment_prev = y_deployed.mean()

   y_pred = deployment_model.predict_proba(X_deployed, prevalence=deployment_prev)
   auroc = roc_auc_score(y_deployed, y_pred)

   # Expect AUROC similar to test set (discrimination unchanged by prevalence adjustment)
   # If AUROC drops >5%, investigate data shift
   ```

3. **Validate calibration**
   ```python
   from sklearn.calibration import calibration_curve

   prob_true, prob_pred = calibration_curve(y_deployed, y_pred, n_bins=10)

   # Ideally: prob_true ≈ prob_pred (points near diagonal)
   # If calibration slope << 1: model overconfident (even after adjustment)
   # If calibration slope >> 1: model underconfident
   ```

4. **Check for data shift**
   ```python
   # Compare feature distributions between test and deployment
   # Use statistical tests (KS, Wasserstein) or visual inspection
   # Large shifts may require retraining or drift detection
   ```

---

## 6. Handling Model Updates

### Incremental Learning Pattern

```
1. Train on full dataset (including new deployment data)
2. Validate on held-out test set
3. Compare performance to previous model
4. Deploy only if improvement is clinically meaningful (e.g., >2% AUROC improvement)
5. Monitor performance on deployed model
```

### Drift Detection

Monitor key metrics on incoming data:

```python
def detect_drift(new_data, reference_data, feature_cols, threshold=0.05):
    """
    Detect feature distribution shift.

    Args:
        new_data: Recent patient cohort
        reference_data: Historical reference (from training)
        feature_cols: Protein biomarkers
        threshold: p-value threshold for significance
    """
    from scipy.stats import ks_2samp

    drifts = []
    for col in feature_cols:
        stat, pval = ks_2samp(reference_data[col], new_data[col])
        if pval < threshold:
            drifts.append((col, stat, pval))

    return drifts
```

---

## 7. Batch Processing

For large-scale clinical screening:

```python
import numpy as np
from pathlib import Path

def batch_predict(model, data_loader, batch_size=1000):
    """
    Process patients in batches to manage memory.

    Args:
        model: DeploymentModel instance
        data_loader: Iterator yielding patient batches
        batch_size: Patients per batch

    Returns:
        list of predictions
    """
    predictions = []

    for batch in data_loader:
        X_batch = batch[feature_cols]
        preds = model.predict_proba(X_batch, deployment_prevalence=0.0034)
        predictions.extend(preds)

    return np.array(predictions)
```

---

## 8. Clinical Decision Integration

### Example: Risk Stratification

```python
def stratify_patients(risk_scores, thresholds):
    """
    Assign patients to risk tiers for clinical action.

    Risk tiers (example, should be clinically justified):
    - Screen-low (< 0.5%): Standard interval (5 years)
    - Screen-medium (0.5% - 2%): Accelerated interval (2 years)
    - Screen-high (> 2%): Urgent referral (immediate)
    """
    actions = []
    for score in risk_scores:
        if score < thresholds['low']:
            action = 'Standard screening interval (5 years)'
        elif score < thresholds['high']:
            action = 'Accelerated screening (2 years)'
        else:
            action = 'Urgent referral for gastroenterology'
        actions.append(action)

    return actions
```

---

## 9. Regulatory and Documentation Requirements

### Required Documentation

1. **Model Card**
   - Model description and intended use
   - Training data characteristics
   - Performance metrics (on test set at training prevalence)
   - Known limitations

2. **Clinical Deployment Protocol**
   - Prevalence adjustment methodology
   - Threshold selection rationale
   - Required validation on deployment data
   - Drift monitoring procedures

3. **Version Control**
   - Git commit hash of trained model
   - Package versions (scikit-learn, numpy, etc.)
   - Data version / split seed
   - Calibration state

### Example Model Card

```yaml
model_name: "CeliacRisk-LR-EN-Ensemble"
version: "1.0.0"
purpose: "Predict incident celiac disease risk from proteomics"
training_date: "2026-01-22"
git_commit: "abc123def456"  # pragma: allowlist secret

training_data:
  samples: 21980  # 50% of 43,960
  prevalence: 0.167  # After downsampling
  features: 127  # After feature selection

performance_at_training_prevalence:
  test_auroc: 0.89
  test_auprc: 0.48
  calibration_brier: 0.012
  calibration_slope: 0.98

deployment_info:
  training_prevalence: 0.167  # 5:1 case:control after downsampling
  uk_biobank_prevalence: 0.0034  # UK Biobank cohort (148/43,960)
  global_prevalence_range: "0.007-0.014"  # Singh et al. 2018 meta-analysis
  adjustment_method: "logit-scale"
  required_validation: "Yes - must re-validate on target deployment cohort"
  notes: "Prevalence varies by population; estimate from target cohort before deployment"

limitations:
  - "Trained on UK Biobank (European ancestry majority); generalization to other populations unknown"
  - "Protein measurements from single platform (Olink); cross-platform portability untested"
  - "No temporal validation; model assumes stable biomarker distributions over time"
  - "Prevalence in UK Biobank (0.34%) may differ from clinical screening populations"
  - "Requires prevalence estimation from target population for accurate risk calibration"
```

---

## 10. Example Deployment Workflow

```python
import joblib
import pandas as pd
import numpy as np

# 1. Load trained model
model_bundle = joblib.load("results/LR_EN/split_seed0/core/final_model.pkl")
base_model = model_bundle["model"]

# 2. Initialize deployment wrapper
deployment_model = DeploymentModel(
    base_model=base_model,
    training_prevalence=0.167
)

# 3. Load new patient data
patient_data = pd.read_csv("new_patients_proteomics.csv")
X_patients = patient_data[feature_cols]

# 4. Get risk predictions (at estimated deployment prevalence)
# NOTE: Estimate prevalence from target population, don't assume 0.0034
estimated_prevalence = 0.0034  # Example: UK Biobank cohort
risk_scores = deployment_model.predict_proba(
    X_patients,
    deployment_prevalence=estimated_prevalence
)

# 5. Stratify into risk tiers
thresholds = {'low': 0.005, 'high': 0.02}  # 0.5% and 2%
risk_categories = deployment_model.predict_risk_category(
    X_patients,
    thresholds=thresholds
)

# 6. Generate clinical report
results = pd.DataFrame({
    "patient_id": patient_data["eid"],
    "risk_score": risk_scores,
    "risk_category": risk_categories,
    "deployment_date": pd.Timestamp.now(),
    "model_version": "LR_EN-1.0.0"
})

results.to_csv("clinical_screening_results.csv", index=False)
print(f"Screened {len(results)} patients")
print(f"High-risk: {(risk_categories == 'high').sum()}")
```

---

## 11. References

### Prevalence and Epidemiology
1. **Singh P, et al.** (2018). Global Prevalence of Celiac Disease: Systematic Review and Meta-analysis. *Clinical Gastroenterology and Hepatology*, 16(6):823-836. [PubMed](https://pubmed.ncbi.nlm.nih.gov/29551598/)
   - Pooled global prevalence: 1.4% (serologic), 0.7% (biopsy-confirmed)
   - Regional variation: Europe 0.8%, North America 0.5%, Asia 0.6%

2. **Mahadev S, et al.** (2024). Celiac Disease Affects 1% of Global Population: Who Will Manage All These Patients? *Gastroenterology*, 156(5):1199-1202. [PubMed](https://pubmed.ncbi.nlm.nih.gov/38290622/)
   - Updated estimate: 0.7-2.9% prevalence globally
   - Higher in females and children

3. **Ashtari S, et al.** (2020). Incidence of Celiac Disease Is Increasing Over Time: A Systematic Review and Meta-analysis. *Clinical Gastroenterology and Hepatology*, 18(3):748-759. [PubMed](https://pubmed.ncbi.nlm.nih.gov/32022718/)
   - Incidence: 17.4 per 100,000 person-years (women), 7.8 (men)
   - Children: 21.3 per 100,000 person-years
   - Increasing 7.5% per year

4. **Lebwohl B, et al.** (2024). Patient and Community Health Global Burden in a World With More Celiac Disease. *Gastroenterology*, 166(5):786-799. [PubMed](https://pubmed.ncbi.nlm.nih.gov/38309629/)
   - Rapidly increasing global prevalence
   - Growing public health management challenge

### Statistical Methods
5. **Steyerberg, E. W.** (2019). *Clinical Prediction Models: A Practical Approach to Development, Validation, and Updating* (2nd ed.), Chapter 13 (Updating for a New Setting).
   - Logit-scale prevalence adjustment methodology

### Related ADRs
- [ADR-010: Prevalence Adjustment via Logit Shift](adr/ADR-010-prevalence-adjustment.md)
- [ADR-011: Threshold Optimization on Validation Set](adr/ADR-011-threshold-on-val.md)
- [ADR-012: Fixed Specificity 0.95 for Clinical Use](adr/ADR-012-fixed-spec-95.md)

---

**Last Updated:** 2026-01-24
**Status:** Speculative best-practices guide (not integrated into current pipeline)
