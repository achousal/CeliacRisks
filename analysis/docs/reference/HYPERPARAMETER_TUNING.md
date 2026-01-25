# Hyperparameter Tuning and Training Configuration Guide

## Overview

This guide explains how to configure model training, hyperparameter optimization, and evaluation for machine learning pipelines. While examples are drawn from the CeliacRisks project, the concepts apply broadly to scikit-learn-based ML workflows.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Training Configuration Sections](#training-configuration-sections)
3. [Hyperparameter Optimization Methods](#hyperparameter-optimization-methods)
4. [Model-Specific Parameters](#model-specific-parameters)
5. [Feature Selection Pipeline](#feature-selection-pipeline)
6. [Calibration Strategies](#calibration-strategies)
7. [Threshold Optimization](#threshold-optimization)
8. [Parallelization and Performance](#parallelization-and-performance)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Core Concepts

### Fixed vs Tuned Parameters

Understanding which parameters are fixed versus tuned is critical for effective optimization:

**Fixed Parameters** (baseline settings):
- Set once when the model is instantiated
- Apply uniformly to all hyperparameter trials
- Examples: `max_iter`, `solver`, `tree_method`, `random_state`
- **When to adjust:** Convergence issues, algorithm changes, hardware optimization

**Tuned Parameters** (search space):
- Explored by hyperparameter optimization algorithms
- Each trial tests different combinations
- Examples: `C`, `l1_ratio`, `n_estimators`, `learning_rate`, `max_depth`
- **When to adjust:** Expanding search ranges, refining around promising regions

### Parameter Flow Architecture

```
training_config.yaml
        ↓
┌───────────────────────────────────┐
│  Fixed Parameters  │  Tuned Parameters  │
│  (max_iter,        │  (C, l1_ratio,     │
│   solver, etc)     │   n_estimators,    │
│                    │   learning_rate)   │
└───────┬────────────┴────────────────────┘
        ↓
Model Registry (registry.py)
- Instantiates base estimators with fixed params
        ↓
Hyperparameter Module (hyperparams.py)
- Builds search space from tuned params
- Converts to sklearn or Optuna format
        ↓
┌───────────────┬────────────────────┐
│               │                    │
RandomizedSearchCV    Optuna
- Random sampling     - Bayesian TPE
- Fixed iterations    - Adaptive pruning
- No study storage    - Study persistence
│               │                    │
└───────────────┴────────────────────┘
        ↓
Best Parameters Found
- Optimal C, l1_ratio, n_estimators, etc
- Fixed max_iter, solver remain unchanged
```

---

## Training Configuration Sections

### 1. Scenario Selection

```yaml
scenario: IncidentPlusPrevalent
```

**Purpose:** Controls which samples are included in training.

**Options:**
- `IncidentOnly`: Train only on prospective cases (biomarkers before diagnosis)
- `PrevalentOnly`: Train only on cross-sectional cases (biomarkers at/after diagnosis)
- `IncidentPlusPrevalent`: Combine both for increased statistical power

**Use cases:**
- Incident-only: Prospective risk prediction (most rigorous)
- Prevalent: Diagnostic screening (concurrent biomarkers)
- Combined: Maximize sample size when biological assumption holds (incident ≈ prevalent)

---

### 2. Cross-Validation Configuration

```yaml
cv:
  folds: 3                # Number of CV folds (stratified by outcome)
  repeats: 2              # Number of repeated CV runs
  scoring: roc_auc        # Optimization metric
  inner_folds: 3          # Nested CV for hyperparameter tuning
  n_jobs: -1              # CPU parallelization (-1 = all cores)
  random_state: 42        # Reproducibility seed
  grid_randomize: true    # Add jitter to grid values for diversity
  verbose: 1              # Logging verbosity (0=silent, 1=progress, 2=debug)
```

#### Key Parameters Explained

**`folds` (3-10)**
- **What it controls:** Number of train/validation splits per repeat
- **Trade-off:** More folds = less bias but more compute time
- **Recommendations:**
  - Small datasets (n < 1000): 5-10 folds
  - Medium datasets (1000-10000): 3-5 folds
  - Large datasets (n > 10000): 3 folds
  - Imbalanced data: Use stratified folds (automatic)

**`repeats` (1-5)**
- **What it controls:** Number of times entire CV is repeated with different splits
- **Trade-off:** More repeats = lower variance estimates but longer runtime
- **Recommendations:**
  - Exploratory: 1-2 repeats
  - Production: 3-5 repeats
  - High variance models (RF, XGBoost): Use more repeats

**`scoring` (optimization objective)**
- **What it controls:** Metric used to select best hyperparameters
- **Options:**
  - `roc_auc`: Area under ROC curve (rank-based, prevalence-invariant)
  - `average_precision`: Area under PR curve (better for rare outcomes)
  - `neg_brier_score`: Calibration quality (penalizes miscalibration)
  - `f1`: Harmonic mean of precision/recall (requires threshold)
  - `balanced_accuracy`: Average of sensitivity and specificity

**Choosing the right metric:**
| Use Case | Recommended Metric | Why |
|----------|-------------------|-----|
| Rare disease screening | `average_precision` | PR curves better for class imbalance |
| Risk stratification | `roc_auc` | Prevalence-independent discrimination |
| Calibrated probabilities | `neg_brier_score` | Rewards well-calibrated predictions |
| Binary classification balance | `balanced_accuracy` | Equal weight to sensitivity/specificity |

**`inner_folds` (nested CV)**
- **What it controls:** CV folds for hyperparameter tuning (nested within outer folds)
- **Purpose:** Prevents overfitting to validation set
- **Architecture:**
  ```
  Outer CV (folds=5, repeats=3) → 15 outer iterations
    ├── Each outer fold splits data into: TRAIN_OUTER + VAL
    └── Inner CV (inner_folds=3) runs on TRAIN_OUTER only
        ├── Tunes hyperparameters via 3-fold CV
        └── Returns best params → applied to VAL
  ```
- **Recommendations:**
  - Always use nested CV for unbiased performance estimates
  - `inner_folds=3` is standard (balance bias/compute)
  - Increase to 5 for small datasets or when hyperparameter sensitivity is high

**`grid_randomize` (diversity enhancement)**
- **What it controls:** Adds small random jitter to numeric grid values
- **Example:**
  ```yaml
  # Original grid
  C_grid: [0.01, 0.1, 1.0]

  # With grid_randomize=true
  C_sampled: [0.0098, 0.103, 0.97]  # Slight perturbations
  ```
- **Benefits:**
  - Explores regions between grid points
  - Reduces risk of missing optimal values due to coarse grid
  - Particularly useful with small `n_iter` or limited `n_trials`

---

### 3. Ensemble Configuration

```yaml
ensemble:
  method: stacking                          # Ensemble strategy
  base_models: [LR_EN, RF, XGBoost, LinSVM_cal]  # Models to combine
  meta_model:
    type: logistic_regression               # Meta-learner algorithm
    penalty: l2                             # Regularization type
    C: 1.0                                  # Inverse regularization strength
    max_iter: 1000                          # Convergence limit
    solver: lbfgs                           # Optimization algorithm
  use_probabilities: true                   # Input format for meta-learner
  passthrough: false                        # Include original features
  cv_for_meta: 5                            # CV folds for meta-model calibration
```

#### Ensemble Methods Explained

**Stacking** (recommended for heterogeneous models):
- **How it works:**
  1. Train base models on K-fold CV (generates out-of-fold predictions)
  2. Use OOF predictions as features for meta-learner
  3. Meta-learner learns optimal combination weights
- **Strengths:** Captures complex interactions, often best performance
- **Weaknesses:** Requires training N+1 models, risk of overfitting

**Blending** (simpler alternative):
- **How it works:**
  1. Split data into TRAIN/HOLDOUT
  2. Train base models on TRAIN
  3. Make predictions on HOLDOUT
  4. Train meta-learner on HOLDOUT predictions
- **Strengths:** Simpler, less overfitting risk
- **Weaknesses:** Uses less data, single holdout split (higher variance)

**Weighted Average** (linear combination):
- **How it works:**
  - Final prediction = w1·model1 + w2·model2 + w3·model3
  - Weights optimized via grid search or constraint optimization
- **Strengths:** Simple, interpretable, no additional model
- **Weaknesses:** Cannot capture nonlinear interactions

#### Meta-Model Configuration

**`use_probabilities` vs logits**
- `true`: Meta-learner receives `[P(class=1) for model1, model2, ...]`
- `false`: Meta-learner receives `[log(P/(1-P)) for model1, model2, ...]`
- **Recommendation:** Use `true` for calibrated base models, `false` for uncalibrated

**`passthrough` (feature pass-through)**
- `true`: Meta-learner receives both base predictions AND original features
- `false`: Meta-learner receives only base predictions
- **Use case:** Enable when base models might miss important features
- **Risk:** Increases overfitting potential, requires more data

---

### 4. Optuna Hyperparameter Optimization

```yaml
optuna:
  enabled: true                # Use Optuna instead of RandomizedSearchCV
  n_trials: 20                 # Number of hyperparameter trials
  timeout: null                # Optional time limit (seconds)
  sampler: tpe                 # Sampling algorithm
  sampler_seed: 42             # Reproducibility
  pruner: hyperband            # Early stopping strategy
  pruner_n_startup_trials: 5   # Trials before pruning starts
  pruner_percentile: 20.0      # Percentile threshold for pruner
  n_jobs: -1                   # Parallel trials
  save_study: true             # Persist Optuna study object
  save_trials_csv: true        # Export trials as CSV
```

#### Optuna vs RandomizedSearchCV

| Feature | RandomizedSearchCV | Optuna |
|---------|-------------------|--------|
| **Search strategy** | Random uniform sampling | Bayesian optimization (TPE) |
| **Adaptivity** | Fixed trials | Learns from past trials |
| **Early stopping** | None | Prunes unpromising trials |
| **Best for** | Quick exploration, simple models | Expensive models, large search spaces |
| **Speedup** | Baseline | 2-5x faster convergence |
| **Storage** | Not supported | Persistent studies (resume later) |

#### Sampler Options

**`tpe` (Tree-structured Parzen Estimator)** - Recommended
- **How it works:** Builds probabilistic model of good vs bad hyperparameters
- **Best for:** Most use cases, especially with 20+ trials
- **Strengths:** Sample-efficient, handles conditional parameters
- **Weaknesses:** Requires 10+ trials to initialize model

**`random`**
- **How it works:** Uniform random sampling (equivalent to RandomizedSearchCV)
- **Best for:** Baseline comparisons, parallel trials with weak dependencies
- **Strengths:** Simple, parallelizes perfectly
- **Weaknesses:** No learning from past trials

**`cmaes` (Covariance Matrix Adaptation Evolution Strategy)**
- **How it works:** Evolution-based optimization
- **Best for:** Continuous parameter spaces, expensive evaluations
- **Strengths:** Strong theoretical foundation
- **Weaknesses:** Not well-suited for categorical/discrete parameters

**`grid`**
- **How it works:** Exhaustive grid search
- **Best for:** Final refinement with narrow ranges
- **Strengths:** Guarantees coverage
- **Weaknesses:** Exponential cost, curse of dimensionality

#### Pruner Options

**`hyperband`** - Recommended for expensive models
- **How it works:** Adaptive resource allocation (successive halving)
- **Best for:** XGBoost, Random Forest, neural networks
- **Parameters:**
  - `pruner_n_startup_trials`: Burn-in period (default: 5)
  - Automatically allocates more resources to promising trials

**`median`**
- **How it works:** Prune if intermediate result worse than median of past trials
- **Best for:** Stable metrics, many trials
- **Parameters:** `pruner_n_startup_trials` sets burn-in period

**`percentile`**
- **How it works:** Prune if worse than Nth percentile
- **Best for:** Aggressive pruning, limited compute budget
- **Parameters:**
  - `pruner_percentile: 20.0` → Prune bottom 20%
  - `pruner_n_startup_trials`: Burn-in before pruning

**`none`**
- **How it works:** No early stopping
- **Best for:** Fast models, small search spaces

#### Trial Budget Recommendations

| Model Type | n_trials | Sampler | Pruner |
|------------|----------|---------|--------|
| Logistic Regression | 20-50 | tpe | none |
| Linear SVM | 20-50 | tpe | median |
| Random Forest | 50-100 | tpe | hyperband |
| XGBoost | 100-200 | tpe | hyperband |
| Ensemble meta-learner | 20-30 | tpe | none |

**Exploratory phase:** Start with 20 trials, `sampler: random` to map search space
**Refinement phase:** 50-100 trials, `sampler: tpe`, `pruner: hyperband`
**Production:** 100-200 trials for final model

---

### 5. Feature Selection Pipeline

```yaml
features:
  feature_select: hybrid           # Strategy: none, kbest, l1_stability, hybrid
  kbest_scope: protein             # Feature domain for k-best selection
  kbest_max: 800                   # Maximum features to consider
  k_grid: [25, 50, 100, 150, 200, 300, 400]  # k values to tune
  l1_c_min: 0.001                  # L1 stability: min regularization
  l1_c_max: 1.0                    # L1 stability: max regularization
  l1_c_points: 4                   # L1 stability: grid density
  l1_stability_thresh: 0.70        # Stability threshold (0-1)
  hybrid_kbest_first: true         # Hybrid: k-best before stability
  hybrid_k_for_stability: 200      # Features passed to stability stage

  # Screening (stage 1: unsupervised pruning)
  screen_method: mannwhitney       # mannwhitney or f_classif
  screen_top_n: 1000               # Keep top N by univariate test

  # Stability selection
  stability_thresh: 0.70           # Keep features selected in ≥70% of folds

  # Correlation pruning
  stable_corr_thresh: 0.85         # Remove features with r > 0.85
```

#### Feature Selection Strategies

**`none`** (use all features):
- **When to use:** High signal-to-noise, small feature sets (p < 100)
- **Pros:** No information loss
- **Cons:** Overfitting risk, slow training

**`kbest`** (univariate filter):
- **How it works:**
  1. Compute univariate statistic (e.g., Mann-Whitney U) for each feature
  2. Rank features by p-value or effect size
  3. Select top k features
- **When to use:** Large feature sets (p > 1000), interpretability needed
- **Pros:** Fast, simple, interpretable
- **Cons:** Ignores feature interactions, redundancy

**`l1_stability`** (multivariate wrapper):
- **How it works:**
  1. Train L1-regularized models with varying C values
  2. Track which features have non-zero coefficients across CV folds
  3. Select features appearing in ≥ `stability_thresh` of trials
- **When to use:** Medium feature sets (100 < p < 5000), multivariate interactions
- **Pros:** Captures interactions, reduces redundancy
- **Cons:** Slower, requires tuning `l1_c_min/max`, `stability_thresh`

**`hybrid`** (multi-stage pipeline) - Recommended:
- **How it works:**
  1. **Screening:** Mann-Whitney → keep top `screen_top_n` features
  2. **K-best tuning:** Tune k via CV on screened features
  3. **Stability selection:** L1 stability on top k features
  4. **Correlation pruning:** Remove redundant features (r > `stable_corr_thresh`)
- **When to use:** Large feature sets (p > 1000), production pipelines
- **Pros:** Combines speed (screening) + quality (stability) + interpretability (k-best)
- **Cons:** More parameters to tune

#### Parameter Tuning Guide

**`screen_top_n` (screening threshold)**
- **Purpose:** Reduce feature space before expensive stability selection
- **Recommendations:**
  - p < 500: Skip screening (`screen_top_n: null`)
  - 500 < p < 5000: `screen_top_n: 1000`
  - p > 5000: `screen_top_n: 2000`
- **Rule of thumb:** 2-5x larger than target feature count

**`k_grid` (k-best tuning)**
- **Purpose:** Optimize number of features via nested CV
- **Recommendations:**
  - Exploratory: `[10, 25, 50, 100, 200]` (sparse grid)
  - Production: `[25, 50, 75, 100, 150, 200, 300, 400]` (dense grid)
- **Strategy:** Include powers of 2 and round numbers for interpretability

**`stability_thresh` (stability selection cutoff)**
- **Purpose:** Feature must be selected in ≥ X% of CV folds to be retained
- **Recommendations:**
  - Conservative (low false positives): `0.80-0.90`
  - Balanced: `0.70-0.75`
  - Aggressive (maximize signal): `0.50-0.65`
- **Trade-off:** Higher → fewer features, more robust; Lower → more features, more risk

**`stable_corr_thresh` (correlation pruning)**
- **Purpose:** Remove redundant features with pairwise correlation > threshold
- **Recommendations:**
  - Proteomics/genomics: `0.85` (biological redundancy common)
  - Engineered features: `0.90` (stricter, avoid information loss)
  - Interpretability priority: `0.75` (aggressive pruning)

---

### 6. Model-Specific Hyperparameters

#### Logistic Regression (LR_EN, LR_L1)

```yaml
lr:
  # Tuned parameters:
  C_min: 0.0001                    # Min regularization strength (inverse)
  C_max: 10.0                      # Max regularization strength
  C_points: 20                     # Grid density (log-spaced)
  l1_ratio: [0.0, 0.1, 0.5, 0.9, 1.0]  # ElasticNet mixing (0=L2, 1=L1)
  class_weight_options: "balanced" # Handle class imbalance

  # Optuna-specific overrides (optional):
  optuna_C: [1.0e-5, 100.0]        # Log-uniform range
  optuna_l1_ratio: [0.0, 1.0]      # Uniform range

  # Fixed parameters:
  solver: saga                     # Supports L1/ElasticNet
  max_iter: 10000                  # Increase if convergence warnings

  # Search algorithm:
  n_iter: 4                        # RandomizedSearchCV iterations (ignored if Optuna)
```

**Key concepts:**

**`C` (inverse regularization)**
- Higher C = less regularization = more complex model
- Lower C = more regularization = simpler model
- **Choosing range:**
  - High-dimensional (p >> n): Start with `[1e-4, 1.0]`
  - Low-dimensional (p < n): Expand to `[1e-2, 100]`
  - If all trials hit upper bound → increase `C_max`

**`l1_ratio` (ElasticNet mixing)**
- `0.0`: Pure L2 (Ridge) - all features with small coefficients
- `1.0`: Pure L1 (Lasso) - sparse feature selection
- `0.5`: Balanced ElasticNet - compromise
- **Recommendations:**
  - High collinearity: Include `[0.0, 0.1, 0.2]` (L2-heavy)
  - Sparse signals: Include `[0.8, 0.9, 1.0]` (L1-heavy)
  - Exploratory: Use `[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]`

**`solver` (optimization algorithm)**
- `saga`: Supports L1/L2/ElasticNet, fast for large datasets
- `lbfgs`: L2 only, good for small datasets
- `liblinear`: Good for small datasets, supports L1/L2
- **Recommendation:** Use `saga` by default for flexibility

**`max_iter` (convergence limit)**
- Increase if seeing `ConvergenceWarning`
- **Typical values:** 1000 (small data), 5000 (medium), 10000 (large/sparse)

---

#### Linear SVM (LinSVM_cal)

```yaml
svm:
  # Tuned parameters:
  C_min: 0.0001                    # Min inverse regularization
  C_max: 10.0                      # Max inverse regularization
  C_points: 15                     # Grid density
  class_weight_options: "balanced"

  # Optuna-specific:
  optuna_C: [1.0e-4, 100.0]

  # Fixed parameters:
  max_iter: 10000                  # Convergence limit

  # Search algorithm:
  n_iter: 6
```

**SVM-specific notes:**
- SVMs are sensitive to feature scaling (ensure standardization in pipeline)
- `C` interpretation same as logistic regression (inverse regularization)
- LinearSVC uses hinge loss instead of logistic loss
- Often wrapped in `CalibratedClassifierCV` for probability estimates

---

#### Random Forest (RF)

```yaml
rf:
  # Tuned parameters:
  n_estimators_grid: [100, 300, 500, 700]       # Number of trees
  max_depth_grid: [null, 10, 20, 30, 50]        # Tree depth (null=unlimited)
  min_samples_split_grid: [2, 5, 10, 20]        # Min samples to split node
  min_samples_leaf_grid: [1, 2, 4, 8]           # Min samples in leaf
  max_features_grid: ["sqrt", "log2", 0.3, 0.5, 0.7]  # Features per split
  class_weight_options: "balanced"

  # Optuna-specific (continuous ranges):
  optuna_n_estimators: [100, 800]
  optuna_max_depth: [5, 50]
  optuna_min_samples_split: [2, 30]
  optuna_min_samples_leaf: [1, 15]
  optuna_max_features: [0.05, 0.8]

  # Search algorithm:
  n_iter: 2
```

**Key RF hyperparameters:**

**`n_estimators` (number of trees)**
- More trees = better performance (diminishing returns after ~500)
- More trees = longer training, no overfitting risk
- **Recommendations:**
  - Exploratory: `[100, 300]`
  - Production: `[300, 500, 700]`
  - Budget unlimited: `[500, 1000, 1500]`

**`max_depth` (tree depth)**
- `null`: Trees grow until pure or `min_samples_leaf` reached
- Low values (3-10): Prevents overfitting, faster training
- High values (20-50): Captures complex interactions
- **Recommendations:**
  - High-dimensional sparse data: `[10, 20, 30]`
  - Low-dimensional dense data: `[null, 20, 40]`
  - Overfitting observed: Cap at `[5, 10, 15]`

**`min_samples_split` (split threshold)**
- Minimum samples required to split internal node
- Higher values = more regularization
- **Recommendations:**
  - Small datasets (n < 1000): `[5, 10, 20]`
  - Large datasets (n > 10000): `[2, 5, 10]`

**`min_samples_leaf` (leaf size)**
- Minimum samples required in leaf node
- Higher values = smoother decision boundaries
- **Recommendations:**
  - Balanced datasets: `[1, 2, 4]`
  - Imbalanced datasets: `[2, 4, 8]` (prevent pure minority leaves)

**`max_features` (features per split)**
- `"sqrt"`: √p features (default, good baseline)
- `"log2"`: log₂(p) features (more decorrelated trees)
- Fraction (0.3-0.7): Manual control
- **Recommendations:**
  - High-dimensional: `["sqrt", 0.3, 0.5]`
  - Low-dimensional: `["sqrt", "log2", 0.7]`

---

#### XGBoost

```yaml
xgboost:
  # Tuned parameters:
  n_estimators_grid: [100, 200, 400, 800, 1200]     # Boosting rounds
  max_depth_grid: [3, 4, 5, 6, 7, 8]                # Tree depth
  learning_rate_grid: [0.01, 0.03, 0.05, 0.1, 0.2]  # Shrinkage
  subsample_grid: [0.5, 0.7, 0.85, 1.0]             # Row sampling
  colsample_bytree_grid: [0.3, 0.5, 0.7, 0.85, 1.0] # Column sampling
  scale_pos_weight_grid: [3.0, 5.0, 7.0, 10.0]      # Class imbalance
  min_child_weight_grid: [1, 2, 4, 6, 10]           # Min sum of weights
  gamma_grid: [0.0, 0.05, 0.1, 0.5, 1.0, 2.0]       # Min split loss
  reg_alpha_grid: [0.0, 0.001, 0.01, 0.1, 1.0]      # L1 regularization
  reg_lambda_grid: [0.01, 0.1, 1.0, 5.0, 10.0, 50.0]  # L2 regularization

  # Optuna-specific:
  optuna_n_estimators: [50, 1500]
  optuna_max_depth: [3, 10]
  optuna_learning_rate: [0.005, 0.3]
  optuna_min_child_weight: [0.1, 20.0]
  optuna_gamma: [0.0, 2.0]
  optuna_subsample: [0.5, 1.0]
  optuna_colsample_bytree: [0.3, 1.0]
  optuna_reg_alpha: [1.0e-8, 10.0]        # Log-scale
  optuna_reg_lambda: [1.0e-2, 50.0]       # Log-scale

  # Fixed parameters:
  tree_method: hist                        # hist (fast), exact, approx

  # Search algorithm:
  n_iter: 2
```

**Critical XGBoost parameters:**

**`learning_rate` (shrinkage)**
- Controls contribution of each tree
- Lower = more robust but needs more `n_estimators`
- **Typical values:**
  - Fast exploration: `0.1-0.3`
  - Production: `0.01-0.05` (with more trees)
  - Rule of thumb: `learning_rate × n_estimators ≈ constant`

**`n_estimators` (boosting rounds)**
- Number of sequential trees
- More trees = better fit (but risk overfitting without regularization)
- **Recommendations:**
  - learning_rate=0.1: `n_estimators=100-300`
  - learning_rate=0.01: `n_estimators=500-1500`
  - Use early stopping in production

**`max_depth` (tree complexity)**
- XGBoost trees typically shallower than RF
- **Typical values:** 3-8 (vs 10-30 for RF)
- **Recommendations:**
  - Structured data: `[3, 5, 7]`
  - High-dimensional sparse: `[4, 6, 8]`

**`subsample` and `colsample_bytree` (sampling rates)**
- Both reduce overfitting via randomness
- `subsample`: Fraction of rows per tree
- `colsample_bytree`: Fraction of columns per tree
- **Recommendations:**
  - Conservative: `subsample=0.8, colsample=0.8`
  - Aggressive regularization: `subsample=0.5-0.7, colsample=0.5-0.7`
  - Large datasets: Can use `1.0` for both

**`gamma` (minimum split loss)**
- Higher = more conservative splitting
- **Typical values:** 0-2
- **When to increase:** Overfitting, too many leaves

**`reg_alpha` (L1) and `reg_lambda` (L2)**
- Regularization on leaf weights
- L1 induces sparsity, L2 smooths weights
- **Recommendations:**
  - Start with `reg_lambda=1.0` only
  - Add `reg_alpha` if feature selection needed
  - Rarely need both simultaneously

**`scale_pos_weight` (class imbalance)**
- Recommended value: `(n_negative / n_positive)`
- Alternative: Use `class_weight="balanced"` in sklearn wrapper
- **Example:** If 1% positive class → `scale_pos_weight ≈ 99`

---

### 7. Calibration Strategies

```yaml
calibration:
  enabled: true                # Apply calibration
  strategy: oof_posthoc        # per_fold, oof_posthoc, none
  method: isotonic             # isotonic or sigmoid
  cv: 5                        # CV folds for per_fold strategy
```

#### Calibration Strategies Compared

**`per_fold`** (traditional, simple):
- **How it works:** Apply `CalibratedClassifierCV` inside each CV fold
- **Pros:** Simple, widely used, sklearn native
- **Cons:** Optimistic bias (~0.5-1% overestimate of calibration quality)
- **Use case:** Quick exploration, small datasets

**`oof_posthoc`** (recommended, unbiased):
- **How it works:**
  1. Train uncalibrated model via CV → collect out-of-fold predictions
  2. Fit calibrator on pooled OOF predictions (never seen during training)
  3. Apply calibrator to test predictions
- **Pros:** Eliminates optimistic bias, more honest calibration assessment
- **Cons:** Slightly more complex implementation
- **Use case:** Production models, final reporting

**`none`** (no calibration):
- **Use case:** Already well-calibrated models (e.g., logistic regression with proper regularization)

#### Calibration Methods

**`isotonic`** (non-parametric, recommended):
- **How it works:** Monotonic piecewise-constant mapping
- **Pros:** Flexible, no distributional assumptions
- **Cons:** Requires more data (50+ positive cases), can overfit
- **Use case:** Default choice for most applications

**`sigmoid`** (Platt scaling, parametric):
- **How it works:** Logistic function fit to (score, outcome) pairs
- **Pros:** Works with small datasets, smooth mapping
- **Cons:** Assumes sigmoid shape
- **Use case:** Small datasets (n < 500), SVMs

---

### 8. Threshold Optimization

```yaml
thresholds:
  objective: fixed_spec            # Threshold selection criterion
  fixed_spec: 0.95                 # Target specificity
  fbeta: 1.0                       # Beta for F-beta score
  fixed_ppv: 0.10                  # Target positive predictive value
  threshold_source: val            # val or test
  target_prevalence_source: test   # Prevalence for calibration
  target_prevalence_fixed: null    # Manual prevalence override
  risk_prob_source: test           # Dataset for final risk scores
```

#### Threshold Objectives

**`fixed_spec`** (fixed specificity):
- **Use case:** Screening tests (minimize false alarms)
- **Example:** `fixed_spec: 0.95` → 95% specificity, accepts 5% false positive rate
- **Clinical application:** First-line screening before expensive confirmatory tests

**`youden`** (Youden's J statistic):
- **Formula:** J = sensitivity + specificity - 1
- **Use case:** Balanced sensitivity/specificity
- **Optimal when:** Equal cost for false positives and false negatives

**`max_f1`**:
- **Formula:** F1 = 2 × (precision × recall) / (precision + recall)
- **Use case:** Imbalanced datasets where both precision and recall matter
- **Optimal when:** Cost ratio unknown

**`max_fbeta`**:
- **Formula:** Fβ = (1 + β²) × (precision × recall) / (β² × precision + recall)
- **Use case:** Asymmetric costs
- **Examples:**
  - `fbeta: 2.0` → Recall 2x more important than precision (find all cases)
  - `fbeta: 0.5` → Precision 2x more important (avoid false alarms)

**`fixed_ppv`** (fixed positive predictive value):
- **Use case:** Resource-constrained settings (limit false positives)
- **Example:** `fixed_ppv: 0.10` → 10% of positives are true cases (90% FDR acceptable)

#### Threshold Selection Strategy

**`threshold_source: val`** (recommended):
- Select threshold on validation set
- Prevents overfitting to test set
- Use for model selection and comparison

**`threshold_source: test`** (only for final evaluation):
- Optimize threshold on test set
- Provides best achievable performance
- Risk of overfitting, use only for reporting

---

### 9. Evaluation Configuration

```yaml
evaluation:
  test_ci_bootstrap: true                     # Bootstrap confidence intervals
  n_boot: 100                                 # Bootstrap iterations
  boot_random_state: 0                        # Reproducibility
  learning_curve: true                        # Generate learning curves
  lc_train_sizes: [0.1, 0.25, 0.5, 0.75, 1.0] # Training set fractions
  feature_reports: true                       # Feature importance reports
  feature_report_max: 200                     # Max features to report
  control_spec_targets: [0.90, 0.95, 0.99]    # Specificity operating points
  toprisk_fracs: [0.01, 0.05, 0.10]           # Top risk percentiles
```

**`n_boot` recommendations:**
- Exploratory: 100
- Production: 500-1000
- Publication: 2000

---

### 10. Decision Curve Analysis (DCA)

```yaml
dca:
  compute_dca: true                  # Enable DCA
  dca_threshold_min: 0.0005          # Min risk threshold
  dca_threshold_max: 1.0             # Max risk threshold
  dca_threshold_step: 0.001          # Threshold step size
  dca_report_points: [0.01, 0.05, 0.10, 0.20]  # Key thresholds for reporting
```

**DCA threshold range:**
- Should span clinically plausible risk thresholds
- **Rare diseases:** Use `dca_threshold_min: 0.0001` (0.01%)
- **Common diseases:** Use `dca_threshold_min: 0.01` (1%)

---

## Parallelization and Performance

### Nested Parallelism

```yaml
cv:
  n_jobs: -1      # Parallelizes CV folds within each trial

optuna:
  n_jobs: 1       # Parallelizes trials themselves
```

**Interaction:**
- `cv.n_jobs=-1, optuna.n_jobs=1`: Each trial uses all cores (recommended)
- `cv.n_jobs=-1, optuna.n_jobs=-1`: Parallel trials, sequential folds (for `sampler: random`)
- `-1` auto-detects available cores

### Recommended Configurations

**Local development (4-8 cores):**
```yaml
cv:
  n_jobs: -1
optuna:
  n_jobs: 1
  sampler: tpe
```

**HPC (8+ cores):**
```yaml
cv:
  n_jobs: -1      # Uses allocated cores automatically
optuna:
  n_jobs: 1       # Sequential trials for TPE effectiveness
  sampler: tpe
```

**High-throughput (100+ trials):**
```yaml
cv:
  n_jobs: -1
optuna:
  n_jobs: -1      # Parallel trials
  sampler: random  # TPE ineffective when parallelized
```

---

## Best Practices

### 1. Start Simple, Expand Gradually

**Phase 1: Baseline**
```yaml
cv:
  folds: 3
  repeats: 1
optuna:
  enabled: false
lr:
  n_iter: 10
  C_min: 0.001
  C_max: 10.0
  C_points: 5
```

**Phase 2: Expand Search Space**
```yaml
cv:
  repeats: 2
optuna:
  enabled: true
  n_trials: 50
lr:
  C_min: 0.0001
  C_max: 100.0
  C_points: 10
  l1_ratio: [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
```

**Phase 3: Production Refinement**
```yaml
cv:
  folds: 5
  repeats: 3
optuna:
  n_trials: 200
  pruner: hyperband
evaluation:
  n_boot: 1000
```

### 2. Grid Design Principles

- **Use log-spacing for regularization:** `C`, `learning_rate`, `reg_alpha`
- **Use linear-spacing for fractions:** `subsample`, `l1_ratio`, `max_features`
- **Include edge cases:** `max_depth: [null, ...]`, `gamma: [0.0, ...]`
- **Avoid redundant coverage:** Don't over-sample dense regions

### 3. Debugging Strategy

1. **Disable Optuna first:** Use `n_iter: 5` with RandomizedSearchCV
2. **Check single trial:** Verify one hyperparameter set runs successfully
3. **Reduce data:** Test on 10% sample for fast iteration
4. **Enable verbose logging:** `cv.verbose: 2`, check for warnings
5. **Re-enable Optuna:** Start with `n_trials: 10`, `sampler: random`

### 4. Reproducibility Checklist

- Set all `random_state` / `*_seed` parameters
- Log full config with each run
- Save Optuna study objects (`save_study: true`)
- Version control config files
- Record package versions in environment

---

## Troubleshooting

### Convergence Warnings (Logistic Regression)

```
ConvergenceWarning: lbfgs failed to converge
```

**Solutions:**
1. Increase `max_iter: 10000` → `max_iter: 20000`
2. Try different solver: `solver: saga` → `solver: liblinear`
3. Check feature scaling (standardize continuous features)
4. Expand `C` range (try smaller values)

### Optuna Pruning All Trials

```
[optuna] All 50 trials pruned or failed
```

**Causes & solutions:**
1. **Too few trials for TPE:** Use `n_trials: 100` or `sampler: random`
2. **Incompatible hyperparameters:** Test with RandomizedSearchCV first
3. **Pruner too aggressive:** Try `pruner: none` or increase `pruner_n_startup_trials`
4. **Data issues:** Check for NaNs, class imbalance, feature scaling

### Out of Memory (XGBoost/RF)

**Solutions:**
1. Reduce `n_estimators_grid`: `[100, 200, 400]` instead of `[500, 1000, 1500]`
2. Reduce `max_depth_grid`: Cap at `[3, 5, 7]`
3. Enable `tree_method: hist` for XGBoost (uses histograms, less memory)
4. Reduce `cv.n_jobs` or `optuna.n_jobs` (fewer parallel processes)

### Poor Calibration Despite Calibration

**Diagnostics:**
1. Plot calibration curve (should be close to diagonal)
2. Check Brier score decomposition (resolution vs reliability)
3. Verify sufficient positives (need 50+ for isotonic regression)

**Solutions:**
1. Switch `method: sigmoid` if small sample size
2. Try `strategy: oof_posthoc` for less biased estimates
3. Increase CV folds: `calibration.cv: 10`
4. Check for distribution shift between train/test

### Stacking Ensemble Underperforms

**Common causes:**
1. **Homogeneous base models:** Use diverse algorithms (LR + RF + XGBoost)
2. **Overfitting:** Reduce `cv_for_meta` or use `penalty: l2` with lower `C`
3. **Insufficient data:** Need 5-10x minority class size for meta-learner
4. **Correlated base predictions:** Check pairwise correlations, remove redundant models

---

## Related Documentation

- [training_config.yaml](../../configs/training_config.yaml) - Full config reference
- [ADR-008: Optuna Hyperparameter Optimization](../adr/ADR-008-optuna-hyperparameter-optimization.md)
- [ADR-014: OOF-Posthoc Calibration](../adr/ADR-014-oof-posthoc-calibration.md)
- [CLI_REFERENCE.md](CLI_REFERENCE.md) - Command-line interface guide
