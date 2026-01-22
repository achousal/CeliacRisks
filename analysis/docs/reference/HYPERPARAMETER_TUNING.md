# Hyperparameter Tuning Guide

## Overview

The CeliacRisks pipeline supports two hyperparameter optimization methods:
1. **RandomizedSearchCV** (sklearn default)
2. **Optuna** (Bayesian optimization with pruning)

Both methods use the **same parameter configurations** from [training_config.yaml](../configs/training_config.yaml).

## Key Concept: Fixed vs Tuned Parameters

### Fixed Parameters (Baseline)
These are **NOT tuned** by RandomizedSearchCV or Optuna. They are set once when the model is created and remain constant across all trials.

**Examples:**
- `lr.max_iter: 5000` - Maximum iterations for convergence
- `lr.solver: saga` - Optimization algorithm
- `svm.max_iter: 5000` - SVM convergence iterations
- `xgboost.tree_method: hist` - XGBoost tree construction algorithm

**Where they're used:**
- Set in [registry.py:build_models()](../src/ced_ml/models/registry.py) when creating the base estimator
- Apply to ALL hyperparameter trials uniformly

**When to change:**
- Seeing convergence warnings → increase `max_iter`
- Need different solver → change `solver`
- GPU available → change `tree_method: gpu_hist`

### Tuned Parameters (Search Space)
These **ARE explored** by RandomizedSearchCV or Optuna. Each trial tests different combinations.

**Examples:**
- `lr.C_min`, `lr.C_max`, `lr.C_points` → Regularization strength grid
- `lr.l1_ratio: [0.1, 0.5, 0.9]` → ElasticNet mixing ratios
- `rf.n_estimators_grid: [100, 300, 500]` → Number of trees
- `xgboost.learning_rate_grid: [0.01, 0.05, 0.1, 0.3]` → Step size

**Where they're used:**
- Converted to search spaces in [hyperparams.py:get_param_distributions()](../src/ced_ml/models/hyperparams.py)
- For Optuna: converted to suggest specs in [hyperparams.py:get_param_distributions_optuna()](../src/ced_ml/models/hyperparams.py)

**When to change:**
- Expand search range → adjust `_min`/`_max` values
- Add more options → extend grid lists
- Change grid density → adjust `_points` parameter

## Parameter Flow

```
training_config.yaml
        ↓
┌───────┴────────┐
│                │
│  Fixed Params  │  Tuned Params
│  (max_iter,    │  (C, l1_ratio,
│   solver, etc) │   n_estimators, etc)
│                │
└───────┬────────┘
        ↓
    registry.py:build_models()
    - Creates base estimator with fixed params
        ↓
    hyperparams.py:get_param_distributions()
    - Builds search space from tuned params
        ↓
┌───────┴────────┐
│                │
RandomizedSearchCV    Optuna
- Samples from     - Bayesian opt
  param grid         over suggest specs
- Fixed n_iter     - Adaptive trials
  trials             with pruning
│                │
└───────┬────────┘
        ↓
   Best params found
   - Best C, l1_ratio, etc
   - Fixed max_iter, solver unchanged
```

## Configuration Examples

### Logistic Regression (LR_EN, LR_L1)

```yaml
lr:
  # TUNED by both RandomizedSearchCV and Optuna:
  C_min: 0.0001             # Min regularization strength (log-spaced)
  C_max: 100.0              # Max regularization strength
  C_points: 7               # Number of C values in search grid
  l1_ratio: [0.1, 0.5, 0.9] # ElasticNet mixing (LR_EN only)
  class_weight_options: "balanced"

  # FIXED (not tuned, baseline for all trials):
  solver: saga              # Solver supporting L1/ElasticNet
  max_iter: 5000            # Max iterations (increase if convergence warnings)

  # Search algorithm:
  n_iter: 2                 # RandomizedSearchCV iterations (ignored by Optuna)
```

**What gets tuned:**
- `C`: 7 log-spaced values from 0.0001 to 100.0
- `l1_ratio`: 3 discrete values [0.1, 0.5, 0.9]
- `class_weight`: "balanced" (or expand to `[None, "balanced"]`)

**What stays fixed:**
- `solver: saga` (all trials use saga)
- `max_iter: 5000` (all trials use 5000)

### Linear SVM (LinSVM_cal)

```yaml
svm:
  # TUNED:
  C_min: 0.01
  C_max: 10.0
  C_points: 4
  class_weight_options: "balanced"

  # FIXED:
  max_iter: 5000

  # Search algorithm:
  n_iter: 2
```

### Random Forest (RF)

```yaml
rf:
  # TUNED:
  n_estimators_grid: [100, 300, 500]
  max_depth_grid: [null, 10, 20, 30]
  min_samples_split_grid: [2, 5, 10]
  min_samples_leaf_grid: [1, 2, 4]
  max_features_grid: ["sqrt", "log2", 0.5]
  class_weight_options: "balanced"

  # Search algorithm:
  n_iter: 2
```

**Note:** RF has no fixed `max_iter` because tree-based models don't iterate to convergence.

### XGBoost

```yaml
xgboost:
  # TUNED:
  n_estimators_grid: [100, 300, 500]
  max_depth_grid: [3, 5, 7, 10]
  learning_rate_grid: [0.01, 0.05, 0.1, 0.3]
  subsample_grid: [0.7, 0.8, 1.0]
  colsample_bytree_grid: [0.7, 0.8, 1.0]
  scale_pos_weight_grid: [1.0, 2.0, 5.0]
  min_child_weight_grid: [1, 3, 5]
  gamma_grid: [0.0, 0.1, 0.3]
  reg_alpha_grid: [0.0, 0.01, 0.1]
  reg_lambda_grid: [1.0, 2.0, 5.0]

  # FIXED:
  tree_method: hist

  # Search algorithm:
  n_iter: 2
```

## Common Questions

### Q: Does Optuna tune `max_iter`?
**A:** No. `max_iter` is a fixed parameter set when the base estimator is created in `build_models()`. It applies uniformly to all trials.

### Q: I'm seeing convergence warnings. What do I do?
**A:** Increase `max_iter` in the config:
```yaml
lr:
  max_iter: 10000  # Increase from 5000
```

This will apply to ALL RandomizedSearchCV and Optuna trials.

### Q: Can I make `max_iter` tunable?
**A:** Yes, but you'd need to modify the code:

1. Add to hyperparams.py `_get_lr_params()`:
   ```python
   params["clf__max_iter"] = [2000, 5000, 10000]
   ```

2. For Optuna, add to `get_param_distributions_optuna()`:
   ```python
   "clf__max_iter": {
       "type": "categorical",
       "choices": [2000, 5000, 10000]
   }
   ```

However, this is **not recommended** because:
- `max_iter` rarely needs tuning if set high enough
- Increases search space unnecessarily
- Better to fix it at a safe high value (5000-10000)

### Q: What's the difference between `n_iter` and Optuna `n_trials`?
**A:**
- `lr.n_iter: 2` → RandomizedSearchCV samples 2 random parameter combinations
- `optuna.n_trials: 50` → Optuna runs 50 Bayesian optimization trials

When `optuna.enabled: true`, the `n_iter` values are ignored.

### Q: How do I expand the search space?
**A:** Edit the grid ranges in training_config.yaml:

```yaml
lr:
  C_min: 0.00001    # Expand lower bound
  C_max: 1000.0     # Expand upper bound
  C_points: 10      # Denser grid
  l1_ratio: [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]  # More options
```

### Q: Should I use RandomizedSearchCV or Optuna?
**A:**

**Use RandomizedSearchCV when:**
- Quick exploratory runs (few trials, e.g., n_iter=10)
- Simple models (LR, LinSVM)
- Limited compute time

**Use Optuna when:**
- Expensive models (XGBoost, RF with large grids)
- Many trials needed (50-200)
- Want adaptive search with pruning
- Need study persistence for resuming

**Performance comparison:**
- RandomizedSearchCV: Uniform random sampling, no early stopping
- Optuna: Bayesian (TPE), prunes unpromising trials, 2-5x speedup on expensive models

## Switching Between Methods

### Enable Optuna
```yaml
optuna:
  enabled: true
  n_trials: 50
  sampler: tpe
  pruner: hyperband
```

### Disable Optuna (use RandomizedSearchCV)
```yaml
optuna:
  enabled: false

lr:
  n_iter: 20  # Now used (was ignored when optuna.enabled=true)
```

## Parameter Conversion (Optuna)

When `optuna.enabled: true`, sklearn-style grids are auto-converted to Optuna suggest specs:

**Grid format (sklearn/config):**
```yaml
lr:
  C_min: 0.0001
  C_max: 100.0
  C_points: 7
  l1_ratio: [0.1, 0.5, 0.9]
```

**Converted to Optuna specs:**
```python
{
    "clf__C": {
        "type": "float",
        "low": 0.0001,
        "high": 100.0,
        "log": True  # Auto-detected from log-spacing
    },
    "clf__l1_ratio": {
        "type": "categorical",
        "choices": [0.1, 0.5, 0.9]
    }
}
```

This happens automatically in [hyperparams.py:get_param_distributions_optuna()](../src/ced_ml/models/hyperparams.py).

## Best Practices

1. **Set `max_iter` high enough to avoid warnings** (5000-10000 for LR/SVM)
2. **Start with coarse grids** (few values) for exploration
3. **Refine grids** around promising regions in follow-up runs
4. **Use Optuna for expensive models** (RF, XGBoost) with many trials
5. **Use RandomizedSearchCV for quick checks** on simple models (LR, LinSVM)
6. **Log-space for regularization** (C, learning_rate, reg_alpha) - already configured
7. **Linear-space for fractions** (subsample, colsample_bytree, l1_ratio)

## Troubleshooting

### Convergence warnings for LogisticRegression
```
ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
```

**Solution:** Increase `lr.max_iter`:
```yaml
lr:
  max_iter: 10000  # or higher
```

### Optuna pruning all trials
```
[optuna] All 50 trials pruned or failed.
```

**Possible causes:**
- `n_trials` too low for TPE startup (need 40+ for TPE)
- Hyperparameters incompatible
- Data issues (all NaN, wrong shapes)

**Solutions:**
- Increase `n_trials` or switch to `sampler: random`
- Check logs for error messages
- Test with RandomizedSearchCV first to validate grid

### Out of memory (XGBoost/RF)
**Solution:** Reduce grid size or use smaller `n_estimators` values:
```yaml
xgboost:
  n_estimators_grid: [50, 100, 200]  # Smaller trees
```

## Related Documentation

- [training_config.yaml](../configs/training_config.yaml) - Full config reference
- [hyperparams.py](../src/ced_ml/models/hyperparams.py) - Grid generation logic
- [optuna_search.py](../src/ced_ml/models/optuna_search.py) - Optuna integration
- [registry.py](../src/ced_ml/models/registry.py) - Model builders with fixed params
- [ADR-018](../docs/adr/ADR-018-optuna-hyperparameter-optimization.md) - Optuna design decisions
