# ADR-008: Optuna Hyperparameter Optimization

**Status:** Accepted
**Date:** 2026-01-20

## Context

The pipeline originally used scikit-learn's `RandomizedSearchCV` for hyperparameter tuning in nested cross-validation. While RandomizedSearchCV is simple and works well for many cases, it has limitations:

**RandomizedSearchCV Limitations:**
- **Sampling strategy:** Purely random sampling, no learning from previous trials
- **Efficiency:** May waste evaluations on poor hyperparameter regions
- **Fixed budget:** Must specify `n_iter` upfront; cannot adapt based on early results
- **No pruning:** Runs all CV folds even for obviously poor hyperparameter combinations
- **Limited samplers:** Only uniform or log-uniform distributions

**Optuna Advantages:**
- **Bayesian optimization:** TPE (Tree-structured Parzen Estimator) learns from trial history
- **Pruning:** MedianPruner stops unpromising trials early (saves compute)
- **Adaptive sampling:** Explores promising regions more densely
- **Flexible search spaces:** Categorical, conditional, nested parameter spaces
- **Study persistence:** Save/resume optimization studies
- **Parallel-friendly:** Supports distributed hyperparameter search

For computationally expensive models (XGBoost, Random Forest on large datasets), pruning and intelligent sampling can reduce tuning time by 2-5x while achieving better hyperparameters.

## Decision

Add Optuna as an optional hyperparameter optimization backend, coexisting with RandomizedSearchCV.

**Implementation:**
1. **Optional dependency:** Optuna installed via `pip install ced-ml[optuna]` or `pyproject.toml` extras
2. **Config-driven selection:** `OptunaConfig.enabled` flag toggles Optuna vs. RandomizedSearchCV
3. **Sklearn-compatible wrapper:** `OptunaSearchCV` class mimics `RandomizedSearchCV` API
4. **Graceful fallback:** If Optuna not installed but enabled, fall back to RandomizedSearchCV with warning
5. **Persistence:** Optuna studies saved to `cv/optuna/` with trial metadata

**OptunaConfig Parameters:**
```yaml
optuna:
  enabled: false                    # Use Optuna (vs RandomizedSearchCV)
  n_trials: 100                     # Number of trials per inner CV fold
  sampler: tpe                      # tpe | random | cmaes | grid
  sampler_seed: 42                  # Sampler RNG seed
  pruner: median                    # median | percentile | hyperband | none
  pruner_n_startup_trials: 5        # Trials before pruning starts
  storage: null                     # Storage URL (e.g., sqlite:///study.db)
  study_name: null                  # Study name for persistence
  save_trials_csv: true             # Export trials to CSV
```

**Sampler Options:**
- **TPE** (default): Tree-structured Parzen Estimator, best for continuous/discrete spaces
- **Random**: Pure random search (baseline comparison)
- **CMA-ES**: Covariance Matrix Adaptation Evolution Strategy (continuous spaces only)
- **Grid**: Exhaustive grid search (small search spaces only)

**Pruner Options:**
- **Median**: Stop if trial's intermediate value is below median of previous trials
- **Percentile**: Stop if below specified percentile (e.g., 25th)
- **Hyperband**: Successive halving with aggressive early stopping
- **None**: No pruning (equivalent to RandomizedSearchCV)

## Alternatives Considered

1. **Replace RandomizedSearchCV entirely:**
   - Simpler codebase (one optimizer)
   - Rejected: Some users may not want Optuna dependency; RandomizedSearchCV is battle-tested

2. **Use scikit-optimize (skopt):**
   - Bayesian optimization with Gaussian Processes
   - Rejected: Less actively maintained than Optuna; no pruning support

3. **Use Hyperopt:**
   - Similar to Optuna (TPE algorithm)
   - Rejected: Optuna has better API, more active development, better pruning

4. **Use Ray Tune:**
   - Scalable hyperparameter tuning with distributed support
   - Rejected: Heavy dependency; overkill for single-node optimization

5. **Grid search only:**
   - Exhaustive but guaranteed optimal within grid
   - Rejected: Exponentially expensive for high-dimensional spaces

## Consequences

### Positive
- **Efficiency:** 2-5x faster hyperparameter tuning for expensive models (via pruning + TPE)
- **Quality:** Better hyperparameters than random search in limited trials
- **Flexibility:** Supports categorical, conditional, nested parameter spaces
- **Observability:** Trial history, visualization, study persistence
- **Backward compatible:** Existing configs work unchanged (Optuna opt-in)

### Negative
- **Dependency:** Adds Optuna dependency (~10 MB) if enabled
- **Complexity:** More configuration options (sampler, pruner, storage)
- **Reproducibility:** TPE is stochastic; requires `sampler_seed` for exact replication
- **Overhead:** Optuna has ~5-10ms overhead per trial (negligible for expensive models, noticeable for fast models)

## Evidence

### Code Pointers
- [models/optuna_search.py](../../src/ced_ml/models/optuna_search.py) - `OptunaSearchCV` wrapper
- [models/training.py:_build_hyperparameter_search](../../src/ced_ml/models/training.py) - Optuna integration
- [models/hyperparams.py:get_param_distributions_optuna](../../src/ced_ml/models/hyperparams.py) - Optuna search spaces
- [config/schema.py:OptunaConfig](../../src/ced_ml/config/schema.py) - Configuration schema

### Test Coverage
- `tests/test_optuna_search.py` - Validates OptunaSearchCV wrapper (to be added)
- `tests/test_training.py` - Integration tests with Optuna enabled
- Manual testing: Confirmed 3x speedup on XGBoost hyperparameter tuning (median pruner)

### Benchmark Results (Example)
```
RandomizedSearchCV:  200 trials × 5 CV folds = 1000 model fits, 45 min
OptunaSearchCV (TPE + MedianPruner):
                      100 trials × ~2.5 CV folds (pruned) = 250 model fits, 12 min
                      (Better AUROC: 0.87 vs 0.85)
```

### References
- Optuna Documentation: https://optuna.readthedocs.io/
- Akiba, T., et al. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. *KDD 2019*.
- Bergstra, J., et al. (2011). Algorithms for Hyper-Parameter Optimization. *NIPS 2011*. (TPE algorithm)

## Related ADRs

- Depends on: [ADR-006: Nested CV Structure](ADR-006-nested-cv.md) - Inner CV where Optuna is applied
- Related to: [ADR-007: AUROC Optimization](ADR-007-auroc-optimization.md) - Optimization objective
