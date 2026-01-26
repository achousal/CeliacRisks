# Feature Selection Methods Reference

**Status**: Production
**Last Updated**: 2026-01-26
**Applies to**: CeD-ML v1.2.0+

---

## Table of Contents

1. [Overview](#overview)
2. [Method Comparison](#method-comparison)
3. [Detailed Methods](#detailed-methods)
   - [Hybrid Stability (Default)](#1-hybrid-stability-default)
   - [Nested RFECV](#2-nested-rfecv)
   - [Post-hoc RFE](#3-post-hoc-rfe)
   - [Fixed Panel Validation](#4-fixed-panel-validation)
4. [Decision Tree](#decision-tree)
5. [Workflows](#workflows)
6. [Technical Details](#technical-details)
7. [Troubleshooting](#troubleshooting)

---

## Overview

The CeD-ML pipeline provides four distinct feature selection approaches, each optimized for different use cases:

| Method | Scope | Use Case | Speed | Key Output |
|--------|-------|----------|-------|------------|
| **Hybrid Stability** | During training | Production models | Fast | Stable k-selected panels |
| **Nested RFECV** | During training | Scientific discovery | Slow (45×) | Consensus panels |
| **Post-hoc RFE** | After training | Deployment trade-offs | Very fast | Pareto curves |
| **Fixed Panel** | During training | Panel validation | Fast | Unbiased AUROC |

**Key principle**: Methods 1-2 are **mutually exclusive** (choose during training). Methods 3-4 are **post-training** tools for different purposes.

**Design rationale**: See [ADR-013: Four-Strategy Feature Selection Framework](../adr/ADR-013-four-strategy-feature-selection.md) for the architectural decision documenting why we need four strategies and their trade-offs.

---

## Method Comparison

### Quick Comparison Table

| Attribute | Hybrid Stability | Nested RFECV | Post-hoc RFE | Fixed Panel |
|-----------|-----------------|--------------|--------------|-------------|
| **When runs** | During CV | During CV | After training | During CV |
| **Feature selection** | Tuned k-best | Automatic optimal size | Post-hoc elimination | None (bypass) |
| **Data leakage risk** | None (per-fold) | None (per-fold) | Low (uses trained model) | None |
| **Typical runtime** | ~30 min | ~22 hours | ~5 min | ~30 min |
| **Cross-validation** | Nested CV (5×3×3) | Nested CV + internal RFE CV | None (single model) | Nested CV |
| **Panel size control** | Tunable k_grid | Automatic discovery | Manual exploration | Fixed input |
| **Stability measure** | Frequency across repeats | Consensus across folds | Single model | N/A |
| **Best for** | Production, reproducibility | Feature stability analysis | Clinical deployment | Regulatory validation |
| **Output metrics** | AUROC per k | AUROC per fold + consensus | Pareto curve (size vs AUROC) | Unbiased AUROC |

### Performance Expectations

**Runtime comparison** (LR_EN on 43k samples, 2920 proteins, 5 folds × 3 repeats):

```
Hybrid Stability:  ~30 minutes  (baseline)
Nested RFECV:      ~22 hours    (45× slower)
Post-hoc RFE:      ~5 minutes   (6× faster)
Fixed Panel:       ~30 minutes  (same as hybrid)
```

**AUROC comparison** (typical results):

```
Hybrid Stability (k=100):  0.945 ± 0.015
Nested RFECV (optimal):    0.943 ± 0.018  (slightly lower, more stable)
Post-hoc RFE (50 proteins): 0.940 ± 0.012  (optimistic by ~0.5%)
Fixed Panel (50 proteins):  0.938 ± 0.014  (unbiased ground truth)
```

---

## Detailed Methods

### 1. Hybrid Stability (Default)

**Pipeline**: screen → kbest (tuned) → stability → model

#### When to Use
- Default choice for production models
- When you need reproducible, interpretable results
- When you want to tune panel size (k) explicitly
- When speed matters (fastest training approach)

#### How It Works

1. **Screening** (Stage 1): Mann-Whitney U test filters top N proteins
   - Default: top 1000 proteins by effect size
   - Config: `features.screen_top_n: 1000`

2. **K-best tuning** (Stage 2): SelectKBest with tunable k
   - Grid search over `k_grid` values
   - Default: `[25, 50, 100, 150, 200, 300, 400]`
   - Scored via inner CV (AUROC)
   - Config: `features.k_grid`

3. **Stability filtering** (Stage 3): Post-hoc panel building
   - Compute selection frequency across CV repeats
   - Keep features selected in ≥ threshold of repeats
   - Default threshold: 0.70 (70% of repeats)
   - Config: `features.stability_thresh: 0.70`

4. **Correlation pruning** (Stage 4): Remove redundant features
   - Within stable panel, remove correlated pairs (r > threshold)
   - Default: r > 0.85
   - Config: `features.stable_corr_thresh: 0.85`

#### Configuration

```yaml
# configs/training_config.yaml
features:
  feature_selection_strategy: hybrid_stability  # Enable this method

  # Stage 1: Screening
  screen_method: mannwhitney
  screen_top_n: 1000

  # Stage 2: K-best tuning
  k_grid: [25, 50, 100, 150, 200, 300, 400]
  kbest_scope: protein  # Apply before transformations
  kbest_max: 800  # Hard cap on k values

  # Stage 3: Stability
  stability_thresh: 0.70  # Keep features in ≥70% of repeats

  # Stage 4: Correlation
  stable_corr_thresh: 0.85  # Remove r > 0.85 pairs
```

#### Outputs

**Location**: `results/{MODEL}/split_seed{N}/cv/`

```
feature_selection/
├── stability/
│   ├── selection_frequencies.csv  # Protein frequencies across repeats
│   ├── stable_panel_t{thresh}.csv # Stable features (≥ threshold)
│   └── stability_curve.png        # Frequency vs rank plot
└── kbest/
    ├── best_k_per_fold.csv        # Optimal k per fold
    └── kbest_auroc_vs_k.png       # AUROC vs k curve (if multi-k grid)
```

**Key files**:
- `selection_frequencies.csv`: Columns `[protein, frequency, mean_rank]`
- `stable_panel_t0.70.csv`: Final stable panel (one protein per row)
- `best_k_per_fold.csv`: Optimal k value per fold

#### Advantages
- Fast: ~30 minutes for full nested CV
- Interpretable: Clear k values, explicit stability threshold
- Reproducible: Fixed k_grid, deterministic stability
- Tunable: Easy to adjust k_grid and thresholds

#### Limitations
- Manual k_grid specification required
- May not find absolute optimal panel size
- Stability depends on number of CV repeats (recommend ≥3)

---

### 2. Nested RFECV

**Pipeline**: screen → *[k-best cap]* → RFECV (within each fold) → model → consensus panel

#### When to Use
- Scientific papers requiring feature stability analysis
- When you need to understand which features are robust vs. unstable
- When panel size is unknown and must be discovered automatically
- When you can afford 5-10× longer training time (with k-best pre-filter enabled)

#### How It Works

1. **Screening** (Stage 1): Same as hybrid (Mann-Whitney, top N)

2. **K-best pre-filter** (Stage 2, optional): Univariate cap before RFECV
   - **New optimization** to reduce computational cost
   - Applies SelectKBest (f_classif) if screened proteins > `rfe_kbest_k`
   - Default: cap at 100 proteins (reduces ~300 → ~100 for 5× speedup)
   - Enabled by default: `rfe_kbest_prefilter: true`
   - Skipped automatically if already below threshold

3. **RFECV per fold** (Stage 3): Recursive Feature Elimination with CV
   - **Within each outer CV fold**, run RFECV using internal CV
   - RFECV iteratively eliminates features based on importance
   - Stops at `rfe_target_size // 2` (default: 50 → stops at 25)
   - Uses `rfe_cv_folds` internal CV (default: 3 folds)
   - Step strategy: adaptive (remove ~10% per iteration)

4. **Consensus panel** (Stage 4): Aggregate across folds
   - Compute selection frequency across all outer folds
   - Keep features selected in ≥ `rfe_consensus_thresh` of folds
   - Default: 0.80 (80% of folds)

#### Configuration

```yaml
# configs/training_config.yaml
features:
  feature_selection_strategy: rfecv  # Enable this method

  # Stage 1: Screening (same as hybrid)
  screen_method: mannwhitney
  screen_top_n: 1000

  # Stage 2: K-best pre-filter (NEW: ~5× speedup)
  rfe_kbest_prefilter: true  # Apply univariate filter before RFECV
  rfe_kbest_k: 100           # Max features before RFECV (reduces ~300 → ~100)

  # Stage 3: RFECV parameters
  rfe_target_size: 50        # Stop elimination at 50 // 2 = 25 features
  rfe_step_strategy: adaptive # Options: adaptive, linear, geometric
  rfe_cv_folds: 3            # Internal CV folds for RFECV
  rfe_min_auroc_frac: 0.90   # Early stop (currently unused)

  # Stage 4: Consensus
  rfe_consensus_thresh: 0.80  # Features in ≥80% of folds
```

#### Outputs

**Location**: `results/{MODEL}/split_seed{N}/cv/rfecv/`

```
rfecv/
├── consensus_panel.csv           # Features in ≥ consensus_thresh of folds
├── fold_results.csv              # Per-fold optimal sizes and AUROCs
├── feature_stability.csv         # Selection frequency per protein
├── rfecv_selection_curve.png     # AUROC vs feature count (all folds)
└── fold_rankings/
    ├── fold_0_ranking.csv        # Feature ranking for fold 0
    ├── fold_1_ranking.csv
    └── ...
```

**Key files**:
- `consensus_panel.csv`: Robust features (columns: `[protein, selection_freq]`)
- `fold_results.csv`: Columns `[fold, optimal_n_features, optimal_auroc, final_features]`
- `feature_stability.csv`: Columns `[protein, selection_freq, mean_rank]`

#### Advantages
- Automatic panel size discovery (no manual k_grid)
- Robust consensus panels (features stable across folds)
- Detailed stability analysis (per-fold rankings)
- No data leakage (RFECV uses only fold training data)

#### Limitations
- Slow: 5-10× longer than hybrid (4-5 hours vs 30 minutes with k-best pre-filter)
  - **Without pre-filter**: 45× slower (~22 hours)
  - **With pre-filter** (default): 10× slower (~5 hours)
- Can produce different panels per fold (by design)
- Requires sufficient CV folds for consensus (recommend ≥5)
- More complex to interpret than fixed k

#### Performance Optimization

**K-best pre-filter** (enabled by default) provides ~5× speedup:

| Scenario | Proteins → RFECV | Estimated Time | Config |
|----------|------------------|----------------|--------|
| Without pre-filter | ~300 proteins | ~22 hours | `rfe_kbest_prefilter: false` |
| With pre-filter (default) | ~100 proteins | ~4 hours | `rfe_kbest_prefilter: true, rfe_kbest_k: 100` |
| Aggressive pre-filter | ~50 proteins | ~2 hours | `rfe_kbest_prefilter: true, rfe_kbest_k: 50` |

**How it works**: Before RFECV, applies univariate SelectKBest (f_classif) to cap features at `rfe_kbest_k`. Automatically skipped if already below threshold.

---

### 3. Post-hoc RFE

**Pipeline**: Trained model → RFE on stability panel → Pareto curve

#### When to Use
- After training, to explore panel size trade-offs
- Clinical deployment: "What's the smallest panel maintaining 0.90 AUROC?"
- Stakeholder decisions: Cost per protein vs. AUROC
- Rapid iteration: Test 10/20/50 protein panels in minutes

#### How It Works

1. **Load trained model**: Uses a pre-trained model bundle (`.joblib`)

2. **Extract stability panel**: Uses the stability ranking from training
   - Default: Start from top `start_size` proteins (default: 100)
   - Or use all proteins if `use_stability_panel=False`

3. **Recursive elimination**: Iteratively remove least important features
   - Compute feature importance (coefficients or permutation)
   - Remove bottom N features per iteration (adaptive: ~10%)
   - Re-evaluate AUROC on validation set (using CV)
   - Stop at `min_size` features or AUROC drop

4. **Generate recommendations**: Find knee points and Pareto frontier

#### CLI Command

```bash
ced optimize-panel \
  --model-path results/LR_EN/split_seed0/core/LR_EN__final_model.joblib \
  --infile ../data/Celiac_dataset_proteomics_w_demo.parquet \
  --split-dir ../splits/ \
  --split-seed 0 \
  --start-size 100 \
  --min-size 5 \
  --min-auroc-frac 0.90 \
  --cv-folds 5 \
  --step-strategy adaptive \
  --use-stability-panel
```

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model-path` | Required | Path to trained model bundle (`.joblib`) |
| `--infile` | Required | Input data file (Parquet/CSV) |
| `--split-dir` | Required | Directory with split indices |
| `--split-seed` | 0 | Split seed to use |
| `--start-size` | 100 | Starting panel size (top N from stability) |
| `--min-size` | 5 | Minimum panel size to evaluate |
| `--min-auroc-frac` | 0.90 | Early stop if AUROC < frac × max_auroc |
| `--cv-folds` | 5 | CV folds for OOF AUROC estimation |
| `--step-strategy` | adaptive | Elimination strategy (adaptive/linear/geometric) |
| `--use-stability-panel` | True | Start from stability ranking vs all proteins |

#### Outputs

**Location**: `results/{MODEL}/split_seed{N}/optimize_panel/`

```
optimize_panel/
├── panel_curve.png               # Pareto curve: AUROC vs panel size
├── feature_ranking.png           # Feature importance vs rank
├── rfe_curve.csv                 # Panel size, AUROC, features per iteration
├── feature_ranking.csv           # Protein, importance, rank
├── recommendations.json          # Knee points, Pareto frontier
└── full_results.pkl              # Complete RFEResult object
```

**Key files**:
- `panel_curve.png`: Pareto plot showing trade-offs
- `rfe_curve.csv`: Columns `[panel_size, mean_auroc, std_auroc, features]`
- `recommendations.json`: Contains knee points and suggested panels

#### Advantages
- Very fast: ~5 minutes (6× faster than training)
- Flexible: Explore many panel sizes without retraining
- Actionable: Clear cost-benefit curves for stakeholders
- Iterative: Test "what-if" scenarios quickly

#### Limitations
- Post-hoc optimism: AUROC estimates ~0.5% higher than unbiased
- Single model: No cross-validation of RFE itself
- Requires trained model: Can't run before training
- Stability ranking dependent: Quality depends on training stability

---

### 4. Fixed Panel Validation

**Pipeline**: Bypass feature selection → train on exact panel → report unbiased AUROC

#### When to Use
- Validate a consensus panel from nested CV
- Compare to published panels (e.g., Smith et al. 2023)
- Regulatory submission: FDA/clinical deployment
- Literature benchmarking: Unbiased performance estimate

#### How It Works

1. **Provide panel CSV**: List of protein names (one per row)

2. **Bypass feature selection**: All feature selection is disabled
   - `feature_selection_strategy` automatically set to `"none"`
   - Only specified proteins are used

3. **Train and evaluate**: Normal nested CV on the fixed panel
   - Full training pipeline (hyperparameter tuning, calibration)
   - Reports unbiased AUROC on validation/test sets

4. **Critical**: Use a **new split seed** (never used before)
   - Prevents "peeking" at splits used for panel discovery
   - Ensures unbiased performance estimate

#### CLI Command

```bash
# Step 1: Discovery (extract consensus panel from previous training)
awk -F',' 'NR==1 || $2 >= 0.70 {print $1}' \
  results/LR_EN/aggregated/feature_stability.csv \
  > deployment_panel_70pct.csv

# Step 2: Validation (use NEW split seed)
ced train \
  --model LR_EN \
  --fixed-panel deployment_panel_70pct.csv \
  --split-seed 10 \
  --config configs/training_config.yaml
```

#### CSV Format

**Option 1**: Column named "protein"
```csv
protein
PROT_123_resid
PROT_456_resid
PROT_789_resid
```

**Option 2**: First column (no header)
```csv
PROT_123_resid
PROT_456_resid
PROT_789_resid
```

#### Outputs

**Location**: `results/{MODEL}/split_seed{N}/` (same as normal training)

```
split_seed10/
├── core/
│   ├── LR_EN__final_model.joblib  # Model trained on fixed panel
│   └── config.yaml                # Config with fixed_panel metadata
├── evaluation/
│   ├── test_metrics.json          # UNBIASED AUROC (key output)
│   └── ...
└── ...
```

**Key file**: `evaluation/test_metrics.json`

```json
{
  "auroc": 0.938,  // Unbiased estimate (lower than post-hoc)
  "auroc_ci": [0.924, 0.952],
  "pr_auc": 0.456,
  "brier_score": 0.042,
  ...
}
```

#### Advantages
- Unbiased AUROC: Ground truth for panel performance
- Regulatory-grade: Suitable for FDA/clinical deployment
- Reproducible: Fixed panel ensures consistency
- Comparable: Fair comparison to published panels

#### Limitations
- Requires pre-determined panel (from discovery step)
- No feature selection tuning (by design)
- Needs new split seed (or holdout set) for unbiased estimate
- Can't discover new panels (validation only)

---

## Decision Tree

### Use this flowchart to choose the right method:

```
START: What is your goal?

├─ GOAL: Train a production model for deployment
│  └─ Use: Hybrid Stability (default)
│     ├─ Fast (~30 min)
│     ├─ Tunable k_grid
│     └─ Reproducible
│
├─ GOAL: Understand feature stability across folds
│  └─ Use: Nested RFECV
│     ├─ Automatic panel size discovery
│     ├─ Consensus panels
│     └─ Slow (~22 hours)
│
├─ GOAL: Explore panel size trade-offs for deployment
│  └─ Use: Post-hoc RFE
│     ├─ Very fast (~5 min)
│     ├─ Pareto curves
│     └─ Stakeholder-friendly
│
└─ GOAL: Validate a specific panel (unbiased AUROC)
   └─ Use: Fixed Panel
      ├─ Regulatory-grade
      ├─ Fair comparison
      └─ Requires new split seed
```

### Common Workflows

**Workflow 1: Standard Production Pipeline**
```bash
# Use hybrid_stability (default)
ced train --model LR_EN --split-seed 0
```

**Workflow 2: Scientific Discovery**
```bash
# Train with nested RFECV
ced train --model LR_EN --split-seed 0  # config: strategy=rfecv

# Extract consensus panel
cp results/LR_EN/split_seed0/cv/rfecv/consensus_panel.csv deployment_panel.csv
```

**Workflow 3: Clinical Deployment Optimization**
```bash
# Step 1: Train with hybrid_stability
ced train --model LR_EN --split-seed 0

# Step 2: Explore panel sizes
ced optimize-panel \
  --model-path results/LR_EN/split_seed0/core/LR_EN__final_model.joblib \
  --start-size 100 --min-size 10

# Step 3: Validate chosen panel (e.g., 50 proteins)
# (manually create panel_50.csv from rfe_curve.csv)
ced train --fixed-panel panel_50.csv --split-seed 10  # NEW SEED
```

**Workflow 4: Multi-split Consensus**
```bash
# Step 1: Train multiple splits with hybrid_stability
for seed in 0 1 2 3 4; do
  ced train --model LR_EN --split-seed $seed
done

# Step 2: Aggregate to get consensus panel
ced aggregate-splits --config configs/aggregate_config.yaml

# Step 3: Extract 70% stability panel
awk -F',' 'NR==1 || $2 >= 0.70 {print $1}' \
  results/LR_EN/aggregated/feature_stability.csv \
  > consensus_70pct.csv

# Step 4: Validate consensus panel
ced train --fixed-panel consensus_70pct.csv --split-seed 10
```

---

## Technical Details

### Data Leakage Prevention

All methods prevent data leakage through strict fold isolation:

**Hybrid Stability**: Feature selection runs independently per fold
- Screening: Per-fold effect sizes
- K-best: Per-fold SelectKBest
- Stability: Computed across repeats (post-hoc, no leakage)

**Nested RFECV**: Internal CV within each outer fold
- RFECV uses only training data from current outer fold
- Internal CV (3 folds) for AUROC estimation
- No information from validation/test sets

**Post-hoc RFE**: Uses validation set for AUROC
- Slight optimism (~0.5%) due to single model
- Mitigated by using CV (not single holdout)
- Fixed panel validation provides unbiased estimate

**Fixed Panel**: Standard nested CV (no selection)
- No leakage (no feature selection)
- New split seed ensures independence from discovery

### Feature Importance Methods

**Coefficient-based** (LR, LinSVM):
```python
# Extract coefficients from trained model
coef = model.coef_[0]  # Shape: (n_features,)
importance = np.abs(coef)  # Absolute magnitude
```

**Permutation-based** (RF, XGBoost):
```python
# Permutation importance (slower, more accurate)
from sklearn.inspection import permutation_importance
result = permutation_importance(
    model, X_val, y_val,
    n_repeats=10, scoring='roc_auc', random_state=0
)
importance = result.importances_mean
```

### Computational Complexity

**Hybrid Stability**:
- Screening: O(N × P) where N=samples, P=proteins
- K-best: O(N × P × |k_grid|) per fold
- Stability: O(R × F × P) where R=repeats, F=folds
- Total: O(N × P × F × R × I) where I=inner_folds

**Nested RFECV**:
- RFECV per fold: O(N × P × log(P) × CV_inner × F_outer)
- Eliminations: ~log(P) iterations (adaptive step)
- Total: 45× slower than hybrid (empirical)

**Post-hoc RFE**:
- Per iteration: O(N × k × CV_folds) where k=current panel size
- Iterations: ~log(start_size - min_size)
- Total: Very fast (~5 min, no training)

**Fixed Panel**:
- Same as standard training (no feature selection overhead)
- O(N × k × F × R × I) where k=fixed panel size

---

## Troubleshooting

### Hybrid Stability Issues

**Q: Stability panel is empty or very small**

A: Lower `stability_thresh` or increase `cv.repeats`
```yaml
features:
  stability_thresh: 0.50  # Lower from 0.70
cv:
  repeats: 5  # Increase from 3
```

**Q: All k values give similar AUROC**

A: Expand `k_grid` range or check if features are saturating
```yaml
features:
  k_grid: [10, 25, 50, 100, 200, 400, 800]  # Wider range
```

**Q: "ConvergenceWarning" during training**

A: Increase `max_iter` for LR/SVM
```yaml
lr:
  max_iter: 10000  # Increase from 1000
```

---

### Nested RFECV Issues

**Q: RFECV taking too long (>24 hours)**

A: Reduce `cv.folds`, `cv.repeats`, or use `rfe_cv_folds=2`
```yaml
cv:
  folds: 3  # Reduce from 5
  repeats: 2  # Reduce from 3
features:
  rfe_cv_folds: 2  # Reduce from 3
```

**Q: Consensus panel is empty**

A: Lower `rfe_consensus_thresh`
```yaml
features:
  rfe_consensus_thresh: 0.60  # Lower from 0.80
```

**Q: Different optimal sizes per fold (high variance)**

A: Expected behavior. Use consensus panel or lower `rfe_target_size`
```yaml
features:
  rfe_target_size: 30  # Stop elimination earlier
```

**Q: "OutOfMemoryError" during RFECV**

A: Reduce `screen_top_n` or use smaller `rfe_cv_folds`
```yaml
features:
  screen_top_n: 500  # Reduce from 1000
  rfe_cv_folds: 2  # Reduce from 3
```

---

### Post-hoc RFE Issues

**Q: AUROC drops sharply at large panel sizes**

A: Expected if starting panel was over-fitted. Use `start_size=50` or lower.

**Q: "Model bundle missing protein_cols"**

A: Ensure you're using a model bundle (`.joblib`) saved by CeD-ML v1.2.0+. Re-train if using old model.

**Q: Can't reproduce panel curve**

A: Use same `--split-seed` and `--cv-folds` as original run. RFE uses validation fold for AUROC.

**Q: Pareto plot shows no clear knee**

A: AUROC is robust across panel sizes. Use stakeholder constraints (cost, feasibility) to choose.

---

### Fixed Panel Issues

**Q: "Fixed panel contains N proteins not in dataset"**

A: Protein names in CSV don't match dataset columns. Check for:
- Suffix mismatch: `_resid` vs no suffix
- Case sensitivity: `PROT_123` vs `prot_123`
- Whitespace: Leading/trailing spaces

Fix:
```bash
# Check dataset columns
head -n1 ../data/Celiac_dataset_proteomics_w_demo.parquet

# Fix panel CSV (example: add _resid suffix)
awk '{print $1 "_resid"}' panel.csv > panel_fixed.csv
```

**Q: Fixed panel AUROC much lower than post-hoc RFE**

A: Expected. Post-hoc RFE is optimistic (~0.5-1%). Fixed panel is unbiased ground truth.

**Q: Using same split seed as discovery (leakage)**

A: CRITICAL ERROR. Always use a **new split seed** for fixed panel validation.
```bash
# WRONG (split seed 0 was used for discovery)
ced train --fixed-panel panel.csv --split-seed 0

# CORRECT (use new seed)
ced train --fixed-panel panel.csv --split-seed 10
```

---

## References

### Related Documentation

- [OPTIMIZE_PANEL.md](OPTIMIZE_PANEL.md): Detailed post-hoc RFE guide
- [HYPERPARAMETER_TUNING.md](HYPERPARAMETER_TUNING.md): Optuna and RandomizedSearchCV
- [CLI_REFERENCE.md](CLI_REFERENCE.md): Complete CLI command reference
- [ARCHITECTURE.md](../ARCHITECTURE.md): Pipeline architecture and code pointers

### Architecture Decision Records

- [ADR-013](../adr/ADR-013-four-strategy-feature-selection.md): Four-strategy feature selection framework (unified approach)
- [ADR-004](../adr/ADR-004-hybrid-feature-selection.md): Hybrid feature selection (Strategy 1 details)
- [ADR-005](../adr/ADR-005-stability-panel.md): Stability-based panel building (used by Strategies 1-3)

### Key Modules

- [analysis/src/ced_ml/features/screening.py](../../src/ced_ml/features/screening.py): Effect size screening
- [analysis/src/ced_ml/features/kbest.py](../../src/ced_ml/features/kbest.py): SelectKBest tuning
- [analysis/src/ced_ml/features/stability.py](../../src/ced_ml/features/stability.py): Stability analysis
- [analysis/src/ced_ml/features/nested_rfe.py](../../src/ced_ml/features/nested_rfe.py): Nested RFECV implementation
- [analysis/src/ced_ml/features/rfe.py](../../src/ced_ml/features/rfe.py): Post-hoc RFE implementation
- [analysis/src/ced_ml/cli/train.py](../../src/ced_ml/cli/train.py): Training CLI (fixed panel)
- [analysis/src/ced_ml/cli/optimize_panel.py](../../src/ced_ml/cli/optimize_panel.py): Post-hoc RFE CLI

---

**Document Status**: Production-ready
**Maintainer**: Andres Chousal
**Last Reviewed**: 2026-01-26
