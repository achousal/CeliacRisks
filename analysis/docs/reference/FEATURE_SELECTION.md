# Feature Selection Reference

**Status**: Production
**Last Updated**: 2026-01-26
**Applies to**: CeD-ML v1.2.0+

---

## Quick Reference

### Decision Tree (Start Here)

**First question: Does panel size/cost matter?**

```
Do you have budget/cost constraints on the final panel?

YES (Clinical deployment, assay costs matter)
│
├─ Need to see AUROC vs panel size trade-offs?
│  │
│  ├─ YES → Two-Stage Post-hoc RFE (Workflow 3)
│  │         Stage 1: ced optimize-panel (5 min)
│  │         Stage 2: ced train --fixed-panel --split-seed 10 (30 min)
│  │         Output: Full Pareto curve + unbiased AUROC for chosen k
│  │
│  └─ NO (just want fast model) → Hybrid Stability (Strategy 1)
│                                   ced train (30 min)
│                                   Output: Stable panel at tuned k
│
NO (Research, feature discovery, cost not a factor)
│
├─ Need feature stability analysis?
│  │
│  ├─ YES → Single-Stage RFECV (Strategy 2 / Workflow 4)
│  │         ced train with rfecv (5-22 hours)
│  │         Output: Consensus panel + unbiased AUROC
│  │
│  └─ NO → Hybrid Stability (Strategy 1)
│             ced train (30 min)
│             Output: Stable panel at tuned k
│
Already have a panel to validate? → Fixed Panel (Strategy 4)
                                      ced train --fixed-panel panel.csv --split-seed 10
                                      Output: Unbiased AUROC only
```

**Quick summary**:

| Your situation | Recommended approach | Rationale |
|----------------|---------------------|-----------|
| Clinical deployment, cost matters | Two-stage post-hoc RFE | See AUROC vs cost curve, choose k |
| Research paper, feature discovery | Single-stage RFECV | Automatic, unbiased, detailed stability |
| Fast production model | Hybrid stability | Default, 30 min, good balance |
| Validate specific panel | Fixed panel | Unbiased AUROC for given proteins |

### Original Decision Tree (by strategy type)

```
What do you need?

├─ FAST PRODUCTION MODEL (default)
│  └─ Strategy 1: Hybrid Stability
│     Command: ced train --model LR_EN --split-seed 0
│     Config: feature_selection_strategy: hybrid_stability
│     Runtime: ~30 minutes
│
├─ FEATURE STABILITY ANALYSIS (scientific papers)
│  └─ Strategy 2: Nested RFECV
│     Config: feature_selection_strategy: rfecv
│     Runtime: ~22 hours (or ~5 hours with k-best pre-filter)
│
├─ DEPLOYMENT PANEL SIZING (cost vs. performance)
│  └─ Strategy 3: Post-hoc RFE
│     Command: ced optimize-panel --model-path ... --start-size 100
│     Runtime: ~5 minutes (run after training)
│
└─ UNBIASED VALIDATION (regulatory, literature comparison)
   └─ Strategy 4: Fixed Panel
      Command: ced train --fixed-panel panel.csv --split-seed 10
      Runtime: ~30 minutes
```

### Strategy Comparison

| Attribute | Hybrid Stability | Nested RFECV | Post-hoc RFE | Fixed Panel |
|-----------|------------------|--------------|--------------|-------------|
| **When** | During training | During training | After training | During training |
| **Speed** | Fast (~30 min) | Slow (~22 hrs) | Very fast (~5 min) | Fast (~30 min) |
| **Use for** | Production | Discovery | **Deployment sizing** | Validation |
| **Panel size** | Tuned (k_grid) | Automatic (max AUROC) | **User-optimized** | Fixed (input) |
| **Cost consideration** | Partial (k_grid) | ❌ None (always max) | ✅ **Full visibility** | N/A |
| **Trade-off curve** | No | No | ✅ **Yes (Pareto)** | No |
| **Leakage** | None | None | Low (~0.5%) | None |
| **Output** | Stable k-panels | Consensus panels | **AUROC vs k curve** | Unbiased AUROC |
| **Best for** | Fast baseline | Feature stability | **Clinical deployment** | Regulatory filing |

**Design rationale**: See [ADR-013](../adr/ADR-013-four-strategy-feature-selection.md) for why we need four strategies.

---

## Table of Contents

1. [Quick Reference](#quick-reference) (above)
2. [Strategy 1: Hybrid Stability](#strategy-1-hybrid-stability-default)
3. [Strategy 2: Nested RFECV](#strategy-2-nested-rfecv)
4. [Strategy 3: Post-hoc RFE](#strategy-3-post-hoc-rfe)
5. [Strategy 4: Fixed Panel](#strategy-4-fixed-panel-validation)
6. [Common Workflows](#common-workflows)
7. [Troubleshooting](#troubleshooting)

---

## Strategy 1: Hybrid Stability (Default)

**Pipeline**: screen → kbest (tuned) → stability → correlation pruning → model

### When to Use
- Default choice for production models
- Need reproducible, interpretable results
- Want to tune panel size (k) explicitly
- Speed matters (fastest training approach)

### Quick Start

```yaml
# configs/training_config.yaml
features:
  feature_selection_strategy: hybrid_stability
  screen_top_n: 1000
  k_grid: [25, 50, 100, 150, 200, 300, 400]
  stability_thresh: 0.70
  stable_corr_thresh: 0.85
```

```bash
ced train --model LR_EN --split-seed 0
```

### How It Works

1. **Screening**: Mann-Whitney U test filters top 1000 proteins by effect size
2. **K-best tuning**: Grid search over k_grid values via inner CV
3. **Stability filtering**: Keep features selected in ≥70% of CV repeats
4. **Correlation pruning**: Remove redundant features (r > 0.85)

### Outputs

```
results/{MODEL}/split_seed{N}/feature_selection/
├── stability/
│   ├── selection_frequencies.csv  # Protein frequencies across repeats
│   ├── stable_panel_t0.70.csv     # Final stable panel
│   └── stability_curve.png
└── kbest/
    ├── best_k_per_fold.csv
    └── kbest_auroc_vs_k.png
```

### Advantages / Limitations

**Pros**: Fast (~30 min), interpretable, reproducible, tunable k_grid
**Cons**: Manual k_grid specification, may not find absolute optimal size

---

## Strategy 2: Nested RFECV

**Pipeline**: screen → [k-best cap] → RFECV (per fold) → consensus panel → model

### When to Use
- Scientific papers requiring feature stability analysis
- Need to understand robust vs. unstable features
- Panel size unknown and must be discovered
- Can afford longer training time (5-22 hours)

### Quick Start

```yaml
# configs/training_config.yaml
features:
  feature_selection_strategy: rfecv
  screen_top_n: 1000
  rfe_kbest_prefilter: true   # 5× speedup (recommended)
  rfe_kbest_k: 100            # Cap at 100 before RFECV
  rfe_target_size: 50         # Stop at 50 // 2 = 25 features
  rfe_consensus_thresh: 0.80  # Features in ≥80% of folds
```

```bash
ced train --model LR_EN --split-seed 0
```

### How It Works

1. **Screening**: Same as Hybrid (top 1000 proteins)
2. **K-best pre-filter** (optional, recommended): Cap at 100 proteins for 5× speedup
3. **RFECV per fold**: Within each outer CV fold, run RFECV with internal CV
   - Iteratively eliminates features based on importance
   - Stops at rfe_target_size // 2 (default: 25)
4. **Consensus panel**: Keep features selected in ≥80% of folds

### Outputs

```
results/{MODEL}/split_seed{N}/cv/rfecv/
├── consensus_panel.csv          # Robust features (≥consensus_thresh)
├── feature_stability.csv        # Selection frequency per protein
├── fold_results.csv             # Per-fold optimal sizes and AUROCs
├── rfecv_selection_curve.png
└── fold_rankings/
    └── fold_*_ranking.csv
```

### Performance Optimization

| Scenario | Proteins → RFECV | Estimated Time | Config |
|----------|------------------|----------------|--------|
| Without pre-filter | ~300 | ~22 hours | `rfe_kbest_prefilter: false` |
| With pre-filter (default) | ~100 | ~5 hours | `rfe_kbest_prefilter: true, rfe_kbest_k: 100` |
| Aggressive | ~50 | ~2 hours | `rfe_kbest_k: 50` |

### The RFECV Optimization Blind Spot

**CRITICAL**: RFECV has a fundamental limitation for clinical deployment: **it always maximizes AUROC within tolerance, ignoring panel size costs**.

```python
# RFECV stops when CV score FIRST drops below best - tolerance
# Example with tol=0.01:
# k=100: AUROC 0.950 ← RFECV selects this
# k=75:  AUROC 0.948 ← within tolerance, keeps eliminating
# k=50:  AUROC 0.945 ← within tolerance, keeps eliminating
# k=25:  AUROC 0.930 ← DROPS, stops at k=50
```

**The problem**: RFECV picks k=50, but clinical stakeholders might happily accept k=25 (AUROC 0.930) if it reduces assay costs by 50%!

**What each method actually optimizes**:

| Method | Optimization Goal | Considers Cost? | Trade-off Visibility |
|--------|------------------|-----------------|---------------------|
| **RFECV** | Max AUROC - tol | ❌ No | None (black box) |
| **Post-hoc RFE** | **User decision** | ✅ Yes | Full curve visible |
| **Hybrid Stability** | Stability + tuned k | Partial | k_grid only |

**When RFECV is appropriate**:
- Pure feature discovery (research papers)
- Panel size is not a constraint
- Stakeholders want "best possible AUROC"

**When post-hoc RFE is better**:
- Clinical deployment with budget limits
- Stakeholders need AUROC vs. cost trade-off
- Regulatory submissions requiring cost-benefit analysis

### Advantages / Limitations

**Pros**: Automatic panel size, robust consensus, detailed stability analysis, no leakage
**Cons**: Slow (5-22 hours), different panels per fold (by design), complex to interpret, **always maximizes AUROC (ignores panel size costs)**

---

## Strategy 3: Post-hoc RFE

**Pipeline**: Trained model → RFE on stability panel → Pareto curve → recommendations

### When to Use
- After training, explore panel size trade-offs
- Clinical deployment: "What's the smallest panel maintaining 0.90 AUROC?"
- Stakeholder decisions: Cost per protein vs. AUROC
- Rapid iteration: Test 10/20/50 protein panels in minutes

### Quick Start

```bash
# Run AFTER training
ced optimize-panel \
  --model-path results/LR_EN/split_seed0/core/LR_EN__final_model.joblib \
  --infile ../data/input.parquet \
  --split-dir ../splits/ \
  --split-seed 0 \
  --start-size 100 \
  --min-size 5 \
  --min-auroc-frac 0.90
```

### How It Works

1. **Load trained model**: Uses pre-trained model bundle (.joblib)
2. **Extract stability panel**: Start from top N proteins from stability ranking
3. **Recursive elimination**: Iteratively remove least important features
   - Compute feature importance (coefficients or permutation)
   - Remove bottom ~10% per iteration (adaptive)
   - Re-evaluate AUROC on validation set (using CV)
   - Stop at min_size or AUROC drop
4. **Generate recommendations**: Find knee points and Pareto frontier

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model-path` | Required | Path to trained model bundle (.joblib) |
| `--infile` | Required | Input data file (Parquet/CSV) |
| `--split-dir` | Required | Directory with split indices |
| `--split-seed` | 0 | Split seed to use |
| `--start-size` | 100 | Starting panel size (top N from stability) |
| `--min-size` | 5 | Minimum panel size to evaluate |
| `--min-auroc-frac` | 0.90 | Early stop if AUROC < frac × max_auroc |
| `--cv-folds` | 5 | CV folds for OOF AUROC estimation |
| `--step-strategy` | adaptive | Elimination strategy (adaptive/linear/geometric) |

### Outputs

```
results/{MODEL}/split_seed{N}/optimize_panel/
├── panel_curve.png              # Pareto curve: AUROC vs panel size
├── panel_curve.csv              # Panel size, AUROC, features per iteration
├── recommended_panels.json      # Knee points, min sizes for 95%/90%/85%
├── feature_ranking.csv          # Protein, elimination order
├── feature_ranking.png          # Importance visualization
└── pareto_frontier.csv          # Non-dominated solutions
```

**Key file: recommended_panels.json**
```json
{
  "max_auroc": 0.887,
  "recommended_panels": {
    "min_size_95pct": 25,  // 25 proteins for 95% of max AUROC
    "min_size_90pct": 12,  // 12 proteins for 90% of max AUROC
    "knee_point": 18       // Diminishing returns threshold
  }
}
```

### Interpretation Guide

**Scenario 1: Steep decline → Plateau**
```
AUROC
  |     *
  |     |\
  |     | \___
  |_____|_____
       25   50
```
**Interpretation**: Keep 25+ proteins; diminishing returns beyond
**Action**: Use recommended min_size_90pct

**Scenario 2: Gradual decline (no clear knee)**
```
AUROC
  |     *
  |     |\
  |     | \
  |_____|__
       25 50
```
**Interpretation**: Proteins have similar importance
**Action**: Choose based on cost-benefit constraints

**Scenario 3: Collapse at small sizes**
```
AUROC
  |     *
  |     |\
  |     | |__
  |_____|____
        10  5
```
**Interpretation**: Few proteins carry most signal
**Action**: Use larger panel (15+) for robustness

### Practical Decision-Making

Given recommendations:
```json
{"max_auroc": 0.887, "min_size_95pct": 25, "min_size_90pct": 12, "knee_point": 18}
```

**Decision matrix:**
| Panel Size | AUROC | Cost | Recommendation |
|-----------|-------|------|----------------|
| 12 | 0.798 | Very low | High-risk screening |
| 18 | ~0.865 | Low | Good balance (knee point) |
| 25 | 0.842 | Medium | Safe option (95% retention) |
| 50+ | 0.880+ | High | Reference standard |

**Suggested approach:**
1. **Gold standard**: 100-protein panel (research/development)
2. **Clinical deployment**: 18-25 proteins (knee point ± margin)
3. **Rapid screening**: 12 proteins with follow-up testing

### Why Post-hoc RFE is Superior for Deployment

Unlike RFECV, post-hoc RFE gives you **control over the AUROC vs. cost trade-off**:

```bash
# Post-hoc RFE output (example):
# panel_curve.csv:
# n_features,mean_auroc,std_auroc
# 100,0.950,0.012
# 75,0.948,0.013
# 50,0.945,0.014
# 25,0.930,0.018  ← Stakeholder decides!
# 10,0.880,0.025
```

**Decision scenario**:
- Clinical team: "We can only afford 25 proteins"
- You: "AUROC drops from 0.950 → 0.930, acceptable?"
- Clinical team: "Yes, deploy k=25"

**RFECV would force you to k=50** with no visibility into the 25-protein alternative.

### Advantages / Limitations

**Pros**: Very fast (~5 min), flexible exploration, actionable cost-benefit curves, iterative, **stakeholder controls trade-offs**
**Cons**: Post-hoc optimism (~0.5% higher AUROC), single model (no CV of RFE itself), requires trained model

---

## Strategy 4: Fixed Panel Validation

**Pipeline**: Bypass feature selection → train on exact panel → report unbiased AUROC

### When to Use
- Validate consensus panel from nested CV
- Compare to published panels (e.g., literature benchmarking)
- Regulatory submission: FDA/clinical deployment
- Unbiased performance estimate

### Quick Start

```bash
# Step 1: Extract consensus panel from previous training
awk -F',' 'NR==1 || $2 >= 0.70 {print $1}' \
  results/LR_EN/aggregated/feature_stability.csv \
  > deployment_panel.csv

# Step 2: Validate (use NEW split seed - CRITICAL)
ced train \
  --model LR_EN \
  --fixed-panel deployment_panel.csv \
  --split-seed 10 \
  --config configs/training_config.yaml
```

### How It Works

1. **Provide panel CSV**: List of protein names (one per row)
2. **Bypass feature selection**: All feature selection disabled (strategy → "none")
3. **Train and evaluate**: Normal nested CV on the fixed panel
4. **Critical**: Use a **new split seed** never used before (prevents peeking)

### CSV Format

**Option 1: Column named "protein"**
```csv
protein
PROT_123_resid
PROT_456_resid
```

**Option 2: First column (no header)**
```csv
PROT_123_resid
PROT_456_resid
```

### Outputs

```
results/{MODEL}/split_seed{N}/
├── core/
│   ├── LR_EN__final_model.joblib  # Model trained on fixed panel
│   └── config.yaml                # Config with fixed_panel metadata
├── evaluation/
│   ├── test_metrics.json          # UNBIASED AUROC (key output)
│   └── ...
```

**Key file: evaluation/test_metrics.json**
```json
{
  "auroc": 0.938,  // Unbiased estimate (lower than post-hoc RFE)
  "auroc_ci": [0.924, 0.952],
  "pr_auc": 0.456,
  "brier_score": 0.042
}
```

### Advantages / Limitations

**Pros**: Unbiased AUROC (ground truth), regulatory-grade, reproducible, comparable to literature
**Cons**: Requires pre-determined panel, no feature selection tuning, needs new split seed, can't discover panels

---

## Common Workflows

### Workflow 1: Production Pipeline (Fastest)
```bash
# Use hybrid_stability (default)
ced train --model LR_EN --split-seed 0
```

### Workflow 2: Scientific Publication
```bash
# Train with nested RFECV
ced train --model LR_EN --split-seed 0  # config: rfecv
# Output: results/LR_EN/split_seed0/cv/rfecv/consensus_panel.csv
```

### Workflow 3: Two-Stage Clinical Deployment (Recommended)

**Goal**: Find optimal panel size with cost-benefit trade-off, then validate with unbiased AUROC.

```bash
# ═══════════════════════════════════════════════════════════════
# STAGE 1: DISCOVERY (Explore panel sizes)
# ═══════════════════════════════════════════════════════════════

# Step 1: Train with hybrid_stability (fast baseline)
ced train --model LR_EN --split-seed 0  # ~30 min

# Step 2: Run post-hoc RFE to explore panel size trade-offs
ced optimize-panel \
  --model-path results/LR_EN/split_seed0/core/LR_EN__final_model.joblib \
  --infile ../data/input.parquet \
  --split-dir ../splits/ \
  --split-seed 0 \
  --start-size 100 \
  --min-size 5  # ~5 min

# Step 3: Review trade-off curve
cat results/LR_EN/split_seed0/panel_optimization/rfe_results.csv
cat results/LR_EN/split_seed0/panel_optimization/recommended_panels.json

# Example output:
# n_features,mean_auroc,std_auroc
# 100,0.950,0.012
# 50,0.945,0.014
# 25,0.930,0.018  ← Decision point
# 10,0.880,0.025
#
# Recommendations:
# - min_size_95pct: 50 (AUROC 0.945)
# - min_size_90pct: 25 (AUROC 0.930)
# - knee_point: 30 (diminishing returns)

# Step 4: Stakeholder decision
# Clinical team: "We can afford 25 proteins max"
# You: "AUROC drops 0.950 → 0.930 (2%), acceptable?"
# Clinical team: "Yes, deploy k=25"

# ═══════════════════════════════════════════════════════════════
# STAGE 2: VALIDATION (Unbiased estimate on NEW data)
# ═══════════════════════════════════════════════════════════════

# Step 5: Extract chosen panel (e.g., top 25 proteins)
head -n 26 results/LR_EN/split_seed0/panel_optimization/rfe_ranking_full.csv \
  | awk -F',' 'NR==1 || NR<=26 {print $1}' > deployment_panel_k25.csv

# Step 6: CRITICAL - Validate on NEW split seed (prevents data leakage)
ced train \
  --model LR_EN \
  --fixed-panel deployment_panel_k25.csv \
  --split-seed 10 \
  --config configs/training_config.yaml  # ~30 min

# Step 7: Extract unbiased AUROC (regulatory-grade estimate)
cat results/LR_EN/split_seed10/evaluation/test_metrics.json
# {"auroc": 0.928, "auroc_ci": [0.914, 0.942]}
#  ↑ Expected to be ~0.5-1% lower than Stage 1 (0.930 → 0.928)
#  ↑ This is your FDA/publication number
```

**Why this workflow?**

| Aspect | Stage 1 (Discovery) | Stage 2 (Validation) |
|--------|---------------------|---------------------|
| **Purpose** | Find optimal k | Unbiased AUROC estimate |
| **Split Seed** | 0 (arbitrary) | **NEW seed** (e.g., 10) |
| **Feature Selection** | RFE on trained model | None (fixed panel) |
| **AUROC Bias** | +0.5-1% optimistic | Unbiased (ground truth) |
| **Use For** | Stakeholder decisions | Regulatory submission |
| **Speed** | 5 min (after training) | 30 min (full training) |

**Key principles**:
1. **Separation of concerns**: Discovery uses split 0, validation uses split 10+
2. **No data peeking**: Fixed-panel validation must use data that NEVER influenced panel selection
3. **Regulatory readiness**: Stage 2 provides the performance number you'd report to FDA/journals
4. **Practical trade-off**: Stage 1 lets you explore cost vs. performance; Stage 2 locks it in

**When to skip Stage 2**:
- Internal development only (not for deployment)
- Rapidly iterating on panel sizes
- Cost constraints change frequently

**When Stage 2 is mandatory**:
- FDA/regulatory submission
- Clinical deployment
- Literature comparison (fair benchmark)
- Stakeholder sign-off on final panel

### Workflow 4: Single-Stage RFECV (Alternative for Full Automation)

**Goal**: Fully automated panel discovery with unbiased AUROC (no manual intervention).

```bash
# Single command - RFECV discovers panel size automatically
ced train --model LR_EN --split-seed 0  # ~5-22 hours
# (config: feature_selection_strategy: rfecv)

# Output: Consensus panel with unbiased AUROC
cat results/LR_EN/split_seed0/cv/rfecv/consensus_panel.csv
cat results/LR_EN/split_seed0/evaluation/test_metrics.json
```

**Comparison with Two-Stage Post-hoc RFE**:

| Aspect | Two-Stage Post-hoc RFE | Single-Stage RFECV |
|--------|----------------------|-------------------|
| **Total runtime** | 35 min (30 + 5 + validation) | 5-22 hours |
| **User control** | High (choose k from curve) | Low (automatic k) |
| **Cost consideration** | ✅ Full visibility | ❌ None (always max AUROC) |
| **Unbiased AUROC** | Stage 2 only (new seed) | ✅ Built-in |
| **Stakeholder engagement** | High (visual trade-offs) | Low (black box) |
| **Panel size outcome** | User-optimized | RFECV-optimized |
| **Best for** | Clinical deployment | Research papers |

**Example outcome comparison**:

*Scenario: Same dataset, same model (LR_EN)*

**Two-stage approach**:
- Stage 1: "k=25 gives AUROC 0.930 (vs 0.950 at k=100)"
- Stakeholder: "Accept k=25 (saves 75 proteins)"
- Stage 2: Unbiased AUROC = 0.928 (k=25 panel)

**RFECV approach**:
- Automatic: "k=50 gives AUROC 0.945"
- No visibility into k=25 alternative
- Unbiased AUROC = 0.944 (k=50 panel)

**Which is "better"?**
- If cost matters → Two-stage (k=25, AUROC 0.928)
- If only AUROC matters → RFECV (k=50, AUROC 0.944)

**In practice**: Clinical deployment almost always has cost constraints, making two-stage post-hoc RFE the preferred approach.

### Workflow 5: Multi-split Consensus
```bash
# Step 1: Train multiple splits
for seed in 0 1 2 3 4; do
  ced train --model LR_EN --split-seed $seed
done

# Step 2: Aggregate
ced aggregate-splits --config configs/aggregate_config.yaml

# Step 3: Extract consensus
awk -F',' 'NR==1 || $2 >= 0.70 {print $1}' \
  results/LR_EN/aggregated/feature_stability.csv > consensus.csv

# Step 4: Validate
ced train --fixed-panel consensus.csv --split-seed 10
```

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
A: Expand `k_grid` range
```yaml
features:
  k_grid: [10, 25, 50, 100, 200, 400, 800]
```

---

### Nested RFECV Issues

**Q: RFECV taking too long (>24 hours)**
A: Enable k-best pre-filter (5× speedup)
```yaml
features:
  rfe_kbest_prefilter: true
  rfe_kbest_k: 100
cv:
  folds: 3  # Reduce from 5
```

**Q: Consensus panel is empty**
A: Lower `rfe_consensus_thresh`
```yaml
features:
  rfe_consensus_thresh: 0.60  # Lower from 0.80
```

---

### Post-hoc RFE Issues

**Q: AUROC drops sharply at large panel sizes**
A: Starting panel was over-fitted. Use `start_size=50`

**Q: "Model bundle missing protein_cols"**
A: Re-train with CeD-ML v1.2.0+ (old model bundles incompatible)

**Q: Can't reproduce panel curve**
A: Use same `--split-seed` and `--cv-folds` as original run

**Q: Pareto plot shows no clear knee**
A: AUROC is robust across panel sizes. Use stakeholder constraints (cost, feasibility)

---

### Fixed Panel Issues

**Q: "Fixed panel contains N proteins not in dataset"**
A: Protein names don't match dataset columns. Check for:
- Suffix mismatch: `_resid` vs no suffix
- Case sensitivity: `PROT_123` vs `prot_123`
- Whitespace: Leading/trailing spaces

Fix:
```bash
# Check dataset columns
head -n1 ../data/input.parquet

# Fix panel CSV (example: add _resid suffix)
awk '{print $1 "_resid"}' panel.csv > panel_fixed.csv
```

**Q: Fixed panel AUROC much lower than post-hoc RFE**
A: Expected. Post-hoc RFE is optimistic (~0.5-1%). Fixed panel is unbiased ground truth.

**Q: Using same split seed as discovery (CRITICAL ERROR)**
A: Always use **new split seed** for fixed panel validation
```bash
# WRONG (split seed 0 was used for discovery)
ced train --fixed-panel panel.csv --split-seed 0

# CORRECT (use new seed)
ced train --fixed-panel panel.csv --split-seed 10
```

---

## Frequently Asked Questions

### Q: When should I use RFECV vs post-hoc RFE?

**Use post-hoc RFE when**:
- Panel size/cost is a constraint (clinical deployment)
- Stakeholders need to see AUROC vs panel size trade-offs
- You want rapid iteration (5 min per curve)
- Regulatory submission requires cost-benefit analysis

**Use RFECV when**:
- Panel size is not a constraint (research only)
- You want fully automated feature discovery
- You need detailed feature stability analysis
- You can afford 5-22 hours training time

**Key insight**: RFECV always maximizes AUROC within tolerance, ignoring panel size. Post-hoc RFE lets you choose the optimal AUROC/cost balance.

### Q: Why is my fixed-panel AUROC lower than post-hoc RFE?

**Expected behavior**. Example:
- Post-hoc RFE (split 0): AUROC 0.940 for k=50
- Fixed panel (split 10): AUROC 0.938 for k=50

**Reason**: Post-hoc RFE has ~0.5-1% optimistic bias (panel was optimized on that data). Fixed panel with new split seed is the unbiased ground truth.

**Action**: Report the fixed-panel number for regulatory/publication purposes.

### Q: Can I skip Stage 2 validation in two-stage workflow?

**Depends on use case**:

| Use case | Stage 2 required? | Rationale |
|----------|------------------|-----------|
| FDA/regulatory submission | ✅ YES | Unbiased AUROC mandatory |
| Clinical deployment | ✅ YES | Stakeholder sign-off |
| Literature comparison | ✅ YES | Fair benchmark |
| Internal development | ❌ NO | Stage 1 sufficient |
| Rapid prototyping | ❌ NO | Iterate with Stage 1 only |

**Rule of thumb**: If the AUROC number will be reported externally or used for deployment decisions, do Stage 2.

### Q: What if RFECV and post-hoc RFE recommend different panel sizes?

**Expected**. Example:
- RFECV: k=50 (AUROC 0.945, within tolerance of max)
- Post-hoc RFE recommendation: k=25 (AUROC 0.930, knee point)

**Reason**: RFECV maximizes AUROC. Post-hoc RFE finds diminishing returns (knee point).

**Which to use?**
- If cost matters: k=25 (post-hoc RFE knee point)
- If only AUROC matters: k=50 (RFECV selection)

**Pro tip**: Run both, present trade-offs to stakeholders, let them decide.

## Common Pitfalls

### Pitfall 1: Using same split seed for fixed panel
**Wrong**:
```bash
ced train --model LR_EN --split-seed 0  # Discovery
ced train --fixed-panel panel.csv --split-seed 0  # LEAKAGE!
```

**Correct**:
```bash
ced train --model LR_EN --split-seed 0  # Discovery
ced train --fixed-panel panel.csv --split-seed 10  # NEW SEED
```

### Pitfall 2: Expecting post-hoc RFE and fixed panel to match
- **Post-hoc RFE** (50 proteins): AUROC 0.940 (~0.5% optimistic)
- **Fixed panel** (50 proteins): AUROC 0.938 (unbiased ground truth)

This is expected behavior.

### Pitfall 3: Not enabling k-best pre-filter for nested RFECV
**Without pre-filter**: ~22 hours
**With pre-filter** (default): ~5 hours

```yaml
features:
  feature_selection_strategy: rfecv
  rfe_kbest_prefilter: true   # ENABLE THIS
  rfe_kbest_k: 100
```

---

## Technical Details

### Data Leakage Prevention

All strategies prevent leakage through strict fold isolation:
- **Hybrid Stability**: Feature selection runs independently per fold
- **Nested RFECV**: Internal CV within each outer fold
- **Post-hoc RFE**: Uses validation set (slight optimism ~0.5%, mitigated by CV)
- **Fixed Panel**: Standard nested CV (no selection, no leakage with new seed)

### Feature Importance Methods

**Coefficient-based** (LR, LinSVM):
```python
coef = model.coef_[0]
importance = np.abs(coef)
```

**Permutation-based** (RF, XGBoost):
```python
from sklearn.inspection import permutation_importance
result = permutation_importance(model, X_val, y_val, n_repeats=10, scoring='roc_auc')
importance = result.importances_mean
```

### Computational Complexity

- **Hybrid Stability**: O(N × P × F × R × I) where F=folds, R=repeats, I=inner_folds
- **Nested RFECV**: O(N × P × log(P) × CV_inner × F_outer) — 45× slower empirically
- **Post-hoc RFE**: O(N × k × CV_folds × log(k)) — very fast (~5 min, no training)
- **Fixed Panel**: O(N × k × F × R × I) where k=fixed panel size

---

## Related Documentation

**Architecture decisions**:
- [ADR-013](../adr/ADR-013-four-strategy-feature-selection.md): Four-strategy framework rationale (read this for "why four strategies")
- [ADR-004](../adr/ADR-004-hybrid-feature-selection.md): Hybrid feature selection implementation details
- [ADR-005](../adr/ADR-005-stability-panel.md): Stability-based panel building

**High-level overview**:
- [CLAUDE.md](../../../CLAUDE.md): Project overview and quick start
- [ARCHITECTURE.md](../ARCHITECTURE.md): Pipeline architecture and code pointers
- [CLI_REFERENCE.md](CLI_REFERENCE.md): Complete CLI command reference

**Key modules**:
- [src/ced_ml/features/screening.py](../../src/ced_ml/features/screening.py): Effect size screening
- [src/ced_ml/features/kbest.py](../../src/ced_ml/features/kbest.py): SelectKBest tuning
- [src/ced_ml/features/stability.py](../../src/ced_ml/features/stability.py): Stability analysis
- [src/ced_ml/features/nested_rfe.py](../../src/ced_ml/features/nested_rfe.py): Nested RFECV implementation
- [src/ced_ml/features/rfe.py](../../src/ced_ml/features/rfe.py): Post-hoc RFE implementation
- [src/ced_ml/cli/train.py](../../src/ced_ml/cli/train.py): Training CLI (fixed panel)
- [src/ced_ml/cli/optimize_panel.py](../../src/ced_ml/cli/optimize_panel.py): Post-hoc RFE CLI

---

**Document Status**: Production-ready
**Maintainer**: Andres Chousal
**Last Reviewed**: 2026-01-26
