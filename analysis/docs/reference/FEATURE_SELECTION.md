# Feature Selection Reference

**Status**: Production
**Last Updated**: 2026-01-28
**Applies to**: CeD-ML v1.3.0+

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
Already have a panel to validate? → Fixed Panel (Strategy 5)
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
├─ SINGLE-MODEL DEPLOYMENT PANEL SIZING (cost vs. performance)
│  └─ Strategy 3: Aggregated RFE
│     Command: ced optimize-panel --run-id <RUN_ID> --model LR_EN
│     Runtime: ~10 minutes (run after aggregation)
│
├─ CROSS-MODEL CONSENSUS PANEL (robust deployment)
│  └─ Strategy 4: Consensus Panel
│     Command: ced consensus-panel --run-id <RUN_ID>
│     Runtime: ~15 minutes (run after aggregation)
│
└─ UNBIASED VALIDATION (regulatory, literature comparison)
   └─ Strategy 5: Fixed Panel
      Command: ced train --fixed-panel panel.csv --split-seed 10
      Runtime: ~30 minutes
```

### Strategy Comparison

| Attribute | Hybrid Stability | Nested RFECV | Aggregated RFE | Consensus Panel | Fixed Panel |
|-----------|------------------|--------------|--------------|-----------------|-------------|
| **When** | During training | During training | After aggregation | After aggregation | During training |
| **Speed** | Fast (~30 min) | Slow (~22 hrs) | Fast (~10 min) | Fast (~15 min) | Fast (~30 min) |
| **Use for** | Production | Discovery | Single-model sizing | Cross-model sizing | Validation |
| **Panel size** | Tuned (k_grid) | Automatic (max AUROC) | User-optimized | RRA consensus | Fixed (input) |
| **Model scope** | Single | Single | Single | Multiple | Single |
| **Cost consideration** | Partial (k_grid) | ❌ None (always max) | ✅ Full visibility | ✅ Consensus trade-off | N/A |
| **Trade-off curve** | No | No | ✅ Yes (Pareto) | Via per-model RFE | No |
| **Leakage** | None | None | None (consensus) | None (consensus) | None |
| **Output** | Stable k-panels | Consensus panels | AUROC vs k curve | Cross-model panel | Unbiased AUROC |
| **Best for** | Fast baseline | Feature stability | Clinical deployment | Robust deployment | Regulatory filing |

**Design rationale**: See [ADR-013](../adr/ADR-013-four-strategy-feature-selection.md) for why we need five strategies.

---

## Table of Contents

1. [Quick Reference](#quick-reference) (above)
2. [Strategy 1: Hybrid Stability](#strategy-1-hybrid-stability-default)
3. [Strategy 2: Nested RFECV](#strategy-2-nested-rfecv)
4. [Strategy 3: Post-hoc RFE](#strategy-3-post-hoc-rfe)
5. [Strategy 4: Consensus Panel](#strategy-4-consensus-panel-cross-model-deployment)
6. [Strategy 5: Fixed Panel](#strategy-5-fixed-panel-validation)
7. [Common Workflows](#common-workflows)
8. [Ensemble Feature Selection Workflows](#ensemble-feature-selection-workflows)
9. [Troubleshooting](#troubleshooting)

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
# Run AFTER aggregation (auto-detection, RECOMMENDED)
ced optimize-panel \
  --run-id 20260127_115115 \
  --model LR_EN \
  --stability-threshold 0.75 \
  --min-size 5 \
  --min-auroc-frac 0.90

# Alternative: Explicit paths (legacy)
ced optimize-panel \
  --results-dir results/LR_EN/run_20260127_115115 \
  --infile ../data/input.parquet \
  --split-dir ../splits/
```

### How It Works

1. **Load consensus stable proteins**: Extract proteins stable across all splits (≥75% selection frequency)
2. **Pool data**: Combine train/val data from all splits for maximum robustness
3. **Recursive elimination**: Iteratively remove least important features
   - Compute feature importance (coefficients or permutation)
   - Remove bottom ~10% per iteration (adaptive)
   - Re-evaluate AUROC using cross-validation on pooled data
   - Stop at min_size or AUROC drop threshold
4. **Generate recommendations**: Find knee points and Pareto frontier

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--results-dir` | Required | Path to aggregated results directory |
| `--infile` | Required | Input data file (Parquet/CSV) |
| `--split-dir` | Required | Directory with split indices |
| `--stability-threshold` | 0.75 | Minimum selection fraction for stable proteins |
| `--min-size` | 5 | Minimum panel size to evaluate |
| `--min-auroc-frac` | 0.90 | Early stop if AUROC < frac × max_auroc |
| `--cv-folds` | 5 | CV folds for OOF AUROC estimation |
| `--step-strategy` | adaptive | Elimination strategy (adaptive/geometric/fine/linear) |

**Step Strategy Comparison:**

| Strategy | Evaluation Points | Speed | Use Case |
|----------|-------------------|-------|----------|
| **geometric** (default) | ~6 points | Fastest | Quick exploration (100→50→25→12→6→5) |
| **fine** | ~10 points (1.67x more) | Fast | More granular data (100→75→50→37→25→18→12→9→6→5) |
| **linear** | Every size | Slowest | Maximum detail (100→99→98...→6→5) |

**Recommendation**: Use `--step-strategy fine` for publication-quality Pareto curves with better resolution at intermediate panel sizes.

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

## Strategy 4: Consensus Panel (Cross-Model Deployment)

**Pipeline**: Aggregate multiple models → RRA ranking → correlation clustering → final panel

### When to Use
- Clinical deployment requiring robust feature selection across multiple algorithms
- Reduce model-specific bias in feature selection
- Generate deployment panel from ensemble of base models (LR_EN, RF, XGBoost, LinSVM)
- Before fixed-panel validation training

### Quick Start

```bash
# Step 1: Train base models across splits
ced train --model LR_EN --split-seed 0
ced train --model RF --split-seed 0
ced train --model XGBoost --split-seed 0
ced train --model LinSVM_cal --split-seed 0

# Step 2: Aggregate results for each model
ced aggregate-splits --config configs/aggregate_config.yaml

# Step 3: (Optional) Run panel optimization per model
ced optimize-panel --run-id 20260127_115115

# Step 4: Generate cross-model consensus panel
ced consensus-panel --run-id 20260127_115115

# Step 5: Validate consensus panel (use NEW split seed)
ced train --model LR_EN \
  --fixed-panel results/consensus_panel/run_20260127_115115/final_panel.txt \
  --split-seed 10
```

### How It Works

**Robust Rank Aggregation (RRA):**
1. **Per-model composite ranking**: Combines stability frequency (0-1) + RFE importance (elimination order)
   - If RFE available: weighted average of stability rank + RFE rank
   - If RFE missing: uses stability rank only
2. **Cross-model aggregation**: Geometric mean of reciprocal ranks (Stuart et al. 2003)
   - Proteins ranked highly by multiple models receive lower (better) consensus scores
3. **Correlation clustering**: Spearman correlation matrix → hierarchical clustering → keep one representative per cluster
4. **Final panel**: Top N proteins by consensus score after redundancy removal

**Parameters:**
- `--target-size 25` - Desired panel size (default: 25)
- `--stability-threshold 0.75` - Minimum selection frequency per model (default: 0.75)
- `--corr-threshold 0.85` - Correlation threshold for redundancy removal (default: 0.85)
- `--rfe-weight 0.5` - Weight for RFE vs stability (0=stability only, 1=RFE only)
- `--rra-method geometric_mean` - Aggregation method (default: geometric_mean)

### Outputs

```
results/consensus_panel/run_<RUN_ID>/
├── final_panel.txt                 # One protein per line (for --fixed-panel)
├── final_panel.csv                 # Panel with consensus scores
├── consensus_ranking.csv           # All proteins with RRA scores
├── per_model_rankings.csv          # Per-model composite rankings
├── correlation_clusters.csv        # Cluster assignments
└── consensus_metadata.json         # Run parameters and statistics
```

### Advantages / Limitations

**Pros**: Model-agnostic robustness, reduces single-model bias, incorporates both stability and importance, automatic redundancy removal
**Cons**: Requires multiple trained models, 15 min runtime, more complex interpretation than single-model selection

---

## Strategy 5: Fixed Panel Validation

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

# Step 1: Train across all splits with hybrid_stability
ced train --model LR_EN --all-splits  # ~30 min per split

# Step 2: Aggregate results (auto-detection)
ced aggregate-splits --run-id 20260127_115115 --model LR_EN  # ~5 min

# Step 3: Run aggregated RFE to explore panel size trade-offs (auto-detection)
ced optimize-panel \
  --run-id 20260127_115115 \
  --model LR_EN \
  --stability-threshold 0.75 \
  --min-size 5  # ~10 min

# Step 4: Review trade-off curve
cat results/LR_EN/run_20260127_115115/aggregated/optimize_panel/panel_curve_aggregated.csv
cat results/LR_EN/run_20260127_115115/aggregated/optimize_panel/recommended_panels_aggregated.json

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

# Step 5: Stakeholder decision
# Clinical team: "We can afford 25 proteins max"
# You: "AUROC drops 0.950 → 0.930 (2%), acceptable?"
# Clinical team: "Yes, deploy k=25"

# ═══════════════════════════════════════════════════════════════
# STAGE 2: VALIDATION (Unbiased estimate on NEW data)
# ═══════════════════════════════════════════════════════════════

# Step 6: Extract chosen panel (e.g., top 25 proteins)
head -n 26 results/LR_EN/run_XXXXXX/aggregated/optimize_panel/feature_ranking_aggregated.csv \
  | awk -F',' 'NR==1 || NR<=26 {print $1}' > deployment_panel_k25.csv

# Step 7: CRITICAL - Validate on NEW split seed (prevents data leakage)
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

## Ensemble Feature Selection Workflows

**Context**: See [ADR-009: OOF Stacking Ensemble](../adr/ADR-009-oof-stacking-ensemble.md) for ensemble architecture.

The pipeline provides two specialized workflows for ensemble-based feature selection:

### Workflow 6: Two-Pass Ensemble Panel Optimization

**Goal**: Find minimal viable panel size for ensemble model via consensus-based RFE.

**When to use**:
- Clinical deployment of ensemble model
- Need to see AUROC vs panel size trade-offs
- Cost/assay constraints matter

**Why two-pass**: The ensemble meta-learner operates on base model predictions (4 features), not raw proteins (2,920 features). To optimize protein panels for deployment, we use a **two-pass approach**:
1. **Pass 1 (Discovery)**: Run RFE on best single model to identify minimal panel sizes
2. **Pass 2 (Ensemble Retraining)**: Retrain base models + ensemble on optimized panel

**Expected benefit**: Ensemble typically achieves +2-5% AUROC over best single model. Two-pass RFE identifies smallest panel maintaining this gain.

```bash
# PASS 1: Discover optimal panel sizes via single-model RFE
# ============================================================

# Step 1a: Train base models with feature selection
ced train --model LR_EN --split-seed 0
ced train --model RF --split-seed 0
ced train --model XGBoost --split-seed 0

# Step 1b: Aggregate to get consensus panel (auto-detection)
ced aggregate-splits --run-id 20260127_115115 --model LR_EN

# Step 1c: Run aggregated RFE on best single model (using consensus panel)
# Uses the model with highest AUROC (typically LR_EN or XGBoost)
ced optimize-panel \
  --run-id 20260127_115115 \
  --model LR_EN \
  --stability-threshold 0.75 \
  --min-size 10

# Output: Pareto curve showing AUROC vs panel size
# Expected runtime: ~10 minutes


# PASS 2: Retrain ensemble on optimized panel
# ============================================

# Step 2a: View RFE results to choose panel size
cat results/LR_EN/run_XXXXXX/aggregated/optimize_panel/panel_curve_aggregated.csv

# Example output:
# size,auroc_cv,auroc_val,prauc_val
# 150,0.945,0.943,0.80  (baseline: consensus panel)
# 75,0.942,0.940,0.79   (99.7% of max)
# 50,0.937,0.935,0.77   (99.2% of max)  ← Knee point
# 25,0.925,0.923,0.74   (97.9% of max)
# 10,0.902,0.900,0.68   (95.4% of max)

# Step 2b: Stakeholder decision
# "Panel size 50 maintains 99% of consensus AUROC but saves 100 proteins"

# Step 2c: Extract chosen panel
head -51 results/LR_EN/run_*/split_seed0/optimize_panel/feature_ranking.csv | \
  tail -50 > optimized_panel_k50.csv

# Step 2d: Retrain base models on optimized panel
ced train --model LR_EN --fixed-panel optimized_panel_k50.csv --split-seed 0
ced train --model RF --fixed-panel optimized_panel_k50.csv --split-seed 0
ced train --model XGBoost --fixed-panel optimized_panel_k50.csv --split-seed 0

# Step 2e: Train ensemble on retrained base models
ced train-ensemble --base-models LR_EN,RF,XGBoost --split-seed 0

# Expected ensemble AUROC: ~0.940-0.945 (ensemble boost on optimized panel)
# Expected runtime: ~90 minutes total (3 models × 30 min)
```

**Outputs**:
```
# Pass 1 (RFE Discovery):
results/LR_EN/run_*/split_seed0/optimize_panel/
  ├── panel_curve.csv              # AUROC vs panel size (single model)
  ├── panel_curve.png              # Pareto visualization
  ├── recommended_panels.json      # Knee points (95%/90%/85% of max)
  ├── feature_ranking.csv          # Protein elimination order
  └── metrics_summary.csv          # Full metrics at each panel size

# Pass 2 (Ensemble on optimized panel):
results/ENSEMBLE/run_*/split_seed0/core/
  ├── ENSEMBLE__final_model.joblib # Retrained ensemble
  └── metrics.json                 # Ensemble AUROC on k=50 panel
```

**Cost-benefit calculation**:
```python
# Example: Assay cost $50/protein
Consensus panel (k=150): $7,500/sample, Single AUROC=0.943, Ensemble=0.950
Optimized panel (k=50):  $2,500/sample, Single AUROC=0.935, Ensemble=0.942
  → 67% cost reduction
  → Single model: -0.8% AUROC loss
  → Ensemble: -0.8% AUROC loss (maintains +0.7% boost over single model)

Aggressive panel (k=25): $1,250/sample, Single AUROC=0.923, Ensemble=0.930
  → 83% cost reduction
  → Single model: -2.0% AUROC loss
  → Ensemble: -2.0% AUROC loss (maintains +0.7% boost over single model)
```

**Key insight**: The ensemble boost (+2-5% AUROC) is **preserved** when using optimized panels, because the meta-learner combines diverse base model predictions regardless of panel size.

### Workflow 7: Consensus Panel Aggregation

**Goal**: Identify algorithm-invariant biomarkers selected by multiple base models.

**When to use**:
- Scientific discovery: "Which proteins are robustly selected across algorithms?"
- Ultra-robust panels: Intersection of multiple selection methods
- Feature stability analysis across model families (linear vs tree-based)

**Expected benefit**: Proteins selected by diverse models (LR, RF, XGBoost) are likely biologically robust, not algorithm artifacts.

**IMPORTANT**: This workflow is **automatically executed** by the aggregation pipeline. You do NOT need to run a separate command.

```bash
# Step 1: Train base models with feature selection
ced train --model LR_EN --split-seed 0
ced train --model RF --split-seed 0
ced train --model XGBoost --split-seed 0

# Step 2: Consensus aggregation happens automatically via:
# LOCAL: ced aggregate-splits --config configs/aggregate_config.yaml
# HPC:   bash scripts/post_training_pipeline.sh --run-id <RUN_ID>

# No separate consensus-panel command needed!
```

**Consensus methods**:

| Method | Description | Use Case | Panel Size |
|--------|-------------|----------|------------|
| **intersection** | Features in ALL models | Ultra-conservative, high confidence | Smallest |
| **union** | Features in ANY model | Comprehensive, exploratory | Largest |
| **frequency** | Features in ≥67% of models (default) | Balanced, robust | Medium |
| **weighted** | Weighted by model AUROC | Performance-driven | Medium |

**Outputs** (from `ced aggregate-splits`):
```
results/{MODEL}/aggregated/
  ├── feature_stability.csv            # Per-protein selection frequencies across splits
  ├── feature_importances.csv          # Mean feature importances
  └── aggregation_metadata.json        # Metadata and statistics
```

**To extract consensus panel from aggregation output**:
```bash
# Extract proteins selected in ≥70% of splits (frequency-based consensus)
awk -F',' 'NR==1 || $2 >= 0.70 {print $1}' \
  results/LR_EN/aggregated/feature_stability.csv \
  > consensus_panel_70pct.csv
```

**Example**: Frequency method (≥2 of 3 models)
```bash
# Input (from model training):
# LR_EN:   [P1, P2, P3, P5, P7]        (5 proteins)
# RF:      [P2, P3, P4, P6, P7, P8]    (6 proteins)
# XGBoost: [P2, P5, P7, P9]            (4 proteins)

# Consensus output (≥0.67 = ≥2 models):
# P2: 3/3 models = 100% → KEEP
# P3: 2/3 models = 67%  → KEEP
# P7: 3/3 models = 100% → KEEP
# P5: 2/3 models = 67%  → KEEP

# Consensus panel: [P2, P3, P5, P7] (4 proteins)
```

**Validation workflow** (recommended):
```bash
# Step 1: Extract consensus panel from aggregation output
awk -F',' 'NR==1 || $2 >= 0.80 {print $1}' \
  results/LR_EN/aggregated/feature_stability.csv \
  > consensus_panel_intersection.csv

# Step 2: Validate on NEW split (unbiased)
ced train --model LR_EN --fixed-panel consensus_panel_intersection.csv --split-seed 10
ced train --model RF --fixed-panel consensus_panel_intersection.csv --split-seed 10
ced train --model XGBoost --fixed-panel consensus_panel_intersection.csv --split-seed 10

# Step 3: Compare per-model AUROC
# If all models perform well with consensus panel → robust biomarkers
```

### Workflow 8: Deprecated - See Workflow 6

**Note**: This workflow has been deprecated. Use **Workflow 6 (Two-Pass Ensemble Panel Optimization)** instead.

The two-pass approach is necessary because the ensemble meta-learner operates on base model predictions (4 features), not raw proteins (2,920 features). Direct RFE on the ensemble model is not supported.

### Workflow Comparison: Two-Pass Ensemble RFE vs Aggregation Pipeline

| Attribute | Two-Pass Ensemble RFE (Workflow 6) | Aggregation Pipeline (Automatic) |
|-----------|-------------------------------------|----------------------------------|
| **When** | After base model training | After base model training |
| **Tool** | `ced optimize-panel` (Pass 1) + `ced train-ensemble` (Pass 2) | `ced aggregate-splits` (auto via HPC) |
| **Input** | Best single model (Pass 1) → Retrained base models (Pass 2) | All base models across splits |
| **Goal** | Optimize panel size (cost vs. AUROC) for ensemble deployment | Compute consensus features (stability) |
| **Output** | Pareto curve + ensemble on optimized panel | feature_stability.csv |
| **AUROC** | Highest (ensemble on optimized panel) | Per-model (LR_EN, RF, etc.) |
| **Runtime** | ~5 min (Pass 1) + ~90 min (Pass 2) | <1 minute (part of aggregation) |
| **Use for** | Clinical deployment with cost constraints | Scientific discovery, robustness |

**Decision guide**:
- **Two-Pass Ensemble RFE** (Workflow 6): For clinical deployment with cost constraints (stakeholder-driven panel sizing, preserves ensemble boost)
- **Aggregation Pipeline** (Automatic): For scientific discovery and feature stability analysis (runs automatically)

### Panel Optimization Best Practices

1. **Always train ensemble AFTER base models**:
   ```bash
   # Correct order (auto-detection)
   ced train --model LR_EN --split-seed 0  # Base models first
   ced train --model RF --split-seed 0
   ced train --model XGBoost --split-seed 0
   ced train-ensemble --run-id 20260127_115115 --split-seed 0  # Ensemble second (auto-detects base models)
   ced aggregate-splits --run-id 20260127_115115 --model ENSEMBLE
   ced optimize-panel --run-id 20260127_115115 --model ENSEMBLE  # Optimize third
   ```

2. **Use geometric step strategy** (default for RFE):
   - Fast: ~5 min vs ~45 min for linear
   - Sufficient granularity for clinical decisions
   - Fine strategy for high-stakes deployments (more Pareto points)

3. **Validate with new split seed** (regulatory/publication):
   ```bash
   # Discovery: Split seed 0 (auto-detection)
   ced optimize-panel --run-id 20260127_115115 --model ENSEMBLE

   # Extract chosen panel (e.g., k=25)
   head -26 results/ENSEMBLE/run_20260127_115115/aggregated/optimize_panel/feature_ranking_aggregated.csv | \
     tail -25 > deployment_panel_k25.csv

   # Validation: Split seed 10 (NEW data, no peeking)
   ced train --model ENSEMBLE --fixed-panel deployment_panel_k25.csv --split-seed 10
   ```

4. **Document cost-performance trade-offs**:
   - Save `panel_curve.csv` with stakeholder decisions
   - Include Pareto plots in regulatory submissions
   - Record rationale: "Chose k=25: 98% AUROC retained, 75% cost reduction"

5. **Consensus panel thresholds** (from aggregation output):
   ```bash
   # Extract high-stability proteins (≥80% of splits)
   awk -F',' 'NR==1 || $2 >= 0.80 {print $1}' \
     results/LR_EN/aggregated/feature_stability.csv \
     > consensus_panel_intersection.csv  # Ultra-conservative

   # Extract moderate-stability proteins (≥67% of splits)
   awk -F',' 'NR==1 || $2 >= 0.67 {print $1}' \
     results/LR_EN/aggregated/feature_stability.csv \
     > consensus_panel_frequency.csv  # Balanced (default)
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
- [src/ced_ml/features/consensus.py](../../src/ced_ml/features/consensus.py): Consensus panel aggregation (NEW)
- [src/ced_ml/cli/train.py](../../src/ced_ml/cli/train.py): Training CLI (fixed panel)
- [src/ced_ml/cli/optimize_panel.py](../../src/ced_ml/cli/optimize_panel.py): Post-hoc RFE and consensus workflows CLI

---

**Document Status**: Production-ready
**Maintainer**: Andres Chousal
**Last Reviewed**: 2026-01-27 (Added ensemble workflows)
