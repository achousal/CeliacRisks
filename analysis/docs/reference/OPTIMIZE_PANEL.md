# Panel Size Optimization via RFE

## Overview

The `ced optimize-panel` command finds the **minimum viable protein panel** for cost-effective clinical screening through Recursive Feature Elimination (RFE).

**Use case**: After training ML models, you want to reduce the number of proteins tested per patient—from 2,920 features to a practical 5–50 protein panel—while maintaining acceptable discrimination (AUROC).

**Output**: A Pareto curve showing the size-performance trade-off, with recommendations for practical panel sizes.

---

## Quick Start

```bash
# After training a model (e.g., ced train --model LR_EN --split-seed 0)

cd analysis/

ced optimize-panel \
  --model-path results/LR_EN/split_seed0/core/LR_EN__final_model.joblib \
  --infile ../data/Celiac_dataset_proteomics_w_demo.parquet \
  --split-dir ../splits/ \
  --split-seed 0
```

Output directory: `results/LR_EN/split_seed0/optimize_panel/`

---

## Command Syntax

```bash
ced optimize-panel \
  --model-path PATH                 # Trained model (.joblib) [REQUIRED]
  --infile PATH                     # Input data (Parquet/CSV) [REQUIRED]
  --split-dir PATH                  # Split indices directory [REQUIRED]
  --split-seed INT                  # Split seed (default: 0)
  --start-size INT                  # Starting panel size (default: 100)
  --min-size INT                    # Minimum to evaluate (default: 5)
  --min-auroc-frac FLOAT            # Early stop threshold (default: 0.90)
  --cv-folds INT                    # CV folds for AUROC (default: 5)
  --step-strategy {adaptive|linear|geometric}  # Elimination strategy (default: adaptive)
  --outdir PATH                     # Output directory (optional)
  --use-stability-panel/--use-all-features  # Start from ranking (default: True)
```

---

## Parameters Explained

### Required
| Parameter | Description |
|-----------|-------------|
| `--model-path` | Path to trained model bundle (saved from `ced train`) |
| `--infile` | Original proteomics data file |
| `--split-dir` | Directory containing split indices (from `ced save-splits`) |

### Optional
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--split-seed` | 0 | Which split seed to use |
| `--start-size` | 100 | Start RFE from top-N proteins (from stability ranking) |
| `--min-size` | 5 | Stop when panel reaches this size |
| `--min-auroc-frac` | 0.90 | Early stop if AUROC < 90% of max |
| `--cv-folds` | 5 | CV folds for unbiased AUROC estimate |
| `--step-strategy` | adaptive | Geometric (fast), linear (thorough), or adaptive |
| `--use-stability-panel` | True | Start from stability ranking vs all features |

---

## Examples

### Example 1: Rapid screening (adaptive strategy)
```bash
ced optimize-panel \
  --model-path results/LR_EN/split_seed0/core/LR_EN__final_model.joblib \
  --infile ../data/input.parquet \
  --split-dir ../splits/ \
  --start-size 100 \
  --min-size 5 \
  --step-strategy adaptive
```
**Expected time**: 3–5 minutes for LR_EN, 15–25 for RF
**Evaluations**: ~10–15 panel sizes

### Example 2: Thorough exploration (linear strategy)
```bash
ced optimize-panel \
  --model-path results/RF/split_seed0/core/RF__final_model.joblib \
  --infile ../data/input.parquet \
  --split-dir ../splits/ \
  --start-size 50 \
  --min-size 3 \
  --step-strategy linear \
  --cv-folds 10
```
**Expected time**: 60–120 minutes for RF
**Evaluations**: 48 panel sizes
**Use**: When cost difference between 25 and 30 proteins matters

### Example 3: Aggressive minimization
```bash
ced optimize-panel \
  --model-path results/LR_EN/split_seed0/core/LR_EN__final_model.joblib \
  --infile ../data/input.parquet \
  --split-dir ../splits/ \
  --start-size 50 \
  --min-size 3 \
  --min-auroc-frac 0.85 \
  --step-strategy geometric
```
**Goal**: Accept 15% AUROC loss for maximum panel reduction
**Use**: Ultra-low-cost screening

---

## Output Files

All outputs saved to `{results}/{MODEL}/split_seed{N}/optimize_panel/`

### 1. `panel_curve.csv` - Evaluation points
```csv
size,auroc_cv,auroc_cv_std,auroc_val,proteins
100,0.892,0.015,0.887,"[""PROT1"",""PROT2"",...]"
50,0.889,0.018,0.884,"[""PROT1"",""PROT3"",...]"
25,0.879,0.022,0.871,"[""PROT1"",""PROT5"",...]"
12,0.852,0.028,0.845,"[""PROT1"",""PROT8"",...]"
```

**Columns:**
- `size`: Number of proteins in panel
- `auroc_cv`: AUROC from 5-fold CV (OOF estimate)
- `auroc_cv_std`: Standard deviation across CV folds
- `auroc_val`: AUROC on validation set (used for elimination decisions)
- `proteins`: JSON list of selected protein names

### 2. `panel_curve.png` - Pareto frontier plot
Shows AUROC vs panel size with:
- Blue line: validation AUROC (primary curve)
- Gray dashed line: CV AUROC with error bars
- Threshold lines: 95%, 90%, 85% of max AUROC
- Knee point: marked with purple star
- Recommended sizes: marked with colored diamonds

**How to read:**
- Steep drops = proteins with high importance
- Flat regions = proteins with low importance (safe to remove)
- Knee point = diminishing returns threshold

### 3. `recommended_panels.json` - Actionable recommendations
```json
{
  "model": "LR_EN",
  "split_seed": 0,
  "max_auroc": 0.887,
  "recommended_panels": {
    "min_size_95pct": 25,
    "min_size_90pct": 12,
    "min_size_85pct": 6,
    "knee_point": 18
  },
  "pareto_points": [
    {"size": 100, "auroc_val": 0.887},
    {"size": 50, "auroc_val": 0.884},
    {"size": 25, "auroc_val": 0.871},
    ...
  ],
  "timestamp": "2026-01-26T14:30:00"
}
```

**Interpretation:**
- `min_size_95pct: 25` → Use 25 proteins to maintain 95% of max AUROC (0.842)
- `min_size_90pct: 12` → Use 12 proteins to maintain 90% of max AUROC (0.798)
- `knee_point: 18` → Diminishing returns beyond 18 proteins

### 4. `feature_ranking.csv` - Elimination order
```csv
protein,elimination_order
PROT99,0
PROT87,1
PROT43,2
PROT56,3
...
PROT1,97
PROT2,98
```

**Interpretation:**
- Order 0 = removed first (least important)
- Order 98 = removed last (most important)
- Use to prioritize protein assay development

### 5. `pareto_frontier.csv` - Non-dominated solutions
```csv
size,auroc_val
100,0.887
50,0.884
25,0.871
```

All points satisfy: no other point has both smaller size AND higher AUROC.

### 6. `feature_ranking.png` - Importance visualization
Horizontal bar chart, proteins sorted by elimination order (latest removals at top).

---

## Interpretation Guide

### Scenario 1: Steep decline → Early plateau
```
AUROC
  |     *
  |     |\
  |     | \
  |     |  \___
  |     |      \___
  |_____|__________|_____ panel size
        25       50
```

**Interpretation**: Keep 25+ proteins; diminishing returns beyond
**Action**: Use recommended min_size_90pct (likely 25-30)

### Scenario 2: Gradual decline (no clear knee)
```
AUROC
  |     *
  |     |\
  |     | \
  |     |  \
  |     |   \
  |_____|_____|_____ panel size
        25   50
```

**Interpretation**: Proteins have similar importance; no clear cutoff
**Action**: Choose based on cost-benefit (e.g., 20 proteins if feasible)

### Scenario 3: Collapse at small sizes
```
AUROC
  |     *
  |     |\
  |     | \
  |     |  \
  |     |   |__
  |_____|______|\_ panel size
        10    5
```

**Interpretation**: Few proteins carry most signal; risky for clinical use
**Action**: Use larger panel (e.g., 15+) for robustness

---

## Practical Decision-Making

### Cost vs Performance Trade-off

Given recommendations for LR_EN:
```json
{
  "max_auroc": 0.887,
  "min_size_95pct": 25,
  "min_size_90pct": 12,
  "knee_point": 18
}
```

**Decision matrix:**
| Panel Size | AUROC | Cost | Recommendation |
|-----------|-------|------|-----------------|
| 12 | 0.798 | Very low | High-risk screening |
| 18 | ~0.865 | Low | Good balance |
| 25 | 0.842 | Medium | Safe option |
| 50+ | 0.880+ | High | Reference standard |

**Suggested approach:**
1. **Gold standard**: Use full 100-protein panel in research/development
2. **Clinical deployment**: Use 18–25 proteins (knee point ± margin)
3. **Rapid screening**: Use 12 proteins with follow-up testing

---

## Technical Details

### Algorithm: Recursive Feature Elimination
1. Start with top-N proteins from stability ranking (or all proteins)
2. Train model with 5-fold CV, compute feature importances
3. Remove least important protein
4. Evaluate AUROC on validation set
5. Repeat until min_size reached or AUROC drops > threshold

### Feature Importance
- **Linear models (LR_EN, LinSVM_cal)**: Absolute coefficient values
- **Tree models (RF, XGBoost)**: Permutation importance

### Stopping Criteria
Stop if:
- Panel size reaches `--min-size` (default: 5), OR
- AUROC < `--min-auroc-frac` × max_auroc (default: 0.90 × 0.887 = 0.798)

### Efficiency
- **Adaptive (default)**: ~10–15 evaluations via geometric progression (100→50→25→12→6→5)
- **Linear**: 95 evaluations (every size from 100 to 5)
- **Geometric**: Same as adaptive

**Runtime estimates (split_seed=0, start_size=100):**
- LR_EN: 3–5 min
- RF: 15–25 min
- XGBoost: 20–35 min

---

## Troubleshooting

### Error: "Model not found"
```
FileNotFoundError: Model not found: results/LR_EN/split_seed0/core/LR_EN__final_model.joblib
```
**Fix**: Verify model path and ensure training completed successfully
```bash
ls results/LR_EN/split_seed0/core/
```

### Error: "Split file not found"
```
FileNotFoundError: Split file not found. Tried: [...]
```
**Fix**: Run `ced save-splits` first, or check split_seed matches training
```bash
ls splits/split_*.pkl
```

### Error: "Only N valid proteins, less than min_size"
```
ValueError: Only 8 valid proteins, less than min_size=5
```
**Fix**: Reduce `--start-size` or `--min-size`
```bash
--start-size 20 --min-size 3
```

### Slow execution
**Symptom**: Command running for > 30 min on LR_EN
**Fixes**:
- Use `--step-strategy adaptive` (default, faster)
- Increase `--cv-folds` is slower; try `--cv-folds 3`
- Reduce `--start-size` to 50 instead of 100

### AUROC collapses below threshold
**Symptom**: Stops at panel size 50 with AUROC 0.75
**Interpretation**: Last removed protein was crucial
**Fix**: Check `feature_ranking.csv` at elimination order ~45; likely misclassified

---

## Integration with Workflows

### After single model training
```bash
# Train single model
ced train --model LR_EN --split-seed 0

# Optimize panel
ced optimize-panel \
  --model-path results/LR_EN/split_seed0/core/LR_EN__final_model.joblib \
  --infile ../data/input.parquet \
  --split-dir ../splits/ \
  --split-seed 0
```

### Across multiple split seeds
```bash
for seed in 0 1 2 3 4; do
  ced optimize-panel \
    --model-path results/LR_EN/split_seed${seed}/core/LR_EN__final_model.joblib \
    --infile ../data/input.parquet \
    --split-dir ../splits/ \
    --split-seed $seed
done

# Results in: results/LR_EN/split_seed{0,1,2,3,4}/optimize_panel/
```

### Across multiple models
```bash
for model in LR_EN RF XGBoost; do
  ced optimize-panel \
    --model-path results/${model}/split_seed0/core/${model}__final_model.joblib \
    --infile ../data/input.parquet \
    --split-dir ../splits/ \
    --split-seed 0
done
```

---

## Example: Real-world case study

### Setup
- **Model**: LR_EN, trained on split_seed=0
- **Goal**: Reduce from 2920 proteins to practical panel

### Run
```bash
ced optimize-panel \
  --model-path results/LR_EN/split_seed0/core/LR_EN__final_model.joblib \
  --infile ../data/Celiac_dataset_proteomics_w_demo.parquet \
  --split-dir ../splits/ \
  --split-seed 0 \
  --start-size 100 \
  --min-size 5
```

### Results interpretation
**recommended_panels.json:**
```json
{
  "max_auroc": 0.892,
  "min_size_95pct": 28,
  "min_size_90pct": 14,
  "knee_point": 22
}
```

**Decision:**
- **Research**: Use full 100 proteins (AUROC 0.892)
- **Clinical trial**: Use 22 proteins at knee point (AUROC ~0.880, 2% loss)
- **Rapid screening**: Use 14 proteins (AUROC 0.803, 10% loss)

**Cost impact** (hypothetical):
- 100 proteins: $500/patient
- 22 proteins: $110/patient (78% reduction)
- 14 proteins: $70/patient (86% reduction)

**Final recommendation**: Deploy 22-protein panel for clinical use—good balance of cost and performance.

---

## References

- Recursive Feature Elimination: Guyon et al. (2002)
- Knee-point detection: Satopaa et al. (2011)
- Panel size trade-offs: CeD-ML ADRs (see `docs/adr/`)

---

**Last Updated**: 2026-01-26
**Version**: 1.0
