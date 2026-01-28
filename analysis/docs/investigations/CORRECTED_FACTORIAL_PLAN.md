# Corrected Factorial Investigation Plan

**Date**: 2026-01-27
**Status**: Ready to implement
**Objective**: Re-run factorial design with proper config isolation to answer: Is the incident vs prevalent score difference an artifact or biology?

---

## Problem Summary

Investigation 1 produced duplicate results because `investigate.py` uses hardcoded `project_root / "splits"` instead of config-specific directories.

**What worked**: Split generation and training correctly used `--split-dir` per config
**What failed**: Analysis script read from wrong splits directory

**Evidence**: All 4 configs showed identical sample sizes (73 incident, 76 prevalent) despite different configurations.

---

## Root Cause

```python
# investigate.py line 77 - THE BUG
self.splits_dir = project_root / "splits"  # HARDCODED - ignores config-specific directories
```

The splits ARE correctly generated in `splits_experiments/{pf}_{ccr}/`:
- `0.5_1/`: 298 train samples (1:1 ratio)
- `0.5_5/`: 894 train samples (1:5 ratio)
- `1.0_1/`: 298 train samples (1:1 ratio)
- `1.0_5/`: 894 train samples (1:5 ratio)

But analysis reads from `splits/` (default), not the config-specific directories.

---

## Factorial Design

### 2x2 Design Matrix

| Config | prevalent_frac | case:control | Expected n_train | Expected train_prev |
|--------|----------------|--------------|------------------|---------------------|
| 0.5_1  | 0.5 (50%)      | 1:1          | 298              | 50%                 |
| 0.5_5  | 0.5 (50%)      | 1:5          | 894              | 17%                 |
| 1.0_1  | 1.0 (100%)     | 1:1          | 298              | 50%                 |
| 1.0_5  | 1.0 (100%)     | 1:5          | 894              | 17%                 |

### Key Parameters

- `prevalent_train_frac`: Controls how many prevalent cases included in training (0.5 = half, 1.0 = all)
- `train_control_per_case`: Controls case:control ratio (1 = balanced, 5 = imbalanced)

### Experimental Controls (Fixed Across All Configs)

To isolate sampling effects, the following parameters are **frozen**:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Feature selection** | Fixed 100 top features | No k-best tuning; eliminates FS variability |
| **Hyperparameters** | Optuna per config | Adaptive HP allows each config to find optimal model |
| **Calibration** | OOF-posthoc isotonic | Matches production pipeline (ADR-014) |
| **CV structure** | 5-fold outer × 3 repeats × 3-fold inner | Standard nested CV |
| **Optuna trials** | 100 trials per config | Sufficient for convergence |
| **Val/test sets** | Same across configs | Only training set varies |
| **Random seeds** | 5 independent splits | Robust variance estimation |

**Feature selection detail:**
- Screen to top 1000 (Mann-Whitney)
- Select top 100 by univariate AUC
- Skip k-best tuning and stability selection
- Config: `feature_selection_strategy: none` + `fixed_panel: top100_panel.csv`

### Interpretation Framework

| Phase 1 Result | Phase 2 Result | Conclusion |
|----------------|----------------|------------|
| Score diff 25% | Score diff 5%  | **Artifact** (80% reduction) |
| Score diff 25% | Score diff 22% | **Biology** (persistent) |
| Score diff 25% | Score diff 12% | **Mixed** (partial reduction) |

---

## Implementation Steps

### Step 1: Create New Analysis Script

**File**: `analysis/docs/investigations/investigate_factorial.py`

**Why new script**: Reading from run outputs (not splits) avoids the hardcoded path bug.

**Key functions**:
```python
def discover_runs(results_root: Path) -> List[Path]
    """Find all run_* directories"""

def identify_config(run_dir: Path) -> str
    """Map run to config via n_train/prevalence in metadata"""
    # 298 samples + 50% prev -> 1:1 ratio
    # 894 samples + 17% prev -> 1:5 ratio

def extract_metrics(run_dir: Path) -> Dict
    """Read AUROC, PR-AUC, calibration from run outputs"""

def generate_comparison(metrics: List[Dict]) -> pd.DataFrame
    """Aggregate by config with mean/std across seeds"""

def compute_effect_size(group1: np.ndarray, group2: np.ndarray) -> float
    """Cohen's d for AUROC differences"""
    pooled_std = np.sqrt((group1.std()**2 + group2.std()**2) / 2)
    return (group1.mean() - group2.mean()) / pooled_std

def paired_comparison(metrics_df: pd.DataFrame) -> pd.DataFrame
    """Paired t-tests between configs with Bonferroni correction"""
    # Compare: 0.5_1 vs 0.5_5 (case:control effect at 50% prevalent)
    # Compare: 1.0_1 vs 1.0_5 (case:control effect at 100% prevalent)
    # Compare: 0.5_1 vs 1.0_1 (prevalent sampling effect at 1:1 ratio)
    # Compare: 0.5_5 vs 1.0_5 (prevalent sampling effect at 1:5 ratio)
    # 4 comparisons × 2 models = 8 tests -> alpha_corrected = 0.05/8 = 0.00625

def power_analysis(n_seeds: int, effect_size: float, alpha: float) -> float
    """Post-hoc power for detecting AUROC differences"""
    # Use statsmodels.stats.power for paired t-test power
```

**Metrics to extract**:
- From `core/test_metrics.csv`: AUROC, PR_AUC, sens_ctrl_95
- From `core/val_metrics.csv`: validation metrics
- From `config_metadata.json`: n_train, n_val, n_test, train_prevalence

**Statistical testing**:
- **Primary outcome**: Test-set AUROC (paired t-test across seeds)
- **Secondary outcomes**: PR-AUC, Brier score, sensitivity@95%spec
- **Correction**: Bonferroni (8 tests per outcome, alpha = 0.05/8 = 0.00625)
- **Effect size**: Cohen's d (small: 0.2, medium: 0.5, large: 0.8)
- **Power**: Post-hoc calculation for observed effects

---

### Step 2: Modify run_experiment.sh

**Changes**:

1. Add multi-seed support (line ~50):
```bash
SPLIT_SEEDS=(0 1 2 3 4)  # 5 seeds for robust variance estimation
```

2. Add experiment tracking:
```bash
EXPERIMENT_ID=$(date +%Y%m%d_%H%M%S)
```

3. Generate fixed panel (before training loop):
```bash
# Extract top 100 features by univariate AUC (after Mann-Whitney screening)
python -c "
from ced_ml.features.screening import screen_features
from ced_ml.features.kbest import select_k_best
# Load data, screen to 1000, select top 100, save to top100_panel.csv
" > top100_panel.csv
```

4. Loop over seeds in training phase with frozen config:
```bash
for SEED in "${SPLIT_SEEDS[@]}"; do
    ced train \
        --split-seed "$SEED" \
        --fixed-panel top100_panel.csv \
        --config configs/training_config_frozen.yaml \
        ...
done
```

**training_config_frozen.yaml overrides**:
```yaml
features:
  feature_selection_strategy: none  # Skip FS, use fixed panel

optuna:
  enabled: true
  n_trials: 100
  sampler: tpe
  pruner: median

calibration:
  enabled: true
  strategy: oof_posthoc
  method: isotonic
```

5. Replace Phase 3 with new analysis:
```bash
python investigate_factorial.py \
    --results-dir ../../../results \
    --output-dir "$RESULTS_DIR/experiment_${EXPERIMENT_ID}"
```

---

### Step 3: Run Corrected Experiment

```bash
cd analysis/docs/investigations

# Option A: Full experiment (40 runs, ~4-5 hours)
bash run_experiment.sh \
    --skip-splits \
    --models LR_EN,RF \
    --split-seeds 0,1,2,3,4

# Option B: Quick test (8 runs, ~45 min)
bash run_experiment.sh \
    --skip-splits \
    --models LR_EN,RF \
    --split-seeds 0
```

**Expected runs**: 4 configs × 5 seeds × 2 models = 40 training runs (full experiment)

---

### Step 4: Analyze Results

```bash
python investigate_factorial.py --compare-configs
```

**Output files**:
```
results/investigations/experiment_{ID}/
  metrics_all.csv              # Raw metrics per run (40 rows)
  comparison_table.csv         # Config x Model summary (8 rows)
  statistical_tests.csv        # Paired comparisons with p-values and effect sizes
  power_analysis.csv           # Post-hoc power for observed effects
  summary.md                   # Human-readable findings with statistical interpretation
```

---

## Expected Output Format

### comparison_table.csv

| Config | Model | n_runs | AUROC_mean | AUROC_std | AUROC_95CI | PR_AUC_mean | Sens95_mean | Brier_mean |
|--------|-------|--------|------------|-----------|------------|-------------|-------------|------------|
| 0.5_1  | LR_EN | 5      | ?          | ?         | [?, ?]     | ?           | ?           | ?          |
| 0.5_1  | RF    | 5      | ?          | ?         | [?, ?]     | ?           | ?           | ?          |
| 0.5_5  | LR_EN | 5      | ?          | ?         | [?, ?]     | ?           | ?           | ?          |
| 0.5_5  | RF    | 5      | ?          | ?         | [?, ?]     | ?           | ?           | ?          |
| 1.0_1  | LR_EN | 5      | ?          | ?         | [?, ?]     | ?           | ?           | ?          |
| 1.0_1  | RF    | 5      | ?          | ?         | [?, ?]     | ?           | ?           | ?          |
| 1.0_5  | LR_EN | 5      | ?          | ?         | [?, ?]     | ?           | ?           | ?          |
| 1.0_5  | RF    | 5      | ?          | ?         | [?, ?]     | ?           | ?           | ?          |

### statistical_tests.csv

| Comparison | Model | Metric | Mean_diff | Cohen_d | t_stat | p_value | p_adj | Significant |
|------------|-------|--------|-----------|---------|--------|---------|-------|-------------|
| 0.5_1 vs 0.5_5 | LR_EN | AUROC | ? | ? | ? | ? | ? | ? |
| 0.5_1 vs 0.5_5 | RF | AUROC | ? | ? | ? | ? | ? | ? |
| 1.0_1 vs 1.0_5 | LR_EN | AUROC | ? | ? | ? | ? | ? | ? |
| 1.0_1 vs 1.0_5 | RF | AUROC | ? | ? | ? | ? | ? | ? |
| 0.5_1 vs 1.0_1 | LR_EN | AUROC | ? | ? | ? | ? | ? | ? |
| 0.5_1 vs 1.0_1 | RF | AUROC | ? | ? | ? | ? | ? | ? |
| 0.5_5 vs 1.0_5 | LR_EN | AUROC | ? | ? | ? | ? | ? | ? |
| 0.5_5 vs 1.0_5 | RF | AUROC | ? | ? | ? | ? | ? | ? |

Note: p_adj uses Bonferroni correction (alpha = 0.05/8 = 0.00625 per test)

---

## Verification Checklist

Before running:
- [ ] Config-specific splits exist in `splits_experiments/{config}/`
- [ ] Sample sizes differ: 298 (1:1) vs 894 (1:5)
- [ ] `top100_panel.csv` generated with 100 proteins
- [ ] `training_config_frozen.yaml` created with frozen settings

After running:
- [ ] 40 run directories created (4 configs × 5 seeds × 2 models)
- [ ] `config_metadata.json` shows correct n_train per config
- [ ] Metrics differ between configs (not duplicates)
- [ ] Comparison table has 8 rows (4 configs × 2 models)
- [ ] Statistical tests show 8 comparisons (4 paired × 2 models)
- [ ] All p-values adjusted with Bonferroni correction
- [ ] Effect sizes (Cohen's d) calculated for all comparisons

---

## Decision Matrix

After collecting results, use this matrix:

### If AUROC improves significantly with 1:5 ratio (for LR_EN):
- **Artifact hypothesis supported**
- Action: Use 1:5 ratio for production
- LR_EN needs more controls for stable training

### If AUROC similar across ratios:
- **Biology hypothesis supported**
- Action: Use 1:5 for more training data (no downside)
- Document case-type differences

### If prevalent_frac (0.5 vs 1.0) affects results:
- **Prevalent sampling matters**
- Action: Decide based on clinical use case
- More prevalent = better generalization to diagnosed patients

### If RF consistently outperforms LR_EN:
- **Model selection insight**
- Action: Prefer RF for production
- LR_EN is more sensitive to class imbalance

---

## Timeline

| Task | Duration | Priority |
|------|----------|----------|
| Generate top100_panel.csv | 15 min | HIGH |
| Create training_config_frozen.yaml | 10 min | HIGH |
| Create investigate_factorial.py | 60 min | HIGH |
| Modify run_experiment.sh | 20 min | HIGH |
| Run experiment (40 runs) | 4-5 hours | HIGH |
| Statistical analysis | 30 min | HIGH |
| Document findings | 45 min | MEDIUM |
| **Total** | **~7 hours** | |

**Note**: Full experiment is 40 runs (4 configs × 5 seeds × 2 models). Quick test with 1 seed reduces to ~1 hour runtime.

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `investigate_factorial.py` | CREATE | New analysis script with statistical testing |
| `run_experiment.sh` | MODIFY | Add 5 seeds, frozen config, panel generation |
| `training_config_frozen.yaml` | CREATE | Frozen experimental controls |
| `top100_panel.csv` | GENERATE | Fixed 100-protein panel for all configs |
| `CORRECTED_FACTORIAL_PLAN.md` | MODIFY | This document (updated) |

---

## Success Criteria

1. **4 unique configurations** with different sample sizes confirmed
2. **Metrics vary by config** (not duplicates)
3. **Variance estimates** from 5 seeds per config (95% CIs)
4. **Statistical significance** assessed with Bonferroni correction
5. **Effect sizes** quantified (Cohen's d) for all comparisons
6. **Power analysis** confirms adequate sample size
7. **Clear conclusion** on artifact vs biology hypothesis
8. **Production recommendation** for optimal config (statistically justified)

---

## Questions Answered After This Investigation

1. Does case:control ratio (1:1 vs 1:5) affect model performance?
2. Does prevalent sampling (50% vs 100%) affect model performance?
3. Is the incident vs prevalent score difference an artifact or biology?
4. Which model (LR_EN vs RF) is more robust to configuration changes?
5. What is the optimal configuration for production deployment?

---

**Prepared by**: Claude
**Last Updated**: 2026-01-27 (v2 with statistical rigor)

---

## Changelog

**v2 (2026-01-27)**:
- Increased seeds from 3 to 5 for robust variance estimation
- Added experimental controls (fixed 100 features, frozen config)
- Added statistical testing framework (paired t-tests, Bonferroni, Cohen's d)
- Added power analysis for sample size justification
- Added 95% confidence intervals to all summary statistics
- Updated timeline to 7 hours (more realistic)

**v1 (2026-01-27)**:
- Initial plan identifying root cause and factorial design
