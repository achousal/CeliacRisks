# Aggregate Splits Enhancement Plan

## Goal
Mirror the same output structure as single-model runs, producing summary artifacts and plots that describe combined model performance across multiple split seeds.

## Current State
- `aggregate_splits.py` collects and summarizes metrics (mean/std/CI)
- Outputs: `all_test_metrics.csv`, `test_metrics_summary.csv`, `aggregation_metadata.json`

## Key Insight
The existing plotting functions (`plot_roc_curve`, `plot_pr_curve`, `plot_calibration_curve`) already support `split_ids` parameter for multi-split visualization with CI bands.

---

## Output Structure for `aggregated/`

```
aggregated/
  core/
    test_metrics_summary.csv      # Mean/std/CI of test metrics
    val_metrics_summary.csv       # Mean/std/CI of val metrics
    pooled_test_metrics.csv       # Metrics computed on pooled predictions
    pooled_val_metrics.csv        # Metrics computed on pooled predictions
  cv/
    cv_metrics_summary.csv        # Mean/std/CI of CV repeat metrics
    all_cv_repeat_metrics.csv     # Concatenated CV metrics
  preds/
    test_preds/
      pooled_test_preds.csv       # All test preds with split_seed column
    val_preds/
      pooled_val_preds.csv        # All val preds with split_seed column
    train_oof/
      pooled_train_oof.csv        # All OOF preds with split_seed column
    plots/
      aggregated_test_risk_dist.{fmt}
      aggregated_val_risk_dist.{fmt}
  reports/
    feature_reports/
      feature_stability_summary.csv    # Selection freq across all splits
    stable_panel/
      consensus_stable_panel.csv       # Proteins stable in >= threshold splits
    panels/
      consensus_panel_N{size}.json     # Consensus panels
  diagnostics/
    plots/
      aggregated_test_roc.{fmt}        # ROC with CI bands
      aggregated_val_roc.{fmt}
      aggregated_test_pr.{fmt}         # PR with CI bands
      aggregated_val_pr.{fmt}
      aggregated_test_calibration.{fmt}
      aggregated_val_calibration.{fmt}
    calibration/
      aggregated_calibration.csv       # Combined calibration data
    dca/
      aggregated_dca.csv               # Combined DCA data
      aggregated_dca_plot.{fmt}
  all_test_metrics.csv                 # Raw concatenated (existing)
  all_val_metrics.csv                  # Raw concatenated (existing)
  aggregation_metadata.json            # Metadata (existing, enhanced)
```

---

## Implementation Tasks

### 1. Collect Pooled Predictions
- Read `preds/test_preds/*.csv` from each split
- Add `split_seed` column
- Concatenate into `pooled_test_preds.csv`
- Same for val and train_oof predictions

### 2. Compute Pooled Metrics
- Compute AUROC, PR-AUC, Brier on pooled predictions
- Compute metrics per-split then average (current approach)
- Report both in summary

### 3. Generate Aggregated Plots
Use existing plot functions with pooled data and `split_ids`:

```python
plot_roc_curve(
    y_true=pooled_y_true,
    y_pred=pooled_y_pred,
    split_ids=pooled_split_ids,  # This triggers CI band computation
    out_path=...,
    title="Aggregated Test Set ROC",
    meta_lines=[f"n={n_splits} splits", ...]
)
```

- ROC with mean curve + 95% CI bands
- PR with mean curve + 95% CI bands
- Calibration with mean curve + 95% CI bands
- Risk distribution (overlay or faceted)

### 4. Feature Stability Analysis
- Read `cv/selected_proteins_per_split.csv` from each split
- Compute cross-split selection frequency:
  - protein X selected in K out of N splits
  - protein X average selection frequency within CV
- Identify consensus features (selected in >= 75% of splits)

### 5. Panel Consensus
- Read panel manifests from each split
- For each panel size N:
  - Identify proteins appearing in >= threshold splits
  - Create consensus panel manifest

### 6. DCA Aggregation
- Read DCA CSV from each split
- Compute mean/CI at each threshold
- Generate combined DCA plot

### 7. Enhanced Metadata
```json
{
  "timestamp": "...",
  "n_splits": 5,
  "split_seeds": [0, 1, 2, 3, 4],
  "pooled_n_test": 500,
  "pooled_n_val": 500,
  "metrics_summary": {
    "AUROC_mean": 0.85,
    "AUROC_std": 0.02,
    "AUROC_ci95": [0.81, 0.89]
  },
  "feature_consensus": {
    "n_stable_75pct": 45,
    "top_10_features": ["..."]
  }
}
```

---

## Function Signatures (New)

```python
def collect_predictions(
    split_dirs: List[Path],
    pred_type: str,  # "test", "val", "train_oof"
) -> pd.DataFrame:
    """Collect predictions from all splits, add split_seed column."""

def compute_pooled_metrics(
    pooled_df: pd.DataFrame,
) -> Dict[str, float]:
    """Compute metrics on pooled predictions."""

def aggregate_feature_stability(
    split_dirs: List[Path],
) -> pd.DataFrame:
    """Aggregate feature selection across splits."""

def build_consensus_panels(
    split_dirs: List[Path],
    panel_sizes: List[int],
    threshold: float = 0.75,
) -> Dict[int, List[str]]:
    """Build consensus panels from per-split panels."""

def generate_aggregated_plots(
    pooled_preds: pd.DataFrame,
    out_dir: Path,
    config: AggregationConfig,
) -> None:
    """Generate all aggregated diagnostic plots."""
```

---

## Questions to Resolve

1. **Pooled vs averaged metrics**: Report both? Pooled gives single estimate; averaged gives variance.
pooled, to get SD+- and 95% CI in the plots

2. **Threshold selection**: Use Youden from which split? Or from pooled data?
from pooled

3. **Feature stability threshold**: Default 75%? Configurable?
configurable

4. **DCA aggregation**: Mean net benefit at each threshold?
sure, whats a other uption
---

## Unresolved Questions
- None that block implementation; defaults are sensible.
what does this mean
