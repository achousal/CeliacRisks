# CLI Reference - CeD-ML Pipeline

Complete command-line interface reference for the Celiac Disease risk prediction pipeline.

## Table of Contents
- [Basic Usage](#basic-usage)
- [Config Overrides](#config-overrides)
- [Common Workflows](#common-workflows)
- [Optuna Integration](#optuna-integration)
- [HPC Usage](#hpc-usage)

---

## Basic Usage

### Training a Model

```bash
cd analysis
python -m ced_ml.cli.main train --config configs/training_config.yaml
```

### Required Config Parameters

The `training_config.yaml` must specify:
- `infile`: Path to input data (CSV or Parquet)
- `split_dir`: Path to split indices directory
- `scenario`: One of `IncidentOnly`, `PrevalentOnly`, `IncidentPlusPrevalent`
- `model`: One of `LR_EN`, `RF`, `XGBoost`, `LinSVM_cal`
- `outdir`: Output directory for results

### Environment

Always run from the `analysis/` directory:
```bash
cd analysis
python -m ced_ml.cli.main <command> [options]
```

---

## Config Overrides

### Syntax

Override any config parameter without editing YAML files:

```bash
--override path.to.param=value
```

The path follows the YAML nesting structure. Examples:

```yaml
# YAML structure
cv:
  folds: 5
  scoring: roc_auc
optuna:
  enabled: false
  n_trials: 100
```

```bash
# CLI overrides
--override cv.folds=10
--override cv.scoring=average_precision
--override optuna.enabled=true
--override optuna.n_trials=200
```

### Common Override Examples

#### Quick Development Run
```bash
python -m ced_ml.cli.main train \
  --config configs/training_config.yaml \
  --override cv.folds=2 \
  --override cv.repeats=3 \
  --override cv.n_iter=10 \
  --override cv.inner_folds=2
```

#### Change Scoring Metric
```bash
python -m ced_ml.cli.main train \
  --config configs/training_config.yaml \
  --override cv.scoring=average_precision
```

#### Adjust Feature Selection
```bash
python -m ced_ml.cli.main train \
  --config configs/training_config.yaml \
  --override features.feature_select=kbest \
  --override features.kbest_max=500 \
  --override features.screen_top_n=800
```

#### Enable Bootstrap CI
```bash
python -m ced_ml.cli.main train \
  --config configs/training_config.yaml \
  --override evaluation.test_ci_bootstrap=true \
  --override evaluation.n_boot=1000
```

#### Change Threshold Selection
```bash
python -m ced_ml.cli.main train \
  --config configs/training_config.yaml \
  --override thresholds.objective=youden \
  --override thresholds.threshold_source=val
```

---

## Common Workflows

### 1. Fast Local Test (< 5 minutes)

Minimal CV for quick iteration:

```bash
python -m ced_ml.cli.main train \
  --config configs/training_config.yaml \
  --override cv.folds=2 \
  --override cv.repeats=2 \
  --override cv.n_iter=5 \
  --override cv.inner_folds=2 \
  --override features.feature_select=none \
  --override output.save_plots=false
```

### 2. Single-Split Quality Run (Production Settings)

Full nested CV with all features:

```bash
python -m ced_ml.cli.main train \
  --config configs/training_config.yaml \
  --override cv.folds=5 \
  --override cv.repeats=10 \
  --override cv.n_iter=200 \
  --override cv.scoring=neg_brier_score \
  --override features.feature_select=hybrid
```

Expected runtime:
- LR_EN: ~30-60 minutes
- RF: ~2-4 hours (high memory)
- XGBoost: ~1-2 hours
- LinSVM_cal: ~20-40 minutes

### 3. Compare Multiple Models

Run same config across different models (sequential):

```bash
for model in LR_EN RF XGBoost LinSVM_cal; do
  python -m ced_ml.cli.main train \
    --config configs/training_config.yaml \
    --override model=$model \
    --override outdir=results_compare/$model
done
```

### 4. Hyperparameter Search Comparison

Compare RandomizedSearchCV vs Optuna:

```bash
# Baseline: RandomizedSearchCV
python -m ced_ml.cli.main train \
  --config configs/training_config.yaml \
  --override cv.n_iter=200 \
  --override optuna.enabled=false \
  --override outdir=results_random

# Optuna: TPE + Median Pruner
python -m ced_ml.cli.main train \
  --config configs/training_config.yaml \
  --override optuna.enabled=true \
  --override optuna.n_trials=200 \
  --override optuna.sampler=tpe \
  --override optuna.pruner=median \
  --override outdir=results_optuna
```

Compare outputs:
```bash
# Metrics
diff results_random/*/core/test_metrics.csv results_optuna/*/core/test_metrics.csv

# Optuna trials
head -20 results_optuna/*/cv/optuna/*_trials.csv
```

---

## Optuna Integration

### Enable Optuna

**Method 1: Edit config file**
```yaml
# training_config.yaml
optuna:
  enabled: true
  n_trials: 100
```

**Method 2: CLI override** (recommended for experimentation)
```bash
--override optuna.enabled=true
```

### Optuna Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `false` | Use Optuna (vs RandomizedSearchCV) |
| `n_trials` | `100` | Number of hyperparameter trials |
| `timeout` | `null` | Max seconds (null = unlimited) |
| `sampler` | `"tpe"` | Sampling algorithm |
| `sampler_seed` | `null` | RNG seed for reproducibility |
| `pruner` | `"median"` | Early stopping strategy |
| `pruner_n_startup_trials` | `5` | Trials before pruning starts |
| `pruner_percentile` | `25.0` | Percentile for percentile pruner |
| `n_jobs` | `1` | Parallel jobs for CV |
| `storage` | `null` | Database URL (e.g., `sqlite:///study.db`) |
| `study_name` | `null` | Study identifier |
| `load_if_exists` | `false` | Resume existing study |
| `save_study` | `true` | Save study object |
| `save_trials_csv` | `true` | Export trials as CSV |
| `direction` | `null` | `"maximize"` or `"minimize"` (auto-detected) |

### Optuna Examples

#### Fast Exploration (Development)
```bash
python -m ced_ml.cli.main train \
  --config configs/training_config.yaml \
  --override optuna.enabled=true \
  --override optuna.n_trials=50 \
  --override optuna.sampler=random \
  --override optuna.pruner=median \
  --override optuna.sampler_seed=42
```

#### Production Optimization
```bash
python -m ced_ml.cli.main train \
  --config configs/training_config.yaml \
  --override optuna.enabled=true \
  --override optuna.n_trials=200 \
  --override optuna.sampler=tpe \
  --override optuna.pruner=median \
  --override optuna.sampler_seed=42 \
  --override optuna.save_trials_csv=true
```

#### Maximum Efficiency (XGBoost/RF)
```bash
python -m ced_ml.cli.main train \
  --config configs/training_config.yaml \
  --override model=XGBoost \
  --override optuna.enabled=true \
  --override optuna.n_trials=100 \
  --override optuna.sampler=tpe \
  --override optuna.pruner=hyperband \
  --override optuna.timeout=3600  # 1 hour limit
```

#### Resumable Study (HPC)
```bash
python -m ced_ml.cli.main train \
  --config configs/training_config.yaml \
  --override optuna.enabled=true \
  --override optuna.storage=sqlite:///cv/optuna_study.db \
  --override optuna.study_name=xgboost_incident_split0 \
  --override optuna.load_if_exists=true
```

### Optuna Sampler Options

| Sampler | Best For | Notes |
|---------|----------|-------|
| `tpe` | General use (default) | Bayesian optimization, learns from history |
| `random` | Baseline comparison | Pure random search |
| `cmaes` | Continuous params only | Evolution strategy |
| `grid` | Small search spaces | Exhaustive grid search |

### Optuna Pruner Options

| Pruner | Speedup | Aggressiveness | Best For |
|--------|---------|----------------|----------|
| `median` | 2-3x | Moderate | Default choice |
| `percentile` | 3-5x | High | Fast exploration |
| `hyperband` | 3-5x | Adaptive | Large-scale search (n_trials >= 40) |
| `none` | 1x | None | Full evaluation (like RandomizedSearchCV) |

### Optuna Outputs

When Optuna is enabled, additional files are saved:

```
results/
└── seed_0/
    └── IncidentPlusPrevalent/
        └── cv/
            ├── best_params_per_split.csv      # Standard output
            ├── selected_proteins_per_split.csv
            └── optuna/
                ├── optuna_config.json         # Optuna settings
                ├── best_params_optuna.csv     # Best params with trial metadata
                └── <model>_<scenario>_trials.csv  # Full trial history
```

**Trials CSV columns:**
- `trial_number`: Trial ID
- `param_*`: Hyperparameter values
- `value`: Objective score
- `state`: `COMPLETE`, `PRUNED`, or `FAIL`
- `datetime_start`, `datetime_complete`: Timing

---

## HPC Usage

### Job Submission Scripts

#### Local (single model, single split)
```bash
bash run_local.sh
```

See: `configs/pipeline_local.yaml`

#### HPC (job array, multiple seeds)
```bash
bash run_hpc.sh
```

See: `configs/pipeline_hpc.yaml`

### HPC Configuration

The HPC script reads from `pipeline_hpc.yaml`:

```yaml
# configs/pipeline_hpc.yaml
seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 10 repeated splits
models: [LR_EN, RF, XGBoost, LinSVM_cal]

training_config_base: configs/training_config.yaml

# Override for HPC
overrides:
  cv:
    folds: 5
    repeats: 10
    n_iter: 200
  optuna:
    enabled: true
    n_trials: 150
```

### Monitoring Jobs

```bash
# Check status
bjobs -w | grep CeD

# Check logs
tail -f logs/CeD_LR_EN_seed0.log

# Check completed models
ls results_seed*/*/core/test_metrics.csv

# Count completions
ls results_seed*/*/core/test_metrics.csv | wc -l
```

### Aggregating Results

After all jobs complete:

```bash
python -m ced_ml.cli.aggregate_splits \
  --results-root results_hpc \
  --n-splits 10 \
  --models LR_EN,RF,XGBoost,LinSVM_cal \
  --outdir results_aggregated
```

---

## Advanced Usage

### Multiple Config Files

Use different configs for different experiments:

```bash
# Development config
python -m ced_ml.cli.main train --config configs/training_dev.yaml

# Production config
python -m ced_ml.cli.main train --config configs/training_prod.yaml

# Optuna config
python -m ced_ml.cli.main train --config configs/training_optuna.yaml
```

### Environment Variables

Set environment variables for consistent runs:

```bash
export CED_ML_DATA=/path/to/data.csv
export CED_ML_SPLITS=/path/to/splits
export CED_ML_RESULTS=/path/to/results

python -m ced_ml.cli.main train \
  --config configs/training_config.yaml \
  --override infile=$CED_ML_DATA \
  --override split_dir=$CED_ML_SPLITS \
  --override outdir=$CED_ML_RESULTS
```

### Debugging

Enable verbose output:

```bash
python -m ced_ml.cli.main train \
  --config configs/training_config.yaml \
  --override verbose=2  # 0=WARNING, 1=INFO, 2=DEBUG
```

Check logs:
```bash
tail -f results/*/run.log
```

---

## Troubleshooting

### Common Issues

**1. "Optuna not installed" warning**
```bash
pip install optuna
```

**2. Memory errors (RF)**
- Reduce `rf.n_estimators_grid`
- Request high-memory node on HPC

**3. Slow hyperparameter search**
- Enable Optuna: `--override optuna.enabled=true`
- Reduce trials: `--override optuna.n_trials=50`
- Use pruning: `--override optuna.pruner=median`

**4. Split files not found**
```bash
# Generate splits first
python -m ced_ml.cli.main save-splits --config configs/splits_config.yaml
```

**5. Config validation errors**
- Check YAML syntax
- Verify parameter ranges (see schema.py)
- Use `--override` to test values

---

## Reference

### Key Files
- `configs/training_config.yaml` - Main training configuration
- `configs/splits_config.yaml` - Split generation configuration
- `configs/pipeline_hpc.yaml` - HPC job array configuration
- `run_local.sh` - Local execution script
- `run_hpc.sh` - HPC submission script

### Documentation
- `docs/reference/KNOBS_CHEATSHEET.txt` - Quick reference
- `docs/reference/PARAMETERS_REFERENCE.txt` - Full parameter list
- `docs/adr/` - Architecture decision records
- `docs/ARCHITECTURE.md` - Pipeline architecture

### Getting Help
```bash
python -m ced_ml.cli.main --help
python -m ced_ml.cli.main train --help
```

---

**Last Updated:** 2026-01-20
**Pipeline Version:** ced_ml v2.0 (Pydantic config + Optuna support)
