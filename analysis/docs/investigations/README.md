# Factorial Experiments

**See**: [FACTORIAL_EXPERIMENTS.md](FACTORIAL_EXPERIMENTS.md) for complete documentation.

## Quick Start

```bash
# Quick test (30 minutes)
bash run_factorial_experiment.sh --quick

# Overnight run (6-8 hours)
bash run_factorial_experiment.sh --overnight

# Full experiment (24-30 hours)
bash run_factorial_experiment.sh --full
```

## Key Files

| File | Purpose |
|------|---------|
| **FACTORIAL_EXPERIMENTS.md** | Complete documentation |
| **run_factorial_experiment.sh** | Consolidated runner script |
| **top100_panel.csv** | Fixed 100-protein panel |
| **training_config_frozen.yaml** | Frozen hyperparameter config |
| **generate_fixed_panel.py** | Panel generation utility |
| **investigate_factorial.py** | Statistical analysis script |
