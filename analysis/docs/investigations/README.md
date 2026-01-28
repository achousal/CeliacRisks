# Feature Selection Factorial Experiment

**Objective**: Determine optimal feature selection strategy by systematically testing:
- Prevalent case sampling (0.5 vs 1.0)
- Case:control ratios (1:1 vs 1:5)

**Status**: Ready to run
**Last Updated**: 2026-01-27

---

## Quick Start

### Run Full Experiment (5 seeds, ~5 hours)

```bash
cd analysis/docs/investigations
bash run_experiment_v2.sh --skip-splits --split-seeds 0,1,2,3,4
```

### Quick Test (1 seed, ~1 hour)

```bash
bash run_experiment_v2.sh --skip-splits --split-seeds 0
```

---

## What It Does

1. **Generates fixed 100-protein panel** (Mann-Whitney → k-best, eliminates FS variability)
2. **Trains models** with frozen config (OOF-posthoc calibration, 100 Optuna trials)
3. **Tests 4 configurations**:
   - `0.5_1`: 50% prevalent, 1:1 ratio
   - `0.5_5`: 50% prevalent, 1:5 ratio (production default)
   - `1.0_1`: 100% prevalent, 1:1 ratio
   - `1.0_5`: 100% prevalent, 1:5 ratio
4. **Statistical analysis**: Paired t-tests, Bonferroni correction, Cohen's d, power analysis

---

## Key Files

| File | Purpose |
|------|---------|
| `QUICKSTART.md` | Quick reference guide |
| `READY_TO_RUN.md` | Pre-flight checklist |
| `CORRECTED_FACTORIAL_PLAN.md` | Detailed experiment design |
| `run_experiment_v2.sh` | Main execution script |
| `investigate_factorial.py` | Statistical analysis code |
| `generate_fixed_panel.py` | Panel generation utility |
| `training_config_frozen.yaml` | Frozen training config |
| `splits_config_experiment_*.yaml` | Split configs (4 files) |

---

## Expected Output

```
results/investigations/experiment_{TIMESTAMP}/
├── metrics_all.csv          # All runs (40 for full, 8 for quick test)
├── comparison_table.csv     # 8 config-model combinations with 95% CIs
├── statistical_tests.csv    # Paired comparisons (p-values, effect sizes)
├── power_analysis.csv       # Post-hoc power calculations
└── summary.md              # Human-readable findings
```

---

## Design Highlights

- **Fixed 100-protein panel**: Eliminates feature selection variability
- **Frozen config**: Same hyperparameters/calibration across all runs
- **5 random seeds**: Robust variance estimation
- **Bonferroni α = 0.00625**: Controls family-wise error (0.05/8 tests)
- **OOF-posthoc calibration**: Unbiased calibration strategy

---

## Prerequisites

Splits must exist in `../../../splits_experiments/`:
- `0.5_1/`
- `0.5_5/`
- `1.0_1/`
- `1.0_5/`

If missing, remove `--skip-splits` flag from commands above.

---

## Documentation

- **QUICKSTART.md**: Quick reference for common tasks
- **READY_TO_RUN.md**: Pre-flight checklist and verification
- **CORRECTED_FACTORIAL_PLAN.md**: Full experiment design and rationale

---

## Archived Files

Previous prevalent vs incident investigation files moved to `_archive/`:
- Original investigation scripts and documentation
- Legacy calibration analysis

---

**Need help?** See [QUICKSTART.md](QUICKSTART.md) for quick reference or [CORRECTED_FACTORIAL_PLAN.md](CORRECTED_FACTORIAL_PLAN.md) for full details.
