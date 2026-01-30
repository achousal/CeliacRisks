# Test Suite Audit and Improvement Plan

**Audited**: 2026-01-30
**Suite**: 1,350 test items across `analysis/tests/`
**Passing**: 1,317 (97.6%) | **Failing**: 13 (1.0%) | **Skipped**: 16 (1.2%)
**Coverage**: 13% overall

---

## Status: 80 -> 13 Failing Tests (2026-01-30)

**Root cause (FIXED)**: `utils/logging.py` `auto_log_path()` assumed a
`CeliacRisks` parent directory exists, which broke in pytest temp directories.

**Fix applied**: added fallback to `outdir.parent / "logs"` when project root
is not found. Also fixed `verbose` -> `log_level` API mismatch in
`test_config_tools.py` and `test_models_stacking.py`.

### Remaining 13 failures (pre-existing, unrelated to audit)

- `test_e2e_output_structure.py` (9): consensus panel output structure
  assertions do not match current CLI output paths.
- `test_e2e_run_id_workflows.py` (2): ensemble/consensus error message
  assertions do not match current CLI wording.
- `test_e2e_runner.py` (1): temporal splits generation test has incorrect
  seed expectation (expects seed0, CLI uses seed_start from config).
- `test_metrics_thresholds.py` (1): `pytest.warns` does not capture
  `logging.warning` (test uses wrong capture mechanism).
- `test_models_optuna.py` (1): same `pytest.warns` vs `logging` issue.
- `test_models_optuna_multiobjective.py` (1): same issue.

---

## Coverage Gaps

### 0% Coverage (Core Production Workflows)

| Module | Lines | Purpose |
|--------|-------|---------|
| `cli/aggregate_splits.py` | 340 | Aggregation workflow |
| `cli/consensus_panel.py` | 203 | Cross-model consensus |
| `cli/optimize_panel.py` | 375 | Panel optimization |
| `cli/train_ensemble.py` | 385 | Ensemble training |
| `cli/run_pipeline.py` | 272 | Main orchestration |
| `features/rfe.py` | 287 | RFE feature selection |
| `features/consensus.py` | 236 | Consensus aggregation |
| `features/nested_rfe.py` | 141 | Nested RFE |
| `models/stacking.py` | 326 | Ensemble stacking |
| `models/optuna_search.py` | 320 | Hyperparameter optimization |
| `hpc/lsf.py` | 147 | HPC job submission |

### Coverage by Layer

| Layer | Coverage | Critical Gaps |
|-------|----------|---------------|
| CLI | 0-69% | All workflow orchestration |
| Data | 32-95% | Splits (35%), persistence (32%) |
| Features | 0-50% | RFE (0%), consensus (0%), nested_rfe (0%) |
| Models | 6-26% | Optuna (0%), stacking (0%), training (7%) |
| Metrics | 8-30% | DCA (8%), thresholds (15%), discrimination (14%) |
| Plotting | 0-22% | Most modules 0-8% |
| Utils | 10-40% | Logging (10%), paths (29%) |

---

## Quality Issues

### Mega-tests

`test_e2e_runner.py::test_full_pipeline_single_model` (114 lines) validates split
generation, training, and output structure in one test. If split generation fails,
all downstream assertions are unreachable.

**Action**: split into focused single-behavior tests.

### File-existence-only plot tests

~~Plots only checked file existence and size > 0.~~

**DONE** (2026-01-30): `test_basic_roc_plot` and `test_basic_pr_plot` now
validate image dimensions via `matplotlib.image.imread`. Further content
validation (axes labels, legend entries) remains a future improvement.

### Unconditionally skipped tests

~~`test_e2e_pipeline.py` contained an unconditional `pytest.skip()` placeholder.~~

**DONE** (2026-01-30): deleted `TestSlowEndToEndPipeline` class and duplicate
`toy_proteomics_csv` fixture. File consolidated into `test_e2e_runner.py`
(see P4 below).

### Vague assertions

~~`test_data_splits.py:109` accepted two different behaviors with `or`.~~

**DONE** (2026-01-30): replaced with exact assertions matching `make_strata`
behavior: `assert strata[0] == "None_Male"` and `assert strata[1] == "Incident_None"`.

### Implementation-coupled comments

~~`test_training.py:150` referenced `training.py:189`.~~

**DONE** (2026-01-30): docstring now describes behavior ("probability clipping
prevents NaN propagation") without referencing source line numbers.

### Duplicate coverage

~~`test_e2e_runner.py` and `test_e2e_pipeline.py` tested the same workflow.~~

**DONE** (2026-01-30): deleted `test_e2e_pipeline.py`. Its sole real test
(config roundtrip) is already covered by `TestE2EConfigValidation` in
`test_e2e_runner.py`.

---

## Good Practices (Keep)

- **AAA structure**: clean Arrange-Act-Assert in `test_data_splits.py`,
  `test_features_stability.py`, `test_models_calibration.py`.
- **Deterministic RNG**: `conftest.py` provides `rng` fixtures via
  `np.random.default_rng(42)` -- no global state pollution.
- **Descriptive naming**: `test_basic_frequency_computation`,
  `test_empty_log`, `test_malformed_json`, `test_duplicate_proteins_in_split`.
- **Edge case coverage**: NaN handling and boundary values in calibration tests.

---

## Prioritized Action Plan

### P0 -- Unblocks 80 tests -- DONE

1. ~~Fix path detection in `utils/logging.py`.~~
   **DONE**: added fallback for missing `CeliacRisks` parent directory.
   Also fixed `verbose` -> `log_level` API mismatch in two test files.

### P1 -- Cover core workflows

2. Add smoke tests for CLI commands with 0% coverage:
   `aggregate-splits`, `optimize-panel`, `consensus-panel`, `train-ensemble`.
3. Create pre-generated fixtures (small trained model artifacts) so downstream
   tests skip retraining.

### P2 -- Improve existing tests -- PARTIALLY DONE

4. ~~Delete or fix unconditionally skipped tests in `test_e2e_pipeline.py`.~~
   **DONE**: deleted placeholder class and file.
5. Split mega-tests in `test_e2e_runner.py` into single-behavior tests.
6. ~~Strengthen plot tests with content validation.~~
   **DONE**: added image dimension validation. Axes/label checks remain future work.

### P3 -- Harden correctness -- PARTIALLY DONE

7. Add property-based tests for:
   - Stratified splits always preserve class ratios.
   - Calibration always produces valid probabilities in [0, 1].
   - Bootstrap CIs always contain the point estimate.
8. ~~Replace vague assertions with exact expected values.~~
   **DONE**: `test_data_splits.py` now asserts exact strata strings.
9. ~~Remove implementation-detail references from test docstrings.~~
   **DONE**: `test_training.py` docstring no longer references source lines.

### P4 -- Cleanup -- PARTIALLY DONE

10. ~~Consolidate `test_e2e_runner.py` and `test_e2e_pipeline.py`.~~
    **DONE**: deleted `test_e2e_pipeline.py`.
11. Add mock-based coverage for HPC job submission.
