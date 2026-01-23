# E2E Test Guide

End-to-end tests for user-critical flows in the CeliacRisks ML pipeline.

## Overview

E2E tests validate complete workflows with realistic small fixtures:

1. **Full Pipeline**: splits → train → aggregate
2. **Ensemble Workflow**: base models → ensemble → aggregate
3. **HPC Workflow**: config validation → dry-run
4. **Temporal Validation**: temporal splits → train → evaluate

## Running Tests

### Quick run (fast tests only, ~30s)
```bash
cd analysis/
pytest tests/test_e2e_runner.py -v -m "not slow"
```

### Full run (all tests, ~5 min)
```bash
pytest tests/test_e2e_runner.py -v
```

### Specific test class
```bash
pytest tests/test_e2e_runner.py::TestE2EFullPipeline -v
```

### Single test
```bash
pytest tests/test_e2e_runner.py::TestE2EFullPipeline::test_splits_generation_basic -v
```

### Run locally and in CI
```bash
# Local (all tests)
pytest tests/test_e2e_runner.py -v

# CI (fast only)
pytest tests/test_e2e_runner.py -v -m "not slow" --cov=ced_ml
```

## Test Organization

### Fast Tests (< 5s each)
- `test_splits_generation_basic`: Split file creation and structure
- `test_reproducibility_same_seed`: Deterministic split verification
- `test_output_file_structure`: Directory structure validation
- `test_hpc_config_validation`: Config loading
- `test_temporal_splits_generation`: Temporal ordering (if implemented)
- All error handling tests

### Slow Tests (30s - 3 min each)
Marked with `@pytest.mark.slow`:
- `test_full_pipeline_single_model`: Complete pipeline with one model
- `test_ensemble_training_workflow`: Base models + ensemble training

## Test Structure

### Fixtures

#### `minimal_proteomics_data`
- 200 samples: 150 controls, 30 incident, 20 prevalent
- 15 protein features with realistic signal
- Demographics: age, BMI, sex, ethnicity
- Small enough for fast tests, realistic structure

#### `temporal_proteomics_data`
- 200 samples with chronological sample dates (2020-2023)
- For testing temporal validation

#### `minimal_training_config`
- Reduced CV: 2 folds, 1 repeat, 2 inner folds
- Fast feature selection: top 10 proteins, k=[3,5]
- Minimal model configs for speed
- Optuna disabled

#### `minimal_splits_config`
- 2 splits, seed=42
- 50/25/25 train/val/test
- Control downsampling 5:1

#### `hpc_config`
- HPC-specific settings for dry-run testing
- Project allocation, queue, resources

### Test Classes

#### `TestE2EFullPipeline`
Tests the core workflow:
1. Generate splits with CLI
2. Train single model
3. Verify output files and metrics
4. Validate reproducibility

**Coverage:**
- Split generation and persistence
- Training integration
- Output directory structure
- Metrics file structure
- Reproducibility with fixed seeds

#### `TestE2EEnsembleWorkflow`
Tests ensemble stacking:
1. Generate splits
2. Train 2 base models (LR_EN, RF)
3. Train ensemble meta-learner
4. Verify ensemble outputs

**Coverage:**
- Base model training
- OOF prediction collection
- Meta-learner training
- Ensemble-specific outputs
- Error handling for missing base models

#### `TestE2EHPCWorkflow`
Tests HPC configuration:
1. Config validation
2. Required fields verification
3. Type checking

**Coverage:**
- HPC config structure
- Resource specification
- Dry-run mode readiness

#### `TestE2ETemporalValidation`
Tests temporal split functionality:
1. Generate chronological splits
2. Verify train < val < test ordering
3. No future data leakage

**Coverage:**
- Temporal split logic
- Chronological ordering
- Date handling

#### `TestE2EErrorHandling`
Tests failure modes:
- Missing input files
- Invalid model names
- Missing splits directory
- Corrupted config files

**Coverage:**
- Graceful error messages
- Exit code handling
- User-friendly error reporting

## Deterministic Testing Strategy

### Fixtures are deterministic
- Fixed RNG seeds (`np.random.default_rng(42)`)
- Same inputs → same outputs
- No network calls, no external dependencies

### Reproducibility checks
- `test_reproducibility_same_seed`: Verifies identical splits with same seed
- `test_full_pipeline_single_model`: Checks metrics reproducibility

### Validation approach
1. Create small realistic datasets
2. Run minimal configs (fast but realistic)
3. Verify output structure and content
4. Compare with expected ranges (not exact values due to stochasticity)

## Output Validation

### Split files
```
splits/
  train_idx_IncidentOnly_seed42.csv
  val_idx_IncidentOnly_seed42.csv
  test_idx_IncidentOnly_seed42.csv
  split_meta_IncidentOnly_seed42.json
```

Validated:
- Files exist
- No index overlap between sets
- Metadata matches expectations
- Sample counts correct

### Training outputs
```
results/LR_EN/split_seed42/
  metrics.json
  config.yaml
  metadata.json
  preds/
    train_oof/preds_train_oof_rep0.csv
    val/preds_val_rep0.csv
    test/preds_test_rep0.csv
  models/
    model_rep0_fold0.pkl
  cv/
    cv_results.csv
```

Validated:
- All required files present
- Metrics structure correct
- AUROC in [0, 1] range
- Config saved correctly

### Ensemble outputs
```
results/ENSEMBLE/split_seed42/
  metrics.json
  config.yaml
  preds/test/preds_test_rep0.csv
  models/meta_learner.pkl
```

Validated:
- Ensemble-specific files present
- Meta-learner artifact exists
- Metrics combine base model results

## Common Failure Modes Tested

### Missing files
- Input data not found → clear error message
- Splits directory missing → informative error
- Base models not trained → ensemble fails gracefully

### Invalid inputs
- Corrupted config → YAML parsing error caught
- Invalid model name → registry lookup fails with message
- Wrong file format → format detection error

### Configuration errors
- Missing required fields → validation error
- Type mismatches → schema validation catches
- Conflicting settings → logical validation fails

## Adding New E2E Tests

### Checklist for new tests
1. Use existing fixtures or add minimal new ones
2. Mark slow tests with `@pytest.mark.slow`
3. Test with deterministic data (fixed RNG)
4. Validate outputs, not internals
5. Check error handling, not just happy path
6. Keep fixtures small (<500 samples)
7. Document what flow is being tested

### Template for new test
```python
def test_new_workflow(self, minimal_proteomics_data, minimal_training_config, tmp_path):
    """
    Test: Description of workflow.

    Validates X, Y, Z.
    """
    # Setup
    splits_dir = tmp_path / "splits"
    results_dir = tmp_path / "results"

    runner = CliRunner()

    # Step 1: ...
    result = runner.invoke(cli, [...])
    assert result.exit_code == 0

    # Step 2: ...

    # Validate
    assert (results_dir / "expected_file.json").exists()
```

## Selector Strategy

### Stable selectors for file paths
- Use `Path` objects for cross-platform compatibility
- Construct paths relative to `tmp_path` fixture
- Avoid hardcoded absolute paths

### CLI command selectors
- Use `CliRunner().invoke(cli, [args])`
- Validate via `result.exit_code` and `result.output`
- Catch exceptions with `catch_exceptions=False` for debugging

### Output validation selectors
- JSON files: `json.load()` and key checks
- CSV files: `pd.read_csv()` and column/index checks
- YAML configs: `yaml.safe_load()` and schema checks
- Metrics ranges: `assert 0.0 <= value <= 1.0`

## Resilience Strategy

### Handling test flakiness
- All tests use deterministic fixtures
- No network calls or external services
- Isolated tmp directories per test
- Fixed random seeds throughout

### Test isolation
- Each test gets fresh `tmp_path`
- No shared state between tests
- Clean fixtures created per test
- No side effects on filesystem

### CI/CD considerations
- Fast tests run on every commit
- Slow tests run on PR or scheduled
- Coverage thresholds enforced
- Artifacts saved for debugging

## Integration with Existing Tests

### Relationship to unit tests
- Unit tests: test individual functions/modules
- E2E tests: test complete user workflows
- E2E tests complement, don't replace unit tests

### Coverage targets
- Unit tests: ≥80% line coverage on core modules
- E2E tests: 100% coverage of critical user flows
- Combined: comprehensive validation

### When to use E2E vs unit
- E2E: user-facing workflows, CLI integration, multi-step pipelines
- Unit: algorithm correctness, edge cases, error conditions

## Debugging Failed E2E Tests

### Check test output
```bash
pytest tests/test_e2e_runner.py::test_name -v -s
```
The `-s` flag shows print statements and detailed output.

### Inspect tmp directory
Failed tests preserve `tmp_path` contents:
```python
# In test:
if result.exit_code != 0:
    print(f"Temp dir: {tmp_path}")
    print(f"Files: {list(tmp_path.rglob('*'))}")
```

### Enable exception tracebacks
```python
result = runner.invoke(cli, [...], catch_exceptions=False)
```

### Check logs
```bash
cat logs/*.err  # HPC job errors
cat logs/*.out  # HPC job output
```

## Performance Guidelines

### Fast test criteria
- Complete in < 5 seconds
- Use minimal data (< 500 samples)
- Reduce CV folds (2 folds, 1 repeat)
- Disable Optuna
- Test structure, not accuracy

### Slow test criteria
- Complete in < 5 minutes
- Use small realistic data (200-1000 samples)
- Minimal CV (2-3 folds)
- Test integration and correctness
- Mark with `@pytest.mark.slow`

### Keep tests deterministic
- Fixed seeds everywhere
- No random timeouts
- No network dependencies
- No filesystem race conditions

## Future Enhancements

### Planned additions
1. Aggregation E2E test (post multiple splits)
2. Holdout evaluation E2E test
3. Multi-scenario workflow test
4. Config validation E2E test
5. HPC job submission test (mocked)

### Test maintenance
- Update fixtures when schema changes
- Add tests for new CLI commands
- Deprecate tests for removed features
- Keep documentation synchronized

## Summary

E2E tests provide confidence that user workflows work end-to-end:
- Deterministic fixtures ensure reproducibility
- Fast tests run on every commit
- Slow tests validate complex integrations
- Clear error handling prevents silent failures

Run fast tests locally before commit. Run full suite before PR.
