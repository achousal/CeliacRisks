# ADR-012: Pydantic Config Schema

**Status:** Accepted
**Date:** 2026-01-20
**Decision Makers:** Elahi Lab + Computational Team

## Context

The pipeline has ~200 configuration parameters across:
- Data splits (17 params)
- Cross-validation (8 params)
- Feature selection (14 params)
- Model hyperparameters (~60 params across 4 models)
- Thresholds (8 params)
- Evaluation (12 params)
- Output control (8 params)

Configuration management requires:
- Type validation
- Default values
- Cross-field validation (e.g., "VAL + TEST fractions < 1.0")
- YAML serialization/deserialization

## Decision

Use **Pydantic** for configuration schema (all config classes inherit `BaseModel`).

Pydantic provides:
- Runtime type validation + coercion
- Nested config objects
- Cross-field validation via `@model_validator`
- YAML/JSON serialization
- IDE autocomplete + type hints

## Alternatives Considered

### Alternative A: dataclasses
- Standard library (no dependencies)
- **Rejected:** No runtime validation; requires manual validation logic

### Alternative B: attrs
- Similar to dataclasses with validation
- **Rejected:** Less popular than Pydantic in ML community; weaker validation

### Alternative C: Plain Dictionaries
- Simplest (no schema)
- **Rejected:** No type safety, no validation, error-prone

### Alternative D: Hydra (Facebook)
- Config framework with composition
- **Rejected:** Overkill for this use case; Pydantic simpler and more Pythonic

## Consequences

### Positive
- Runtime validation catches config errors early
- Type hints improve IDE support and readability
- Cross-field validation via `@model_validator` (e.g., split fractions)
- YAML roundtrip preserves types (int, float, bool)
- Nested config objects (e.g., `TrainingConfig.cv.folds`)

### Negative
- Additional dependency (pydantic)
- Slight learning curve for Pydantic validators

## Evidence

### Code Pointers
- [config/schema.py:292-343](../../src/ced_ml/config/schema.py#L292-L343) - `TrainingConfig` (top-level config)
- [config/schema.py](../../src/ced_ml/config/schema.py) - All config classes (inherit `BaseModel`)
- [config/validation.py](../../src/ced_ml/config/validation.py) - Custom validators

### Test Coverage
- `tests/test_config.py::test_training_config_validation` - Validates Pydantic validation
- `tests/test_config.py::test_config_yaml_roundtrip` - Validates YAML serialization

### References
- Pydantic documentation: https://docs.pydantic.dev/

## Related ADRs

- None (foundational design decision)
