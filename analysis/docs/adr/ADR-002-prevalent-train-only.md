# ADR-002: Prevalent Cases in TRAIN Only

**Status:** Accepted  
**Date:** 2026-01-20  
**Decision Makers:** Elahi Lab + Computational Team  

## Context

The dataset contains two types of Celiac Disease (CeD) cases:
- **Incident cases** (n=148): Diagnosed *after* plasma sample collection (prospective)
- **Prevalent cases** (n=150): Diagnosed *before* plasma sample collection (retrospective)

Prevalent cases provide additional positive signal but represent a different distribution than the prospective screening scenario. Including prevalent cases in VAL/TEST would make evaluation non-representative of real-world prospective screening.

## Decision

**Add prevalent cases to TRAIN set only**, sampled at 50% to balance signal enrichment vs. distribution shift. VAL and TEST sets remain incident-only to maintain prospective evaluation.

## Alternatives Considered

### Alternative A: Exclude Prevalent Cases Entirely
- Simplest approach (incident-only throughout)
- **Rejected:** Wastes 150 positive samples with signal enrichment potential

### Alternative B: Include Prevalent in All Splits
- More training data for VAL/TEST as well
- **Rejected:** VAL/TEST no longer representative of prospective screening; evaluation not clinically relevant

### Alternative C: Separate Prevalent Model
- Train two models: incident-only and incident+prevalent
- **Rejected:** Adds complexity; prevalent signal can be leveraged in training without contaminating evaluation

### Alternative D: 100% Prevalent Sampling (All Prevalent to TRAIN)
- Maximum signal enrichment
- **Rejected:** Larger distribution shift between TRAIN and VAL/TEST; 50% sampling balances signal vs. shift

## Consequences

### Positive
- Training benefits from additional positive signal (150 → 148+75 = 223 total TRAIN positives)
- VAL/TEST remain prospective (incident-only) → clinically relevant evaluation
- 50% sampling balances signal enrichment vs. distribution shift

### Negative
- Distribution mismatch between TRAIN (incident+prevalent) and VAL/TEST (incident-only)
- Requires careful prevalence adjustment for deployment (see [ADR-005](ADR-005-prevalence-adjustment.md))

## Evidence

### Code Pointers
- [data/splits.py:326-366](../../src/ced_ml/data/splits.py#L326-L366) - `add_prevalent_to_train` function
- [data/schema.py:49](../../src/ced_ml/data/schema.py#L49) - `SCENARIO_DEFINITIONS` constant
- [cli/save_splits.py](../../src/ced_ml/cli/save_splits.py) - Split generation CLI

### Test Coverage
- `tests/test_data_splits.py::test_add_prevalent_to_train` - Validates prevalent handling
- `tests/test_data_splits.py::test_prevalent_never_in_val_test` - Enforces leakage prevention

### References
- Project memory: `project_overview` - "IncidentPlusPrevalent scenario: Prevalent in TRAIN only"

## Related ADRs

- Depends on: [ADR-001: Split Strategy](ADR-001-split-strategy.md)
- Supports: [ADR-005: Prevalence Adjustment](ADR-005-prevalence-adjustment.md)
