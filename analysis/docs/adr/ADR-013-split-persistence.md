# ADR-013: Split Persistence Format (CSV Index Files)

**Status:** Accepted  
**Date:** 2026-01-20  
**Decision Makers:** Elahi Lab + Computational Team  

## Context

Data splits must be persisted to ensure:
- Exact reproducibility across runs
- Comparable results across different models
- Auditability (which samples in which split?)

Persistence format options:
- **CSV:** Human-readable text files
- **Pickle:** Binary Python-specific format
- **HDF5:** Binary format for large datasets
- **Parquet:** Columnar binary format

## Decision

Save split indices as **CSV files** with single column of row indices (int64).

**Filename Pattern:** `{scenario}_{split}_idx_seed{N}.csv`

**Example:**
```
IncidentPlusPrevalent_train_idx_seed42.csv
IncidentPlusPrevalent_val_idx_seed42.csv
IncidentPlusPrevalent_test_idx_seed42.csv
```

**Contents:**
```csv
row_idx
0
5
12
...
```

## Alternatives Considered

### Alternative A: Pickle
- Binary format, Python-specific
- **Rejected:** Not human-readable, not version-controllable, fragile across Python versions

### Alternative B: HDF5
- Efficient for large datasets
- **Rejected:** Overkill for small index arrays (148 incident + 740 controls = 888 samples per split)

### Alternative C: Parquet
- Columnar format, efficient compression
- **Rejected:** Overkill for small index arrays; CSV sufficient

### Alternative D: NumPy `.npy` Files
- Binary format, efficient
- **Rejected:** Not human-readable, not version-controllable

### Alternative E: JSON
- Human-readable, widely supported
- **Rejected:** Verbose for large index lists; CSV more concise

## Consequences

### Positive
- Human-readable and inspectable (can view in text editor or Excel)
- Version-controllable (git-friendly)
- Language-agnostic (Python, R, etc. can read CSV)
- Small file size (hundreds of integers)

### Negative
- Slightly slower I/O than binary formats (negligible for small arrays)

## Evidence

### Code Pointers
- [data/persistence.py](../../src/ced_ml/data/persistence.py) - `save_split_indices`, `load_split_indices`
- [cli/save_splits.py](../../src/ced_ml/cli/save_splits.py) - Split generation CLI

### Test Coverage
- `tests/test_data_persistence.py::test_save_load_split_indices` - Validates CSV I/O roundtrip
- `tests/test_data_persistence.py::test_split_index_format` - Validates CSV format

### References
- None (standard CSV format)

## Related ADRs

- Supports: [ADR-001: Split Strategy](ADR-001-split-strategy.md) (persistence of splits)
