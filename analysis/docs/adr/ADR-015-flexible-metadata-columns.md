# ADR-015: Flexible Metadata Column Configuration

**Status:** Accepted
**Date:** 2026-01-20
**Decision Makers:** Elahi Lab + Computational Team

## Context

The original pipeline assumed fixed metadata columns (age, BMI, sex, ethnicity) in addition to proteins. However:
- **Protein-only datasets:** Some datasets have no metadata (only protein columns)
- **Custom metadata:** Users may have different metadata columns (e.g., `cohort`, `custom_score`)
- **Missing columns:** Not all datasets have all default metadata columns

Hard-coding metadata columns causes:
- **KeyError** on protein-only datasets
- Inflexibility for custom metadata
- Difficulty adapting to new data sources

## Decision

Implement **flexible metadata column configuration** via `ColumnsConfig` class with two modes:

### Mode 1: Auto (Default)
- Auto-detect available metadata columns from data file
- Check for default metadata columns: `["age", "BMI", "sex", "ethnicity"]`
- Use only those present
- If none present, use 0 metadata columns (protein-only)

### Mode 2: Explicit
- User specifies custom metadata column lists:
  - `metadata_features`: Columns to include as features
  - `metadata_filters`: Columns used for row filtering
  - `metadata_stratify`: Columns used for stratification
- Validate that all specified columns exist in data

**Resolution Process:**
1. Before data loading, resolve columns via `resolve_columns()`
2. Returns `ResolvedColumns` object with:
   - `protein_cols`: All `*_resid` columns
   - `metadata_feature_cols`: Resolved metadata feature columns
   - `metadata_filter_cols`: Resolved filter columns
   - `metadata_stratify_cols`: Resolved stratify columns
3. Use resolved columns in `load_data()`, `apply_row_filters()`, and `stratified_split()`
4. Save resolved columns to `run_settings.json` for reproducibility

## Alternatives Considered

### Alternative A: Hard-Coded Metadata Columns
- Simplest implementation
- **Rejected:** Fails on protein-only datasets; inflexible

### Alternative B: Optional Metadata Flag
- User sets `use_metadata=True/False`
- **Rejected:** Binary choice insufficient for custom metadata

### Alternative C: Separate Config Files
- Protein-only vs. metadata configs
- **Rejected:** Duplicates configuration; `ColumnsConfig` cleaner

### Alternative D: Runtime Column Detection Only
- Always auto-detect, no explicit mode
- **Rejected:** Insufficient for custom metadata columns

## Consequences

### Positive
- Protein-only datasets work automatically (0 metadata columns)
- Custom metadata columns supported via explicit mode
- Backward compatible (existing configs use auto mode with defaults)
- Resolved columns saved for reproducibility
- Auto mode robust to missing columns

### Negative
- Adds configuration complexity (`ColumnsConfig` class)
- Requires column resolution step before data loading

## Evidence

### Code Pointers
- [config/schema.py](../../src/ced_ml/config/schema.py) - `ColumnsConfig` class
- [data/columns.py](../../src/ced_ml/data/columns.py) - `resolve_columns`, `ResolvedColumns`, `get_available_columns_from_file`
- [data/io.py](../../src/ced_ml/data/io.py) - `usecols_for_proteomics` (uses resolved columns)
- [data/filters.py](../../src/ced_ml/data/filters.py) - `apply_row_filters` (uses resolved columns)
- [cli/train.py](../../src/ced_ml/cli/train.py) - Column resolution before data loading

### Test Coverage
- `tests/test_data_columns.py` - 11 test cases covering:
  - Auto mode with all metadata present
  - Auto mode with partial metadata
  - Auto mode with no metadata (protein-only)
  - Explicit mode with custom columns
  - Explicit mode validation (missing columns)
  - Integration with `load_data()` and `apply_row_filters()`

### References
- None (custom design for this pipeline)

## Related ADRs

- Supports: [ADR-001: Split Strategy](ADR-001-split-strategy.md) (stratification uses resolved columns)
- Mentioned in: ARCHITECTURE.md Section 5.1 (Column Schema), Section 5.2 (Metadata Column Resolution)
