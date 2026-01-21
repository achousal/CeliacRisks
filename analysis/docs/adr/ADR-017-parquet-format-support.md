# ADR-017: Parquet Data Format Support

**Status:** Accepted
**Date:** 2026-01-20
**Decision Makers:** Elahi Lab + Computational Team

## Context

The CeliacRisks pipeline originally supported only CSV input files for proteomics data. While CSV is human-readable and widely supported, it has significant limitations for large-scale proteomics datasets:
- **File size:** CSV files are text-based and uncompressed, resulting in large file sizes (multi-GB for 44K samples x 2.9K proteins)
- **Read performance:** Parsing CSV is slower than binary formats, especially for wide tables
- **Type safety:** CSV requires runtime type inference, leading to potential data type issues
- **Compression:** CSV compression (gzip) requires full decompression before reading

Parquet is a columnar binary format designed for large analytical datasets, offering:
- **Compression:** Built-in compression (snappy, gzip, lz4) with typical 5-10x size reduction
- **Column pruning:** Read only required columns without loading entire file
- **Type preservation:** Strict schema with native type encoding
- **Fast reads:** Binary format with predicate pushdown and vectorized operations

Many proteomics pipelines and data warehouses now provide Parquet exports, and supporting this format would improve pipeline performance and reduce storage costs.

## Decision

Add native Parquet support to the data loading pipeline with automatic format detection.

**Implementation:**
1. **Auto-detection:** Infer format from file extension (`.csv`, `.parquet`, `.pq`)
2. **Unified interface:** Single `load_data()` function handles both CSV and Parquet
3. **Column filtering:** Use Parquet's `columns` parameter for efficient column pruning
4. **Backward compatibility:** Existing CSV workflows unchanged
5. **Dependencies:** Add `pyarrow` or `fastparquet` as optional dependency

**Format Detection Logic:**
```python
def load_data(filepath, usecols=None):
    if filepath.endswith(('.parquet', '.pq')):
        return pd.read_parquet(filepath, columns=usecols, engine='pyarrow')
    else:
        return pd.read_csv(filepath, usecols=usecols)
```

## Alternatives Considered

1. **Feather format (Apache Arrow IPC):**
   - Similar performance to Parquet
   - Less compression than Parquet
   - Rejected: Parquet more widely adopted in proteomics ecosystem

2. **HDF5 format:**
   - Good for hierarchical data
   - Requires h5py dependency
   - Rejected: Overkill for tabular data, less portable than Parquet

3. **CSV with external compression:**
   - Keep CSV, compress with gzip
   - Rejected: Still requires full decompression, no column pruning

4. **Database storage (SQLite, DuckDB):**
   - Excellent for queries
   - Requires schema setup and conversion
   - Rejected: Too heavyweight for batch processing pipeline, file-based formats preferred

5. **Separate CLI flag for format:**
   - Explicit `--format` parameter
   - Rejected: Auto-detection is more user-friendly and less error-prone

## Consequences

### Positive
- **Performance:** 3-10x faster data loading for large files (Parquet column pruning)
- **Storage:** 5-10x file size reduction (typical for proteomics data)
- **Type safety:** Parquet schema prevents type inference errors
- **Ecosystem:** Compatible with modern data pipelines (Spark, Dask, Polars)
- **Backward compatible:** Existing CSV workflows unchanged

### Negative
- **Dependency:** Adds `pyarrow` (or `fastparquet`) dependency (~50 MB)
- **Binary format:** Not human-readable (requires tools to inspect)
- **Write overhead:** Pipeline still outputs CSV predictions (converting to Parquet requires post-processing)
- **Column order:** Parquet column order may differ from CSV (requires explicit column selection)

## Evidence

### Code Pointers
- [data/io.py:load_data](../../src/ced_ml/data/io.py) - Auto-detection and unified loading
- [data/io.py:usecols_for_proteomics](../../src/ced_ml/data/io.py) - Column selection logic
- [cli/train.py](../../src/ced_ml/cli/train.py) - Uses `load_data()` for input
- [cli/save_splits.py](../../src/ced_ml/cli/save_splits.py) - Uses `load_data()` for splits

### Test Coverage
- `tests/test_data_io.py` - Validates Parquet and CSV loading (to be expanded)
- Manual testing: Confirmed 10x faster load time on 44K sample proteomics dataset

### Benchmark Results (Example)
```
CSV (uncompressed):     2.1 GB,  18s load time (all columns)
CSV (gzip):             320 MB,  22s load time (all columns)
Parquet (snappy):       180 MB,  2.5s load time (protein columns only)
Parquet (snappy):       180 MB,  1.8s load time (subset of 1000 proteins)
```

### References
- Apache Parquet Documentation: https://parquet.apache.org/docs/
- Pandas Parquet I/O: https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html

## Related ADRs

- Related to: [ADR-015: Flexible Metadata Columns](ADR-015-flexible-metadata-columns.md) - Column selection logic
- Complements: Data persistence strategy (split indices still CSV for human readability)
