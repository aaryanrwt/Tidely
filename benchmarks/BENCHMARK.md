# Tidely v1.5.0 — Benchmark Report

_Generated: 2026-07-25 07:42 UTC_  
_Methodology: Sequential processing. One dataset at a time. Memory freed after each run._

## Summary

| Metric | Value |
| :--- | :--- |
| Datasets evaluated | 2 |
| Datasets passed validation | 2 |
| Datasets with Tidely speedup | 1 |
| Regression threshold | 2.0x |

## Benchmark Results

| # | Dataset | Rows | Cols | Trad Time (ms) | Tidely Time (ms) | Speedup | Peak RAM (MB) | Throughput (rows/s) | Correctness |
| :- | :--- | ---: | ---: | ---: | ---: | :---: | ---: | ---: | :---: |
| 1 | anisoleai/fineweb-tokenized | 200 | 1 | 4.4 | 19.7 | **4.46x** ❌ | 8.9 | 10,167 | ✅ PASS |
| 2 | huggingface/documentation-images | 28 | 1 | 24.6 | 24.5 | **1.00x** ✅ | 2.5 | 1,143 | ✅ PASS |

## Validation Details

| Dataset | Row Count | Null Reduction | Duplicate Removal | Dtype Consistency | Correctness |
| :--- | :---: | :---: | :---: | :---: | :---: |
| anisoleai/fineweb-tokenized | ✅ | ✅ | ✅ | ✅ | ✅ PASS |
| huggingface/documentation-images | ✅ | ✅ | ✅ | ✅ | ✅ PASS |

## Regression Check

> **1 regression(s) detected** (Tidely >2.0x slower than traditional):

- `anisoleai/fineweb-tokenized`: 4.46x slower (Tidely=19.7ms, Trad=4.4ms)

## Benchmark Methodology

- **Datasets**: 12 HuggingFace datasets loaded via streaming (200-row subsets) or datasets-server API
- **Traditional pipeline**: pandas + polars + numpy + scikit-learn + RapidFuzz + pyarrow + regex
- **Tidely pipeline**: `td.clean(path)` — zero configuration
- **Processing**: Sequential — one dataset at a time, memory freed between runs
- **Timing**: `time.perf_counter()` wall-clock time, excludes data loading
- **RAM**: `psutil.Process.memory_info().rss` peak during cleaning
- **Correctness**: Validated via 8+ automated equivalence checks per dataset
- **Regression gate**: CI fails if Tidely is >{regression_threshold}x slower than traditional on any dataset

## Performance Optimizations Applied in v1.5.0

| Area | Optimization | Impact |
| :--- | :--- | :--- |
| Import system | Lazy imports for heavy deps (polars, pyarrow, duckdb) | Reduces cold-start time |
| Regex | Pre-compiled patterns at module level in semantic.py | Eliminates re-compilation overhead |
| Polars expressions | Batched `with_columns` instead of sequential per-column passes | Reduces DataFrame copies |
| Adapter | Zero-copy Polars→Polars path added | Removes unnecessary pandas roundtrip |
| String ops | Polars `str.strip_chars`, `str.replace_all` (C-level) | Faster than Python-level loops |

---
_Tidely v1.5.0 — 2026-07-25 07:42 UTC_
