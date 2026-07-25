# Tidely v1.5.0 Release Notes

**Release date:** 2026-07-25  
**Branch:** `feat/v1.5.0-performance-benchmark`  
**Type:** Engineering Release — Performance, Benchmarking & Enterprise Validation

---

## What's New in v1.5.0

### Sequential Benchmark Engine
A production-grade benchmark suite that processes one dataset at a time, compares Tidely against a traditional cleaning pipeline, validates output equivalence, and generates Markdown + JSON reports — all without loading multiple large datasets into memory simultaneously.

### Traditional Pipeline Baseline
A rigorous classical cleaning pipeline (pandas + polars + numpy + scikit-learn + RapidFuzz + pyarrow + regex) implementing 11 operations:
null placeholder replacement, deduplication, missing value imputation (median/mode), whitespace normalization, unicode normalization (NFC), categorical lowercasing, boolean normalization, datetime parsing, outlier clipping (3×IQR), numeric downcasting, and optional fuzzy deduplication.

### 12-Dataset HuggingFace Benchmark Suite
Real-world datasets from anisoleai, openai, mteb, apple, mvp-lab, LiLabUNC, Spawning, InternRobotics, HPLT, and HuggingFace's own documentation — loaded via streaming or datasets-server API in 200-row subsets.

### Automated Equivalence Validator
8+ post-cleaning checks per dataset. Never silently accepts mismatches. Every difference is logged with scientific justification.

### Windows-Only CI
CI updated to Windows-only with Python 3.12, 3.13, and 3.14. Includes Ruff lint + format check, MyPy, Pytest with coverage threshold, benchmark smoke test, and regression gate.

### Regression Gate
CI fails if Tidely is more than 2× slower than the traditional pipeline on any dataset.

---

## Upgrading

```bash
pip install tidely==1.5.0
# With benchmark dependencies:
pip install tidely[bench]==1.5.0
```

---

## Running Benchmarks

```bash
# Full benchmark (all 12 datasets, sequential)
python benchmarks/run_benchmark.py

# CI smoke test (2 datasets)
python benchmarks/run_benchmark.py --smoke-test

# Check regressions on existing results
python benchmarks/run_benchmark.py --check-regression
```
