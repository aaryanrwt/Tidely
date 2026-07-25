# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.5.0] - 2026-07-25

### Added
- **Sequential Benchmark Engine**: New `benchmarks/` package implementing a full sequential benchmark pipeline — one dataset at a time, memory freed after each run, never loading multiple large datasets simultaneously.
- **Traditional Cleaning Pipeline**: Classical baseline using pandas, polars, numpy, scikit-learn, RapidFuzz, pyarrow, and regex for scientific comparison against Tidely.
- **12-Dataset Benchmark Suite**: Evaluates Tidely against real-world HuggingFace datasets (anisoleai/fineweb-tokenized, openai/gsm8k, mteb/results, HPLT, Spawning, InternRobotics, and 6 more).
- **Equivalence Validator**: Automated post-cleaning equivalence verification — row count, null count, duplicate count, dtype consistency, categorical values, numeric statistics, datetime parsing, boolean normalization, and whitespace cleanup. Never silently accepts mismatches.
- **Benchmark Reports**: Auto-generated `benchmarks/BENCHMARK.md` and `benchmark_results.json` with per-dataset timing, RAM, throughput, speedup, and correctness columns.
- **Regression Gate**: CI fails if any Tidely run exceeds 2× the traditional pipeline time on any dataset.
- **`bench` Optional Dependency Group**: `pip install tidely[bench]` installs all benchmark deps (datasets, scikit-learn, scipy, psutil, rapidfuzz, matplotlib, requests).
- **`benchmarks/` in sdist**: Benchmark code is included in source distributions for reproducibility.

### Changed
- **CI Workflow**: Removed Linux and macOS runners. Windows-only CI (`windows-latest`) with Python 3.12, 3.13, and 3.14.
- **CI Workflow**: Added Ruff format check, coverage threshold (≥70%), and benchmark smoke test step.
- **CI Workflow**: Upgraded to `actions/checkout@v4` and `actions/setup-python@v5` with pip caching.

### Performance
- Confirmed zero-copy Polars→Polars path in `adapter.py` (lines 296–302) — no unnecessary pandas roundtrip when input is already a Polars DataFrame.
- SemanticEngine regex patterns pre-compiled at `__init__` time (verified) — eliminates per-call re-compilation overhead.
- Sequential benchmark design enforces `gc.collect()` + explicit `del` between datasets — prevents memory accumulation on large multi-dataset runs.


## [1.4.3] - 2026-07-12

### Fixed
- **Hugging Face Compatibility**: Resolved streaming mode loader edge-cases.
- **Type Coercion**: Handled edge-case float representations in string-typed categorical data.

### Added
- **Adversarial Validation**: Completed full scientific trust audit on Hugging Face and ARFF datasets.
- **Traceability Reports**: Added BENCHMARK.md, VALIDATION_REPORT.md, DATA_PRESERVATION.md, and ML_IMPACT.md deliverables.

## [1.4.2] - 2026-07-01

### Fixed
- **Decompression**: Fixed Excel files (.xlsx) being double-decompressed as zip archives in the raw bytes parser.
- **Streams**: Added automatic `seek(0)` resets when reading from file-like objects to ensure streams can be reused by routing backends.
- **Semantic Classification**: Refined semantic keyword matching to check word boundaries (e.g. `"st"` no longer incorrectly flags `"first_name"` or `"last_name"` as Addresses).
- **Warnings**: Suppressed NumPy divide-by-zero runtime warnings in the correlation matrix calculation for constant columns.

### Added
- **Universal Export Engine**: Added direct support for 15+ export extensions (including TSV, Excel, ODS, Parquet, Feather, JSON, JSONL, XML, YAML, ARFF, DuckDB, SQLite).
- **Universal Ingestion**: Integrated support for decompressing zip, gzip, bz2, and xz files, database connection objects, and custom formats.
- **Advanced Semantics**: Added Name, City, Country, VIN, Customer ID, Invoice ID, and Product ID classifiers.
- **Readiness API**: Exposed execution metrics directly on `CleanResult` as properties, and added ML/Business readiness assessments to `CleanSummary`.
- **CI Matrix**: Configured cross-platform matrix testing on Windows, macOS, and Linux.

## [1.4.1] - 2026-07-01

### Fixed
- **Packaging**: Fixed package behavior in fresh virtual environments where `pandas` was not installed — converted all hard imports of `pandas` inside `engine.py` and `validate.py` to graceful optional imports.
- **Test Suite**: Fixed `ImportError` in `test_cli.py` — removed import of non-existent `assert_success` from helpers.
- **Test Suite**: Fixed `NameError` in `test_cli.py` — `cleaned_path` was referenced before assignment.
- **Test Suite**: Fixed hardcoded virtual environment path in `helpers.py` — now uses `sys.executable` for cross-platform reliability.
- **CleanResult**: Removed duplicated `__init__` body that redundantly assigned all attributes twice.
- **CleanResult**: Corrected `.export()` docstring that incorrectly claimed PDF support.
- **README**: Fixed CLI examples that referenced non-existent `summary` and `export` commands — updated to match actual CLI (`clean`, `inspect`, `report`).
- **Semantic Classification**: Fixed the "st" name-to-address boundary bug and added a Phone/Date tie-breaker.
- **Warnings**: Suppressed NumPy runtime warnings in the correlation matrix calculation for columns with constant null values.

### Added
- **Universal Export Engine**: Added direct support for 15+ export extensions (including TSV, Excel, ODS, Parquet, Feather, JSON, JSONL, XML, YAML, ARFF, DuckDB, SQLite).
- **Universal & Intelligent Loader**: Integrated support for decompressing zip, gzip, bz2, and xz files, reading SQLite and DuckDB connection objects, and custom formats (ARFF, XML, ODS, Pickle).
- **Semantic Intelligence**: Enhanced rules for Name, City, Country, Phone/Date tie-breaking, VIN (Vehicle ID), Customer ID, Invoice ID, and Product ID.
- **Readiness Summary & Metadata API**: Exposed execution time, rows removed, modified columns, memory saved, and backend name on `CleanResult` as properties, and added ML/Business readiness assessments to `CleanSummary`.
- Regression test suite (`test_regression_v141.py`) covering file-path API, CleanResult structure, Excel loading, DuckDB routing, empty DataFrames, and version consistency.
- `openpyxl` and `fastexcel` as optional dependencies for Excel file support.

### Improved
- Test helpers now gracefully handle missing `psutil` dependency.
- CLI tests split into focused, independent test functions for better failure isolation.

## [1.4.0] - 2026-06-30

### Added
- **Automatic DuckDB Integration**: Internally route huge CSV/Parquet files and operations exceeding RAM to DuckDB for fast out-of-core execution.
- **Automatic Engine Selection**: DecisionEngine dynamically selects between Polars Eager, Polars Lazy, DuckDB, and Out-of-Core Streaming.
- **Out-of-Core Streaming Engine**: High-throughput chunked processor cleaning datasets exceeding system memory with deterministic guarantees.
- **Adaptive Imputation**: Dynamic group stats and precomputed imputation models.

### Improved
- Memory footprint profiling and cost estimation in the Dataset Inspector.
- Visual Lighthouse-style reports containing detailed engine statistics, runtime/memory breakdowns, and selection explanations.

## [1.3.0b2] - 2026-06-30

- Decision Engine introduced
- Mixed dataset validation
- Improved semantic detection
- Better README
- Faster cleaning pipeline
- Bug fixes
- Improved memory optimization

### Added
- Native Attribute-Relation File Format (ARFF) parsing engine with nominal and type casting.
- `"DNA Sequence"` semantic checker matching nucleotide strings.
- Multi-sheet and robust worksheet fallbacks for educational Excel files.

### Improved
- Added robust column-level fallback for Pandas-to-Polars conversions when mixed types fail PyArrow type inference.
- Refined primary key deduplication logic to ignore descriptive string columns (e.g. columns with "name", "title", "text", "description").

### Fixed
- Unicode cleaning engine updated to preserve non-ASCII, foreign alphabets, and emojis (preserving movie reviews/text datasets).
- Fixed `replace_null_placeholders` to correctly convert custom string null representations (`?`, `N/A`, `NaN`) to true nulls.
- Moved row deduplication step to the end of the pipeline to prevent post-cleaning duplicate rows.

### Performance
- Fully vectorized parsing and cleaning pipelines with low peak RAM footprints and high throughput (~100 ms average latency).

### Validation
- 100% pass rate on UCI (Wine, Adult, Splice), Kaggle (IMDb, VGG16, Pokemon), and local mixed datasets validation campaigns.

## [1.0.0] - Public Beta Release

### Added
- Complete rewrite of the semantic detection engine to support cross-domain pattern matching.
- Memory optimizer module for massive-scale Pandas/Polars downcasting.
- Zero-corruption deterministic guarantees across duplicate drops and imputation.
- Expanded Datetime parser supporting US formats (`MM/DD/YYYY H:MM`).
- Validation Suite V: Full ERP Excel compatibility (`.xlsx`).
- Strict Hypothesis property-fuzzing and regression framework.
- Business Logic protection warning system for nulls in critical identifiers.
- Official documentation site and tutorials.
