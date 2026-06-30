# Tidely v1.4.1 Release Notes

Tidely v1.4.1 is a stability and correctness patch release. No new features are introduced. All efforts are focused on test reliability, documentation accuracy, and regression prevention.

## 🔧 What's Fixed in v1.4.1

- **Packaging Stability**: Fixed package behavior in fresh virtual environments where `pandas` was not installed. Converted all hard imports of `pandas` inside `engine.py` and `validate.py` to graceful optional imports.
- **Test Suite Reliability**: Fixed critical import errors and runtime failures in the CLI test module that prevented the full test suite from executing.
- **CleanResult Constructor**: Removed duplicated `__init__` body that redundantly assigned all attributes twice.
- **Documentation Accuracy**: Corrected `.export()` docstring that incorrectly claimed PDF support. Fixed CLI examples in README that referenced non-existent commands.
- **Cross-Platform Tests**: Replaced hardcoded virtual environment paths in test helpers with `sys.executable` for portable execution.

## ✅ What's New in v1.4.1

- **Regression Test Suite**: Added `test_regression_v141.py` covering file-path API, CleanResult structure, Excel loading, DuckDB routing, empty DataFrames, and version consistency.
- **Excel Dependencies**: Added `openpyxl` and `fastexcel` as optional dependencies for Excel file support.
- **Verified 2-Line API**: Confirmed that `td.clean("file.csv").df` works correctly across CSV, Excel, TXT, ARFF, and Parquet formats.

## 🚀 Highlights Carried Forward from v1.4.0
- **Native ARFF Format Support:** Zero-dependency ARFF parser supporting numeric, real, and nominal types.
- **DNA Semantic Protection:** Semantic nucleotide pattern recognition to protect biological sequences.
- **Automatic DuckDB Integration:** Out-of-core execution for large CSV/Parquet files.
- **Engine Auto-Selection:** Dynamic routing between Polars Eager, Polars Lazy, DuckDB, and Streaming.

## 📦 Installation
```bash
pip install tidely==1.4.1
```

## 📖 Known Limitations & Next Steps
- Nested JSON flattening is supported, but advanced auto-relational flattening of deeply nested arrays is scheduled for future milestones.
- Out-of-core streaming for non-CSV/Parquet formats is under active testing.
