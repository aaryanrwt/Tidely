# Tidely v1.4.2 Release Notes

Tidely v1.4.2 is a production-hardening release focusing on quality, stability, correctness, and comprehensive pipeline validation.

## 🔧 What's Fixed in v1.4.2

- **Double Decompression**: Fixed Excel files (.xlsx) being double-decompressed as zip archives in the raw bytes parser.
- **Stream Cursor Resets**: Added automatic `seek(0)` resets when reading from file-like objects so streams can be reused by routing backends.
- **Name-to-Address Classification**: Refined semantic keyword matching to check word boundaries (e.g. `"st"` no longer incorrectly flags `"first_name"` or `"last_name"` as Addresses).
- **Runtime Warnings**: Suppressed NumPy divide-by-zero runtime warnings in the correlation matrix calculation for constant columns.

## ✅ What's New in v1.4.2

- **Universal Export Engine**: Added direct support for 15+ export extensions (including TSV, Excel, ODS, Parquet, Feather, JSON, JSONL, XML, YAML, ARFF, DuckDB, SQLite).
- **Universal & Intelligent Ingestion**: Integrated support for decompressing zip, gzip, bz2, and xz files, reading database connections, and custom formats.
- **Advanced Semantic Classifiers**: Added support for Names, Cities, Countries, VINs, Customer IDs, Invoice IDs, and Product IDs.
- **Readiness Summary & Metadata API**: Exposed execution metrics (time, memory saved, rows removed, backend used) directly on `CleanResult` as properties, and added ML/Business readiness assessments to `CleanSummary`.
- **Packaging & CI Matrix**: Configured cross-platform matrix testing on Windows, macOS, and Linux, and declared optional packaging dependencies.

## 📦 Installation
```bash
pip install tidely==1.4.2
```

## 📖 Known Limitations & Next Steps
- Flat text streams without clear separators default to single-column parses.
- Out-of-core streaming for JSON/XML formats is under active development.
