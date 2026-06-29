<p align="center">
  <img src="assets/banner.png" alt="Tidely Banner" width="100%">
</p>

# Tidely

[![PyPI version](https://img.shields.io/pypi/v/tidely)](https://pypi.org/project/tidely/)
[![Python Versions](https://img.shields.io/pypi/pyversions/tidely)](https://pypi.org/project/tidely/)
[![Downloads](https://img.shields.io/pypi/dm/tidely)](https://pypi.org/project/tidely/)
[![License](https://img.shields.io/github/license/aaryanrwt/Tidely)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/aaryanrwt/Tidely?style=social)](https://github.com/aaryanrwt/Tidely)
[![GitHub Issues](https://img.shields.io/github/issues/aaryanrwt/Tidely)](https://github.com/aaryanrwt/Tidely/issues)
[![Last Commit](https://img.shields.io/github/last-commit/aaryanrwt/Tidely)](https://github.com/aaryanrwt/Tidely)
[![CI](https://github.com/aaryanrwt/Tidely/actions/workflows/ci.yml/badge.svg)](https://github.com/aaryanrwt/Tidely/actions)

Zero-configuration, explainable data cleaning for Python.

---

## What is Tidely?

Tidely is a local-first, deterministic data cleaning library designed to replace hundreds of lines of fragile Pandas preprocessing code with a single, highly optimized command. 

Tidely automatically profiles your dataset, infers semantic types (Dates, Emails, Currency, IDs), safely downcasts memory footprint by up to 85%, and structures unstructured text—all without silently mutating your business logic or randomly dropping values.

## Why Tidely Exists

Data scientists and engineers spend 80% of their time writing repetitive data cleaning boilerplate: fixing `M/D/YYYY` dates, trimming whitespaces, downcasting 64-bit floats to save memory, parsing currency symbols, and dropping exact duplicate rows. 

Tidely eliminates this boilerplate entirely. It is built on three core philosophies:
1. **Never silently delete data.** Every transformation is tracked, explained, and non-destructive.
2. **Local-first and Secure.** Tidely runs entirely on your CPU. No API keys, no LLMs, no cloud processing.
3. **Deterministic.** The same dirty DataFrame yields the exact same clean DataFrame, every single time.

---

## ⚡ Quick Start

### Installation

```bash
pip install tidely
```

### The One-Minute Example

```python
import pandas as pd
import tidely as td

# 1. Load your dirty data
df = pd.read_csv("dirty_data.csv")

# 2. Clean it automatically
result = td.clean(df)

# 3. Retrieve the clean, memory-optimized DataFrame
clean_df = result.df

# 4. View a detailed, explainable summary of what changed
print(result.summary())
```

---

## 🔍 Before vs After

**Before Tidely:**
```python
df = pd.read_csv("data.csv")
df.drop_duplicates(inplace=True)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['price'] = df['price'].str.replace('$', '').astype(float)
df['category'] = df['category'].astype('category')
df['is_active'] = df['is_active'].map({'yes': True, 'no': False})
# ... 50 more lines of boilerplate ...
```

**After Tidely:**
```python
import tidely as td
df = td.clean(pd.read_csv("data.csv")).df
```

---

## 🚀 Core Features

- **Semantic Intelligence**: Natively infers and standardizes Emails, URLs, Currencies, Boolean permutations (yes/y/true/1), IPv4, SSNs, and Dates (including US formats like `MM/DD/YYYY`).
- **Memory Optimization**: Automatically downcasts over-provisioned 64-bit integers/floats to 16/32-bit types, and converts low-cardinality strings to Categorical pointers. Safely reduces Pandas memory footprints by 40-85%.
- **Zero-Corruption Duplicate Removal**: Identifies and drops exact duplicate rows that skew statistical modeling.
- **Deep Explainability**: Generates an exhaustive `summary()` explaining *what* was changed, *why* it was changed, and the *impact* of the change.
- **Business Logic Protection**: Explicitly issues `Warnings` for missing financial or identifier data rather than blindly imputing zeros.

### Supported DataFrames
Tidely currently supports:
* `pandas.DataFrame`
* `polars.DataFrame`
* `polars.LazyFrame`
* `pyarrow.Table`

---

## 🏎️ Performance Philosophy

Tidely is designed for enterprise scale. It operates heavily via vectorized operations backed by `pandas` and `polars`. 

During internal benchmarking, Tidely processed 10,000,000 rows across mixed-types in **under 26 seconds**, safely shrinking the DataFrame from 591 MB down to 85 MB without corrupting type definitions. We rely purely on algorithmic inference—no slow machine learning heuristics or network latency.

---

## 🛡️ Validation Summary (Public Beta)

Tidely v1.0 has completed an extensive internal validation campaign covering more than twenty real-world datasets across healthcare, finance, retail, manufacturing, government, environmental science, e-commerce, and enterprise Excel workflows.

The library has also passed property-based testing (Hypothesis), fuzz testing, large-scale stress testing up to 10 million rows, API stability checks, and cross-version compatibility testing. 

Based on these results, Tidely is now entering **Public Beta**, where broader community feedback will continue to strengthen its reliability.

---

## 📚 Documentation

Detailed documentation is available in the `docs/` directory:
- [Introduction & Philosophy](docs/introduction.md)
- [Installation Guide](docs/installation.md)
- [Cleaning Guide](docs/cleaning_guide.md)
- [Semantic Detection Engine](docs/semantic_detection.md)
- [Memory & Performance](docs/performance.md)
- [Validation Guide](docs/validation_guide.md)
- [FAQ](docs/faq.md)

---

## 🛣️ Roadmap
- Multi-threaded processing for CSV batch-cleaning.
- Out-of-core chunked processing for data exceeding local RAM.
- Geographic coordinate standardization (Lat/Lon).
- Enhanced HTML extraction capabilities.

---

## 🤝 Contributing

Tidely is an open-source project and community contributions are highly welcome. Please review our [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before submitting pull requests.

## License

Tidely is released under the [MIT License](LICENSE).
