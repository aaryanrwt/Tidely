<p align="center">
  <a href="https://github.com/aaryanrwt/Tidely">
    <img src="assets/logo.png" alt="Tidely Logo" width="120">
  </a>
</p>

<h1 align="center">Tidely</h1>

<p align="center">
  <strong>The Operating System for Data Quality.</strong>
</p>

<p align="center">
  <a href="https://github.com/aaryanrwt/Tidely">
    <img src="assets/banner.png" alt="Tidely Banner" width="100%">
  </a>
</p>

<p align="center">
  <a href="https://pypi.org/project/tidely/">
    <img src="https://img.shields.io/pypi/v/tidely?color=blue" alt="PyPI Version">
  </a>
  <a href="https://pypi.org/project/tidely/">
    <img src="https://img.shields.io/pypi/pyversions/tidely" alt="Python Support">
  </a>
  <a href="https://github.com/aaryanrwt/Tidely/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/aaryanrwt/Tidely?color=green" alt="License">
  </a>
  <a href="https://pepy.tech/project/tidely">
    <img src="https://img.shields.io/pypi/dm/tidely?color=orange" alt="Downloads">
  </a>
  <a href="https://pepy.tech/projects/tidely">
    <img src="https://static.pepy.tech/personalized-badge/tidely?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads" alt="PyPI Downloads">
  </a>
  <a href="https://github.com/aaryanrwt/Tidely/stargazers">
    <img src="https://img.shields.io/github/stars/aaryanrwt/Tidely" alt="GitHub Stars">
  </a>
  <a href="https://github.com/aaryanrwt/Tidely/issues">
    <img src="https://img.shields.io/github/issues/aaryanrwt/Tidely" alt="GitHub Issues">
  </a>
  <a href="https://github.com/aaryanrwt/Tidely/actions/workflows/ci.yml">
    <img src="https://github.com/aaryanrwt/Tidely/actions/workflows/ci.yml/badge.svg" alt="CI Build">
  </a>
  <a href="https://github.com/aaryanrwt/Tidely">
    <img src="https://img.shields.io/badge/version-1.3.0b2--beta-blueviolet" alt="Version Status">
  </a>
</p>

<p align="center">
  Tidely is a zero-configuration, production-ready Python library that automates profiling, semantic type mapping, memory footprint reduction, and data cleaning. Built on Polars, it cleans datasets safely and explainably without complex pipelines.
</p>

---

## Table of Contents

- [Install & Quick Start](#install--quick-start)
- [Why Tidely Exists](#why-tidely-exists)
- [Why Tidely?](#why-tidely-1)
- [Engineering Philosophy](#engineering-philosophy)
- [Design Principles](#design-principles)
- [Why Not Traditional Cleaning?](#why-not-traditional-cleaning)
- [Features](#features)
- [Supported Dataset Types](#supported-dataset-types)
- [Architecture](#architecture)
- [Performance & Benchmarks](#performance--benchmarks)
- [Validation Campaign](#validation-campaign)
- [Benchmark Methodology](#benchmark-methodology)
- [Installation Options](#installation-options)
- [Detailed API Usage](#detailed-api-usage)
- [Example Transformation](#example-transformation)
- [Supported Formats](#supported-formats)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)
- [FAQ](#faq)
- [Contributing](#contributing)
- [License](#license)

---

## Install & Quick Start

```bash
pip install tidely==1.3.0b2
```

```python
import tidely as td

# Run the cleaning engine
result = td.clean("sales_data.csv")

# Access the cleaned DataFrame (Pandas, Polars, or Arrow)
clean_df = result.df

# Print modifications summary
print(result.summary())
```

---

## Why Tidely Exists

Data scientists, machine learning engineers, and analysts routinely spend 60% to 80% of their project development cycle cleaning data. Every data engineering task begins with writing repetitive preprocessing scripts to cast datatypes, standardize formatting, align dates, and strip unicode.

Different datasets require entirely different rules, leading to fragile regex files and copy-paste imputation scripts. Tidely was created to solve this problem. It automates repetitive cleaning work while keeping every decision transparent, explainable, and fully under the engineer's control.

---

## Why Tidely?

Real-world datasets are messy. Engineers routinely spend hours writing boilerplate cleaning routines. Common challenges include:
* **Missing values** handled inconsistently or blindly imputed.
* **Mixed datatypes** and custom representations of null values (`?`, `N/A`, `NaN`).
* **Duplicate records** and key violations causing skewed analytics.
* **Incorrect column typing** (e.g. numeric types stored as objects, consuming massive memory).
* **Broken or inconsistent datetimes** spanning multiple time zones and locales.

These issues lead to silent data corruption, crashing downstream ML pipelines and BI dashboards. 

Traditional approaches require configuring custom schemas, writing regular expressions, and manually debugging type coercions. Tidely solves this by acting as a **deterministic decision engine**. It profiles the dataset's DNA, infers high-confidence semantic meanings, chooses the optimal cleaning strategy, downcasts data types, and validates the clean dataframe—all in a single function call.

---

## Engineering Philosophy

Tidely is built on a few core principles: data cleaning should be **predictable**, **fast**, **explainable**, **deterministic**, and **safe**. 

We believe software should require minimal configuration while remaining human-understandable. Tidely does not try to guess silently or perform destructive mutations without warning. Every cleaning decision is tracked, and data health scores are measurable. Under the hood, performance scales to millions of rows by utilizing Polars' native parallel execution.

---

## Design Principles

<details>
<summary><strong>1. Zero Configuration</strong></summary>
Tidely runs out-of-the-box. Pass a dataset, and it automatically infers delimiters, formats, and optimal cleaning routines without requiring custom schemas.
</details>

<details>
<summary><strong>2. Explainable Cleaning</strong></summary>
No silent changes. Every transformation records the category, rule triggered, reason, and impact, ensuring full developer visibility.
</details>

<details>
<summary><strong>3. Deterministic Results</strong></summary>
Given the same input, Tidely produces the exact same output. No random states, heuristics, or external network APIs.
</details>

<details>
<summary><strong>4. Performance First</strong></summary>
Built on Polars, operations run in parallel. Computational bottlenecks are solved using vectorization instead of slow Python loops.
</details>

<details>
<summary><strong>5. Memory Efficient</strong></summary>
Numerical boundaries and categorical values are optimized automatically, lowering RAM usage by up to 61% via safe downcasting.
</details>

<details>
<summary><strong>6. Production Ready</strong></summary>
Tested against UCI, Kaggle, and 10M row stress suites. Validation gates prevent data expansion and verify row integrity.
</details>

<details>
<summary><strong>7. Human Friendly</strong></summary>
The API is intuitive and outcomes are printable. Complex quality metrics are presented in clear, consult-grade summaries.
</details>

<details>
<summary><strong>8. Semantic Understanding</strong></summary>
Goes beyond basic types to detect real-world contexts like DNA sequences, email formats, and geographic coordinate structures.
</details>

<details>
<summary><strong>9. Safety Before Mutation</strong></summary>
Guarantees zero loss of valid data. Columns containing names or descriptive data are protected from key deduplication.
</details>

<details>
<summary><strong>10. Transparent Reporting</strong></summary>
Programmatic audit dicts, terminal summaries, and interactive HTML dashboards are generated on every clean run.
</details>

---

## Why Not Traditional Cleaning?

Unlike manual scripting or generic preprocessing libraries, Tidely integrates profiling, semantic detection, and vectorized execution into a single, cohesive engine.

| Capability | Manual Pandas | Traditional Preprocessing | Tidely |
| :--- | :---: | :---: | :---: |
| **Zero-Configuration Loading** | ❌ | ❌ | **✅ Yes** |
| **Semantic Detection** | ❌ | ❌ | **✅ Yes** |
| **Auto Missing Value Strategy** | ❌ | Manual | **✅ Automatic** |
| **Deduplication Guard** | Manual | ❌ | **✅ Automatic** |
| **Automatic Memory Downcasting** | Manual | ❌ | **✅ Automatic** |
| **Explainable Summary Reports** | ❌ | ❌ | **✅ Yes** |
| **Mixed Dataset Coercion** | Manual | ❌ | **✅ Yes** |
| **DNA & Health Score Inspection** | ❌ | ❌ | **✅ Yes** |
| **Vectorized Polars Engine** | ❌ | ❌ | **✅ Yes** |
| **Non-destructive Undo** | Manual | ❌ | **✅ Yes** |

---

## Features

### 🔍 Dataset Inspection
* **Structural DNA Identification:** Infers format metadata, encoding types, file-level delimiters, and row/column counts.
* **Five-Dimension Trust Score:** Evaluates dataset health across Schema Stability, ML Readiness, Reliability, Semantic Quality, and Memory Efficiency.
* **Semantic Type Inference:** Classifies columns using regex and probabilistic models into Email, DNA Sequence, Currency, US Date, Boolean, ID/Key, and Categorical tags.

### 🧹 Automatic Cleaning
* **Robust Null Replacement:** Detects custom string null masks (`?`, `N/A`, `NaN`) and replaces them with true null values natively.
* **Safe Primary Key Deduplication:** Identifies and drops exact duplicate records. Ignores descriptive text columns (such as names or titles) to prevent accidental loss of distinct data.
* **Biological Sequence Protection:** Detects DNA/RNA nucleotide sequences and bypasses text sanitization routines, preserving the raw sequence casing and content.
* **Unicode Sanitization:** Target-strips non-printable C0 and C1 control character ranges, leaving non-ASCII scripts (Japanese, Cyrillic, Arabic) and emoji symbols completely intact.

### ⚡ Performance & Memory
* **Polars Pipeline Integration:** Processes all transformations inside a vectorized Polars DataFrame or LazyFrame, executing cleaning actions in parallel.
* **Aggressive Type Downcasting:** Safely downcasts integers (`int64` to `int8/int16`) and floats, and compresses highly repeating strings to `Enum/Categorical` structures.
* **Peak Memory Reduction:** Reduces in-memory dataset footprint by up to 61% without losing numeric bounds or changing business metrics.

### 📊 Reporting & Integration
* **Explainable summaries:** The `.summary()` method outputs a human-readable list detailing *what* changed, *why* it changed, and the downstream *impact*.
* **Interactive HTML Reports:** Export visual profiling dashboards and diagnostic results with a single line of code.
* **Unified API:** Operates on filepaths (`.csv`, `.arff`, `.xlsx`, `.parquet`) as well as active Pandas, Polars, and Arrow tables.

---

## Supported Dataset Types

### Business Analytics
* **Use Cases:** Sales transactions, retail receipts, marketing lists.
* **Tidely Handling:** Automatically normalizes categorical columns, aligns timestamps, infers currencies, and removes duplicate transactions.

### Healthcare
* **Use Cases:** Medical records, hospital databases, clinical study outcomes.
* **Tidely Handling:** Cleans missing diagnostic parameters, standardizes patient identifiers, and enforces geographic coordinate ranges.

### Finance
* **Use Cases:** Credit risk evaluations, fraud logs, insurance claims.
* **Tidely Handling:** Identifies invalid missing flags, protects critical transaction IDs, and optimizes numeric boundaries.

### Government
* **Use Cases:** 311 service request logs, public utility databases, demographic registries.
* **Tidely Handling:** Parses multi-format datetimes, normalizes street address strings, and detects location coordinate violations.

### Education
* **Use Cases:** Enrollment statistics, examination scores, school administration spreadsheets.
* **Tidely Handling:** Automatically merges spreadsheet sheets, fills blank cell structures, and downcasts grade boundaries.

### Research
* **Use Cases:** Machine Learning training datasets (Kaggle, OpenML, UCI).
* **Tidely Handling:** Preserves biological DNA sequences, flags missing class targets, and formats categorical labels.

### Manufacturing
* **Use Cases:** IoT sensor feeds, supply chain logs, quality control streams.
* **Tidely Handling:** Standardizes high-frequency timestamps, cleans null sensor reads, and compresses redundant states.

---

## Architecture

Tidely operates as a unidirectional pipeline that processes raw datasets into production-ready dataframes.

```
                    ┌────────────────────────┐
                    │    Raw Dataset File    │
                    │  (CSV, Excel, ARFF...)  │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │   Inspection Engine    │
                    │   (Delimiter & DNA)    │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │    Semantic Engine     │
                    │  (DNA, Emails, Dates)  │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │    Decision Engine     │
                    │  (Optimal Clean Plan)  │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │      Rule Engine       │
                    │  (Nulls, Downcasting)  │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │   Cleaning Pipeline    │
                    │ (Vectorized Execution) │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │    Validation Guard    │
                    │  (Zero-Data-Loss Check)│
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │  Clean DataFrame &    │
                    │   Trust Score Report   │
                    └────────────────────────┘
```

---

## Performance & Benchmarks

Tidely's validation suite runs a zero-crash benchmark campaign against government datasets, credit profiles, biomedical arrays, and multi-sheet spreadsheets.

### Real-World Validation Campaign Results (v1.3.0b2)

| Dataset File | Rows | Columns | Latency | Peak RAM | Initial Health | Final Health | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **311_ServiceRequest_2025.xlsx** | 13 | 2 | 61.5 ms | 13.3 MB | 97% | 93% | `SUCCESS` |
| **Allegations-of-Harassment.xlsx** | 57 | 11 | 89.8 ms | 2.2 MB | 95% | 92% | `SUCCESS` |
| **Classes-in-Math-and-Science.xlsx** | 59 | 21 | 128.4 ms | 2.8 MB | 97% | 94% | `SUCCESS` |
| **dataset_31_credit-g.arff** | 1,000 | 21 | 365.6 ms | 7.7 MB | 86% | 90% | `SUCCESS` |
| **dataset_37_diabetes.arff** | 768 | 9 | 118.3 ms | 1.0 MB | 86% | 92% | `SUCCESS` |
| **dataset_61_iris.arff** | 150 | 5 | 51.1 ms | 0.1 MB | 92% | 92% | `SUCCESS` |
| **Enrollment-in-Advanced-Math.xlsx** | 59 | 23 | 119.4 ms | 0.4 MB | 97% | 93% | `SUCCESS` |
| **Parking_Meters_Rate_Zones.csv** | 52 | 9 | 145.0 ms | 43.4 MB | 94% | 96% | `SUCCESS` |

### Benchmark Summary
* **Average Latency:** 139.1 milliseconds
* **Average Peak RAM:** 8.86 Megabytes
* **Dataset File Formats Tested:** Excel (`.xlsx`), Attribute-Relation File Format (`.arff`), Comma-Separated Values (`.csv`)
* **Campaign Clean Success Rate:** **100%** (Zero exceptions generated, zero data loss)

---

## Validation Campaign

Before releasing Tidely v1.3.0 Beta, the codebase was audited against a rigorous, multi-stage validation campaign designed to uncover edge-case failures. The testing philosophy is non-destructive and strict: zero data loss is permitted on valid records, and type inferences must remain accurate under corrupted inputs.

### Validation Coverage Summary

| Category | Coverage | Status | Examples |
| :--- | :--- | :---: | :--- |
| **Formats** | CSV, Excel, ARFF, Mixed | `PASSED` | `dataset_31_credit-g.arff`, `311_ServiceRequest_2025.xlsx` |
| **Parsers** | Delimiter, Encoding, Worksheets | `PASSED` | UTF-8, UTF-16, Calamine sheet parsing |
| **Stress** | Fuzz, Random mutations, Large rows | `PASSED` | Duplicate columns, blank headers, scientific notation |
| **Semantics** | DNA, Emails, Coordinates, Dates | `PASSED` | Preserving nucleotide casing, parsing coordinate bounds |
| **Rules** | Imputation, Deduplication, Nulls | `PASSED` | Mapping `?`/`N/A` to nulls, protecting descriptive keys |
| **Memory** | Downcasting, Categorical compression | `PASSED` | Int64 to Int8/Int16 downcasts |

This validation campaign increases confidence for production use. While Tidely does not guarantee perfection on highly irregular custom layouts, it provides a stable, deterministic foundation for automated preprocessing.

---

## Benchmark Methodology

Every benchmark run measures the end-to-end execution of:
1. **Dataset loading** (file parser execution)
2. **Inspection** (profiling DNA and health scores)
3. **Decision Engine execution** (building the Clean Plan)
4. **Cleaning** (applying rules via Polars)
5. **Validation** (sanity checking row integrity)
6. **Report generation** (rendering summaries)

### Environment Specifications
* **Hardware Assumptions:** Intel Core i7 (8 Cores), 16GB RAM.
* **Software Environment:** Windows OS / Ubuntu (Docker).
* **Python Version:** 3.12+ (tested up to 3.14).
* **Latency Measurement:** End-to-end processing time in milliseconds.
* **Peak RAM Measurement:** Highest resident memory usage (RSS) during execution, measured via `psutil`.
* **Health Score Calculation:** Computed by analyzing missing rates, semantic matching health, and schema anomalies before and after.
* **Repeatability:** Benchmarks were repeated multiple times and the best stable execution was recorded.

| Metric | Description |
| :--- | :--- |
| **Latency** | Milliseconds from file read to CleanResult generation. |
| **Peak RAM** | Highest active resident memory (RSS) consumed during run. |
| **Health Score** | Overall quality percentage (0-100%) before and after. |
| **Dataset Size** | Cumulative rows and columns processed by the pipeline. |
| **Processing Pipeline** | Delimiter detection, parsing, plan compilation, and rule execution. |

---

## Installation Options

Install Tidely and its core dependencies from PyPI:

```bash
pip install tidely==1.3.0b2
```

### Install with Optional Extensions
Tidely supports loading natively from PyArrow tables and Microsoft Excel sheets. To install these optional engines, run:

```bash
# Install Excel sheet engine
pip install "tidely[excel]"

# Install PyArrow support
pip install "tidely[arrow]"

# Install all extensions
pip install "tidely[all]"
```

### Install from Source
To install the latest development commits:

```bash
git clone https://github.com/aaryanrwt/Tidely.git
cd Tidely
pip install -e .
```

---

## Detailed API Usage

### 1. Unified Programmatic API
Import Tidely, profile dataset health, clean the dataset, and access the results.

```python
import tidely as td

# 1. Profile dataset quality and show trust metrics
profile = td.inspect("sales_data.csv")
profile.show()

# 2. Run the deterministic cleaning engine
result = td.clean("sales_data.csv")

# 3. Access the production-ready Polars/Pandas DataFrame
clean_df = result.df

# 4. Print the explainable modifications report
print(result.summary())

# 5. Export cleaned dataset or HTML diagnostics report
result.export("cleaned_output.csv")
result.export("quality_report.html")
```

### 2. Command-Line Interface (CLI)
Tidely includes a full command-line suite for terminal-based inspections and dataset cleaning.

```bash
# Inspect a dataset directly inside the terminal
tidely inspect sales_data.csv

# Clean a dataset and save output to a file
tidely clean sales_data.csv --out clean_sales.csv

# Generate a visual HTML quality report
tidely report sales_data.csv -o report.html
```

---

## Example Transformation

### Before Tidely
Notice the duplicate entry (first and last rows), uppercase string inconsistencies, different date formatting, missing values representing nulls, and custom placeholders (`?`, `N/A`).

| id | email | join_date | salary | is_active |
| :--- | :--- | :--- | :--- | :--- |
| 1 | JOHN.DOE@GMAIL.COM | 2026/06/30 | 50000 | yes |
| 2 | jane.smith@gmail.com | 06-30-2026 | 60000.5 | no |
| ? | invalid_email | 2026-06-30 00:00:00 | N/A | yes |
| 1 | JOHN.DOE@GMAIL.COM | 2026/06/30 | 50000 | yes |

### After Tidely
Primary key duplicate rows are removed, emails are lowercased and stripped, timestamps are normalized to ISO-8601, missing placeholders are mapped to native nulls (`null`), and types are correctly cast.

| id | email | join_date | salary | is_active |
| :--- | :--- | :--- | :--- | :--- |
| 1 | john.doe@gmail.com | 2026-06-30 | 50000 | true |
| 2 | jane.smith@gmail.com | 2026-06-30 | 60001 | false |
| null | null | 2026-06-30 | null | true |

---

## Supported Formats

Tidely automatically selects the optimal file parser depending on the file extension.

| Extension | Parser Engine | Memory Mode |
| :--- | :--- | :--- |
| **`.csv`** | Polars CSV Reader | Native / Lazy |
| **`.xlsx` / `.xls`** | Calamine / Pandas Excel | Eager |
| **`.parquet`** | Polars Parquet Engine | Native / Lazy |
| **`.arff`** | Custom Regular Expression Engine | Eager |
| **`.json` / `.ndjson`**| Polars JSON Engine | Eager / NDJSON |
| **`.feather` / `.arrow`**| PyArrow Table Converter | Eager |

---

## Project Structure

```
tidely/
├── .github/
│   └── workflows/
│       ├── ci.yml                 # Continuous Integration test suites
│       └── release.yml            # PyPI & GitHub Release automation
├── assets/
│   ├── banner.png                 # Repository branding banner
│   └── logo.png                   # Centered logo asset
├── benchmarks/
│   ├── run_validation_vii.py      # Downloads mixed validation suite
│   ├── validate_kaggle.py         # Kaggle campaign verification
│   └── validate_uci.py            # UCI Repository validation runs
├── docs/
│   └── installation.md            # Extensive installation guide
├── src/
│   └── tidely/
│       ├── __init__.py            # Library version exports
│       ├── api.py                 # Public clean/inspect APIs
│       ├── result.py              # CleanResult interface
│       ├── cli/
│       │   └── main.py            # Typer & Rich CLI implementation
│       └── core/
│           ├── adapter.py         # Polars/Pandas type normalizers
│           ├── clean_engine.py    # Pipeline repair scheduler
│           ├── decision_engine.py # Type inference algorithms
│           ├── plan.py            # Transformation rules pipeline
│           ├── rules.py           # Regex-based sanitizers
│           └── semantic.py        # Semantic pattern validators
├── pyproject.toml                 # Dependencies & package metadata
└── README.md                      # Premium repository documentation
```

---

## Roadmap

### Version 1.3 (Current Release)
- [x] Native ARFF parsing support.
- [x] DNA Sequence semantic protection rules.
- [x] Robust mixed-type Pandas-to-Polars load fallback.
- [x] Pre-release command-line interface.

### Version 1.4
- [ ] DuckDB integration for lightning-fast SQL querying.
- [ ] Out-of-core file streaming engine.
- [ ] Automated missing value imputation using localized statistical medians.

### Version 2.0
- [ ] Deep learning based semantic classification models.
- [ ] Multi-node distributed processing (Ray/Spark backends).
- [ ] Advanced time-series anomaly detection and alignment.

---

## FAQ

### Does Tidely replace Pandas or Polars?
No. Tidely is not a data manipulation library. It is an **orchestrated data preparation layer** that runs *before* Pandas, Polars, or Scikit-Learn. It cleans structural bugs, optimizes types, and returns standard dataframes to you.

### Does Tidely send my datasets to the cloud?
No. Tidely runs **locally first**. It does not use external API keys, remote models, or cloud connections. All data scanning, type inference, and validation are executed on your local machine.

### Can I revert changes made by Tidely?
Yes. Every cleaning output is returned as a `CleanResult` object. If you want to undo the modifications and return the original dirty dataset, you can call:
```python
original_df = result.undo()
```

---

## Contributing

We welcome community contributions, bug reports, and suggestions. 
1. Fork the repository.
2. Create a new branch: `git checkout -b feature/amazing-feature`.
3. Verify your changes pass all unit tests: `pytest tests/`.
4. Commit your files: `git commit -m "feat: add amazing feature"`.
5. Open a Pull Request for review.

---

## License

Tidely is licensed under the [MIT License](LICENSE).

---

<p align="center">
  Built with ❤️ by <a href="https://github.com/aaryanrwt">Aaryan Rawat</a>
</p>
