# Tidely: The Operating System for Data Quality

Tidely is a production-grade Python package that acts as **"The Operating System for Data Quality."** Instead of introducing custom data wrappers, Tidely integrates seamlessly into existing pipelines by accepting and returning standard Pandas DataFrames, Polars DataFrames/LazyFrames, and PyArrow Tables.

Tidely relies on two primitives to drastically improve data workflows:
1. `td.inspect(df)`: Generates a stunning Dataset Intelligence Report detailing Trust Scores, DNA signatures, and Semantics.
2. `td.clean(df)`: Generates an explainable, deterministic cleaning plan to sanitize missing data, duplicate rows, memory bloat, and semantically noisy strings (Dates, Emails, Phones).

---

## 🚀 Key Features

1. **Zero Friction API**: Call `td.inspect()` or `td.clean()` on any Polars, Pandas, or PyArrow dataframe.
2. **Lighthouse Dataset Trust Scores**: Computes multi-dimensional quality scores (Overall, Reliability, ML Readiness, Memory Efficiency, Schema Stability, and Semantic Quality).
3. **Deep Semantic Engine**: Heuristic regexes and checksum algorithms (Luhn for Credit Cards, Verhoeff for Aadhaar) to validate PAN, GSTIN, IP addresses, emails, phone numbers, and currencies.
4. **Explainable Cleaning**: Automatically converts types, normalizes PII formats, imputes missing values, and drops exact duplicates—explaining exactly *what* changed, *why* it changed, and how much it bumped the Trust Score. By default, Tidely avoids forward-filling missing values (to prevent hallucinating metadata in cross-sectional data) and uses constant/mode imputation instead.
5. **Streaming Native**: Built on Polars, `td.clean()` natively supports `collect(streaming=True)` on massive out-of-core datasets.

---

## 📦 Installation

To install Tidely in your project, use `pip` or `uv`:

```bash
pip install tidely
```

or

```bash
uv add tidely
```

*(Note: On Windows systems, Tidely automatically includes the `tzdata` package to support timezone-aware datetime validation).*

---

## ⚡ Quick Start

### 1. Dataset Inspection

```python
import tidely as sp
import polars as pl

# 1. Load your standard dataframe
df = pl.read_csv("messy_sales.csv")

# 2. Inspect the dataset
profile = td.inspect(df)

# Retrieve metrics programmatically
print(f"Overall Trust Score: {profile.trust_score.overall}/100")
print(f"ML Readiness: {profile.trust_score.ml_readiness}/100")

# 3. Display the stunning visual report in your terminal
profile.show()
```

### 2. Explainable Cleaning

```python
import tidely as sp
import polars as pl

df = pl.read_csv("messy_sales.csv")

# Generate the plan, show it in the terminal, and execute it
clean_df = td.clean(df)

# Alternatively, step through it manually:
plan = td.plan(df)
plan.show()

# Dry run to see exactly what rows will be affected before mutating
plan.execute(dry_run=True)

# Execute
clean_df = plan.execute()
```

### 3. Command Line Interface (CLI)

Tidely exposes a Typer-based CLI for instant dataset diagnostics directly from your terminal:

```bash
# Get a stunning visual diagnostic report
tidely inspect --input messy_sales.csv
```

![Tidely Demo](demo.gif)

---

## 🛠️ Benchmarks

Tidely is brutally fast. Check out our benchmarking suite to see how we stack up against `PyJanitor`, `Pandera`, `ydata-profiling`, and `Great Expectations`.

### 100,000 Rows (19MB DataFrame)
| Tool | Time (s) | Memory Peak (MB) |
|------|----------|-----------------|
| **Tidely** | **1.02s** | **113.79** |
| Pandera | 1.18s | 14.38 |
| PyJanitor | N/A* | N/A* |
| Great Expectations | N/A* | N/A* |
| ydata-profiling | N/A* | N/A* |

*\*Note: As of Pandas 2.x/3.x, `pyjanitor` and `ydata-profiling` have severe internal breaking changes that cause crashes. Great Expectations V1.0+ has completely removed its standard `from_pandas` API.*

Despite producing a massively detailed heuristic semantic analysis AND executing data transformations, **Tidely is still faster than pure schema-validation libraries like Pandera**.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on how to set up your development environment, run tests, and submit pull requests.

---

## 📚 API Reference

### `tidely.inspect(df: Any) -> DatasetProfile`
Generates a comprehensive diagnostic profile.
- **df**: The input data (Pandas DataFrame, Polars DataFrame/LazyFrame, PyArrow Table).
- **Returns**: A `DatasetProfile` object. Call `.show()` to render it in the terminal.

### `tidely.plan(df: Any) -> RepairPlan`
Generates a deterministic cleaning plan without mutating the data.
- **df**: The input data.
- **Returns**: A `RepairPlan` object. Call `.show()` to view the plan, and `.execute()` to run the transformations.

### `tidely.clean(df: Any) -> pl.DataFrame`
Automatically plans and executes all recommended data cleaning transformations.
- **df**: The input data.
- **Returns**: A pristine Polars DataFrame.

---

## ❓ FAQ

**Q: Does Tidely overwrite my original data?**
No. Tidely always returns a new, sanitized DataFrame. It never mutates your data in place.

**Q: Why does Tidely use Polars internally?**
Polars is written in Rust, utilizes lazy execution graphs, and is inherently multi-threaded. This allows Tidely to inspect and clean datasets magnitudes faster than native Pandas.

**Q: Can I run this on huge datasets?**
Yes. You can pass a Polars `LazyFrame` to `tidely.clean()` and it will utilize streaming `collect(streaming=True)` if the queries fit out-of-core memory bounds.

**Q: How does it know a column is a GSTIN or PAN?**
Tidely uses a deep semantic engine combining specialized regex heuristics and checksum algorithms (like Luhn and Verhoeff) to deterministically validate PII/Financial tokens.
