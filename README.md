<p align="center">
  <img src="assets/logo.png" width="120">
</p>

<h1 align="center">Tidely</h1>

<p align="center">
The Operating System for Data Quality
</p>

<p align="center">
<img src="assets/banner.png" width="100%">
</p>

<p align="center">

![PyPI](https://img.shields.io/pypi/v/tidely)
![Python](https://img.shields.io/pypi/pyversions/tidely)
![License](https://img.shields.io/github/license/aaryanrwt/Tidely)
![Stars](https://img.shields.io/github/stars/aaryanrwt/Tidely)
![Issues](https://img.shields.io/github/issues/aaryanrwt/Tidely)
![Downloads](https://img.shields.io/pypi/dm/tidely)

</p>

<p align="center">
Zero-Configuration • Explainable • Deterministic • Fast
</p>

---

## Install

```bash
pip install tidely
```

## The Magic

```python
import tidely as td

result = td.clean("sales.csv")

clean_df = result.df

print(result.summary())
```

---

# Why Tidely?

Real-world datasets are messy.

Missing values.

Broken dates.

Mixed datatypes.

Duplicate records.

Memory waste.

Encoding issues.

Schema drift.

Normally you spend hours writing cleaning scripts.

Tidely turns all of that into a single function call.

---

# Dataset Intelligence

```python
profile = td.inspect("sales.csv")

profile.show()
```

Output

✔ Trust Score

✔ Dataset DNA

✔ Semantic Detection

✔ Missing Values

✔ Duplicate Analysis

✔ Memory Analysis

✔ ML Readiness

✔ Data Quality Score

---

# Why Use Tidely?

| Feature | Pandas | Tidely |
|----------|---------|---------|
| Read CSV | ✅ | ✅ |
| Auto Detect Dates | ❌ | ✅ |
| Auto Clean Dataset | ❌ | ✅ |
| Memory Optimization | Manual | Automatic |
| Duplicate Detection | Manual | Automatic |
| Missing Value Strategy | Manual | Automatic |
| Semantic Column Detection | ❌ | ✅ |
| Explain Every Change | ❌ | ✅ |
| Health Score | ❌ | ✅ |
| Trust Score | ❌ | ✅ |
| Production Summary | ❌ | ✅ |

---

# Production Validation

Tidely has been validated on

| Dataset Type | Status |
|--------------|--------|
| CSV | ✅ |
| Excel (.xlsx) | ✅ |
| ARFF | ✅ |
| Government Open Data | ✅ |
| Educational Data | ✅ |
| ML Benchmark Datasets | ✅ |
| Large CSV (>3 Million Rows) | ✅ |
| Time Series | ✅ |
| Mixed Datatypes | ✅ |
| Corrupted Data | ✅ |

---

# Validation Results

Version

**v1.3.0b2**

| Dataset | Rows | Health Before | Health After |
|-----------|---------|----------------|---------------|
| Parking Meters | 52 | 94 | 96 |
| Credit-G | 1000 | 86 | 90 |
| Diabetes | 768 | 86 | 92 |
| Iris | 150 | 92 | 92 |
| Allegations | 57 | 95 | 92 |
| Mathematics | 59 | 97 | 94 |

---

# Benchmarks

3,055,000 Row Dataset

| Metric | Result |
|----------|----------|
| Runtime | 2.37 sec |
| Original Memory | 148 MB |
| Final Memory | 58 MB |
| Memory Saved | 61% |

---

# Supported Formats

- CSV

- Excel

- Parquet

- JSON

- TSV

- Feather

- ARFF

More coming soon.

---

# Explainable Cleaning

Tidely never silently changes your data.

Every transformation is documented.

Example

✓ Converted "Order Date" to datetime

Reason

Detected temporal values.

Impact

Allows time-series operations.

---

✓ Downcasted int64 → int16

Reason

Values fit inside Int16.

Impact

61% lower memory.

---

# Philosophy

Tidely follows three principles.

## Never silently modify data.

Every transformation is visible.

## Deterministic.

Same input.

Same output.

Every time.

## Local First.

Runs entirely on your machine.

No cloud.

No API keys.

No LLMs.

---

# Roadmap

- [x] CSV Cleaning

- [x] Explainable Reports

- [x] Memory Optimization

- [x] Semantic Detection

- [x] ARFF Support

- [x] Excel Support

- [ ] Intelligent Missing Value Imputation

- [ ] Fuzzy Duplicate Detection

- [ ] Streaming Engine

- [ ] DuckDB Integration

- [ ] Out-of-Core Cleaning

- [ ] Auto Feature Engineering

- [ ] SQL Dataset Support

- [ ] Distributed Processing

---

# Contributing

PRs are welcome.

Bug reports are welcome.

Feature requests are welcome.

---

# License

MIT
