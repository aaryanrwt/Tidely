# The Cleaning Guide

The `td.clean()` function is the engine of Tidely. It orchestrates the semantic detection, memory optimization, and structural cleaning of your dataset.

## Basic Usage

To clean a dataset, simply pass a Pandas or Polars DataFrame into `td.clean()`.

```python
import pandas as pd
import tidely as td

df = pd.read_csv("dirty_dataset.csv")

# Clean the dataset
result = td.clean(df)

# Retrieve the optimized DataFrame
clean_df = result.df
```

## What Does Tidely Actually Do?

When you call `td.clean()`, Tidely performs the following deterministic sequence:

1. **Dataset Profiling**: It analyzes the shape, missing values, and memory footprint of the DataFrame.
2. **Semantic Inference**: It executes regex-based structural typing on all `object` and `string` columns to determine if they contain Dates, Emails, Currencies, Booleans, or Identifiers.
3. **Duplicate Removal**: Exact row duplicates (where every single column matches another row identically) are dropped.
4. **Structural Normalization**:
   - Dates (`12/1/2020 8:26`) are cast to `pd.to_datetime`.
   - Currencies (`$1,000.50`) are stripped of symbols and cast to numeric floats.
   - Booleans (`yes`/`True`/`1`) are unified into strict `True`/`False` masks.
5. **Memory Downcasting**: `int64` and `float64` columns are analyzed. If their maximum and minimum bounds can safely fit inside a smaller bit-bucket (e.g., `int16`), they are downcasted.
6. **Categorical Compression**: String columns with low cardinality (e.g., `Country` or `Category`) are converted into integer-backed Categorical pointers, often reducing string memory weight by 90%.

## The Cleaning Summary

Tidely strongly believes in Explainability. You should never run an automated pipeline without knowing *exactly* what it changed.

Always print the `.summary()`:

```python
print(result.summary())
```

This will print a beautiful, structured report explaining every action taken, the logic behind the action, and the memory saved. It will also print **Warnings** for actions Tidely explicitly refused to take (e.g. imputing missing Customer IDs).
