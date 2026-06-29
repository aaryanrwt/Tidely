# Introduction to Tidely

Tidely is a local-first, deterministic data cleaning library designed to replace fragile Pandas boilerplate with a single, highly optimized command. 

## The Problem

Data Scientists and Engineers spend up to 80% of their time cleaning data. A typical Pandas script involves:
1. Deduplicating rows.
2. Dropping empty columns.
3. Guessing and converting Date strings (`M/D/YYYY`).
4. Parsing currency `$100.50` -> `100.5`.
5. Downcasting `int64` and `float64` to `int16`/`float32` to prevent Out-Of-Memory errors.
6. Converting `yes/no/t/f` to boolean `True/False`.

This code is almost identical across every project. However, it is usually fragile, un-tested, and leads to silent data corruption (e.g., falsely imputing missing customer IDs with zeros).

## The Tidely Solution

Tidely automates the exact workflow above without guessing. It uses a **Deep Semantic Engine** backed by precise Regular Expressions to identify the *business meaning* of a column.

Instead of writing 50 lines of Pandas apply/replace functions, you write:
```python
import tidely as td
import pandas as pd

df = td.clean(pd.read_csv("dirty_data.csv")).df
```

## Core Philosophies

1. **Never Silently Delete Data**: Tidely will never drop a row because it contains missing values, and it will never automatically one-hot encode targets without permission. It only removes exact duplicate rows.
2. **Local First**: Tidely runs 100% locally on your machine. No API keys, no network calls. It is completely safe for healthcare and finance data.
3. **Deterministic**: Given the exact same dirty DataFrame, Tidely will always produce the exact same clean DataFrame.
4. **Explainable**: Every action Tidely takes is documented in the `.summary()` output, explaining exactly *why* a column was transformed and how much memory was saved.
