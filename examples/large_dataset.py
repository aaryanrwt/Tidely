"""Tidely Example 6: Large Datasets (Polars Native)
This script demonstrates how Tidely natively wraps Polars for hyper-fast
processing of large datasets that bypass the Pandas GIL.
"""

import polars as pl

import tidely as td

# Generate a large dataset natively in Polars
# (e.g. reading a 10-million row Parquet file)
df = pl.DataFrame({"id": range(1_000_000), "status": ["active", "inactive"] * 500_000})

print("Profiling massive Polars dataset...")

# Tidely detects Polars DataFrames and runs in 'polars_eager' format
result = td.clean(df)

# The returned DataFrame is natively preserved as a Polars object!
print(f"Returned object type: {type(result.df)}")
print(result.summary())
