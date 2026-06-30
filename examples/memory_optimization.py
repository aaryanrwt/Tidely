"""Tidely Example 3: Memory Optimization
This script demonstrates how Tidely automatically downcasts types
and compresses low-cardinality strings to save memory.
"""

import numpy as np
import pandas as pd

import tidely as td

# Generate a heavily bloated DataFrame
n_rows = 1_000_000
df = pd.DataFrame(
    {
        # Int64 by default, but only needs Int8 (0-100)
        "age": np.random.randint(0, 100, size=n_rows),
        # Float64 by default, but only needs Float32
        "price": np.random.uniform(10.0, 50.0, size=n_rows),
        # High memory strings, low cardinality
        "country": np.random.choice(["USA", "UK", "Canada", "India"], size=n_rows),
    }
)

print(f"Memory Before Tidely: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")

# Clean and Optimize
result = td.clean(df)

print(f"Memory After Tidely: {result.df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
print("\nLook at the Summary to see exactly what downcasted:")
print(result.summary())
