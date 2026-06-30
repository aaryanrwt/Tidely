"""Tidely Example 1: Basic CSV Cleaning
This script demonstrates the simplest way to clean a Pandas DataFrame.
"""

import pandas as pd

import tidely as td

# 1. Load your dirty dataset using Pandas
df = pd.DataFrame(
    {
        "id": [1, 2, 2, 3],  # Includes a duplicate row
        "email": [
            "test@test.com",
            "admin@domain.org",
            "admin@domain.org",
            "invalid_email",
        ],
        "age": [25, 30, 30, None],
    }
)

print("Before Tidely:")
print(df)

# 2. Clean the dataset automatically
result = td.clean(df)

# 3. Retrieve the clean dataset
clean_df = result.df

print("\nAfter Tidely:")
print(clean_df)

# 4. View exactly what changed
print("\nCleaning Summary:")
print(result.summary())
