"""Tidely Example 5: Post-Cleaning Validation
This script demonstrates how to run strict schema validation
after cleaning to ensure no anomalies slipped through.
"""

import pandas as pd

import tidely as td

df = pd.DataFrame(
    {
        "user_id": [1, 2, 3],
        "is_active": ["yes", "no", "yes"],
        "score": ["99.5", "80.0", "75.5"],
    }
)

# 1. Clean the dataset
clean_df = td.clean(df).df

# 2. Define your strict downstream schema
# This represents what your ML model or Dashboard expects.
schema = {"user_id": "int", "is_active": "bool", "score": "float"}

# 3. Validate
# If the clean_df does not match this schema, a TidelyValidationError is raised.
td.validate(clean_df, schema)
print("Dataset successfully validated against the schema!")
