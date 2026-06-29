"""
Tidely Example 4: Semantic Detection
This script demonstrates how Tidely infers the business meaning 
of text columns (like Emails, Currencies, and Dates).
"""

import pandas as pd
import tidely as td

df = pd.DataFrame({
    # Non-standard US Dates
    "signup_date": ["12/1/2020 8:26", "1/15/2021 14:00", "05/05/2022 09:30"],
    # Currencies with symbols
    "revenue": ["$1,050.50", "$2,000.00", "$500"],
    # Dirty Booleans
    "is_active": ["yes", "T", "0"]
})

# Run the Inspector to profile the semantic types
profile = td.inspect(df)

print("Semantic Intelligence Detected:")
for col, metadata in profile.semantic_types.items():
    print(f"- {col}: {metadata['type']} (Confidence: {metadata['match_rate']*100:.1f}%)")

# Let Tidely automatically cast these columns to strict primitives
clean_df = td.clean(df).df
print("\nStructurally cleaned DataFrame types:")
print(clean_df.dtypes)
