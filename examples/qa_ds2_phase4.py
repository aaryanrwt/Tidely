import pandas as pd

import tidely as td

df = pd.read_csv("qa_ds2_temp.csv")

result = td.clean(df)

print(f"Max Price Before: {df['price'].max()}")
print(f"Max Price After: {result.df['price'].max()}")
print(f"Duplicates Before: {df.duplicated().sum()}")
print(f"Duplicates After: {result.df.duplicated().sum()}")
print(f"Rows Before: {len(df)}")
print(f"Rows After: {len(result.df)}")
