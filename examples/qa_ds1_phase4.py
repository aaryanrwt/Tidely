import pandas as pd

import tidely as td

df = pd.read_csv(
    "C:\\Users\\Aaryan Rawat\\.cache\\kagglehub\\datasets\\uciml\\breast-cancer-wisconsin-data\\versions\\2\\data.csv"
)

result = td.clean(df)

print(f"Max ID Before: {df['id'].max()}")
print(f"Max ID After: {result.df['id'].max()}")
print(f"Duplicates Before: {df.duplicated().sum()}")
print(f"Duplicates After: {result.df.duplicated().sum()}")
print(f"Rows Before: {len(df)}")
print(f"Rows After: {len(result.df)}")
