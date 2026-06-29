import pandas as pd
import re

df = pd.DataFrame({"Churn": ["Yes", "No", "Yes", "No"]})
sample_list = df["Churn"].tolist()

pattern = re.compile(r"^(yes|no|true|false|t|f|y|n|0|1)$", re.IGNORECASE)
matches = sum(1 for val in sample_list if isinstance(val, str) and pattern.match(str(val).strip()))
print(f"Matches: {matches}")
