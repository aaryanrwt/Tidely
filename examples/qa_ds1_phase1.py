import glob
import json
import os

import kagglehub
import pandas as pd


def manual_inspection():
    print("Downloading Dataset 1: Breast Cancer Wisconsin...")
    path = kagglehub.dataset_download("uciml/breast-cancer-wisconsin-data")

    csv_files = glob.glob(os.path.join(path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV found in the downloaded dataset.")

    df = pd.read_csv(csv_files[0])
    print(f"Loaded CSV: {csv_files[0]}")

    # 1. Basic properties
    num_rows = len(df)
    num_cols = len(df.columns)
    mem_size = df.memory_usage(deep=True).sum()

    # 2. Duplicates
    dup_rows = int(df.duplicated().sum())

    # Identify potential IDs (unique string or int columns)
    dup_ids = 0
    if "id" in df.columns:
        dup_ids = int(df["id"].duplicated().sum())

    # 3. Missing values
    missing_vals = df.isna().sum().to_dict()

    # 4. Data types
    dtypes = {col: str(dt) for col, dt in df.dtypes.items()}

    # 5. Cardinality
    cardinality = {col: int(df[col].nunique(dropna=False)) for col in df.columns}

    # Classify categoricals
    high_cardinality = [
        col for col, count in cardinality.items() if count > 1000 and count < num_rows
    ]
    low_cardinality = [
        col for col, count in cardinality.items() if count <= 100 and count > 2
    ]
    binary_cols = [col for col, count in cardinality.items() if count == 2]

    report = {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "memory_bytes": int(mem_size),
        "duplicate_rows": dup_rows,
        "duplicate_ids": dup_ids,
        "missing_values": missing_vals,
        "dtypes": dtypes,
        "cardinality": cardinality,
        "binary_cols": binary_cols,
        "low_card_cols": low_cardinality,
        "high_card_cols": high_cardinality,
        "csv_path": csv_files[0],
    }

    with open("qa_ds1_baseline.json", "w") as f:
        json.dump(report, f, indent=4)

    print("Saved manual baseline report to qa_ds1_baseline.json")


if __name__ == "__main__":
    manual_inspection()
