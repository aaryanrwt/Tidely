import time

import numpy as np
import pandas as pd

import tidely as td


def generate_large_dataset(rows=1_000_000):
    """Generates a large dataset for benchmarking."""
    return pd.DataFrame({
        "id": np.random.randint(0, 1000000, size=rows),
        "name": [" User \u200b "] * rows,
        "email": ["USER@DOMAIN.COM"] * rows,
        "is_active": (["Yes", "No", "True", "False", "t", "f"] * (rows // 6 + 1))[
            :rows
        ],
        "category": (["A", "B", "C", "D"] * (rows // 4 + 1))[:rows],
        "val": np.random.randn(rows),
    })


def benchmark_pandas(df):
    """Manual pandas cleaning pipeline."""
    start = time.time()

    # 1. Deduplicate
    df = df.drop_duplicates()

    # 2. Clean Strings
    df["name"] = df["name"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df["name"] = df["name"].str.replace(r"[^\x20-\x7E]", "", regex=True)

    # 3. Clean Emails
    df["email"] = df["email"].str.lower().str.strip()

    # 4. Clean Booleans
    df["is_active"] = (
        df["is_active"]
        .astype(str)
        .str.lower()
        .map({
            "true": True,
            "yes": True,
            "1": True,
            "y": True,
            "t": True,
            "false": False,
            "no": False,
            "0": False,
            "n": False,
            "f": False,
        })
        .fillna(df["is_active"])
    )

    # 5. Optimize memory
    df["category"] = df["category"].astype("category")

    end = time.time()
    return end - start


def benchmark_tidely(df):
    """Tidely automatic cleaning pipeline."""
    start = time.time()
    result = td.clean(df)
    end = time.time()
    return end - start


if __name__ == "__main__":
    print("Generating dataset (1M rows)...")
    df = generate_large_dataset(1_000_000)

    print("Benchmarking Pandas...")
    df_pandas = df.copy()
    pd_time = benchmark_pandas(df_pandas)

    print("Benchmarking Tidely...")
    df_tidely = df.copy()
    td_time = benchmark_tidely(df_tidely)

    print("-" * 30)
    print(f"Pandas Manual Time : {pd_time:.2f}s")
    print(f"Tidely Auto Time   : {td_time:.2f}s")
    print(f"Overhead Ratio     : {td_time / pd_time:.2f}x")

    if td_time < pd_time * 1.5:
        print("[SUCCESS] Performance is acceptable (< 1.5x manual pandas).")
    else:
        print("[FAIL] Tidely overhead is too high!")
