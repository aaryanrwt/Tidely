import json
import time

import kagglehub
import pandas as pd
from kagglehub import KaggleDatasetAdapter

import tidely as td


def process_ds4():
    print("Downloading Dataset 4: Car Evaluation...")
    try:
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "elikplim/car-evaluation-data-set",
            "car_evaluation.csv",
        )
    except Exception:
        path = kagglehub.dataset_download("elikplim/car-evaluation-data-set")
        import glob
        import os

        csv_files = glob.glob(os.path.join(path, "*.csv"))
        df = pd.read_csv(csv_files[0])

    print(f"Loaded DataFrame with shape: {df.shape}")

    # PHASE 1: Baseline
    cardinality = {col: int(df[col].nunique(dropna=False)) for col in df.columns}
    report_p1 = {
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": df.isna().sum().to_dict(),
        "cardinality": cardinality,
    }
    with open("qa_ds4_baseline.json", "w") as f:
        json.dump(report_p1, f, indent=4)

    # PHASE 2: Tidely Inspection
    profile = td.inspect(df)
    report_p2 = {
        "trust_score": profile.trust_score.__dict__,
        "semantic_types": profile.semantic_types,
    }
    with open("qa_ds4_inspection.json", "w") as f:
        json.dump(report_p2, f, indent=4)

    # PHASE 3: Tidely Cleaning
    df_clean = df.copy()
    result = td.clean(df_clean)
    with open("qa_ds4_cleaning.json", "w") as f:
        json.dump({"summary": result.summary()}, f, indent=4)

    # PHASE 7: Benchmark
    start_pd = time.time()
    df_pd = df.copy().drop_duplicates()
    for col in df_pd.columns:
        if df_pd[col].dtype == "object":
            if df_pd[col].nunique() < len(df_pd) * 0.05:
                df_pd[col] = df_pd[col].astype("category")
    time_pd = time.time() - start_pd

    start_td = time.time()
    res = td.clean(df.copy())
    time_td = time.time() - start_td

    print(f"DS4 Pandas Time: {time_pd:.4f}s")
    print(f"DS4 Tidely Time: {time_td:.4f}s")
    print(f"DS4 Ratio:       {time_td / max(time_pd, 0.0001):.2f}x")

    print("Completed DS4 Phase 1, 2, 3, 7.")


if __name__ == "__main__":
    process_ds4()
