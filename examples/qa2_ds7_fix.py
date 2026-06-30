import json
import time

import kagglehub
import pandas as pd

import tidely as td


def process_qa2_ds7():
    ds_id = 7
    print(f"Downloading Dataset {ds_id}: Online Retail II...")
    try:
        path = kagglehub.dataset_download(
            "mathchi/online-retail-ii-data-set-from-ml-repository"
        )
        import glob
        import os

        csv_files = glob.glob(os.path.join(path, "*.csv"))
        if not csv_files:
            raise Exception("No CSV files found")
        csv_path = csv_files[0]
        print(f"Loading CSV: {os.path.basename(csv_path)}")
        df = pd.read_csv(csv_path, encoding="latin1", low_memory=False)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    print(f"Loaded DataFrame with shape: {df.shape}")

    # PHASE 1: Baseline
    cardinality = {col: int(df[col].nunique(dropna=False)) for col in df.columns}
    report_p1 = {
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": df.isna().sum().to_dict(),
        "dtypes": {col: str(dt) for col, dt in df.dtypes.items()},
        "cardinality": cardinality,
    }
    with open(f"qa2_ds{ds_id}_baseline.json", "w") as f:
        json.dump(report_p1, f, indent=4)

    # PHASE 2: Tidely Inspection
    profile = td.inspect(df)
    report_p2 = {
        "trust_score": profile.trust_score.__dict__,
        "semantic_types": profile.semantic_types,
        "diagnoses": [str(d) for d in profile.diagnoses],
    }
    with open(f"qa2_ds{ds_id}_inspection.json", "w") as f:
        json.dump(report_p2, f, indent=4)

    # PHASE 3: Tidely Cleaning
    df_clean = df.copy()
    result = td.clean(df_clean)
    with open(f"qa2_ds{ds_id}_cleaning.json", "w") as f:
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

    print(f"DS{ds_id} Pandas Time: {time_pd:.4f}s")
    print(f"DS{ds_id} Tidely Time: {time_td:.4f}s")
    print(f"DS{ds_id} Ratio:       {time_td / max(time_pd, 0.0001):.2f}x")

    print(f"Completed QA2 DS{ds_id} Phases 1, 2, 3, 7.")


if __name__ == "__main__":
    process_qa2_ds7()
