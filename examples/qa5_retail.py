import pandas as pd
import json
import time
import os
import tidely as td
from ucimlrepo import fetch_ucirepo

def process_dataset():
    print(f"\n======================================")
    print(f"Loading UCI Dataset 352 (Online Retail)...")
    
    try:
        online_retail = fetch_ucirepo(id=352)
        df = online_retail.data.original
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    print(f"Loaded DataFrame with shape: {df.shape}")
    
    # PHASE 1: Baseline
    cardinality = {str(col): int(df[col].nunique(dropna=False)) for col in df.columns}
    report_p1 = {
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": df.isna().sum().to_dict(),
        "dtypes": {str(col): str(dt) for col, dt in df.dtypes.items()},
        "cardinality": cardinality
    }
    with open("qa5_retail_baseline.json", "w") as f:
        json.dump(report_p1, f, indent=4)
        
    # PHASE 2: Tidely Inspection
    profile = td.inspect(df)
    report_p2 = {
        "trust_score": profile.trust_score.__dict__,
        "semantic_types": profile.semantic_types,
        "diagnoses": [str(d) for d in profile.diagnoses]
    }
    with open("qa5_retail_inspection.json", "w") as f:
        json.dump(report_p2, f, indent=4)
        
    # PHASE 3: Tidely Cleaning
    df_clean = df.copy()
    start_td = time.time()
    result = td.clean(df_clean)
    time_td = time.time() - start_td
    
    with open("qa5_retail_cleaning.json", "w", encoding="utf-8") as f:
        json.dump({"summary": result.summary()}, f, indent=4)
        
    # PHASE 7: Benchmark
    start_pd = time.time()
    df_pd = df.copy().drop_duplicates()
    for col in df_pd.columns:
        if df_pd[col].dtype == "object":
            if df_pd[col].nunique() < len(df_pd) * 0.05:
                df_pd[col] = df_pd[col].astype("category")
    time_pd = time.time() - start_pd
    
    print(f"Pandas Time: {time_pd:.4f}s")
    print(f"Tidely Time: {time_td:.4f}s")
    print(f"Ratio:       {time_td / max(time_pd, 0.0001):.2f}x")
    print(f"Completed QA5 Retail Phases 1, 2, 3, 7.")

if __name__ == "__main__":
    process_dataset()
