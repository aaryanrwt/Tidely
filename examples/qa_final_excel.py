import pandas as pd
import json
import time
import os
import tidely as td

def process_excel(file_path, alias):
    print(f"\n======================================")
    print(f"Loading Excel: {alias}")
    
    try:
        df = pd.read_excel(file_path)
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
    with open(f"qa_final_{alias}_baseline.json", "w") as f:
        json.dump(report_p1, f, indent=4)
        
    # PHASE 2: Tidely Inspection
    profile = td.inspect(df)
    report_p2 = {
        "trust_score": profile.trust_score.__dict__,
        "semantic_types": profile.semantic_types,
        "diagnoses": [str(d) for d in profile.diagnoses]
    }
    with open(f"qa_final_{alias}_inspection.json", "w") as f:
        json.dump(report_p2, f, indent=4)
        
    # PHASE 3: Tidely Cleaning
    df_clean = df.copy()
    start_td = time.time()
    result = td.clean(df_clean)
    time_td = time.time() - start_td
    
    with open(f"qa_final_{alias}_cleaning.json", "w", encoding="utf-8") as f:
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
    print(f"Completed Final QA: {alias}.")

if __name__ == "__main__":
    file1 = r"C:\Users\Aaryan Rawat\Downloads\New folder\Crunchy Corner Actual - Unclean Data.xlsx"
    file2 = r"C:\Users\Aaryan Rawat\Downloads\New folder\Crunchy Corner Budget-Unclean Data.xlsx"
    
    process_excel(file1, "Actual")
    process_excel(file2, "Budget")
