from kagglehub import KaggleDatasetAdapter
import kagglehub
import pandas as pd
import json
import tidely as td

def process_ds3():
    print("Downloading Dataset 3: Loan Dataset...")
    try:
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "mirzahasnine/loan-data-set",
            "loan_train.csv"
        )
    except Exception:
        path = kagglehub.dataset_download("mirzahasnine/loan-data-set")
        import glob, os
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
        "cardinality": cardinality
    }
    with open("qa_ds3_baseline.json", "w") as f:
        json.dump(report_p1, f, indent=4)
        
    # PHASE 2: Tidely Inspection
    profile = td.inspect(df)
    report_p2 = {
        "trust_score": profile.trust_score.__dict__,
        "semantic_types": profile.semantic_types
    }
    with open("qa_ds3_inspection.json", "w") as f:
        json.dump(report_p2, f, indent=4)
        
    # PHASE 3: Tidely Cleaning
    df_clean = df.copy()
    result = td.clean(df_clean)
    with open("qa_ds3_cleaning.json", "w") as f:
        json.dump({"summary": result.summary()}, f, indent=4)
        
    print("Completed DS3 Phase 1, 2, and 3.")
    
if __name__ == "__main__":
    process_ds3()
