import time
import pandas as pd
import tidely as td

def benchmark():
    df = pd.read_csv("C:\\Users\\Aaryan Rawat\\.cache\\kagglehub\\datasets\\mirzahasnine\\loan-data-set\\versions\\1\\loan_train.csv")
    
    # 1. Pandas baseline
    df_pd = df.copy()
    start_pd = time.time()
    df_pd = df_pd.drop_duplicates()
    for col in df_pd.columns:
        if df_pd[col].dtype == "float64":
            df_pd[col] = pd.to_numeric(df_pd[col], downcast="float")
        elif df_pd[col].dtype == "int64":
            df_pd[col] = pd.to_numeric(df_pd[col], downcast="integer")
        elif df_pd[col].dtype == "object":
            if df_pd[col].nunique() < len(df_pd) * 0.05:
                df_pd[col] = df_pd[col].astype("category")
    time_pd = time.time() - start_pd
    
    # 2. Tidely
    df_td = df.copy()
    start_td = time.time()
    res = td.clean(df_td)
    time_td = time.time() - start_td
    
    print(f"Pandas Manual Time: {time_pd:.4f}s")
    print(f"Tidely Auto Time:   {time_td:.4f}s")
    print(f"Ratio:              {time_td / max(time_pd, 0.0001):.2f}x")

if __name__ == "__main__":
    benchmark()
