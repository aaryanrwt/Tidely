import pandas as pd
import numpy as np
import tidely as td
import time
import os
import psutil

def get_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def run_scale(n_rows):
    print(f"\n--- Testing Massive Scale: {n_rows:,} rows ---")
    
    np.random.seed(42)
    # 5 columns: 2 categorical, 1 numeric int, 1 numeric float, 1 boolean
    data = {
        "cat1": np.random.choice(["A", "B", "C", "D"], n_rows),
        "cat2": np.random.choice(["X", "Y", "Z"], n_rows),
        "num_int": np.random.randint(0, 1000, n_rows),
        "num_float": np.random.randn(n_rows) * 1000,
        "bool_val": np.random.choice([True, False, None], n_rows)
    }
    
    print("Generating DataFrame...")
    df = pd.DataFrame(data)
    
    mem_before = get_memory()
    print(f"Memory Before tidely: {mem_before:.2f} MB")
    
    start = time.time()
    res = td.clean(df)
    duration = time.time() - start
    
    mem_after = get_memory()
    print(f"Memory After tidely: {mem_after:.2f} MB")
    
    print(f"Tidely execution time: {duration:.4f} seconds")
    print(f"Throughput: {n_rows/duration:,.0f} rows/second")
    # Replace unicode characters for Windows console
    text = res.summary().replace("→", "->").replace("✓", "V").replace("•", "-")
    print(text)

if __name__ == "__main__":
    for size in [1_000_000, 5_000_000, 10_000_000]:
        run_scale(size)
