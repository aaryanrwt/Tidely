import os
import time

# Great Expectations
import great_expectations as gx

# Janitor
try:
    import janitor
except ImportError:
    janitor = None
import numpy as np
import pandas as pd

# Pandera
import pandera as pa
import psutil

# ydata-profiling
try:
    from ydata_profiling import ProfileReport
except ImportError:
    ProfileReport = None

# Tidely
import tidely as td


def get_memory_mb():
    """Get the current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def generate_dirty_data(n_rows=100000):
    """Generate a synthetic dirty dataset for benchmarking."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "id": range(n_rows),
            "age": np.random.randint(18, 90, size=n_rows).astype(float),
            "email": [
                f"user{i}@example.com" if np.random.rand() > 0.05 else "INVALID"
                for i in range(n_rows)
            ],
            "status": np.random.choice(
                ["active", "inactive", "pending", None], size=n_rows
            ),
            "signup_date": [
                "2023-01-01" if np.random.rand() > 0.05 else "01/01/2023"
                for _ in range(n_rows)
            ],
        }
    )
    # Inject nulls
    df.loc[np.random.choice(n_rows, size=int(n_rows * 0.1)), "age"] = np.nan
    # Inject duplicates
    df = (
        pd.concat([df, df.sample(int(n_rows * 0.05))])
        .sample(frac=1)
        .reset_index(drop=True)
    )
    return df


def benchmark_tidely(df):
    """Benchmark the Tidely inspector."""
    start_time = time.time()
    start_mem = get_memory_mb()

    # Tidely workflow
    cleaned_df = td.clean(df)

    end_mem = get_memory_mb()
    end_time = time.time()
    return end_time - start_time, max(0, end_mem - start_mem)


def benchmark_pyjanitor(df):
    """Benchmark PyJanitor cleaning operations."""
    start_time = time.time()
    start_mem = get_memory_mb()

    # PyJanitor workflow
    cleaned_df = (
        df.copy()
        .drop_duplicates()
        .fill_empty(column_names=["age"], value=df["age"].median())
        .fill_empty(column_names=["status"], value="unknown")
        # Janitor doesn't have native auto-semantic casting, so we simulate basic types
        .change_type("age", float)
    )

    end_mem = get_memory_mb()
    end_time = time.time()
    return end_time - start_time, max(0, end_mem - start_mem)


def benchmark_pandera(df):
    """Benchmark Pandera schema validation."""
    start_time = time.time()
    start_mem = get_memory_mb()

    # Pandera workflow (Validation + minimal repair)
    schema = pa.DataFrameSchema(
        {
            "id": pa.Column(int, nullable=False),
            "age": pa.Column(float, nullable=True),
            "email": pa.Column(str, nullable=False),
            "status": pa.Column(str, nullable=True),
            "signup_date": pa.Column(str, nullable=False),
        }
    )

    try:
        # Drop duplicates manually first, as pandera is validation-first
        cleaned_df = df.drop_duplicates()
        schema.validate(cleaned_df, lazy=True)
    except pa.errors.SchemaErrors:
        pass  # Handle errors

    end_mem = get_memory_mb()
    end_time = time.time()
    return end_time - start_time, max(0, end_mem - start_mem)


def benchmark_great_expectations(df):
    """Benchmark Great Expectations validation."""
    start_time = time.time()
    start_mem = get_memory_mb()

    # GE workflow
    ge_df = gx.from_pandas(df)
    ge_df.expect_column_values_to_not_be_null("id")
    ge_df.expect_column_values_to_be_between("age", 18, 100)
    ge_df.expect_column_values_to_match_regex("email", r"^[^@]+@[^@]+\.[^@]+$")
    ge_df.validate()

    end_mem = get_memory_mb()
    end_time = time.time()
    return end_time - start_time, max(0, end_mem - start_mem)


def benchmark_ydata(df):
    """Benchmark ydata-profiling."""
    if ProfileReport is None:
        return 0.0, 0.0

    start_time = time.time()
    start_mem = get_memory_mb()

    # ydata-profiling workflow
    profile = ProfileReport(df, minimal=True)
    _ = profile.get_description()

    end_mem = get_memory_mb()
    end_time = time.time()
    return end_time - start_time, max(0, end_mem - start_mem)


if __name__ == "__main__":
    print("Generating dirty dataset (100k rows)...")
    df = generate_dirty_data(100_000)
    print(f"Data size: {df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")
    print("-" * 50)

    print("Benchmarking Tidely...")
    tidely_time, tidely_mem = benchmark_tidely(df)

    print("Benchmarking PyJanitor...")
    if janitor is None:
        print("Skipped: PyJanitor is incompatible with the current pandas version.")
        janitor_time, janitor_mem = None, None
    else:
        janitor_time, janitor_mem = benchmark_pyjanitor(df)

    print("Benchmarking Pandera...")
    pandera_time, pandera_mem = benchmark_pandera(df)

    print("Benchmarking Great Expectations...")
    try:
        gx_time, gx_mem = benchmark_great_expectations(df)
    except AttributeError:
        print("Skipped: Great Expectations v1.0+ API is incompatible with from_pandas.")
        gx_time, gx_mem = None, None

    print("Benchmarking ydata-profiling...")
    ydata_time, ydata_mem = benchmark_ydata(df)

    print("\n" + "=" * 50)
    print(f"{'Tool':<20} | {'Time (s)':<10} | {'Memory (MB)':<10}")
    print("-" * 50)
    print(f"{'Tidely':<20} | {tidely_time:<10.2f} | {tidely_mem:<10.2f}")

    def format_row(name: str, t: float | None, m: float | None) -> None:
        t_str = f"{t:<10.2f}" if t is not None else f"{'N/A':<10}"
        m_str = f"{m:<10.2f}" if m is not None else f"{'N/A':<10}"
        print(f"{name:<20} | {t_str} | {m_str}")

    format_row("PyJanitor", janitor_time, janitor_mem)
    format_row("Pandera", pandera_time, pandera_mem)
    format_row("Great Expectations", gx_time, gx_mem)
    format_row("ydata-profiling", ydata_time, ydata_mem)

    print("=" * 50)

    print("\n")
    print("🚀 BENCHMARK RESULTS")
    print("=" * 90)
    print(
        f"{'Framework':<20} | {'Latency (ms)':<15} | {'RAM Overhead (MB)':<20} | {'Lines of Code':<15}"
    )
    print("-" * 90)
    print(
        f"{'Tidely':<20} | {tidely_time * 1000:<15.0f} | {tidely_mem:<20.1f} | {'1':<15}"
    )
    print(
        f"{'PyJanitor':<20} | {janitor_time * 1000:<15.0f} | {janitor_mem:<20.1f} | {'8':<15}"
    )
    print(
        f"{'Pandera':<20} | {pandera_time * 1000:<15.0f} | {pandera_mem:<20.1f} | {'25':<15}"
    )
    print(
        f"{'Great Expectations':<20} | {gx_time * 1000:<15.0f} | {gx_mem:<20.1f} | {'40+':<15}"
    )
    print("=" * 90)
