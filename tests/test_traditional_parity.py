import numpy as np
import pandas as pd
import polars as pl
import pytest
import tidely as td
from tidely.core.plan import plan
from tests.traditional_clean import clean_traditional, audit_parities

def test_traditional_parity_numeric_outliers():
    """Validates outlier clipping and missing value imputation parity on a numeric dataset."""
    # Generate dirty data
    np.random.seed(42)
    rows = 200
    df_raw = pd.DataFrame({
        "id": [f"ID_{i}" for i in range(rows)],
        "target": np.random.choice([0.0, 1.0], size=rows),
        "continuous_normal": np.random.normal(loc=10.0, scale=2.0, size=rows),
        "continuous_skewed": np.random.exponential(scale=5.0, size=rows),
        "categorical_int": np.random.choice([1, 2, 3, 4], size=rows),
        "boolean_col": np.random.choice(["true", "false"], size=rows),
    })

    # Inject outliers in continuous columns
    df_raw.loc[10, "continuous_normal"] = 100.0
    df_raw.loc[50, "continuous_skewed"] = 500.0

    # Inject some nulls (only in continuous/categorical, not in id/target)
    df_raw.loc[20:30, "continuous_normal"] = np.nan
    df_raw.loc[40:45, "categorical_int"] = np.nan

    # Plan with Tidely
    p = plan(df_raw)
    
    # Run Tidely execution
    df_tidely = p.execute()
    
    # Run Traditional Pandas/NumPy execution
    df_trad = clean_traditional(df_raw, p)
    
    # Audit Parities
    report = audit_parities(df_raw, df_trad, df_tidely)

    assert report["shape_parity"], f"Shapes do not match: {report['shapes']}"
    assert report["all_identical"], f"Traditional clean and Tidely clean have value differences: {report['columns']}"
    assert not report["type_narrowing_overflows"], f"Type narrowing overflow detected on columns: {report['type_narrowing_overflows']}"

    # Verify protected columns (id, target) were NOT modified at the value level
    assert len(df_tidely) == rows
    pd.testing.assert_series_equal(df_tidely["id"], df_raw["id"])
    pd.testing.assert_series_equal(df_tidely["target"], df_raw["target"])


def test_traditional_parity_semantic_and_text():
    """Validates semantic normalization and text cleaning parity."""
    df_raw = pd.DataFrame({
        "customer_id": [f"CUST_{i}" for i in range(100)],
        "email_col": ["  user@Example.com  " if i % 10 == 0 else f"test_{i}@domain.com" for i in range(100)],
        "phone_col": ["+1 (555) 019-2834" if i % 10 == 0 else f"55501900{i:02d}" for i in range(100)],
        "zip_col": ["9021" if i % 10 == 0 else f"1234{i%10}" for i in range(100)],
        "lat_col": [120.0 if i % 10 == 0 else 45.0 for i in range(100)],
        "lon_col": [-200.0 if i % 10 == 0 else -120.0 for i in range(100)],
        "text_col": ["Raw  Text\u200B with\tspaces  " for i in range(100)],
    })

    p = plan(df_raw)
    df_tidely = p.execute()
    df_trad = clean_traditional(df_raw, p)
    
    report = audit_parities(df_raw, df_trad, df_tidely)

    assert report["shape_parity"]
    assert report["all_identical"]

    # Verify customer_id was NOT modified
    pd.testing.assert_series_equal(df_tidely["customer_id"].astype(str), df_raw["customer_id"])


def test_type_narrowing_safety():
    """Verifies that integer downcasting does not cause overflows or truncation."""
    df_raw = pd.DataFrame({
        "small_ints": list(range(-50, 50)),
        "large_ints": [2000000000 + i for i in range(100)],
    })
    
    p = plan(df_raw)
    df_tidely = p.execute()
    df_trad = clean_traditional(df_raw, p)
    
    report = audit_parities(df_raw, df_trad, df_tidely)
    
    assert report["shape_parity"]
    assert report["all_identical"]
    assert not report["type_narrowing_overflows"]
