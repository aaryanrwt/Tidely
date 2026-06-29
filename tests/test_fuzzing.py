import pytest
import pandas as pd
import numpy as np
import tidely as td

def test_fuzz_i18n():
    """Inject random toxic strings, unicode, and multi-language text."""
    toxic_strings = [
        "こんにちは", # Japanese
        "你好", # Chinese
        "مرحبا", # Arabic
        "नमस्ते", # Hindi
        "Привет", # Cyrillic
        "Español", # Accents
        "Café",
        "über",
        "👨‍👩‍👧‍👦", # Emojis
        "🔥🚀",
        "\x00\x01\x02", # Invisible bytes
        " ",
        "",
        np.nan,
        None,
        "NaN",
        "NULL",
        "\"nested\" 'quotes'",
        "a" * 10000, # Huge string
    ]
    
    # Generate 1000 random rows by sampling
    np.random.seed(42)
    df = pd.DataFrame({
        "toxic_col_1": np.random.choice(toxic_strings, 1000),
        "toxic_col_2": np.random.choice(toxic_strings, 1000),
        "numeric_mixed": np.random.choice([1, 2, "1", "2", np.nan, "NaN"], 1000),
        "date_mixed": np.random.choice(["2020-01-01", "01/01/2020", "invalid", None, "2021-02-30"], 1000)
    })
    
    try:
        res = td.clean(df)
        assert res is not None
        # Verify it didn't crash
    except Exception as e:
        pytest.fail(f"Tidely crashed on toxic i18n input: {e}")

def test_fuzz_empty_df():
    """Test completely empty DataFrames."""
    df = pd.DataFrame()
    res = td.clean(df)
    assert len(res.df) == 0

def test_fuzz_all_nulls():
    """Test DataFrame with only NULLs."""
    df = pd.DataFrame({
        "a": [np.nan, np.nan, np.nan],
        "b": [None, None, None]
    })
    res = td.clean(df)
    assert len(res.df.columns) == 2

def test_fuzz_single_row():
    """Test DataFrame with a single row."""
    df = pd.DataFrame({"col": ["value"]})
    res = td.clean(df)
    assert len(res.df) == 1

def test_fuzz_duplicate_columns():
    """Test DataFrame with duplicate column names."""
    df = pd.DataFrame([[1, 2, 3]], columns=["a", "a", "b"])
    res = td.clean(df)
    assert res is not None
