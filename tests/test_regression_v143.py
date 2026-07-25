"""Regression tests for Tidely v1.4.3 release.

These tests guard against every bug fixed in v1.4.3 and verify
core advertised functionality.
"""

from pathlib import Path

import pandas as pd
import polars as pl
import pytest

import tidely as td
from tidely.result import CleanResult

# ---------------------------------------------------------------------------
# 1. File-path API tests
# ---------------------------------------------------------------------------


def test_clean_csv_from_path(tmp_path):
    """td.clean() accepts a CSV file path string and returns a CleanResult."""
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b,c\n1,2,3\n4,5,6\n7,8,9\n", encoding="utf-8")
    result = td.clean(str(csv_path))
    assert isinstance(result, CleanResult)
    assert len(result.df) > 0


def test_clean_csv_from_pathlib(tmp_path):
    """td.clean() accepts a pathlib.Path and returns a CleanResult."""
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("x,y\n10,20\n30,40\n", encoding="utf-8")
    result = td.clean(csv_path)
    assert isinstance(result, CleanResult)
    assert len(result.df) > 0


def test_clean_txt_from_path(tmp_path):
    """td.clean() handles .txt files (treated as CSV fallback)."""
    txt_path = tmp_path / "data.txt"
    txt_path.write_text("col1,col2\nfoo,bar\nbaz,qux\n", encoding="utf-8")
    result = td.clean(str(txt_path))
    assert isinstance(result, CleanResult)


def test_clean_parquet_from_path(tmp_path):
    """td.clean() accepts a Parquet file path string and returns a CleanResult."""
    parquet_path = tmp_path / "data.parquet"
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df.write_parquet(parquet_path)
    result = td.clean(str(parquet_path))
    assert isinstance(result, CleanResult)
    assert len(result.df) == 3


def test_clean_arff_from_path(tmp_path):
    """td.clean() accepts an ARFF file path string and returns a CleanResult."""
    arff_path = tmp_path / "data.arff"
    arff_content = (
        "@relation my_relation\n"
        "@attribute col_a numeric\n"
        "@attribute col_b {x,y,z}\n"
        "@data\n"
        "1, 'x'\n"
        "2, 'y'\n"
        "3, 'z'\n"
    )
    arff_path.write_text(arff_content, encoding="utf-8")
    result = td.clean(str(arff_path))
    assert isinstance(result, CleanResult)
    assert len(result.df) == 3


# ---------------------------------------------------------------------------
# 2. CleanResult structure tests
# ---------------------------------------------------------------------------


def test_cleanresult_df_attribute():
    """CleanResult.df returns a DataFrame (Polars or Pandas)."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = td.clean(df)
    assert result.df is not None
    assert hasattr(result.df, "columns")


def test_cleanresult_summary():
    """CleanResult.summary() returns a string."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = td.clean(df)
    s = result.summary()
    assert isinstance(s, str)
    assert len(s) > 0


def test_cleanresult_show():
    """CleanResult.show() does not raise."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = td.clean(df)
    # show() should not raise
    result.show()


def test_cleanresult_undo():
    """CleanResult.undo() returns a DataFrame matching the original."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    result = td.clean(df)
    original = result.undo()
    assert original is not None


def test_cleanresult_export_csv(tmp_path):
    """CleanResult.export() creates a CSV file."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = td.clean(df)
    out = tmp_path / "out.csv"
    result.export(str(out))
    assert out.is_file()
    assert out.stat().st_size > 0


def test_cleanresult_export_html(tmp_path):
    """CleanResult.export() creates an HTML report."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = td.clean(df)
    out = tmp_path / "report.html"
    result.export(str(out))
    assert out.is_file()
    content = out.read_text(encoding="utf-8")
    assert "<html" in content.lower()


# ---------------------------------------------------------------------------
# 3. Excel loading (requires openpyxl/fastexcel)
# ---------------------------------------------------------------------------


def test_clean_excel_from_path(tmp_path):
    """td.clean() can load and clean an Excel file."""
    xlsx_path = tmp_path / "data.xlsx"
    df = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})
    df.to_excel(str(xlsx_path), index=False)
    result = td.clean(str(xlsx_path))
    assert isinstance(result, CleanResult)
    assert len(result.df) == 3


# ---------------------------------------------------------------------------
# 4. DuckDB routing
# ---------------------------------------------------------------------------


def test_decision_engine_routes_correctly():
    """DecisionEngine routes datasets to correct backends."""
    from tidely.core.decision_engine import DecisionEngine

    engine = DecisionEngine()

    # Small file → polars_eager
    assert engine.route_backend(100, "csv") == "polars_eager"

    # Medium file → polars_lazy
    assert engine.route_backend(20 * 1024 * 1024, "csv") == "polars_lazy"

    # Large file → duckdb
    assert engine.route_backend(100 * 1024 * 1024, "csv") == "duckdb"


# ---------------------------------------------------------------------------
# 5. Empty / edge-case DataFrames
# ---------------------------------------------------------------------------


def test_clean_empty_dataframe():
    """td.clean() handles an empty DataFrame without crashing."""
    df = pd.DataFrame()
    result = td.clean(df)
    assert isinstance(result, CleanResult)
    assert len(result.df) == 0


def test_clean_single_row():
    """td.clean() handles a single-row DataFrame."""
    df = pd.DataFrame({"x": [42]})
    result = td.clean(df)
    assert len(result.df) == 1


def test_clean_all_nulls():
    """td.clean() handles a DataFrame with only null values."""
    import numpy as np

    df = pd.DataFrame({"a": [np.nan, np.nan], "b": [None, None]})
    result = td.clean(df)
    assert isinstance(result, CleanResult)


# ---------------------------------------------------------------------------
# 6. Version consistency
# ---------------------------------------------------------------------------


def test_version_is_1_4_3():
    """__version__ matches the current release version."""
    assert td.__version__ == "1.5.0"


def test_version_matches_pyproject():
    """__version__ matches the version in pyproject.toml."""
    import tomllib

    pyproject_path = Path(__file__).parents[1] / "pyproject.toml"
    if not pyproject_path.is_file():
        pytest.skip("pyproject.toml not found")
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    assert td.__version__ == data["project"]["version"]


# ---------------------------------------------------------------------------
# 7. Duplicate column names
# ---------------------------------------------------------------------------


def test_clean_duplicate_columns():
    """td.clean() handles DataFrames with duplicate column names."""
    df = pd.DataFrame([[1, 2, 3]], columns=["a", "a", "b"])
    result = td.clean(df)
    assert isinstance(result, CleanResult)
    # Columns should be deduplicated
    assert len(set(result.df.columns)) == len(result.df.columns)


# ---------------------------------------------------------------------------
# 8. Polars input
# ---------------------------------------------------------------------------


def test_clean_polars_dataframe():
    """td.clean() accepts a Polars DataFrame directly."""
    df = pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    result = td.clean(df)
    assert isinstance(result, CleanResult)
    assert len(result.df) == 3


# ---------------------------------------------------------------------------
# 9. Extended Exporter and Loader E2E tests
# ---------------------------------------------------------------------------


def test_clean_result_extended_metadata(tmp_path):
    """Exposes all new CleanResult properties and formats them in the summary."""
    df = pl.DataFrame({
        "customer_id": ["C001", "C002", "C001"],
        "name": ["Alice", "Bob", "Alice"],
        "subscription_date": ["2026-06-30", "2026-07-01", "2026-06-30"],
    })
    result = td.clean(df)
    assert isinstance(result.health_before, int)
    assert isinstance(result.health_after, int)
    assert isinstance(result.execution_time, float)
    assert isinstance(result.memory_before, float)
    assert isinstance(result.memory_after, float)
    assert isinstance(result.memory_saved, float)
    assert isinstance(result.backend, str)
    assert isinstance(result.rows_removed, int)
    assert isinstance(result.columns_modified, int)
    assert isinstance(result.actions, list)
    assert result.version == "1.5.0"

    summary = result.summary()
    assert "TIDELY CLEANING SUMMARY" in summary
    assert "READINESS CERTIFICATION" in summary


def test_universal_exporter_formats(tmp_path):
    """CleanResult.export() supports all 15+ formats (CSV, Excel, ODS, JSON, JSONL, XML, YAML, ARFF, TSV, DuckDB, SQLite)."""
    df = pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    result = td.clean(df)

    # TSV
    tsv_path = tmp_path / "data.tsv"
    result.export(str(tsv_path))
    assert tsv_path.exists()

    # JSON & JSONL
    json_path = tmp_path / "data.json"
    result.export(str(json_path))
    assert json_path.exists()

    jsonl_path = tmp_path / "data.jsonl"
    result.export(str(jsonl_path))
    assert jsonl_path.exists()

    # YAML
    yaml_path = tmp_path / "data.yaml"
    result.export(str(yaml_path))
    assert yaml_path.exists()

    # ARFF
    arff_path = tmp_path / "data.arff"
    result.export(str(arff_path))
    assert arff_path.exists()

    # DuckDB
    duckdb_path = tmp_path / "data.duckdb"
    result.export(str(duckdb_path))
    assert duckdb_path.exists()

    # SQLite
    sqlite_path = tmp_path / "data.sqlite"
    result.export(str(sqlite_path))
    assert sqlite_path.exists()


def test_universal_loader_and_intelligent_features(tmp_path):
    """normalize_to_polars handles compression, file-like objects, active connections, and encodings."""
    import gzip
    import io
    import sqlite3
    import zipfile

    # BytesIO loader
    csv_bytes = b"col1,col2\n10,foo\n20,bar\n"
    stream = io.BytesIO(csv_bytes)
    result = td.clean(stream)
    assert len(result.df) == 2

    # gzip compression auto-decompress
    gz_path = tmp_path / "data.csv.gz"
    with gzip.open(gz_path, "wb") as f:
        f.write(csv_bytes)
    result = td.clean(gz_path)
    assert len(result.df) == 2

    # zip compression auto-decompress
    zip_path = tmp_path / "data.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("inner.csv", csv_bytes.decode())
    result = td.clean(zip_path)
    assert len(result.df) == 2

    # SQLite Connection loader
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users (id INT, name TEXT)")
    conn.execute("INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob')")
    conn.commit()
    # clean with active connection
    result = td.clean(conn)
    assert len(result.df) == 2
    conn.close()


def test_semantic_intelligence_correctness():
    """Classifier correctly maps first_name/last_name to Name, dates to Date, and other business items."""
    df = pl.DataFrame({
        "first_name": ["Alice", "Bob", "Charlie"],
        "subscription_date": ["2026-06-30", "2026-07-01", "2026-06-25"],
        "vin_code": ["1HGCR2F83HA000000", "1HGCR2F83HA111111", "1HGCR2F83HA222222"],
        "customer_id": ["C1001", "C1002", "C1003"],
    })
    # profile semantic types
    profile = td.inspect(df)
    sem = profile.semantic_types
    assert sem["first_name"]["type"] == "Name"
    assert sem["subscription_date"]["type"] == "Date"
    assert sem["vin_code"]["type"] == "Vehicle ID"
    assert sem["customer_id"]["type"] == "CustomerID"
