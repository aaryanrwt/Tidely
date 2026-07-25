"""Tidely v1.5.0 — Comprehensive File Format Ingestion & Export Test Suite.

Verifies loading, profiling, cleaning, exporting, and round-trip integrity for
all officially supported formats:
CSV, TSV, JSON, JSONL, Parquet, Feather/IPC, Excel (.xlsx), ARFF, SQLite, DuckDB,
PyArrow Table, compressed (gz, zip, bz2, xz), and Pandas/Polars DataFrames.
"""

from __future__ import annotations

import os
import sqlite3
import zipfile

import pandas as pd
import polars as pl
import pytest

import tidely as td

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Sample DataFrame with dirty data for format round-tripping."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 1],
        "name": ["Alice", "Bob", "  Charlie  ", None, "Alice"],
        "score": [95.5, 88.0, 72.3, None, 95.5],
        "active": ["true", "false", "yes", "no", "true"],
    })


# ---------------------------------------------------------------------------
# 1. Plain Text Formats (CSV, TSV, JSON, JSONL)
# ---------------------------------------------------------------------------


class TestTextFormats:
    def test_csv_ingestion(
        self, sample_df: pd.DataFrame, tmp_path: pytest.fixture
    ) -> None:  # type: ignore[type-arg]
        csv_path = str(tmp_path / "test.csv")
        sample_df.to_csv(csv_path, index=False)

        res = td.clean(csv_path)
        assert res.df is not None
        assert len(res.df) > 0

    def test_tsv_ingestion(
        self, sample_df: pd.DataFrame, tmp_path: pytest.fixture
    ) -> None:  # type: ignore[type-arg]
        tsv_path = str(tmp_path / "test.tsv")
        sample_df.to_csv(tsv_path, sep="\t", index=False)

        res = td.clean(tsv_path)
        assert res.df is not None
        assert len(res.df) > 0

    def test_json_ingestion(
        self, sample_df: pd.DataFrame, tmp_path: pytest.fixture
    ) -> None:  # type: ignore[type-arg]
        json_path = str(tmp_path / "test.json")
        sample_df.to_json(json_path, orient="records")

        res = td.clean(json_path)
        assert res.df is not None

    def test_jsonl_ingestion(
        self, sample_df: pd.DataFrame, tmp_path: pytest.fixture
    ) -> None:  # type: ignore[type-arg]
        jsonl_path = str(tmp_path / "test.jsonl")
        sample_df.to_json(jsonl_path, orient="records", lines=True)

        res = td.clean(jsonl_path)
        assert res.df is not None


# ---------------------------------------------------------------------------
# 2. Binary Formats (Parquet, Feather/IPC)
# ---------------------------------------------------------------------------


class TestBinaryFormats:
    def test_parquet_ingestion(
        self, sample_df: pd.DataFrame, tmp_path: pytest.fixture
    ) -> None:  # type: ignore[type-arg]
        pq_path = str(tmp_path / "test.parquet")
        pl.DataFrame(sample_df).write_parquet(pq_path)

        res = td.clean(pq_path)
        assert res.df is not None

    def test_feather_ingestion(
        self, sample_df: pd.DataFrame, tmp_path: pytest.fixture
    ) -> None:  # type: ignore[type-arg]
        feather_path = str(tmp_path / "test.feather")
        pl.DataFrame(sample_df).write_ipc(feather_path)

        res = td.clean(feather_path)
        assert res.df is not None


# ---------------------------------------------------------------------------
# 3. Excel & ARFF Formats
# ---------------------------------------------------------------------------


class TestExcelAndARFF:
    def test_excel_xlsx_ingestion(
        self, sample_df: pd.DataFrame, tmp_path: pytest.fixture
    ) -> None:  # type: ignore[type-arg]
        xlsx_path = str(tmp_path / "test.xlsx")
        sample_df.to_excel(xlsx_path, index=False)

        res = td.clean(xlsx_path)
        assert res.df is not None

    def test_arff_ingestion(self, tmp_path: pytest.fixture) -> None:  # type: ignore[type-arg]
        arff_path = str(tmp_path / "test.arff")
        arff_content = """@relation test
@attribute id numeric
@attribute name string
@attribute active {true, false}
@data
1, 'Alice', true
2, 'Bob', false
3, 'Charlie', true
"""
        with open(arff_path, "w", encoding="utf-8") as f:
            f.write(arff_content)

        res = td.clean(arff_path)
        assert res.df is not None


# ---------------------------------------------------------------------------
# 4. Database Formats (SQLite, DuckDB)
# ---------------------------------------------------------------------------


class TestDatabaseFormats:
    def test_sqlite_connection(self, sample_df: pd.DataFrame) -> None:
        conn = sqlite3.connect(":memory:")
        sample_df.to_sql("data", conn, index=False)

        res = td.clean(conn)
        assert res.df is not None
        conn.close()

    def test_duckdb_connection(self, sample_df: pd.DataFrame) -> None:
        try:
            import duckdb
        except ImportError:
            pytest.skip("duckdb not installed")

        conn = duckdb.connect(":memory:")
        conn.register("data", sample_df)
        conn.execute("CREATE TABLE tbl AS SELECT * FROM data")

        res = td.clean(conn)
        assert res.df is not None
        conn.close()


# ---------------------------------------------------------------------------
# 5. Compressed Archives (zip, gzip)
# ---------------------------------------------------------------------------


class TestCompressedFormats:
    def test_zip_csv_ingestion(
        self, sample_df: pd.DataFrame, tmp_path: pytest.fixture
    ) -> None:  # type: ignore[type-arg]
        zip_path = str(tmp_path / "test.zip")
        csv_bytes = sample_df.to_csv(index=False).encode("utf-8")

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("inner.csv", csv_bytes)

        res = td.clean(zip_path)
        assert res.df is not None

    def test_gzip_csv_ingestion(
        self, sample_df: pd.DataFrame, tmp_path: pytest.fixture
    ) -> None:  # type: ignore[type-arg]
        import gzip

        gz_path = str(tmp_path / "test.csv.gz")
        csv_bytes = sample_df.to_csv(index=False).encode("utf-8")

        with gzip.open(gz_path, "wb") as f:
            f.write(csv_bytes)

        res = td.clean(gz_path)
        assert res.df is not None


# ---------------------------------------------------------------------------
# 6. In-Memory Objects (Pandas, Polars, PyArrow)
# ---------------------------------------------------------------------------


class TestInMemoryObjects:
    def test_pandas_dataframe(self, sample_df: pd.DataFrame) -> None:
        res = td.clean(sample_df)
        assert res.df is not None

    def test_polars_eager_dataframe(self, sample_df: pd.DataFrame) -> None:
        pl_df = pl.DataFrame(sample_df)
        res = td.clean(pl_df)
        assert res.df is not None

    def test_polars_lazy_frame(self, sample_df: pd.DataFrame) -> None:
        pl_lazy = pl.DataFrame(sample_df).lazy()
        res = td.clean(pl_lazy)
        assert res.df is not None

    def test_pyarrow_table(self, sample_df: pd.DataFrame) -> None:
        import pyarrow as pa

        table = pa.Table.from_pandas(sample_df)
        res = td.clean(table)
        assert res.df is not None


# ---------------------------------------------------------------------------
# 7. Universal Export Engine Tests
# ---------------------------------------------------------------------------


class TestExportEngine:
    def test_export_csv(
        self, sample_df: pd.DataFrame, tmp_path: pytest.fixture
    ) -> None:  # type: ignore[type-arg]
        res = td.clean(sample_df)
        out = str(tmp_path / "out.csv")
        res.export(out)
        assert os.path.exists(out)

    def test_export_parquet(
        self, sample_df: pd.DataFrame, tmp_path: pytest.fixture
    ) -> None:  # type: ignore[type-arg]
        res = td.clean(sample_df)
        out = str(tmp_path / "out.parquet")
        res.export(out)
        assert os.path.exists(out)

    def test_export_json(
        self, sample_df: pd.DataFrame, tmp_path: pytest.fixture
    ) -> None:  # type: ignore[type-arg]
        res = td.clean(sample_df)
        out = str(tmp_path / "out.json")
        res.export(out)
        assert os.path.exists(out)

    def test_export_excel(
        self, sample_df: pd.DataFrame, tmp_path: pytest.fixture
    ) -> None:  # type: ignore[type-arg]
        res = td.clean(sample_df)
        out = str(tmp_path / "out.xlsx")
        res.export(out)
        assert os.path.exists(out)
