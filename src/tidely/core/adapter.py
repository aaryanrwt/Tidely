"""Adapters to normalize inputs (Pandas, Polars, Arrow) to Polars DataFrames/LazyFrames."""

from typing import Any, cast

import polars as pl

from tidely.core.errors import TidelyDataError

# Handle pandas import optionally to prevent failure if pandas isn't installed
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import pyarrow as pa
except ImportError:
    pa = cast(Any, None)

try:
    import duckdb
except ImportError:
    duckdb = cast(Any, None)


def estimate_dataset_size(data: Any) -> int:
    """Estimates the size of the dataset in bytes without loading it into RAM."""
    import os

    if isinstance(data, str):
        if os.path.exists(data):
            return int(os.path.getsize(data))
    elif hasattr(data, "estimated_size"):
        try:
            return int(data.estimated_size())
        except Exception:
            pass
    elif hasattr(data, "memory_usage"):
        try:
            return int(data.memory_usage(deep=True).sum())
        except Exception:
            try:
                return int(data.memory_usage().sum())
            except Exception:
                pass
    elif hasattr(data, "nbytes"):
        try:
            return int(data.nbytes)
        except Exception:
            pass
    return 10 * 1024 * 1024  # Default to 10MB fallback


def decompress_bytes(content: bytes) -> tuple[bytes, str | None]:
    """Auto-detects and decompresses zip, gz, bz2, and xz archives from bytes."""
    import bz2
    import gzip
    import io
    import lzma
    import os
    import zipfile

    if content.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                names = [n for n in z.namelist() if not n.endswith("/")]
                if names:
                    with z.open(names[0]) as f:
                        return f.read(), os.path.splitext(names[0])[1].lower()
        except Exception:
            pass
    elif content.startswith(b"\x1f\x8b"):
        try:
            return gzip.decompress(content), None
        except Exception:
            pass
    elif content.startswith(b"BZh"):
        try:
            return bz2.decompress(content), None
        except Exception:
            pass
    elif content.startswith(b"\xfd7zXZ\x00"):
        try:
            return lzma.decompress(content), None
        except Exception:
            pass
    return content, None


def detect_encoding(content: bytes) -> str:
    """Tries decoding common encodings to find one that works without raising UnicodeDecodeError."""
    for enc in ("utf-8", "latin-1", "utf-16", "cp1252", "ascii"):
        try:
            content.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "utf-8"


def detect_delimiter(text: str) -> str:
    """Intelligently detects delimiters by counting occurrences in the first few lines."""
    lines = [line.strip() for line in text.splitlines()[:5] if line.strip()]
    if not lines:
        return ","
    delims = [",", "\t", ";", "|"]
    counts = dict.fromkeys(delims, 0)
    for line in lines:
        for d in delims:
            counts[d] += line.count(d)
    best_delim = max(counts, key=lambda k: counts[k])
    if counts[best_delim] > 0:
        return best_delim
    return ","


def normalize_to_polars(
    data: Any,
) -> tuple[pl.DataFrame | pl.LazyFrame, str]:
    """Normalizes filepath strings, Pandas, PyArrow, or Polars dataframes to a Polars representation.

    Args:
        data: The input filepath string, dataframe, table, connection object, or file-like stream.

    Returns:
        Tuple: (normalized Polars DataFrame or LazyFrame, original format type string).

    Raises:
        TidelyDataError: If the input data type is unsupported or loading fails.
    """
    import io
    import os
    import pathlib
    import pickle
    import sqlite3

    # Handle pathlib.Path
    if isinstance(data, pathlib.Path):
        data = str(data)

    # Handle Active SQLite Connection
    if isinstance(data, sqlite3.Connection):
        try:
            cursor = data.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [
                r[0] for r in cursor.fetchall() if r[0] not in ("sqlite_sequence",)
            ]
            if not tables:
                raise TidelyDataError("No tables found in SQLite database connection.")
            return pl.read_database(f'SELECT * FROM "{tables[0]}"', data), "sqlite"
        except Exception as e:
            raise TidelyDataError(f"Failed to read from SQLite connection: {e}") from e

    # Handle Active DuckDB Connection
    if duckdb is not None and isinstance(data, duckdb.DuckDBPyConnection):
        try:
            tables = [r[0] for r in data.execute("SHOW TABLES").fetchall()]
            if not tables:
                tables = [
                    r[0]
                    for r in data.execute(
                        "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
                    ).fetchall()
                ]
            if not tables:
                raise TidelyDataError("No tables found in DuckDB database connection.")
            return data.execute(f'SELECT * FROM "{tables[0]}"').pl(), "duckdb"
        except Exception as e:
            raise TidelyDataError(f"Failed to read from DuckDB connection: {e}") from e

    # Handle file-like objects (BytesIO, StringIO, file handles)
    if hasattr(data, "read") or isinstance(data, (io.BytesIO, io.StringIO)):
        try:
            if hasattr(data, "seek"):
                try:
                    data.seek(0)
                except Exception:
                    pass
            content = data.read()
            if hasattr(data, "seek"):
                try:
                    data.seek(0)
                except Exception:
                    pass
            if isinstance(content, str):
                content = content.encode("utf-8")
            data = content
        except Exception as e:
            raise TidelyDataError(f"Failed to read from file-like stream: {e}") from e

    # Handle string file paths
    if isinstance(data, str) and not data.strip().upper().startswith("SELECT "):
        # Auto-detect SQLite/DuckDB files first
        ext = os.path.splitext(data)[1].lower()
        if os.path.exists(data) and ext in (".sqlite", ".sqlite3", ".db", ".duckdb"):
            # Check SQLite
            try:
                conn = sqlite3.connect(data)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [
                    r[0] for r in cursor.fetchall() if r[0] not in ("sqlite_sequence",)
                ]
                if tables:
                    df = pl.read_database(f'SELECT * FROM "{tables[0]}"', conn)
                    conn.close()
                    return df, "sqlite"
                conn.close()
            except Exception:
                pass

            # Check DuckDB
            if duckdb is not None:
                try:
                    duck_conn = duckdb.connect(data)
                    tables = [r[0] for r in duck_conn.execute("SHOW TABLES").fetchall()]
                    if not tables:
                        tables = [
                            r[0]
                            for r in duck_conn.execute(
                                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
                            ).fetchall()
                        ]
                    if tables:
                        df = duck_conn.execute(f'SELECT * FROM "{tables[0]}"').pl()
                        duck_conn.close()
                        return df, "duckdb"
                    duck_conn.close()
                except Exception:
                    pass

        # Load file path to bytes
        if not os.path.exists(data):
            raise TidelyDataError(f"File not found: {data}")
        try:
            with open(data, "rb") as f:
                content = f.read()
            if ext in ("", ".zip", ".gz", ".bz2", ".xz"):
                decompressed_content, inner_ext = decompress_bytes(content)
                target_ext = inner_ext if inner_ext else ext
                data = decompressed_content
            else:
                target_ext = ext
                data = content
        except Exception as e:
            raise TidelyDataError(
                f"Failed to load dataset from path '{data}': {e}"
            ) from e
    else:
        target_ext = None

    # Handle raw bytes data
    if isinstance(data, bytes):
        if target_ext in (
            None,
            "",
            "zip",
            "gz",
            "bz2",
            "xz",
            ".zip",
            ".gz",
            ".bz2",
            ".xz",
        ):
            decompressed_content, inner_ext = decompress_bytes(data)
            if inner_ext:
                target_ext = inner_ext
            data = decompressed_content

        # Route by extension
        ext = target_ext.lower() if target_ext else ""
        try:
            if ext in (".csv", "csv", ".tsv", "tsv", ".txt", "txt", ""):
                enc = detect_encoding(data)
                text = data.decode(enc)
                delim = detect_delimiter(text)
                fmt = "tsv" if delim == "\t" else "csv"
                return pl.read_csv(
                    text.encode("utf-8"), separator=delim, infer_schema_length=10000
                ), fmt
            elif ext in (".parquet", "parquet"):
                return pl.read_parquet(io.BytesIO(data)), "parquet"
            elif ext in (".xlsx", "xlsx", ".xls", "xls", ".ods", "ods"):
                try:
                    return pl.read_excel(data, infer_schema_length=None), "excel"
                except Exception:
                    if pd is not None:
                        return pl.from_pandas(pd.read_excel(io.BytesIO(data))), "excel"
                    raise
            elif ext in (".ipc", "ipc", ".arrow", "arrow", ".feather", "feather"):
                return pl.read_ipc(io.BytesIO(data)), "arrow"
            elif ext in (".json", "json", ".jsonl", "jsonl"):
                try:
                    return pl.read_ndjson(io.BytesIO(data)), "json_nd"
                except Exception:
                    try:
                        return pl.read_json(io.BytesIO(data)), "json"
                    except Exception:
                        if pd is not None:
                            return pl.from_pandas(
                                pd.read_json(io.BytesIO(data))
                            ), "json"
                        raise
            elif ext in (".xml", "xml"):
                if pd is not None:
                    return pl.from_pandas(pd.read_xml(io.BytesIO(data))), "xml"
                raise TidelyDataError("Pandas is required to read XML format.")
            elif ext in (".arff", "arff"):
                enc = detect_encoding(data)
                return pl.from_pandas(parse_arff(data.decode(enc))), "arff"
            elif ext in (".pkl", "pkl", ".pickle", "pickle"):
                obj = pickle.loads(data)
                return normalize_to_polars(obj)
            else:
                # Fallback to smart text parser
                try:
                    enc = detect_encoding(data)
                    text = data.decode(enc)
                    delim = detect_delimiter(text)
                    return pl.read_csv(
                        text.encode("utf-8"), separator=delim, infer_schema_length=10000
                    ), "csv"
                except Exception:
                    raise TidelyDataError(
                        "Could not automatically parse raw dataset format."
                    ) from None
        except Exception as e:
            raise TidelyDataError(f"Failed to parse data bytes: {e}") from e

    # 1. Polars LazyFrame
    if isinstance(data, pl.LazyFrame):
        return data, "polars_lazy"

    # 2. Polars DataFrame
    if isinstance(data, pl.DataFrame):
        return data, "polars_eager"

    # 3. Pandas DataFrame
    if pd is not None and isinstance(data, pd.DataFrame):
        try:
            if pa is not None:
                try:
                    arrow_table = pa.Table.from_pandas(data)
                    from typing import cast

                    return cast(pl.DataFrame, pl.from_arrow(arrow_table)), "pandas"
                except Exception:
                    pass
            res_pandas = pl.from_pandas(data)
            if isinstance(res_pandas, pl.DataFrame):
                return res_pandas, "pandas"
            raise TidelyDataError(
                "Pandas conversion returned a Series instead of a DataFrame."
            )
        except Exception as e:
            try:
                cleaned_cols = {}
                for col in data.columns:
                    series = data[col]
                    try:
                        if pa is not None:
                            pa.array(series, from_pandas=True)
                        cleaned_cols[col] = series
                    except Exception:
                        cleaned_cols[col] = series.astype(str)
                fallback_df = pd.DataFrame(cleaned_cols)
                return pl.from_pandas(fallback_df), "pandas"
            except Exception:
                raise TidelyDataError(
                    f"Failed to convert Pandas DataFrame to Polars: {e}"
                ) from e

    # 4. PyArrow Table
    if pa is not None and isinstance(data, pa.Table):
        try:
            res_arrow = pl.from_arrow(data)
            if isinstance(res_arrow, pl.DataFrame):
                return res_arrow, "arrow"
            raise TidelyDataError(
                "PyArrow conversion returned a Series instead of a DataFrame."
            )
        except Exception as e:
            raise TidelyDataError(
                f"Failed to convert PyArrow Table to Polars: {e}"
            ) from e

    raise TidelyDataError(
        f"Unsupported data type '{type(data).__name__}'. "
        f"Tidely clean()/inspect() accepts filepath strings or Polars, Pandas, or PyArrow DataFrames/Tables."
    )


def parse_arff(data: str) -> "pd.DataFrame":
    """Parses an Attribute-Relation File Format (ARFF) file or content into a Pandas DataFrame."""
    import csv
    import os
    import re

    import pandas as pd

    attributes = []
    data_started = False
    data_lines = []

    attr_re = re.compile(
        r"^\s*@attribute\s+('[^']+'|\"[^\"]+\"|\S+)\s+(.+)$", re.IGNORECASE
    )

    if os.path.exists(data):
        with open(data, encoding="utf-8", errors="ignore") as f:
            lines = f.read().splitlines()
    else:
        lines = data.splitlines()

    for line in lines:
        line_str = line.strip()
        if not line_str or line_str.startswith("%"):
            continue

        if not data_started:
            if line_str.lower().startswith("@relation"):
                continue
            m = attr_re.match(line_str)
            if m:
                attr_name = m.group(1).strip("'\"")
                attr_type = m.group(2).strip()
                attributes.append((attr_name, attr_type))
                continue
            if line_str.lower().startswith("@data"):
                data_started = True
                continue
        else:
            data_lines.append(line_str)

    if not attributes:
        raise TidelyDataError("No attributes found in ARFF data")

    columns = [attr[0] for attr in attributes]
    rows = []

    for line in data_lines:
        try:
            parts_raw = next(csv.reader([line]))
            parts = [None if p.strip() == "?" else p.strip() for p in parts_raw]
            if len(parts) < len(columns):
                parts.extend([None] * (len(columns) - len(parts)))
            elif len(parts) > len(columns):
                parts = parts[: len(columns)]
            rows.append(parts)
        except Exception:
            pass

    df = pd.DataFrame(rows, columns=columns)

    for name, attr_type in attributes:
        type_lower = attr_type.lower()
        if "numeric" in type_lower or "real" in type_lower or "integer" in type_lower:
            df[name] = pd.to_numeric(df[name], errors="coerce")
        elif type_lower.startswith("{") and type_lower.endswith("}"):
            df[name] = df[name].astype(object)

    return df
