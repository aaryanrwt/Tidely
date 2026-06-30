"""Adapters to normalize inputs (Pandas, Polars, Arrow) to Polars DataFrames/LazyFrames."""

from typing import Any

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
    pa = None  # type: ignore[assignment]


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


def normalize_to_polars(
    data: Any,
) -> tuple[pl.DataFrame | pl.LazyFrame, str]:
    """Normalizes filepath strings, Pandas, PyArrow, or Polars dataframes to a Polars representation.

    Args:
        data: The input filepath string, dataframe, or table.

    Returns:
        Tuple: (normalized Polars DataFrame or LazyFrame, original format type string).

    Raises:
        TidelyDataError: If the input data type is unsupported or loading fails.
    """
    # 0. Filepath String
    if isinstance(data, str):
        import os

        if not os.path.exists(data):
            raise TidelyDataError(f"File not found: {data}")
        ext = os.path.splitext(data)[1].lower()
        try:
            if ext == ".csv":
                try:
                    return pl.scan_csv(data), "csv_lazy"
                except Exception:
                    return pl.read_csv(data), "csv"
            elif ext == ".parquet":
                try:
                    return pl.scan_parquet(data), "parquet_lazy"
                except Exception:
                    return pl.read_parquet(data), "parquet"
            elif ext in (".ipc", ".arrow", ".feather"):
                try:
                    return pl.scan_ipc(data), "arrow_lazy"
                except Exception:
                    return pl.read_ipc(data), "arrow"
            elif ext == ".json":
                try:
                    return pl.read_ndjson(data), "json_nd"
                except Exception:
                    return pl.read_json(data), "json"
            elif ext in (".xlsx", ".xls"):
                try:
                    return pl.read_excel(data), "excel"
                except Exception:
                    if pd is not None:
                        return pl.from_pandas(pd.read_excel(data)), "excel"
                    raise
            elif ext == ".arff":
                return pl.from_pandas(parse_arff(data)), "arff"
            else:
                if pd is not None:
                    return pl.from_pandas(pd.read_csv(data)), "pandas"
                raise TidelyDataError(f"Unsupported file extension: {ext}")
        except Exception as e:
            raise TidelyDataError(
                f"Failed to load dataset from path '{data}': {e}"
            ) from e

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
                # Column-by-column fallback casting mixed types to String
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

    # 5. Fallback/Unsupported
    raise TidelyDataError(
        f"Unsupported data type '{type(data).__name__}'. "
        f"Tidely clean()/inspect() accepts filepath strings or Polars, Pandas, or PyArrow DataFrames/Tables."
    )


def parse_arff(filepath: str) -> "pd.DataFrame":
    """Parses an Attribute-Relation File Format (ARFF) file into a Pandas DataFrame."""
    import csv
    import re

    import pandas as pd

    # relation variable removed as unused
    attributes = []
    data_started = False
    data_lines = []

    # Regex to parse attribute line: @attribute <name> <type>
    # Name can be quoted or unquoted
    attr_re = re.compile(
        r"^\s*@attribute\s+('[^']+'|\"[^\"]+\"|\S+)\s+(.+)$", re.IGNORECASE
    )

    with open(filepath, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line_str = line.strip()
            if not line_str or line_str.startswith("%"):
                continue

            if not data_started:
                if line_str.lower().startswith("@relation"):
                    _relation = line_str.split(None, 1)[1].strip()
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
        raise TidelyDataError(f"No attributes found in ARFF file: {filepath}")

    columns = [attr[0] for attr in attributes]
    rows = []

    # Parse data lines
    for line in data_lines:
        try:
            # Respect quotes using standard library csv reader
            parts_raw = next(csv.reader([line]))
            parts: list[Any] = [None if p.strip() == "?" else p.strip() for p in parts_raw]
            # Ensure row matches number of columns
            if len(parts) < len(columns):
                parts.extend([None] * (len(columns) - len(parts)))
            elif len(parts) > len(columns):
                parts = parts[: len(columns)]
            rows.append(parts)
        except Exception:
            pass

    df = pd.DataFrame(rows, columns=columns)

    # Cast types based on attribute metadata
    for name, attr_type in attributes:
        type_lower = attr_type.lower()
        if "numeric" in type_lower or "real" in type_lower or "integer" in type_lower:
            df[name] = pd.to_numeric(df[name], errors="coerce")
        elif type_lower.startswith("{") and type_lower.endswith("}"):
            df[name] = df[name].astype(object)

    return df
