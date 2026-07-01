"""Public API for Tidely."""

from typing import Any

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]

from tidely.core.adapter import normalize_to_polars
from tidely.core.engine import run_pipeline
from tidely.core.errors import TidelyDataError
from tidely.result import CleanResult


def clean(
    data: Any,
) -> CleanResult:
    """Cleans a DataFrame deterministically using deep semantic inference.

    This function now accepts file paths (str or pathlib.Path) and loads them
    using the library's adapter layer before cleaning.

    Args:
        data (Any): A Pandas/Polars DataFrame, a lazy frame, or a file path.

    Returns:
        CleanResult: A proxy object containing the cleaned DataFrame (accessible via `.df`).
            Call `.summary()` on the returned object to see an explainable report
            of every structural and memory optimization performed.
    """
    # Auto-load file paths using the existing adapter without duplicating loader logic
    from pathlib import Path
    if isinstance(data, (str, Path)):
        # Normalize to a string path for the adapter
        data = str(data)
        # Use the load helper to read the file into a Polars/Pandas DataFrame
        from tidely.api import load
        data = load(data)
    return run_pipeline(data)


def inspect(data: Any) -> Any:
    """Profiles a dataset and generates a comprehensive Trust Score and semantic diagnosis.

    Args:
        data (Any): A Pandas or Polars DataFrame.

    Returns:
        DatasetProfile: An object containing the inferred DNA of the dataset,
            column-level semantic types, and a 5-dimension structural Trust Score.

    Example:
        >>> profile = td.inspect(df)
        >>> print(profile.trust_score.overall)
        >>> profile.show()
    """
    from tidely import DatasetProfile
    from tidely.core.detector import DetectionEngine
    from tidely.core.dna import infer_dataset_dna
    from tidely.core.scorer import compute_trust_scores
    from tidely.core.semantic import SemanticEngine

    pl_data, format_name = normalize_to_polars(data)

    if isinstance(pl_data, pl.LazyFrame):
        df = pl_data.collect()
    else:
        df = pl_data

    detector = DetectionEngine()
    metadata = detector.analyze(df)

    semantic_engine = SemanticEngine()
    semantic_types = semantic_engine.infer(df, metadata)

    dna = infer_dataset_dna(df.columns)
    trust_score = compute_trust_scores(df, semantic_types, dna.domain)

    return DatasetProfile(
        row_count=df.height,
        col_count=df.width,
        dna=dna,
        trust_score=trust_score,
        diagnoses=[],
        semantic_types=semantic_types,
        format_name=format_name,
        _df_ref=df,
        metadata=metadata,
    )


def validate(data: Any, schema: dict[str, Any]) -> bool:
    """Validates the dataset against a provided schema dictionary.

    Args:
        data (Any): A Pandas or Polars DataFrame.
        schema (dict): A dictionary describing expected columns and types
            (e.g., `{"user_id": "int", "is_active": "bool"}`).

    Returns:
        bool: True if the dataset completely matches the schema.

    Raises:
        TidelyValidationError: If a column is missing or a type mismatches the schema.

    Example:
        >>> td.validate(df, {"age": "int", "name": "str"})
        True
    """
    from tidely.core.validate import validate_schema

    return validate_schema(data, schema)


def load(filepath: str, **kwargs: Any) -> Any:
    """Helper method to load a dataset into a DataFrame.

    Supports CSV, Parquet, Excel, ARFF, and generic fallback via the adapter.
    """
    # Delegate to the adapter which knows how to handle many formats
    from tidely.core.adapter import normalize_to_polars
    pl_obj, fmt = normalize_to_polars(filepath)
    # Convert Polars object to appropriate Python object (DataFrame or LazyFrame)
    if fmt.endswith('_lazy'):
        # Return LazyFrame for lazy formats
        return pl_obj
    else:
        # Return eager DataFrame
        return pl_obj


def save(data: Any, filepath: str, **kwargs: Any) -> None:
    """Helper method to save a DataFrame or LazyFrame to disk in a wide variety of formats.

    Args:
        data (Any): The DataFrame or LazyFrame to save.
        filepath (str): The destination path.
        **kwargs: Additional arguments passed to the underlying engine.

    Raises:
        TidelyDataError: If the data object or format is unsupported.
    """
    ext = filepath.split(".")[-1].lower()

    # Handle LazyFrame sinking if possible, otherwise collect
    if isinstance(data, pl.LazyFrame):
        if ext == "csv":
            try:
                data.sink_csv(filepath, **kwargs)
                return
            except Exception:
                pass
        elif ext == "parquet":
            try:
                data.sink_parquet(filepath, **kwargs)
                return
            except Exception:
                pass
        # Collect for other non-streaming formats
        try:
            data = data.collect()
        except Exception as e:
            raise TidelyDataError(f"Failed to collect LazyFrame for export: {e}") from e

    # Now handle standard eager DataFrame data (both Polars and Pandas)
    try:
        if ext == "csv":
            if hasattr(data, "write_csv"):
                data.write_csv(filepath, **kwargs)
            elif hasattr(data, "to_csv"):
                data.to_csv(filepath, index=False, **kwargs)
            else:
                raise TidelyDataError("Unsupported CSV data object.")
        elif ext == "tsv":
            if hasattr(data, "write_csv"):
                data.write_csv(filepath, separator="\t", **kwargs)
            elif hasattr(data, "to_csv"):
                data.to_csv(filepath, sep="\t", index=False, **kwargs)
            else:
                raise TidelyDataError("Unsupported TSV data object.")
        elif ext in ("xlsx", "xls"):
            if hasattr(data, "write_excel"):
                try:
                    data.write_excel(filepath, **kwargs)
                    return
                except Exception:
                    pass
            if pd is not None:
                pd_df = data.to_pandas() if hasattr(data, "to_pandas") else data
                pd_df.to_excel(filepath, index=False, **kwargs)
            else:
                raise TidelyDataError("Pandas is required to export to Excel (.xls/.xlsx)")
        elif ext == "ods":
            if pd is not None:
                pd_df = data.to_pandas() if hasattr(data, "to_pandas") else data
                pd_df.to_excel(filepath, engine="odf", index=False, **kwargs)
            else:
                raise TidelyDataError("Pandas and odfpy are required to export to ODS")
        elif ext == "parquet":
            if hasattr(data, "write_parquet"):
                data.write_parquet(filepath, **kwargs)
            elif hasattr(data, "to_parquet"):
                data.to_parquet(filepath, index=False, **kwargs)
            else:
                raise TidelyDataError("Unsupported Parquet data object.")
        elif ext in ("feather", "ipc", "arrow"):
            if hasattr(data, "write_ipc"):
                data.write_ipc(filepath, **kwargs)
            elif hasattr(data, "to_feather"):
                data.to_feather(filepath, **kwargs)
            else:
                raise TidelyDataError("Unsupported Feather data object.")
        elif ext == "json":
            if hasattr(data, "write_json"):
                data.write_json(filepath, **kwargs)
            elif hasattr(data, "to_json"):
                data.to_json(filepath, **kwargs)
            else:
                raise TidelyDataError("Unsupported JSON data object.")
        elif ext == "jsonl":
            if hasattr(data, "write_ndjson"):
                data.write_ndjson(filepath, **kwargs)
            elif hasattr(data, "to_json"):
                data.to_json(filepath, orient="records", lines=True, **kwargs)
            else:
                raise TidelyDataError("Unsupported JSONL data object.")
        elif ext == "xml":
            if pd is not None:
                pd_df = data.to_pandas() if hasattr(data, "to_pandas") else data
                pd_df.to_xml(filepath, index=False, **kwargs)
            else:
                raise TidelyDataError("Pandas is required to export to XML")
        elif ext in ("yaml", "yml"):
            import yaml
            if hasattr(data, "to_dicts"):
                dicts = data.to_dicts()
            elif hasattr(data, "to_dict"):
                dicts = data.to_dict(orient="records")
            else:
                dicts = list(data)
            with open(filepath, "w", encoding="utf-8") as f:
                yaml.dump(dicts, f, default_flow_style=False, **kwargs)
        elif ext == "arff":
            import os
            df_pl = data if hasattr(data, "iter_rows") else pl.from_pandas(data)
            lines = [f"@relation {os.path.basename(filepath)}"]
            for col in df_pl.columns:
                dtype = df_pl[col].dtype
                if dtype.is_numeric():
                    lines.append(f"@attribute {col} numeric")
                elif dtype == pl.Boolean:
                    lines.append(f"@attribute {col} {{false,true}}")
                else:
                    unique_vals = [str(x) for x in df_pl[col].unique().drop_nulls().to_list()]
                    unique_escaped = [f"'{x}'" if " " in x or "," in x else x for x in unique_vals]
                    if not unique_escaped:
                        unique_escaped = ["?"]
                    lines.append(f"@attribute {col} {{{','.join(unique_escaped)}}}")
            lines.append("@data")
            for row in df_pl.iter_rows():
                row_str = []
                for val in row:
                    if val is None:
                        row_str.append("?")
                    elif isinstance(val, bool):
                        row_str.append(str(val).lower())
                    elif isinstance(val, (int, float)):
                        row_str.append(str(val))
                    else:
                        s = str(val)
                        if " " in s or "," in s or "'" in s or '"' in s:
                            s_esc = s.replace("'", "\\'")
                            row_str.append(f"'{s_esc}'")
                        else:
                            row_str.append(s)
                lines.append(",".join(row_str))
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        elif ext == "txt":
            if hasattr(data, "write_csv"):
                data.write_csv(filepath, separator="\t", **kwargs)
            elif hasattr(data, "to_csv"):
                data.to_csv(filepath, sep="\t", index=False, **kwargs)
            else:
                raise TidelyDataError("Unsupported TXT data object.")
        elif ext == "duckdb":
            import duckdb
            df_pl = data if hasattr(data, "lazy") else pl.from_pandas(data)
            conn = duckdb.connect(filepath)
            try:
                conn.register("temp_df", df_pl)
                conn.execute("CREATE TABLE IF NOT EXISTS cleaned_data AS SELECT * FROM temp_df")
            finally:
                conn.close()
        elif ext in ("sqlite", "db"):
            import sqlite3
            df_pl = data if hasattr(data, "iter_rows") else pl.from_pandas(data)
            sqlite_conn = sqlite3.connect(filepath)
            try:
                cursor = sqlite_conn.cursor()
                cols = []
                for col in df_pl.columns:
                    dtype = df_pl[col].dtype
                    sql_type = "REAL" if dtype.is_numeric() else "TEXT"
                    cols.append(f'"{col}" {sql_type}')
                cursor.execute(f"CREATE TABLE IF NOT EXISTS cleaned_data ({', '.join(cols)});")
                placeholders = ", ".join(["?"] * len(df_pl.columns))
                cursor.executemany(f"INSERT INTO cleaned_data VALUES ({placeholders})", df_pl.iter_rows())
                sqlite_conn.commit()
            finally:
                sqlite_conn.close()
        else:
            raise TidelyDataError(f"Unsupported format .{ext} for save().")
    except Exception as e:
        raise TidelyDataError(f"Failed to save data to {filepath}: {e}") from e
