"""Public API for Tidely v1.0."""

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
    """Helper method to save a DataFrame or LazyFrame to disk.

    Args:
        data (Any): The DataFrame or LazyFrame to save.
        filepath (str): The destination path.
        **kwargs: Additional arguments passed to the underlying engine.

    Raises:
        TidelyDataError: If the data object or format is unsupported.
    """
    ext = filepath.split(".")[-1].lower()

    # Handle Polars LazyFrame
    if isinstance(data, pl.LazyFrame):
        try:
            if ext == "csv":
                try:
                    data.sink_csv(filepath, **kwargs)
                except Exception:
                    data.collect().write_csv(filepath, **kwargs)
            elif ext == "parquet":
                try:
                    data.sink_parquet(filepath, **kwargs)
                except Exception:
                    data.collect().write_parquet(filepath, **kwargs)
            else:
                raise TidelyDataError(f"Unsupported lazy export format: .{ext}")
            return
        except Exception as e:
            raise TidelyDataError(f"Failed to save LazyFrame to {filepath}: {e}") from e

    # Handle standard eager dataframes
    if ext == "csv":
        if hasattr(data, "to_csv"):
            data.to_csv(filepath, **kwargs)
        elif hasattr(data, "write_csv"):
            data.write_csv(filepath, **kwargs)
        else:
            raise TidelyDataError("Unsupported CSV data object.")
    elif ext == "parquet":
        if hasattr(data, "to_parquet"):
            data.to_parquet(filepath, **kwargs)
        elif hasattr(data, "write_parquet"):
            data.write_parquet(filepath, **kwargs)
        else:
            raise TidelyDataError("Unsupported Parquet data object.")
    else:
        raise TidelyDataError(f"Unsupported format .{ext} for save().")
