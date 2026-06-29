"""Public API for Tidely v1.0."""

from typing import Any, Optional

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import polars as pl
except ImportError:
    pl = None

from tidely.core.adapter import normalize_to_polars
from tidely.core.engine import run_pipeline
from tidely.result import CleanResult
from tidely.core.errors import TidelyDataError


def clean(
    data: Any,
) -> CleanResult:
    """Cleans a DataFrame deterministically using deep semantic inference.
    
    This is the core engine of Tidely. It automatically profiles the dataset,
    infers semantic types (Dates, Emails, Currency), optimizes memory footprints 
    by downcasting and compressing categoricals, and handles missing values safely 
    without silently dropping user data.
    
    Args:
        data (Any): A Pandas or Polars DataFrame.
        
    Returns:
        CleanResult: A proxy object containing the cleaned DataFrame (accessible via `.df`).
            Call `.summary()` on the returned object to see an explainable report
            of every structural and memory optimization performed.
            
    Example:
        >>> import tidely as td
        >>> import pandas as pd
        >>> df = pd.read_csv("dirty_data.csv")
        >>> result = td.clean(df)
        >>> print(result.summary())
        >>> clean_df = result.df
    """
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
    from tidely.core.adapter import normalize_to_polars
    from tidely.core.dna import infer_dataset_dna
    from tidely.core.scorer import compute_trust_scores
    from tidely.core.semantic import SemanticEngine
    from tidely.core.detector import DetectionEngine

    pl_data, format_name = normalize_to_polars(data)

    if isinstance(pl_data, pl.LazyFrame):
        df = pl_data.collect()
    else:
        df = pl_data

    detector = DetectionEngine()
    metadata = detector.analyze(data)

    semantic_engine = SemanticEngine()
    semantic_types = semantic_engine.infer(data, metadata)

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
    )

def validate(data: Any, schema: dict) -> bool:
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
    
    Args:
        filepath (str): The absolute or relative path to the file.
        **kwargs: Additional arguments passed to the underlying engine (`read_csv`).
        
    Returns:
        Any: A Pandas or Polars DataFrame depending on the installed backend.
        
    Raises:
        TidelyDataError: If the file format is unsupported.
    """
    if filepath.endswith(".csv"):
        if pd is not None:
            return pd.read_csv(filepath, **kwargs)
        elif pl is not None:
            return pl.read_csv(filepath, **kwargs)
    raise TidelyDataError(f"Unsupported file format for load(): {filepath}")


def save(data: Any, filepath: str, **kwargs: Any) -> None:
    """Helper method to save a DataFrame to disk.
    
    Args:
        data (Any): The DataFrame to save.
        filepath (str): The destination path.
        **kwargs: Additional arguments passed to the underlying engine (`to_csv`).
        
    Raises:
        TidelyDataError: If the data object or format is unsupported.
    """
    if hasattr(data, "to_csv") and filepath.endswith(".csv"):
        data.to_csv(filepath, **kwargs)
    elif hasattr(data, "write_csv") and filepath.endswith(".csv"):
        data.write_csv(filepath, **kwargs)
    else:
        raise TidelyDataError("Unsupported data object or format for save().")
