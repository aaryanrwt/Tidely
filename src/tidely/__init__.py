"""Tidely: The Operating System for Data Quality.

This library provides production-grade data quality scoring, semantic type inference,
and Dataset Trust inspection for Pandas and Polars.
"""

from typing import Any

import polars as pl

from tidely.core.adapter import normalize_to_polars
from tidely.core.dna import infer_dataset_dna
from tidely.core.errors import TidelyError
from tidely.core.profile import DatasetProfile
from tidely.core.scorer import compute_trust_scores
from tidely.core.semantic import classify_series
from tidely.core.clean_engine import RepairPlan
from tidely.core.plan import clean, plan

# Handle pandas import optionally
try:
    import pandas as pd
except ImportError:
    pd = None

__version__ = "0.3.0"


def inspect(data: Any) -> DatasetProfile:
    """Profiles a dataset and returns its intelligence, quality metrics and Trust Scores.

    Args:
        data: A Polars DataFrame/LazyFrame, Pandas DataFrame, or PyArrow Table.

    Returns:
        DatasetProfile: The visualizable dataset profile containing Trust Scores, DNA, and semantics.

    Raises:
        TidelyError: If input data fails adapter normalization or profiling fails.
    """
    pl_data, format_name = normalize_to_polars(data)

    # Ensure eager representation for profile calculations
    if isinstance(pl_data, pl.LazyFrame):
        df = pl_data.collect()
    else:
        df = pl_data

    # Sample for semantic classification: up to 200 non-null values per column
    semantic_types = {}
    for col in df.columns:
        # Extract sample
        sample_vals = df[col].head(200).to_list()
        semantic_types[col] = classify_series(sample_vals, col)

    # Infer Dataset DNA
    dna = infer_dataset_dna(df.columns)

    # Compute Trust Score
    trust_score = compute_trust_scores(df, semantic_types, dna.domain)

    # Pass DataFrame reference for on-the-fly profiling in DatasetProfile.show() if needed

    return DatasetProfile(
        row_count=df.height,
        col_count=df.width,
        dna=dna,
        trust_score=trust_score,
        diagnoses=[],
        semantic_types=semantic_types,
        format_name=format_name,
        _df_ref=df,  # Hidden reference to compute missing/duplicate stats in show()
    )




__all__ = [
    "inspect",
    "clean",
    "plan",
    "DatasetProfile",
    "RepairPlan",
    "TidelyError",
]
