"""Adapters to normalize inputs (Pandas, Polars, Arrow) to Polars DataFrames/LazyFrames."""

from typing import Any

import polars as pl

from tidely.core.errors import DatasetError

# Handle pandas import optionally to prevent failure if pandas isn't installed
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import pyarrow as pa
except ImportError:
    pa = None  # type: ignore[assignment]


def normalize_to_polars(
    data: Any,
) -> tuple[pl.DataFrame | pl.LazyFrame, str]:
    """Normalizes Pandas, PyArrow, or Polars dataframes to a Polars representation.

    Args:
        data: The input dataframe/table.

    Returns:
        Tuple: (normalized Polars DataFrame or LazyFrame, original format type string).

    Raises:
        DatasetError: If the input data type is unsupported.
    """
    # 1. Polars LazyFrame
    if isinstance(data, pl.LazyFrame):
        return data, "polars_lazy"

    # 2. Polars DataFrame
    if isinstance(data, pl.DataFrame):
        return data, "polars_eager"

    # 3. Pandas DataFrame
    if pd is not None and isinstance(data, pd.DataFrame):
        try:
            res_pandas = pl.from_pandas(data)
            if isinstance(res_pandas, pl.DataFrame):
                return res_pandas, "pandas"
            raise DatasetError(
                "Pandas conversion returned a Series instead of a DataFrame."
            )
        except Exception as e:
            raise DatasetError(
                f"Failed to convert Pandas DataFrame to Polars: {e}"
            ) from e

    # 4. PyArrow Table
    if pa is not None and isinstance(data, pa.Table):
        try:
            res_arrow = pl.from_arrow(data)
            if isinstance(res_arrow, pl.DataFrame):
                return res_arrow, "arrow"
            raise DatasetError(
                "PyArrow conversion returned a Series instead of a DataFrame."
            )
        except Exception as e:
            raise DatasetError(f"Failed to convert PyArrow Table to Polars: {e}") from e

    # 5. Fallback/Unsupported
    raise DatasetError(
        f"Unsupported data type '{type(data).__name__}'. "
        f"Tidely inspect() accepts Polars, Pandas, or PyArrow DataFrames/Tables."
    )
