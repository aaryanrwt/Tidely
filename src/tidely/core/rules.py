"""Pure functional rules for dataset transformation in Tidely.

All rules return a Callable[[pl.DataFrame], pl.DataFrame] to be executed
lazily or eagerly by the clean engine.
"""

from collections.abc import Callable

import polars as pl


def make_dedup_rows_rule() -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Drops exact duplicate rows."""

    def _rule(df: pl.DataFrame) -> pl.DataFrame:
        return df.unique()

    return _rule


def make_dedup_id_rule(column: str) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Drops rows with duplicate primary keys/IDs, keeping the first occurrence."""

    def _rule(df: pl.DataFrame) -> pl.DataFrame:
        return df.unique(subset=[column], maintain_order=True)

    return _rule


def make_impute_constant_rule(column: str, value: str = "Unknown") -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Imputes missing categorical/string values using a constant value."""

    def _rule(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.col(column).fill_null(value=value)
        )

    return _rule


def make_impute_median_rule(column: str) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Imputes missing numeric values using the median."""

    def _rule(df: pl.DataFrame) -> pl.DataFrame:
        median_val = df.select(pl.col(column).median()).item()
        # If the entire column is null, median_val is None.
        if median_val is None:
            return df
        return df.with_columns(pl.col(column).fill_null(value=median_val))

    return _rule


def make_email_rule(column: str) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Normalizes emails (lowercases and strips whitespace)."""

    def _rule(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.col(column).str.strip_chars().str.to_lowercase().alias(column)
        )

    return _rule


def make_phone_rule(column: str) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Normalizes phones (strips all non-digit characters)."""

    def _rule(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.col(column).str.replace_all(r"[^\d+]", "").alias(column)
        )

    return _rule


def make_date_rule(column: str) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Attempts to cast string dates to native Polars Datetime/Date."""

    def _rule(df: pl.DataFrame) -> pl.DataFrame:
        # Polars str.to_date(strict=False) will parse valid dates and return null for invalid.
        # For a robust clean, we coalesce. We'll simply try basic parsing.
        # Since tidely supports multiple date formats, we can use try_parse or let Polars infer.
        try:
            return df.with_columns(
                pl.col(column).str.to_datetime(strict=False).alias(column)
            )
        except Exception:
            return df

    return _rule


def make_categorical_rule(column: str) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Casts low-cardinality strings to Categorical for memory optimization."""

    def _rule(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.col(column).cast(pl.Categorical))

    return _rule


def make_downcast_rule(
    column: str, target_type: type[pl.DataType]
) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Downcasts integers to smaller precision types (e.g., Int8, Int16)."""

    def _rule(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.col(column).cast(target_type))

    return _rule
