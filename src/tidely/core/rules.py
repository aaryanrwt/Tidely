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


def make_impute_constant_rule(
    column: str, value: str = "Unknown"
) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Imputes missing categorical/string values using a constant value."""

    def _rule(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.col(column).fill_null(value=value))

    return _rule


def make_impute_median_rule(column: str) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Imputes missing numeric values using the median."""

    def _rule(df: pl.DataFrame) -> pl.DataFrame:
        median_val = df.select(pl.col(column).median()).item()
        if median_val is None:
            return df
        return df.with_columns(pl.col(column).fill_null(value=median_val))

    return _rule


def make_impute_mean_rule(column: str) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Imputes missing numeric values using the mean."""

    def _rule(df: pl.DataFrame) -> pl.DataFrame:
        mean_val = df.select(pl.col(column).mean()).item()
        if mean_val is None:
            return df
        return df.with_columns(pl.col(column).fill_null(value=mean_val))

    return _rule


def make_impute_mode_rule(column: str) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Imputes missing values using the most frequent value (mode)."""

    def _rule(df: pl.DataFrame) -> pl.DataFrame:
        mode_series = df.select(pl.col(column).mode())
        if mode_series.height > 0:
            mode_val = mode_series.item(0, 0)
            if mode_val is not None:
                return df.with_columns(pl.col(column).fill_null(value=mode_val))
        return df.with_columns(pl.col(column).fill_null(value="Unknown"))

    return _rule


def make_impute_ffill_rule(column: str) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Imputes missing values using forward fill."""

    def _rule(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.col(column).forward_fill())

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


def make_outlier_iqr_rule(
    column: str, threshold: float = 1.5
) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Clips outlier values based on Interquartile Range (IQR)."""

    def _rule(df: pl.DataFrame) -> pl.DataFrame:
        try:
            q1 = df.select(pl.col(column).quantile(0.25)).item()
            q3 = df.select(pl.col(column).quantile(0.75)).item()
            if q1 is not None and q3 is not None:
                iqr = q3 - q1
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr
                return df.with_columns(pl.col(column).clip(lower, upper))
        except Exception:
            pass
        return df

    return _rule


def make_outlier_zscore_rule(
    column: str, threshold: float = 3.0
) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Clips outliers using Z-score methodology."""

    def _rule(df: pl.DataFrame) -> pl.DataFrame:
        try:
            mean_val = df.select(pl.col(column).mean()).item()
            std_val = df.select(pl.col(column).std()).item()
            if mean_val is not None and std_val is not None and std_val > 0:
                lower = mean_val - threshold * std_val
                upper = mean_val + threshold * std_val
                return df.with_columns(pl.col(column).clip(lower, upper))
        except Exception:
            pass
        return df

    return _rule


def make_unicode_clean_rule(column: str) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Normalizes Unicode text, removes non-printable control characters and extra whitespace."""

    def _rule(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.col(column)
            .cast(pl.String)
            .str.normalize_unicode()
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
            .str.replace_all(r"[\x00-\x1F\x7F-\x9F]", "")
            .alias(column)
        )

    return _rule


def make_zip_code_rule(column: str) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Pads ZIP code strings to exactly 5 digits."""

    def _rule(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.col(column)
            .cast(pl.String)
            .str.strip_chars()
            .str.pad_start(5, fill_char="0")
            .alias(column)
        )

    return _rule


def make_coordinate_clip_rule(
    column: str, is_lat: bool
) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Clips Latitude/Longitude values to their standard geometric bounds."""

    def _rule(df: pl.DataFrame) -> pl.DataFrame:
        min_v, max_v = (-90.0, 90.0) if is_lat else (-180.0, 180.0)
        return df.with_columns(
            pl.col(column).cast(pl.Float64).clip(min_v, max_v).alias(column)
        )

    return _rule


def make_replace_null_placeholders_rule(
    column: str,
) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Replaces common string representations of nulls (e.g., '?', 'N/A') with true nulls."""

    def _rule(df: pl.DataFrame) -> pl.DataFrame:
        # Cast to string safely, strip whitespace, and replace placeholder values with None/Null
        return df.with_columns(
            pl.when(
                pl.col(column)
                .cast(pl.String)
                .str.strip_chars()
                .is_in(["?", "N/A", "n/a", "null", "NULL", "NaN", "nan"])
            )
            .then(None)
            .otherwise(pl.col(column))
            .alias(column)
        )

    return _rule
