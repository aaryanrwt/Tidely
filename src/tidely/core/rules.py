"""Pure functional rules for dataset transformation in Tidely.

All rules return a Callable[[pl.DataFrame], pl.DataFrame] to be executed
lazily or eagerly by the clean engine.
"""

from collections.abc import Callable
from typing import Any

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


def make_impute_median_rule(
    column: str, value: Any = None
) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Imputes missing numeric values using the median."""

    def _rule(df: pl.DataFrame) -> pl.DataFrame:
        val = value
        if val is None:
            val = df.select(pl.col(column).median()).item()
        if val is None:
            return df
        return df.with_columns(pl.col(column).fill_null(value=val))

    return _rule


def make_impute_mean_rule(
    column: str, value: Any = None
) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Imputes missing numeric values using the mean."""

    def _rule(df: pl.DataFrame) -> pl.DataFrame:
        val = value
        if val is None:
            val = df.select(pl.col(column).mean()).item()
        if val is None:
            return df
        return df.with_columns(pl.col(column).fill_null(value=val))

    return _rule


def make_impute_mode_rule(
    column: str, value: Any = None
) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Imputes missing values using the most frequent value (mode)."""

    def _rule(df: pl.DataFrame) -> pl.DataFrame:
        val = value
        if val is None:
            mode_series = df.select(pl.col(column).mode())
            if mode_series.height > 0:
                val = mode_series.item(0, 0)
        if val is not None:
            return df.with_columns(pl.col(column).fill_null(value=val))
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
    column: str,
    threshold: float = 1.5,
    lower_bound: Any = None,
    upper_bound: Any = None,
) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Clips outlier values based on Interquartile Range (IQR)."""

    def _rule(df: pl.DataFrame) -> pl.DataFrame:
        try:
            lower = lower_bound
            upper = upper_bound
            if lower is None or upper is None:
                q1 = df.select(pl.col(column).quantile(0.25)).item()
                q3 = df.select(pl.col(column).quantile(0.75)).item()
                if q1 is not None and q3 is not None:
                    iqr = q3 - q1
                    lower = q1 - threshold * iqr
                    upper = q3 + threshold * iqr
            if lower is not None and upper is not None:
                return df.with_columns(pl.col(column).clip(lower, upper))
        except Exception:
            pass
        return df

    return _rule


def make_outlier_zscore_rule(
    column: str, threshold: float = 3.0, mean_val: Any = None, std_val: Any = None
) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Clips outliers using Z-score methodology."""

    def _rule(df: pl.DataFrame) -> pl.DataFrame:
        try:
            mean_v = mean_val
            std_v = std_val
            if mean_v is None or std_v is None:
                mean_v = df.select(pl.col(column).mean()).item()
                std_v = df.select(pl.col(column).std()).item()
            if mean_v is not None and std_v is not None and std_v > 0:
                lower = mean_v - threshold * std_v
                upper = mean_v + threshold * std_v
                return df.with_columns(pl.col(column).clip(lower, upper))
        except Exception:
            pass
        return df

    return _rule


def make_unicode_clean_rule(column: str) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Normalizes Unicode text, removes non-printable control characters and extra whitespace."""

    def _rule(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl
            .col(column)
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
            pl
            .col(column)
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
            pl
            .when(
                pl
                .col(column)
                .cast(pl.String)
                .str.strip_chars()
                .is_in(["?", "N/A", "n/a", "null", "NULL", "NaN", "nan"])
            )
            .then(None)
            .otherwise(pl.col(column))
            .alias(column)
        )

    return _rule


def make_impute_group_median_rule(
    column: str, group_column: str, global_median: Any = None
) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Imputes nulls based on the median of a correlated group using Polars window functions."""

    def _rule(df: pl.DataFrame) -> pl.DataFrame:
        try:
            g_med = global_median
            if g_med is None:
                g_med = df.select(pl.col(column).median()).item()
            if g_med is None:
                return df
            return df.with_columns(
                pl
                .col(column)
                .fill_null(pl.col(column).median().over(group_column))
                .fill_null(g_med)
            )
        except Exception:
            return make_impute_median_rule(column, value=global_median)(df)

    return _rule


def make_impute_group_mode_rule(
    column: str,
    group_column: str,
    global_mode_val: Any = None,
    group_modes: dict[Any, Any] | None = None,
) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Imputes nulls based on the mode of a correlated group."""

    def _rule(df: pl.DataFrame) -> pl.DataFrame:
        try:
            if group_modes is not None:
                mode_filler = pl.col(group_column).replace(
                    group_modes, default=global_mode_val
                )
                return df.with_columns(pl.col(column).fill_null(mode_filler))
            mode_df = (
                df
                .group_by([group_column, column])
                .count()
                .sort([group_column, "count"], descending=True)
                .unique(subset=[group_column])
            )
            joined = df.join(
                mode_df.select([group_column, pl.col(column).alias("__mode_val")]),
                on=group_column,
                how="left",
            )
            res = joined.with_columns(
                pl.col(column).fill_null(pl.col("__mode_val"))
            ).drop("__mode_val")
            g_mode = global_mode_val
            if g_mode is None:
                global_mode = df.select(pl.col(column).mode())
                if global_mode.height > 0:
                    g_mode = global_mode.item(0, 0)
            if g_mode is not None:
                res = res.with_columns(pl.col(column).fill_null(g_mode))
            return res
        except Exception:
            return make_impute_mode_rule(column, value=global_mode_val)(df)

    return _rule


def make_impute_bfill_rule(column: str) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Imputes missing values using backward fill."""

    def _rule(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.col(column).backward_fill())

    return _rule


def make_impute_interpolate_rule(column: str) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Imputes missing numeric values using linear interpolation."""

    def _rule(df: pl.DataFrame) -> pl.DataFrame:
        try:
            return df.with_columns(pl.col(column).interpolate())
        except Exception:
            return df

    return _rule


def make_outlier_modified_zscore_rule(
    column: str, threshold: float = 3.5, median_val: Any = None, mad_val: Any = None
) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Clips outlier values based on Median Absolute Deviation (MAD)."""

    def _rule(df: pl.DataFrame) -> pl.DataFrame:
        try:
            med = median_val
            mad = mad_val
            if med is None:
                med = df.select(pl.col(column).median()).item()
            if med is not None and mad is None:
                mad = df.select((pl.col(column) - med).abs().median()).item()
            if med is not None and mad is not None and mad > 0:
                lower = med - (threshold * mad / 0.6745)
                upper = med + (threshold * mad / 0.6745)
                return df.with_columns(pl.col(column).clip(lower, upper))
        except Exception:
            pass
        return df

    return _rule


def make_fuzzy_dedup_rule(
    column: str, threshold: float = 90.0, mapping: dict[Any, Any] | None = None
) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Standardizes near-duplicate category strings using fuzzy string similarity."""

    def _rule(df: pl.DataFrame) -> pl.DataFrame:
        try:
            if mapping is not None:
                return df.with_columns(pl.col(column).replace(mapping))
            import rapidfuzz

            counts = df.group_by(column).count().sort("count", descending=True)
            sorted_vals = counts[column].drop_nulls().to_list()
            rule_mapping = {}
            seen: set[str] = set()
            for val in sorted_vals:
                val_str = str(val)
                if val_str in seen:
                    continue
                matched = False
                for canonical in seen:
                    if rapidfuzz.fuzz.ratio(val_str, canonical) >= threshold:
                        rule_mapping[val] = canonical
                        matched = True
                        break
                if not matched:
                    seen.add(val_str)
                    rule_mapping[val] = val
            if rule_mapping:
                return df.with_columns(pl.col(column).replace(rule_mapping))
        except Exception:
            pass
        return df

    return _rule


def make_smart_string_clean_rule(column: str) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Performs advanced string normalization (Unicode normalization, hidden characters, smart quotes)."""

    def _rule(df: pl.DataFrame) -> pl.DataFrame:
        try:
            return df.with_columns(
                pl
                .col(column)
                .cast(pl.String)
                .str.normalize_unicode()
                .str.replace_all(r"[\u200B-\u200D\uFEFF]", "")
                .str.replace_all(r"[\x00-\x1F\x7F-\x9F]", "")
                .str.replace_all(r"[“”]", '"')
                .str.replace_all(r"[‘’]", "'")
                .str.replace_all(r"\s+", " ")
                .str.strip_chars()
                .alias(column)
            )
        except Exception:
            return df

    return _rule


def make_smart_date_rule(column: str) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Standardizes Excel serial dates, Unix timestamps, and mixed date string formats."""

    def _rule(df: pl.DataFrame) -> pl.DataFrame:
        dtype = df[column].dtype
        if dtype.is_numeric():
            try:
                max_val = df.select(pl.col(column).max()).item()
                if max_val is not None:
                    max_val_f = float(max_val)
                    if max_val_f > 1e9:
                        return df.with_columns(
                            pl.from_epoch(pl.col(column)).alias(column)
                        )
                    if 30000 < max_val_f < 60000:
                        return df.with_columns(
                            ((pl.col(column) - 25569.0) * 86400000.0)
                            .cast(pl.Duration("ms"))
                            .cast(pl.Datetime)
                            .alias(column)
                        )
            except Exception:
                pass
            return df
        else:
            try:
                import pandas as pd

                pd_series = pd.to_datetime(
                    df[column].to_pandas(), errors="coerce", format="mixed"
                )
                series_polars = pl.Series(column, pd_series.tolist())
                return df.with_columns(series_polars)
            except Exception:
                return make_date_rule(column)(df)

    return _rule


def make_smart_categorical_rule(column: str) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Merges truthy/falsy category representations and standardizes categorical capitalization."""

    def _rule(df: pl.DataFrame) -> pl.DataFrame:
        try:
            boolean_map = {
                "yes": "True",
                "no": "False",
                "y": "True",
                "n": "False",
                "true": "True",
                "false": "False",
                "t": "True",
                "f": "False",
                "1": "True",
                "0": "False",
            }
            unique_list = (
                df[column]
                .cast(pl.String)
                .str.strip_chars()
                .str.to_lowercase()
                .drop_nulls()
                .unique()
                .to_list()
            )
            if unique_list and all(str(v) in boolean_map for v in unique_list):
                return df.with_columns(
                    pl
                    .col(column)
                    .cast(pl.String)
                    .str.strip_chars()
                    .str.to_lowercase()
                    .replace(boolean_map)
                    .alias(column)
                )
            else:
                return df.with_columns(
                    pl
                    .col(column)
                    .cast(pl.String)
                    .str.strip_chars()
                    .str.to_lowercase()
                    .str.capitalize()
                    .alias(column)
                )
        except Exception:
            pass
        return df

    return _rule


def make_smart_numeric_clean_rule(
    column: str,
) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Cleans numeric values by removing currencies, percentages, commas, and scientific notations."""

    def _rule(df: pl.DataFrame) -> pl.DataFrame:
        try:
            cleaned = pl.col(column).cast(pl.String).str.strip_chars()
            cleaned = cleaned.str.replace_all(r"[\$\€\£\¥\s]", "")
            cleaned = cleaned.str.replace_all(r",", "")
            is_pct = df.select(
                pl
                .col(column)
                .cast(pl.String)
                .str.strip_chars()
                .str.ends_with("%")
                .any()
            ).item()
            if is_pct:
                cleaned = cleaned.str.replace_all(r"%", "")
                expr = cleaned.cast(pl.Float64) / 100.0
            else:
                expr = cleaned.cast(pl.Float64)
            return df.with_columns(expr.alias(column))
        except Exception:
            try:
                extracted = pl.col(column).cast(pl.String).str.extract(r"(-?\d+\.?\d*)")
                return df.with_columns(extracted.cast(pl.Float64).alias(column))
            except Exception:
                return df

    return _rule
