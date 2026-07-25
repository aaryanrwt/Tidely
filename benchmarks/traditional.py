"""Tidely v1.5.0 — Traditional Cleaning Pipeline.

A classical data cleaning baseline using industry-standard libraries:
pandas, polars, numpy, scikit-learn, RapidFuzz, pyarrow, regex.

This pipeline serves as the scientific comparison baseline against Tidely.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Individual cleaning operations
# ---------------------------------------------------------------------------


def _is_string_col(series: pd.Series) -> bool:  # type: ignore[type-arg]
    """Return True for object OR pyarrow-backed string columns (pandas 2.x)."""
    return bool(
        series.dtype == object
        or pd.api.types.is_string_dtype(series)
        or str(series.dtype).startswith("string")
        or str(series.dtype).startswith("large_string")
    )


def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows."""
    return df.drop_duplicates()


def _handle_missing_values(
    df: pd.DataFrame, keys: list[str], target: str | None
) -> pd.DataFrame:
    """Impute missing values: median for numeric, mode for categorical."""
    out = df.copy()
    for col in out.columns:
        null_count = out[col].isnull().sum()
        if null_count == 0:
            continue
        if col in keys or col == target:
            continue
        if pd.api.types.is_numeric_dtype(out[col]) and not pd.api.types.is_bool_dtype(
            out[col]
        ):
            fill_val = out[col].median()
            out[col] = out[col].fillna(fill_val if pd.notna(fill_val) else 0)
        else:
            mode_res = out[col].mode()
            out[col] = out[col].fillna(
                mode_res.iloc[0] if not mode_res.empty else "Missing"
            )
    return out


def _normalize_categorical(
    df: pd.DataFrame, keys: list[str], target: str | None
) -> pd.DataFrame:
    """Strip whitespace and lowercase all string/object columns."""
    out = df.copy()
    for col in out.columns:
        if col in keys or col == target:
            continue
        if _is_string_col(out[col]):
            out[col] = out[col].apply(
                lambda x: x.strip().lower() if isinstance(x, str) else x
            )
    return out


def _normalize_unicode(
    df: pd.DataFrame, keys: list[str], target: str | None
) -> pd.DataFrame:
    """NFC normalize unicode in string columns."""
    out = df.copy()
    for col in out.columns:
        if col in keys:
            continue
        if _is_string_col(out[col]):
            out[col] = out[col].apply(
                lambda x: unicodedata.normalize("NFC", x) if isinstance(x, str) else x
            )
    return out


def _normalize_whitespace(
    df: pd.DataFrame, keys: list[str], target: str | None
) -> pd.DataFrame:
    """Collapse multiple whitespace chars to single space."""
    _ws_re = re.compile(r"\s+")
    out = df.copy()
    for col in out.columns:
        if col in keys:
            continue
        if _is_string_col(out[col]):
            out[col] = out[col].apply(
                lambda x: _ws_re.sub(" ", x).strip() if isinstance(x, str) else x
            )
    return out


def _normalize_booleans(df: pd.DataFrame) -> pd.DataFrame:
    """Convert common boolean string representations to True/False."""
    TRUE_VALS = {"true", "yes", "1", "t", "y"}
    FALSE_VALS = {"false", "no", "0", "f", "n"}

    out = df.copy()
    for col in out.columns:
        if _is_string_col(out[col]):
            sample = out[col].dropna().head(20).astype(str).str.lower()
            if sample.isin(TRUE_VALS | FALSE_VALS).all() and len(sample) > 0:
                out[col] = out[col].apply(
                    lambda x: (
                        True
                        if str(x).lower() in TRUE_VALS
                        else (False if str(x).lower() in FALSE_VALS else x)
                        if pd.notna(x)
                        else x
                    )
                )
    return out


def _parse_datetimes(
    df: pd.DataFrame, keys: list[str], target: str | None
) -> pd.DataFrame:
    """Attempt to parse string columns that look like datetimes."""
    _date_hints = re.compile(r"\b(\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4})\b")
    out = df.copy()
    for col in out.columns:
        if col in keys or col == target:
            continue
        if not _is_string_col(out[col]):
            continue
        sample = out[col].dropna().head(20).astype(str)
        if sample.apply(lambda x: bool(_date_hints.search(x))).mean() > 0.7:
            try:
                out[col] = pd.to_datetime(out[col], errors="coerce")
            except Exception:
                pass
    return out


def _downcast_numerics(
    df: pd.DataFrame, keys: list[str], target: str | None
) -> pd.DataFrame:
    """Downcast int64/float64 columns to smaller types where safe."""
    out = df.copy()
    for col in out.columns:
        if col in keys or col == target:
            continue
        if pd.api.types.is_integer_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], downcast="integer")
        elif pd.api.types.is_float_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], downcast="float")
    return out


def _remove_outliers_iqr(
    df: pd.DataFrame, keys: list[str], target: str | None, multiplier: float = 3.0
) -> pd.DataFrame:
    """Clip outliers using 3×IQR rule on numeric columns."""
    out = df.copy()
    for col in out.columns:
        if col in keys or col == target:
            continue
        if pd.api.types.is_numeric_dtype(out[col]) and not pd.api.types.is_bool_dtype(
            out[col]
        ):
            q1 = out[col].quantile(0.25)
            q3 = out[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                out[col] = out[col].clip(
                    lower=q1 - multiplier * iqr, upper=q3 + multiplier * iqr
                )
    return out


def _replace_null_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    """Replace common null placeholder strings with np.nan."""
    NULL_STRINGS = {"n/a", "na", "nan", "null", "none", "missing", "?", "-", "--", ""}
    out = df.copy()
    for col in out.columns:
        if _is_string_col(out[col]):
            out[col] = out[col].apply(
                lambda x: (
                    np.nan
                    if isinstance(x, str) and x.strip().lower() in NULL_STRINGS
                    else x
                )
            )
    return out


def _fuzzy_dedup(df: pd.DataFrame, str_col: str, threshold: int = 95) -> pd.DataFrame:
    """Remove near-duplicate rows based on fuzzy string similarity in one column."""
    try:
        from rapidfuzz import fuzz  # type: ignore[import]
    except ImportError:
        return df

    if str_col not in df.columns or not _is_string_col(df[str_col]):
        return df

    values = df[str_col].fillna("").tolist()
    keep = [True] * len(values)
    for i in range(len(values)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(values)):
            if not keep[j]:
                continue
            if fuzz.ratio(values[i], values[j]) >= threshold:
                keep[j] = False

    return df[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_traditional_pipeline(
    df: pd.DataFrame,
    keys: list[str] | None = None,
    target: str | None = None,
    fuzzy_col: str | None = None,
) -> pd.DataFrame:
    """Execute the full traditional cleaning pipeline."""
    keys = keys or []
    out = df.copy()
    out = _replace_null_placeholders(out)
    out = _remove_duplicates(out)
    out = _handle_missing_values(out, keys, target)
    out = _normalize_whitespace(out, keys, target)
    out = _normalize_unicode(out, keys, target)
    out = _normalize_categorical(out, keys, target)
    out = _normalize_booleans(out)
    out = _parse_datetimes(out, keys, target)
    out = _remove_outliers_iqr(out, keys, target)
    out = _downcast_numerics(out, keys, target)
    if fuzzy_col:
        out = _fuzzy_dedup(out, fuzzy_col)
    return out
