"""Detection Engine for analyzing base types, missing values, and cardinality."""

import math
from typing import Any


def calculate_entropy(series: Any, is_pandas: bool) -> float:
    """Computes Shannon entropy of column values on a representative sample."""
    try:
        if is_pandas:
            sample_series = series.sample(n=min(len(series), 10000), random_state=42)
            counts = sample_series.value_counts()
            total = counts.sum()
            if total == 0:
                return 0.0
            ent = 0.0
            for c in counts:
                p = c / total
                if p > 0:
                    ent -= p * math.log2(p)
            return float(ent)
        else:
            if series.len() == 0:
                return 0.0
            sample_series = series.sample(n=min(series.len(), 10000), seed=42)
            vc = sample_series.value_counts()
            counts = vc["count"].to_list()
            total = sum(counts)
            if total == 0:
                return 0.0
            ent = 0.0
            for c in counts:
                p = c / total
                if p > 0:
                    ent -= p * math.log2(p)
            return float(ent)
    except Exception:
        return 0.0


def calculate_binary_corr(x_null: list[bool], y_null: list[bool]) -> float:
    """Computes Pearson correlation coefficient between two null indicator vectors."""
    n = len(x_null)
    if n == 0:
        return 0.0
    sum_x = sum(x_null)
    sum_y = sum(y_null)
    if sum_x == 0 or sum_x == n or sum_y == 0 or sum_y == n:
        return 0.0

    sum_xy = sum(1 for xi, yi in zip(x_null, y_null, strict=False) if xi and yi)

    cov = sum_xy - (sum_x * sum_y / n)
    var_x = sum_x - (sum_x * sum_x / n)
    var_y = sum_y - (sum_y * sum_y / n)

    denom = math.sqrt(var_x * var_y)
    if denom == 0:
        return 0.0
    return float(cov / denom)


class DetectionEngine:
    """Scans a DataFrame to collect structural metadata without inferring business logic."""

    def __init__(self, max_sample_size: int = 10000) -> None:
        """Initializes the DetectionEngine with a maximum sample size."""
        self.max_sample_size = max_sample_size

    def analyze(self, df: Any) -> dict[str, Any]:
        """Analyzes the dataframe and returns structural metadata.

        Args:
            df: A Pandas or Polars DataFrame.

        Returns:
            Dictionary containing base types, null counts, unique value counts, and samples.
        """
        metadata: dict[str, Any] = {"columns": {}, "samples": {}}
        is_pandas = hasattr(df, "isna")

        # Check if Pandas
        if hasattr(df, "sample") and is_pandas:
            metadata["duplicate_rows"] = int(df.duplicated().sum())
            for col in df.columns:
                series = df[col]
                null_cnt = int(series.isna().sum())
                unique_cnt = int(series.nunique(dropna=True))
                total_cnt = len(series)

                null_percentage = float(null_cnt / max(total_cnt, 1))
                mem_bytes = 0
                try:
                    mem_bytes = int(series.memory_usage(deep=True))
                except Exception:
                    try:
                        mem_bytes = int(series.memory_usage())
                    except Exception:
                        pass

                metadata["columns"][col] = {
                    "dtype": str(series.dtype),
                    "null_count": null_cnt,
                    "null_percentage": null_percentage,
                    "density": 1.0 - null_percentage,
                    "unique_count": unique_cnt,
                    "cardinality": float(unique_cnt / max(total_cnt, 1)),
                    "uniqueness": bool(unique_cnt == total_cnt),
                    "total_count": total_cnt,
                    "memory_footprint_bytes": mem_bytes,
                }
                # Sample up to max_sample_size non-null values
                valid_vals = series.dropna()
                sample_size = min(len(valid_vals), self.max_sample_size)
                sample_list = (
                    valid_vals.sample(n=sample_size, random_state=42).tolist()
                    if sample_size > 0
                    else []
                )
                metadata["samples"][col] = sample_list

                # Calculate entropy
                metadata["columns"][col]["entropy"] = calculate_entropy(series, is_pandas=True)

                # Skewness and kurtosis
                skew_val = None
                kurt_val = None
                if series.dtype.kind in "biufc":
                    try:
                        skew_val = float(series.skew())
                    except Exception:
                        pass
                    try:
                        kurt_val = float(series.kurtosis())
                    except Exception:
                        pass
                metadata["columns"][col]["skewness"] = skew_val
                metadata["columns"][col]["kurtosis"] = kurt_val

                # Dtype confidence
                dtype_conf = 1.0
                if series.dtype.kind in "OS":
                    numeric_matches = 0
                    for val in sample_list:
                        if val is not None:
                            try:
                                float(str(val).strip())
                                numeric_matches += 1
                            except ValueError:
                                pass
                    if sample_list:
                        numeric_ratio = numeric_matches / len(sample_list)
                        if numeric_ratio > 0.5:
                            dtype_conf = float(1.0 - numeric_ratio)
                metadata["columns"][col]["dtype_confidence"] = dtype_conf

        elif hasattr(df, "sample") and hasattr(df, "null_count"):
            try:
                metadata["duplicate_rows"] = int(df.is_duplicated().sum())
            except Exception:
                metadata["duplicate_rows"] = 0
            for col in df.columns:
                series = df[col]
                null_cnt = int(series.null_count())
                unique_cnt = int(series.n_unique())
                total_cnt = int(series.len())

                null_percentage = float(null_cnt / max(total_cnt, 1))
                mem_bytes = 0
                try:
                    mem_bytes = int(series.estimated_size())
                except Exception:
                    pass

                metadata["columns"][col] = {
                    "dtype": str(series.dtype),
                    "null_count": null_cnt,
                    "null_percentage": null_percentage,
                    "density": 1.0 - null_percentage,
                    "unique_count": unique_cnt,
                    "cardinality": float(unique_cnt / max(total_cnt, 1)),
                    "uniqueness": bool(unique_cnt == total_cnt),
                    "total_count": total_cnt,
                    "memory_footprint_bytes": mem_bytes,
                }
                # Sample up to max_sample_size non-null values
                valid_vals = series.drop_nulls()
                sample_size = min(valid_vals.len(), self.max_sample_size)
                sample_list = (
                    valid_vals.sample(n=sample_size, seed=42).to_list()
                    if sample_size > 0
                    else []
                )
                metadata["samples"][col] = sample_list

                # Calculate entropy
                metadata["columns"][col]["entropy"] = calculate_entropy(series, is_pandas=False)

                # Skewness and kurtosis
                skew_val = None
                kurt_val = None
                if series.dtype.is_numeric():
                    try:
                        skew_val = float(series.skew())
                    except Exception:
                        pass
                    try:
                        if hasattr(series, "kurtosis"):
                            kurt_val = float(series.kurtosis())
                        elif hasattr(series, "kurt"):
                            kurt_val = float(series.kurt())
                    except Exception:
                        pass
                metadata["columns"][col]["skewness"] = skew_val
                metadata["columns"][col]["kurtosis"] = kurt_val

                # Dtype confidence
                dtype_conf = 1.0
                if series.dtype == Any or str(series.dtype).lower() in ("string", "object"):
                    numeric_matches = 0
                    for val in sample_list:
                        if val is not None:
                            try:
                                float(str(val).strip())
                                numeric_matches += 1
                            except ValueError:
                                pass
                    if sample_list:
                        numeric_ratio = numeric_matches / len(sample_list)
                        if numeric_ratio > 0.5:
                            dtype_conf = float(1.0 - numeric_ratio)
                metadata["columns"][col]["dtype_confidence"] = dtype_conf

        # Calculate null indicator correlations (vectorized)
        if hasattr(df, "columns"):
            cols_with_nulls = [c for c in df.columns if metadata["columns"][c]["null_count"] > 0]
            if len(cols_with_nulls) > 1:
                try:
                    import polars as pl
                    sample_len = min(df.height if hasattr(df, "height") else len(df), 5000)
                    if is_pandas:
                        null_df = df[cols_with_nulls].head(sample_len).isna().astype(int)
                        import warnings
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=RuntimeWarning)
                            corr_matrix = null_df.corr()
                        for col in cols_with_nulls:
                            corrs = {}
                            for other in cols_with_nulls:
                                if col == other:
                                    continue
                                val = corr_matrix.at[col, other]
                                if not math.isnan(val) and abs(val) > 0.1:
                                    corrs[other] = float(val)
                            metadata["columns"][col]["null_correlations"] = corrs
                    else:
                        null_df = df.head(sample_len).select([pl.col(c).is_null().cast(pl.Int32).alias(c) for c in cols_with_nulls])
                        import warnings
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=RuntimeWarning)
                            corr_matrix = null_df.corr()
                        corr_dict = corr_matrix.to_dict(as_series=False)
                        for i, col in enumerate(cols_with_nulls):
                            corrs = {}
                            for _j, other in enumerate(cols_with_nulls):
                                if col == other:
                                    continue
                                val = corr_dict[other][i]
                                if val is not None and not math.isnan(val) and abs(val) > 0.1:
                                    corrs[other] = float(val)
                            metadata["columns"][col]["null_correlations"] = corrs
                except Exception:
                    pass

        # Estimate total memory footprint
        total_mem = 0
        if hasattr(df, "estimated_size"):
            try:
                total_mem = int(df.estimated_size())
            except Exception:
                pass
        elif hasattr(df, "memory_usage"):
            try:
                total_mem = int(df.memory_usage(deep=True).sum())
            except Exception:
                try:
                    total_mem = int(df.memory_usage().sum())
                except Exception:
                    pass
        if total_mem == 0:
            total_mem = sum(c_info.get("memory_footprint_bytes", 0) for c_info in metadata["columns"].values())
        metadata["memory_footprint_bytes"] = total_mem

        # Estimate execution cost (roughly 0.01 ms per cell)
        row_count = df.height if hasattr(df, "height") else len(df)
        col_count = len(df.columns) if hasattr(df, "columns") else 1
        metadata["estimated_execution_cost_ms"] = float(row_count * col_count * 0.01)

        return metadata
