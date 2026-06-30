"""Intelligent Decision Engine for automated algorithm selection and backend routing."""

import os
import sys
from typing import Any

try:
    import psutil
except ImportError:
    psutil = None


class DecisionEngine:
    """Central decision engine for Tidely v1.3.

    Dynamically profiles system resources and dataset metadata to select
    the optimal backend, memory strategy, and cleaning algorithms.
    """

    def __init__(self) -> None:
        """Initializes the DecisionEngine and profiles system resources."""
        # Hardware profiling
        self.cpu_cores = os.cpu_count() or 4
        self.total_ram_bytes = self._get_total_ram()
        self.selected_engine = "polars_eager"
        self.selected_reason = "Low-latency default in-memory execution."

    def _get_total_ram(self) -> int:
        """Determines total system memory, falling back to 8GB if psutil is unavailable."""
        if psutil is not None:
            try:
                return int(psutil.virtual_memory().total)
            except Exception:
                pass
        return 8 * 1024 * 1024 * 1024  # Default 8 GB

    def get_available_ram(self) -> int:
        """Determines currently available system memory."""
        if psutil is not None:
            try:
                return int(psutil.virtual_memory().available)
            except Exception:
                pass
        return 4 * 1024 * 1024 * 1024  # Default 4 GB

    def route_backend(self, dataset_size_bytes: int, file_format: str = "csv") -> str:
        """Selects the execution backend and streaming strategy based on system resources and dataset size.

        Rules:
        - If dataset exceeds 50% of available RAM -> 'streaming'
        - If dataset is a large CSV/Parquet file (> 50MB) -> 'duckdb'
        - If dataset is medium (10MB - 50MB) -> 'polars_lazy'
        - Else -> 'polars_eager'
        """
        available_ram = self.get_available_ram()
        fmt_lower = file_format.lower()

        # Exceeds RAM threshold
        if dataset_size_bytes > available_ram * 0.5:
            self.selected_engine = "streaming"
            self.selected_reason = (
                f"Dataset size ({dataset_size_bytes / (1024*1024):.1f} MB) exceeds 50% "
                f"of available RAM ({available_ram / (1024*1024):.1f} MB). Routing to Out-of-Core Streaming."
            )
            return "streaming"

        # Large files
        if dataset_size_bytes > 50 * 1024 * 1024 and ("csv" in fmt_lower or "parquet" in fmt_lower):
            self.selected_engine = "duckdb"
            self.selected_reason = (
                f"Dataset size ({dataset_size_bytes / (1024*1024):.1f} MB) exceeds 50MB limit for "
                f"in-memory processing. Routing to DuckDB query engine for out-of-core acceleration."
            )
            return "duckdb"

        # Medium datasets
        if dataset_size_bytes > 10 * 1024 * 1024:
            self.selected_engine = "polars_lazy"
            self.selected_reason = (
                f"Dataset size ({dataset_size_bytes / (1024*1024):.1f} MB) is between 10MB and 50MB. "
                "Routing to Polars Lazy evaluation for optimized query execution plan."
            )
            return "polars_lazy"

        # Small datasets
        self.selected_engine = "polars_eager"
        self.selected_reason = (
            f"Dataset size ({dataset_size_bytes / (1024*1024):.1f} MB) fits comfortably in memory. "
            "Routing to Polars Eager for low-latency in-memory execution."
        )
        return "polars_eager"

    def select_imputation_strategy(
        self,
        column_name: str,
        dtype_str: str,
        null_count: int,
        total_count: int,
        unique_count: int,
        is_skewed: bool = False,
        null_correlations: dict[str, float] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Selects the optimal imputation strategy by classifying MCAR vs. MAR vs. MNAR.

        Strategies:
        - MAR (Missing At Random): Impute by correlated column's group median/mode.
        - MNAR (Missing Not At Random) or Sequential/Time Date -> Interpolation or bfill/ffill.
        - MCAR (Missing Completely At Random) normal -> Mean.
        - MCAR skewed -> Median.
        - Categorical -> Mode.
        """
        null_ratio = null_count / max(total_count, 1)

        # High missingness threshold
        if null_ratio > 0.8:
            return "impute_constant", {"value": "Unknown"}

        dtype_lower = dtype_str.lower()

        # Check for MAR (Missing At Random) via null correlations
        if null_correlations:
            # Find the column with the highest absolute correlation >= 0.3
            mar_col = None
            max_corr = 0.0
            for col, corr in null_correlations.items():
                if abs(corr) >= 0.3 and abs(corr) > max_corr:
                    max_corr = abs(corr)
                    mar_col = col
            if mar_col is not None:
                if "str" in dtype_lower or "object" in dtype_lower or "cat" in dtype_lower or "enum" in dtype_lower:
                    return "impute_group_mode", {"group_column": mar_col}
                else:
                    return "impute_group_median", {"group_column": mar_col}

        # Check for MNAR / Time-series
        col_lower = column_name.lower()
        if "date" in col_lower or "time" in col_lower or "ts" in col_lower or "year" in col_lower:
            return "forward_fill", {}

        # If low cardinality and string
        if "str" in dtype_lower or "object" in dtype_lower or "cat" in dtype_lower or "enum" in dtype_lower:
            return "impute_mode", {"value": "Unknown"}

        # Numeric checks
        if "int" in dtype_lower or "float" in dtype_lower or "num" in dtype_lower or "double" in dtype_lower or "real" in dtype_lower:
            if "sklearn" in sys.modules and null_ratio > 0.05 and total_count > 1000:
                # If sklearn is available, we can route to MICE/KNN
                return "impute_mice", {}
            if is_skewed:
                return "impute_median", {}
            return "impute_mean", {}

        return "impute_constant", {"value": "Unknown"}

    def select_duplicate_strategy(
        self, row_count: int, col_count: int, semantic_type: str
    ) -> tuple[str, dict[str, Any]]:
        """Selects the deduplication algorithm.

        Strategies:
        - Exact Rows -> Exact Hash Comparison (Polars unique)
        - Near duplicates / Typo emails/phones -> RapidFuzz / TF-IDF
        - Business address/name duplicates -> Record Linkage
        """
        if semantic_type in ("Email", "Phone", "InvoiceID", "CustomerID", "SKU"):
            return "exact_match", {}

        if semantic_type in ("Address", "Name") and "rapidfuzz" in sys.modules:
            return "rapidfuzz_similarity", {"threshold": 90}

        return "exact_match", {}

    def select_outlier_strategy(
        self,
        column_name: str,
        row_count: int,
        is_normal_dist: bool = False,
        skewness: float | None = None,
        kurtosis: float | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Selects outlier detection strategy.

        Strategies:
        - Normal Distribution -> Z-score (Standard threshold = 3.0)
        - Skewed / Unknown distribution -> MAD / Modified Z-score (Standard threshold = 3.5)
        - High dimensional -> Isolation Forest (if sklearn is installed)
        - Local density anomalies -> LOF (if sklearn is installed)
        - Extremely heavy-tailed -> IQR
        """
        # If sklearn is available, use Isolation Forest for large datasets
        if row_count > 5000 and "sklearn" in sys.modules:
            return "isolation_forest", {"contamination": 0.01}

        # Check kurtosis/skewness for heavy tails
        if skewness is not None and abs(skewness) > 1.5:
            return "iqr", {"threshold": 1.5}

        if is_normal_dist:
            return "z_score", {"threshold": 3.0}

        return "modified_zscore", {"threshold": 3.5}

    def select_scaling_strategy(self, is_normal_dist: bool, has_outliers: bool) -> str:
        """Selects scaling method: StandardScaler, RobustScaler, or MinMaxScaler."""
        if has_outliers:
            return "robust_scaler"
        if is_normal_dist:
            return "standard_scaler"
        return "min_max_scaler"

    def select_encoding_strategy(self, unique_count: int, total_count: int) -> str:
        """Selects category encoding: OneHot, Ordinal, Target, or Frequency."""
        cardinality_ratio = unique_count / max(total_count, 1)
        if unique_count <= 2:
            return "column_ordinal"
        if unique_count <= 15:
            return "one_hot"
        if cardinality_ratio < 0.1:
            return "frequency"
        return "target_encoding"
