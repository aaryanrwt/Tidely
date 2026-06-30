"""Intelligent Decision Engine for automated algorithm selection and backend routing."""

import os
import sys
from typing import Any, Dict, List, Tuple

try:
    import psutil
except ImportError:
    psutil = None


class DecisionEngine:
    """Central decision engine for Tidely v1.3.
    
    Dynamically profiles system resources and dataset metadata to select
    the optimal backend, memory strategy, and cleaning algorithms.
    """

    def __init__(self):
        # Hardware profiling
        self.cpu_cores = os.cpu_count() or 4
        self.total_ram_bytes = self._get_total_ram()
        
    def _get_total_ram(self) -> int:
        """Determines total system memory, falling back to 8GB if psutil is unavailable."""
        if psutil is not None:
            try:
                return psutil.virtual_memory().total
            except Exception:
                pass
        return 8 * 1024 * 1024 * 1024  # Default 8 GB

    def get_available_ram(self) -> int:
        """Determines currently available system memory."""
        if psutil is not None:
            try:
                return psutil.virtual_memory().available
            except Exception:
                pass
        return 4 * 1024 * 1024 * 1024  # Default 4 GB

    def route_backend(self, dataset_size_bytes: int, file_format: str = "csv") -> str:
        """Selects the execution backend and streaming strategy.
        
        Rules:
        - If dataset exceeds 80% of available RAM -> 'polars_lazy_streaming'
        - If SQL-heavy or database source -> 'duckdb' (Not yet fully native, fallback to polars)
        - Else -> 'polars_lazy' (Standard optimized path)
        """
        available_ram = self.get_available_ram()
        
        # Exceeds RAM threshold
        if dataset_size_bytes > available_ram * 0.8:
            return "polars_lazy_streaming"
        
        # Check Spark/Dask environment variables
        if "SPARK_HOME" in os.environ or "pyspark" in sys.modules:
            return "spark"
        if "dask" in sys.modules:
            return "dask"
            
        return "polars_lazy"

    def select_imputation_strategy(
        self, 
        column_name: str, 
        dtype_str: str, 
        null_count: int, 
        total_count: int, 
        unique_count: int, 
        is_skewed: bool = False
    ) -> Tuple[str, Dict[str, Any]]:
        """Selects the optimal imputation strategy.
        
        Strategies:
        - Sparse Categorical -> Mode
        - Time-Series columns -> Forward Fill
        - Numeric Skewed -> Median
        - Numeric Normal -> Mean
        - Complex High-Dim / Correlated -> KNN or MICE (if sklearn installed)
        """
        null_ratio = null_count / max(total_count, 1)
        
        # If low cardinality and string
        if "str" in dtype_str or "object" in dtype_str or "cat" in dtype_str:
            if "date" in column_name.lower() or "time" in column_name.lower():
                return "forward_fill", {}
            return "impute_mode", {"value": "Unknown"}
            
        # Numeric checks
        if "int" in dtype_str or "float" in dtype_str or "num" in dtype_str:
            # Check for MICE/KNN opportunities (e.g. moderate nulls, multiple features)
            # To keep startup times low, check if scikit-learn is already imported or available
            if null_ratio > 0.05 and total_count > 500 and "sklearn" in sys.modules:
                return "impute_mice", {}
                
            if is_skewed:
                return "impute_median", {}
            return "impute_mean", {}
            
        return "impute_constant", {"value": "Unknown"}

    def select_duplicate_strategy(
        self, 
        row_count: int, 
        col_count: int, 
        semantic_type: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Selects the deduplication algorithm.
        
        Strategies:
        - Exact Rows -> Exact Hash Comparison (Polars unique)
        - Near duplicates / Typo emails/phones -> RapidFuzz / TF-IDF
        - Business address/name duplicates -> Record Linkage
        """
        if semantic_type in ("Email", "Phone", "InvoiceID", "CustomerID", "SKU"):
            return "exact_match", {}
            
        if semantic_type == "Address" and "rapidfuzz" in sys.modules:
            return "rapidfuzz_similarity", {"threshold": 90}
            
        return "exact_match", {}

    def select_outlier_strategy(
        self, 
        column_name: str, 
        row_count: int, 
        is_normal_dist: bool = False
    ) -> Tuple[str, Dict[str, Any]]:
        """Selects outlier detection strategy.
        
        Strategies:
        - Normal Distribution -> Z-score (Standard threshold = 3.0)
        - Skewed / Unknown distribution -> IQR (Interquartile Range)
        - High dimensional -> Isolation Forest (if sklearn is installed)
        - Local density anomalies -> LOF (if sklearn is installed)
        """
        # If sklearn is available, use Isolation Forest for large datasets
        if row_count > 5000 and "sklearn" in sys.modules:
            return "isolation_forest", {"contamination": 0.01}
            
        if is_normal_dist:
            return "z_score", {"threshold": 3.0}
            
        return "iqr", {"threshold": 1.5}

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
            return "ordinal"
        if unique_count <= 15:
            return "one_hot"
        if cardinality_ratio < 0.1:
            return "frequency"
        return "target_encoding"
