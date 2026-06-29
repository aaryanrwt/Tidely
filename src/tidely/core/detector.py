"""Detection Engine for analyzing base types, missing values, and cardinality."""

from typing import Any, Dict


class DetectionEngine:
    """Scans a DataFrame to collect structural metadata without inferring business logic."""
    
    def __init__(self, max_sample_size: int = 10000):
        self.max_sample_size = max_sample_size
        
    def analyze(self, df: Any) -> Dict[str, Any]:
        """Analyzes the dataframe and returns structural metadata.
        
        Args:
            df: A Pandas or Polars DataFrame.
            
        Returns:
            Dictionary containing base types, null counts, unique value counts, and samples.
        """
        metadata = {"columns": {}, "samples": {}}
        
        # Check if Pandas
        if hasattr(df, "sample") and hasattr(df, "isna"):
            # Pandas path
            metadata["duplicate_rows"] = int(df.duplicated().sum())
            for col in df.columns:
                series = df[col]
                metadata["columns"][col] = {
                    "dtype": str(series.dtype),
                    "null_count": series.isna().sum(),
                    "unique_count": series.nunique(dropna=True),
                    "total_count": len(series)
                }
                # Sample up to max_sample_size non-null values
                valid_vals = series.dropna()
                sample_size = min(len(valid_vals), self.max_sample_size)
                metadata["samples"][col] = valid_vals.sample(n=sample_size, random_state=42).tolist() if sample_size > 0 else []
                
        elif hasattr(df, "sample") and hasattr(df, "null_count"):
            # Polars eager path (approximate check)
            try:
                metadata["duplicate_rows"] = df.is_duplicated().sum()
            except Exception:
                metadata["duplicate_rows"] = 0
            for col in df.columns:
                series = df[col]
                metadata["columns"][col] = {
                    "dtype": str(series.dtype),
                    "null_count": series.null_count(),
                    "unique_count": series.n_unique(),
                    "total_count": series.len()
                }
                # Sample up to max_sample_size non-null values
                valid_vals = series.drop_nulls()
                sample_size = min(valid_vals.len(), self.max_sample_size)
                metadata["samples"][col] = valid_vals.sample(n=sample_size, seed=42).to_list() if sample_size > 0 else []
                
        return metadata
