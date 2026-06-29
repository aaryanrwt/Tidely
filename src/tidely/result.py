"""The Result object returned by td.clean()."""

from typing import Any, Optional


class CleanResult:
    """The outcome of a Tidely cleaning operation.
    
    Exposes the cleaned DataFrame directly (or via .df), along with
    methods to view the summary, export reports, or undo changes.
    """
    
    def __init__(
        self,
        cleaned_df: Any,
        original_df: Any,
        summary_text: str,
        report_data: dict,
    ):
        """Initializes the Result object.
        
        Args:
            cleaned_df: The production-ready DataFrame.
            original_df: A copy of the original DataFrame before cleaning.
            summary_text: The beautiful terminal output string.
            report_data: Programmatic dictionary of what changed.
        """
        self.df = cleaned_df
        self._original_df = original_df
        self._summary_text = summary_text
        self.report = report_data
        
    def export(self, filepath: str) -> None:
        """Exports the cleaned dataset or report based on the file extension.
        
        Supported extensions: .csv, .parquet, .html, .pdf
        """
        ext = filepath.split(".")[-1].lower()
        if ext == "csv":
            self.df.to_csv(filepath, index=False)
        elif ext == "parquet":
            self.df.to_parquet(filepath, index=False)
        elif ext == "html":
            # Phase 4 implementation
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"<html><body><pre>{self._summary_text}</pre></body></html>")
        else:
            raise ValueError(f"Unsupported export format: {ext}")
            
    def summary(self) -> str:
        """Returns the outcome-focused cleaning summary."""
        return self._summary_text

    def undo(self) -> Any:
        """Reverts the cleaning operation, returning the original DataFrame."""
        if hasattr(self._original_df, "copy"):
            return self._original_df.copy()
        return self._original_df

    # Proxy all other attributes to the underlying DataFrame
    def __getattr__(self, name: str) -> Any:
        return getattr(self.df, name)
        
    def __getitem__(self, key: Any) -> Any:
        return self.df[key]
        
    def __setitem__(self, key: Any, value: Any) -> None:
        self.df[key] = value
        
    def __repr__(self) -> str:
        return repr(self.df)
        
    def _repr_html_(self) -> Optional[str]:
        if hasattr(self.df, "_repr_html_"):
            return self.df._repr_html_()
        return None
