"""Custom exception classes for the Tidely data cleaning engine."""

from rich.console import Console
from rich.panel import Panel


class TidelyError(Exception):
    """Base exception for all Tidely library errors."""

    def show(self) -> None:
        """Prints a user-friendly, rich-formatted error panel to the terminal."""
        console = Console(stderr=True)
        panel = Panel(
            f"[bold white]{str(self)}[/bold white]",
            title="[bold red]❌ Tidely Diagnostic Error[/bold red]",
            border_style="red",
            title_align="left",
        )
        console.print(panel)


class DatasetError(TidelyError):
    """Base exception for dataset-related operations."""

    pass


class DatasetLoadError(DatasetError):
    """Raised when a dataset fails to load from a source."""

    pass


class PipelineError(TidelyError):
    """Base exception for data pipeline operations."""

    pass


class PipelineExecutionError(PipelineError):
    """Raised when a pipeline fails during execution."""

    pass


class PluginError(TidelyError):
    """Base exception for plugin-related operations."""

    pass


class PluginLoadError(PluginError):
    """Raised when a plugin fails to register or load."""

    pass


class ValidationError(TidelyError):
    """Raised when data validation fails."""

    pass


class ConfigurationError(TidelyError):
    """Raised when configuration validation or parsing fails."""

    pass
