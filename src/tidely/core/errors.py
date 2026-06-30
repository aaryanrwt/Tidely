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
            subtitle="[dim]Check the docs for more info: https://github.com/aaryanrwt/tidely[/dim]",
        )
        console.print(panel)


class TidelyDataError(TidelyError):
    """Raised when the dataset itself is fundamentally corrupted or malformed.

    This error is highly actionable. The message will typically specify which
    format was received and which formats are explicitly supported (e.g. Pandas, Polars).
    """

    pass


class TidelySchemaError(TidelyError):
    """Raised when the dataset fails to validate against a provided strict schema.

    Provides actionable guidance on which exact column failed validation,
    what the expected type was, and what the actual type is.
    """

    pass


class TidelyBackendError(TidelyError):
    """Raised when a specific backend (e.g. Pandas, Polars) fails an internal operation.

    Typically indicates an environment issue (e.g., PyArrow is missing but required).
    """

    pass


class TidelyExecutionError(TidelyError):
    """Raised when a vectorized operation fails during the execution pipeline.

    Often caused by extreme memory pressure or an unsupported hardware instruction.
    """

    pass


class ConfigurationError(TidelyError):
    """Raised when configuration parsing or validation fails."""

    pass
