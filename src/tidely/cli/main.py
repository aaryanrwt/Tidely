"""Command-line interface implementation for Tidely."""

import os

import polars as pl
import typer
from rich.console import Console

from tidely import inspect
from tidely.core.errors import TidelyError
from tidely.core.logging import setup_logging

app = typer.Typer(
    name="tidely",
    help="Tidely: The Operating System for Data Quality",
    add_completion=False,
)


@app.command("inspect")
def inspect_cmd(
    input_path: str = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path to the dataset file (CSV, Parquet, or JSON).",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    log_level: str = typer.Option(
        "WARNING",
        "--log-level",
        "-l",
        help="Set logging level (DEBUG, INFO, WARNING, ERROR).",
    ),
) -> None:
    """Diagnoses a dataset and displays a stunning visual summary of its DNA, Trust Scores, and Quality."""
    err_console = Console(stderr=True)
    setup_logging(log_level)

    try:
        # Load dataset using Polars
        _, ext = os.path.splitext(input_path.lower())
        if ext == ".csv":
            df = pl.read_csv(input_path)
        elif ext == ".parquet":
            df = pl.read_parquet(input_path)
        elif ext == ".json":
            # Try loading standard or NDJSON
            try:
                df = pl.read_json(input_path)
            except Exception:
                df = pl.read_ndjson(input_path)
        else:
            raise TidelyError(
                f"Unsupported file format '{ext}'. Tidely supports .csv, .parquet, and .json."
            )

        # Run inspection
        profile = inspect(df)
        profile.show()

    except TidelyError as e:
        e.show()
        raise typer.Exit(code=1) from None
    except Exception as e:
        err_console.print(f"[red]Unexpected Error:[/red] {e}")
        raise typer.Exit(code=1) from None


if __name__ == "__main__":
    app()
