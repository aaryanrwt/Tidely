"""Command-line interface for Tidely using Typer and Rich."""

import os
from typing import Optional
import typer
from rich.console import Console

import tidely as td

app = typer.Typer(
    name="tidely",
    help="Tidely: The Intelligent Data Cleaning Engine.",
    no_args_is_help=True
)

console = Console()


@app.command()
def clean(
    input_file: str = typer.Argument(..., help="Path to the raw CSV, Parquet, or JSON file"),
    output: Optional[str] = typer.Option("cleaned_data.csv", "--out", "-o", help="Output file path"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Simulate cleaning without saving")
):
    """Cleans a dataset automatically and outputs a beautiful consult-grade summary."""
    if not os.path.exists(input_file):
        console.print(f"[bold red]Error:[/bold red] File not found: {input_file}")
        raise typer.Exit(code=1)

    console.print(f"[bold cyan]Tidely v{td.__version__}[/bold cyan] [dim]is processing '{input_file}'...[/dim]\n")
    
    try:
        # Load and clean automatically using Tidely unified engine
        result = td.clean(input_file)
        
        # summary is already printed inside plan.show() which clean() triggers,
        # but let's make sure the CleanResult summary is accessible
        
        if not dry_run:
            td.save(result.df, output)
            console.print(f"\n[bold green]Success![/bold green] Cleaned dataset saved to [underline]{output}[/underline]")
    except Exception as e:
        console.print(f"[bold red]Execution failed:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def inspect(
    input_file: str = typer.Argument(..., help="Path to the raw CSV, Parquet, or JSON file")
):
    """Profiles a dataset and generates a comprehensive Trust Score and semantic diagnosis."""
    if not os.path.exists(input_file):
        console.print(f"[bold red]Error:[/bold red] File not found: {input_file}")
        raise typer.Exit(code=1)

    console.print(f"[bold cyan]Tidely v{td.__version__}[/bold cyan] [dim]is profiling '{input_file}'...[/dim]\n")
    
    try:
        profile = td.inspect(input_file)
        profile.show()
    except Exception as e:
        console.print(f"[bold red]Inspection failed:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def report(
    input_file: str = typer.Argument(..., help="Path to the raw CSV, Parquet, or JSON file"),
    output: Optional[str] = typer.Option("tidely_report.html", "--out", "-o", help="Output report path")
):
    """Generates an explainable HTML diagnostic report for a dataset."""
    if not os.path.exists(input_file):
        console.print(f"[bold red]Error:[/bold red] File not found: {input_file}")
        raise typer.Exit(code=1)

    console.print(f"[bold cyan]Tidely v{td.__version__}[/bold cyan] [dim]is generating HTML report for '{input_file}'...[/dim]\n")
    
    try:
        result = td.clean(input_file)
        result.export(output)
        console.print(f"\n[bold green]Success![/bold green] HTML report exported to [underline]{output}[/underline]")
    except Exception as e:
        console.print(f"[bold red]Report generation failed:[/bold red] {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
