from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from tidely.cli.main import app

runner = CliRunner()


def test_cli_inspect_success(tmp_path: Path) -> None:
    """Verifies inspect command successfully runs on a valid dataset."""
    df = pl.DataFrame({"id": [1, 2], "val": ["A", "B"]})
    file_path = tmp_path / "test.csv"
    df.write_csv(file_path)

    result = runner.invoke(app, ["--input", str(file_path)])

    assert result.exit_code == 0
    assert "SPOTLESS DATASET INSPECTOR" in result.stdout
    assert "Dataset DNA & Health" in result.stdout


def test_cli_inspect_unsupported_format(tmp_path: Path) -> None:
    """Verifies inspect command gracefully handles unsupported formats."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("unsupported content")

    result = runner.invoke(app, ["--input", str(file_path)])

    assert result.exit_code == 1
    assert "Unsupported file format" in result.output
