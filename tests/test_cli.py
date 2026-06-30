import subprocess
import sys
import pathlib
import json
import traceback

from .helpers import run_cli, assert_success

def run_cli(command: list, cwd: str = None) -> dict:
    """Run a Tidely CLI command and capture exit code, stdout, stderr.
    Returns a dict with keys: returncode, stdout, stderr.
    """
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def assert_success(result: dict, cmd_desc: str):
    if result["returncode"] != 0:
        raise AssertionError(
            f"CLI command failed ({cmd_desc}). Return code {result['returncode']}\n"
            f"stdout:\n{result['stdout']}\n"
            f"stderr:\n{result['stderr']}"
        )


def test_cli_commands(tmp_path):
    # Use a temporary directory to avoid polluting repo
    work_dir = pathlib.Path(tmp_path)
    # Create a tiny CSV file
    csv_path = work_dir / "sample.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    # 1. tidely clean
    # 1. tidily clean
    res = run_cli(["tidely", "clean", str(csv_path), "--out", str(cleaned_path)], cwd=str(work_dir))
    assert_success(res, "tidely clean")
    # Expect a cleaned file named sample_cleaned.csv or similar
    cleaned_path = work_dir / "sample_cleaned.csv"
    assert cleaned_path.is_file(), f"Cleaned file not created: {cleaned_path}"

    # 2. tidely inspect
    res = run_cli(["tidely", "inspect", str(csv_path)], cwd=str(work_dir))
    assert_success(res, "tidely inspect")
    assert "Inspection" in res["stdout"], "Inspect output missing key word"

    # 3. tidely summary
    res = run_cli(["tidely", "summary", str(csv_path)], cwd=str(work_dir))
    assert_success(res, "tidely summary")
    assert "Summary" in res["stdout"], "Summary output missing key word"

    # 4. tidely export (export to json)
    res = run_cli([
        "tidely", "export", str(csv_path), "--format", "json", "--output", str(work_dir / "out.json")
    ], cwd=str(work_dir))
    assert_success(res, "tidely export")
    assert (work_dir / "out.json").is_file(), "Exported JSON not created"

    # 5. --help
    res = run_cli(["tidely", "--help"], cwd=str(work_dir))
    assert_success(res, "tidely --help")
    assert "Command" in res["stdout"], "Help text missing"

    # 6. --version
    res = run_cli(["tidely", "--version"], cwd=str(work_dir))
    assert_success(res, "tidely --version")
    assert "1.4.0" in res["stdout"], "Version output incorrect"
