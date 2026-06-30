import subprocess
import sys


def run_cli(command: list, cwd: str = None) -> dict:
    """Run a Tidely CLI command and capture exit code, stdout, stderr.

    Returns a dict with keys: returncode, stdout, stderr.
    """
    # Replace 'tidely' with the module entry point for reliable execution
    if command and command[0] == "tidely":
        command = [sys.executable, "-m", "tidely.cli.main"] + command[1:]
    result = subprocess.run(command, cwd=cwd, capture_output=True, encoding="utf-8")
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def assert_success(result: dict, cmd_desc: str):
    """Assert that a CLI command completed successfully."""
    if result["returncode"] != 0:
        raise AssertionError(
            f"CLI command failed ({cmd_desc}). Return code {result['returncode']}\n"
            f"stdout:\n{result['stdout']}\n"
            f"stderr:\n{result['stderr']}"
        )


def test_cli_help():
    """Test --help flag works."""
    res = run_cli(["tidely", "--help"])
    assert_success(res, "tidely --help")


def test_cli_version():
    """Test --version flag reports the correct version."""
    import tidely as td

    res = run_cli(["tidely", "--version"])
    assert_success(res, "tidely --version")
    output = res["stdout"] + res["stderr"]
    assert td.__version__ in output


def test_cli_clean(tmp_path):
    """Test tidely clean command."""
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    cleaned_path = tmp_path / "cleaned_output.csv"
    res = run_cli(
        ["tidely", "clean", str(csv_path), "--out", str(cleaned_path)],
        cwd=str(tmp_path),
    )
    assert_success(res, "tidely clean")
    assert cleaned_path.is_file(), f"Cleaned file not created: {cleaned_path}"


def test_cli_inspect(tmp_path):
    """Test tidely inspect command."""
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    res = run_cli(["tidely", "inspect", str(csv_path)], cwd=str(tmp_path))
    assert_success(res, "tidely inspect")


def test_cli_report(tmp_path):
    """Test tidely report command."""
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    report_path = tmp_path / "report.html"
    res = run_cli(
        ["tidely", "report", str(csv_path), "--out", str(report_path)],
        cwd=str(tmp_path),
    )
    assert_success(res, "tidely report")
    assert report_path.is_file(), f"Report file not created: {report_path}"
