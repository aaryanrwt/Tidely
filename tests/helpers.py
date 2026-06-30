import os
import pathlib
import subprocess
import sys
import time
from typing import Any


def load_dataset(path: str) -> Any:
    """Load a dataset using Tidely's clean API and return the CleanResult.

    This helper is used by many tests to abstract the common pattern.
    """
    import tidely as td

    return td.clean(path)


def metric_counts(df) -> dict:
    """Return a dictionary of basic quality metrics for a DataFrame.

    Supports both Polars and pandas DataFrames.
    """
    if hasattr(df, "shape"):
        rows, cols = df.shape
    else:
        rows = len(df)
        cols = len(df.columns) if hasattr(df, "columns") else None
    # Missing values
    if hasattr(df, "null_count"):
        # Polars DataFrame
        missing = sum(df.null_count())
    else:
        missing = df.isnull().sum().sum()
    # Duplicate rows
    if hasattr(df, "is_duplicated"):
        dup = df.is_duplicated().sum()
    else:
        dup = df.duplicated().sum()
    return {
        "rows": rows,
        "cols": cols,
        "missing": int(missing),
        "duplicates": int(dup),
    }


def capture_performance(func, *args, **kwargs) -> tuple[Any, dict]:
    """Run ``func`` and return its result plus a dict with timing and memory usage."""
    try:
        import psutil

        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss
    except ImportError:
        start_mem = 0
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    try:
        import psutil

        process = psutil.Process(os.getpid())
        end_mem = process.memory_info().rss
    except ImportError:
        end_mem = 0
    return result, {
        "exec_time_s": end - start,
        "peak_mem_mb": (end_mem - start_mem) / (1024 * 1024),
    }


def run_cli(command: list, cwd: str = None) -> dict:
    """Run a Tidely CLI command and capture exit code, stdout, stderr.

    Returns a dict with keys: returncode, stdout, stderr.
    """
    # Use sys.executable for reliable cross-platform execution
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


def read_readme_snippets() -> list:
    """Extract all Python code blocks from README.md.

    Returns a list of (source, code) tuples where ``source`` is the surrounding
    markdown heading for debugging.
    """
    readme_path = pathlib.Path(__file__).parents[1] / "README.md"
    snippets = []
    if not readme_path.is_file():
        return snippets
    in_block = False
    block = []
    heading = ""
    for line in readme_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            heading = line.strip()
        stripped = line.lstrip()
        if stripped.startswith("```python"):
            # start of a python fence
            if not in_block:
                in_block = True
                block = []
                continue
        elif stripped.startswith("```"):
            # end of any fence
            if in_block:
                in_block = False
                snippets.append((heading, "\n".join(block)))
                continue
        if in_block:
            block.append(line)
    return snippets
