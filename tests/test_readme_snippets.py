import json
import os
import sys
import traceback

import numpy as np
import pandas as pd

from .helpers import read_readme_snippets


def exec_snippet(code: str, globals_dict=None):
    """Execute a code snippet in a fresh namespace.
    Returns any exception raised, otherwise None.
    """
    if globals_dict is None:
        globals_dict = {}
    try:
        exec(code, globals_dict)
        return None
    except Exception as e:
        return e


def test_readme_snippets():
    # Set up dummy files in the current working directory so the snippets can run successfully
    dummy_files = {
        "dirty_data.csv": pd.DataFrame({
            "Department": ["HR", "Engineering", "HR"],
            "Salary": [50000.0, 80000.0, np.nan],
            "Zip": ["123.0", "94043.0", "90210"],
            "Email": ["ALICE@GMAIL.COM", "bob@yahoo.com", "charlie"],
            "Latitude": [37.7749, -122.4194, "invalid"]
        }),
        "data.csv": pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "email": ["alice@gmail.com", "bob@yahoo.com", "charlie@gmail.com"]
        })
    }

    # Write dummy files
    for name, df in dummy_files.items():
        df.to_csv(name, index=False)

    snippets = read_readme_snippets()
    assert snippets, "No Python snippets found in README.md"

    failures = []
    try:
        for heading, code in snippets:
            # We don't skip anything now because the dummy files are present!
            err = exec_snippet(code)
            if err:
                failures.append({
                    "heading": heading,
                    "error": repr(err),
                    "traceback": traceback.format_exc()
                })
    finally:
        # Clean up any created files
        files_to_remove = [
            "dirty_data.csv",
            "data.csv",
            "clean_data.csv",
            "clean.csv",
            "report.html"
        ]
        for name in files_to_remove:
            if os.path.exists(name):
                try:
                    os.remove(name)
                except Exception:
                    pass

    if failures:
        print(json.dumps({"readme_snippet_failures": failures}, indent=2), file=sys.stderr)
    assert not failures, f"{len(failures)} README snippets failed"
