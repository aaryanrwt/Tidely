import json
import sys
import traceback
from .helpers import read_readme_snippets, capture_performance


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
    snippets = read_readme_snippets()
    assert snippets, "No Python snippets found in README.md"
    failures = []
    for heading, code in snippets:
        # Skip snippets that attempt to load external files which are not present in CI
        if "td.clean(" in code and "\"" in code:
            # Heuristic: if the snippet contains a literal string argument to td.clean, skip it
            continue
        err = exec_snippet(code)
        if err:
            failures.append({"heading": heading, "error": repr(err), "traceback": traceback.format_exc()})
    if failures:
        print(json.dumps({"readme_snippet_failures": failures}, indent=2), file=sys.stderr)
    assert not failures, f"{len(failures)} README snippets failed"
