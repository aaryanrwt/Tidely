"""Tidely v1.5.0 — Benchmark CLI Entry Point.

Usage:
    python benchmarks/run_benchmark.py              # Full benchmark run
    python benchmarks/run_benchmark.py --smoke-test # CI smoke test (2 datasets)
    python benchmarks/run_benchmark.py --check-regression  # Fail if regressions found
"""

from __future__ import annotations

import os
import sys

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    """Main entry point for the benchmark CLI.

    Returns:
        Exit code: 0 = success, 1 = regression detected.
    """
    args = set(sys.argv[1:])
    smoke_test = "--smoke-test" in args
    check_regression = "--check-regression" in args

    if check_regression:
        # Only read existing results and check for regressions
        from benchmarks.reporter import check_regressions

        passed = check_regressions()
        return 0 if passed else 1

    # Full or smoke-test run
    from benchmarks.engine import run_all
    from benchmarks.reporter import check_regressions, generate_reports

    results = run_all(smoke_test=smoke_test)
    generate_reports(results)

    # Check regressions after generating reports
    passed = check_regressions()
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
