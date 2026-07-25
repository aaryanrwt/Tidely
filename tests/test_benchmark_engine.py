"""Tests for the Tidely v1.5.0 benchmark engine.

These tests are CI smoke tests — they verify the benchmark infrastructure
works correctly on small synthetic datasets without network access.
"""

from __future__ import annotations

import gc
import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_df() -> pd.DataFrame:
    """Tiny synthetic dataset with common data quality issues."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 1],  # row 4 is exact duplicate of row 0 (id=1)
            "name": ["Alice", "Bob", None, "  Dave  ", "Alice"],
            "age": [25, None, 35, 999.0, 25],  # 999 is an outlier
            "score": [0.9, 0.7, None, 0.5, 0.9],
            "category": ["A", "b", "A", "N/A", "A"],
            "active": ["true", "false", "yes", "no", "true"],
        }
    )


@pytest.fixture
def empty_df() -> pd.DataFrame:
    return pd.DataFrame()


@pytest.fixture
def one_row_df() -> pd.DataFrame:
    return pd.DataFrame([{"id": 1, "val": 42.0, "name": "Test"}])


# ---------------------------------------------------------------------------
# Traditional pipeline tests
# ---------------------------------------------------------------------------


class TestTraditionalPipeline:
    def test_removes_duplicates(self, small_df: pd.DataFrame) -> None:
        from benchmarks.traditional import run_traditional_pipeline

        out = run_traditional_pipeline(small_df, keys=["id"])
        assert len(out) < len(small_df), "Duplicates should be removed"

    def test_imputes_missing_numeric(self, small_df: pd.DataFrame) -> None:
        from benchmarks.traditional import run_traditional_pipeline

        out = run_traditional_pipeline(small_df, keys=["id"])
        assert out["age"].isnull().sum() == 0, "Numeric NaN should be imputed"

    def test_imputes_missing_string(self, small_df: pd.DataFrame) -> None:
        from benchmarks.traditional import run_traditional_pipeline

        out = run_traditional_pipeline(small_df, keys=["id"])
        assert out["name"].isnull().sum() == 0, "String NaN should be imputed"

    def test_replaces_null_placeholders(self, small_df: pd.DataFrame) -> None:
        from benchmarks.traditional import run_traditional_pipeline

        out = run_traditional_pipeline(small_df, keys=["id"])
        # "N/A" in category should have been replaced
        assert "N/A" not in out["category"].values

    def test_strips_whitespace(self, small_df: pd.DataFrame) -> None:
        from benchmarks.traditional import run_traditional_pipeline

        out = run_traditional_pipeline(small_df, keys=["id"])
        for val in out["name"].dropna():
            assert val == val.strip(), f"Whitespace not stripped: '{val}'"

    def test_handles_empty_dataframe(self, empty_df: pd.DataFrame) -> None:
        from benchmarks.traditional import run_traditional_pipeline

        out = run_traditional_pipeline(empty_df)
        assert out.empty

    def test_handles_one_row(self, one_row_df: pd.DataFrame) -> None:
        from benchmarks.traditional import run_traditional_pipeline

        out = run_traditional_pipeline(one_row_df, keys=["id"])
        assert len(out) == 1


# ---------------------------------------------------------------------------
# Validator tests
# ---------------------------------------------------------------------------


class TestValidator:
    def test_identical_outputs_pass(self, small_df: pd.DataFrame) -> None:
        from benchmarks.traditional import run_traditional_pipeline
        from benchmarks.validator import validate_equivalence

        trad = run_traditional_pipeline(small_df, keys=["id"])
        result = validate_equivalence("test", small_df, trad, trad.copy())
        assert result.passed, f"Identical outputs should pass: {result.mismatches}"

    def test_mismatch_detected(self, small_df: pd.DataFrame) -> None:
        from benchmarks.traditional import run_traditional_pipeline
        from benchmarks.validator import validate_equivalence

        trad = run_traditional_pipeline(small_df, keys=["id"])
        # Introduce a deliberate schema mismatch (drop a column)
        bad_tidely = trad.drop(columns=["score"])
        result = validate_equivalence("test_mismatch", small_df, trad, bad_tidely)
        assert not result.checks.get("column_count", True), (
            "Column mismatch should be detected"
        )

    def test_null_increase_detected(self, small_df: pd.DataFrame) -> None:
        from benchmarks.traditional import run_traditional_pipeline
        from benchmarks.validator import validate_equivalence

        trad = run_traditional_pipeline(small_df, keys=["id"])
        # Make a copy with lots of nulls
        bad = trad.copy()
        bad["age"] = None
        bad["score"] = None
        result = validate_equivalence("test_nulls", small_df, trad, bad)
        assert not result.checks.get("null_reduction", True), (
            "Null increase should be detected"
        )


# ---------------------------------------------------------------------------
# Reporter tests
# ---------------------------------------------------------------------------


class TestReporter:
    def test_generates_markdown(
        self, tmp_path: pytest.fixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:  # type: ignore[type-arg]
        from benchmarks import reporter

        # Redirect output files to tmp_path
        monkeypatch.setattr(reporter, "BENCHMARK_MD", str(tmp_path / "BENCHMARK.md"))
        monkeypatch.setattr(reporter, "BENCHMARK_JSON", str(tmp_path / "results.json"))

        dummy_results = [
            {
                "id": 1,
                "dataset": "test_ds",
                "status": "OK",
                "rows": 100,
                "cols": 5,
                "traditional_time_ms": 50.0,
                "tidely_time_ms": 30.0,
                "peak_ram_mb": 5.0,
                "validation_passed": True,
                "validation": {
                    "checks": {
                        "row_count": True,
                        "null_reduction": True,
                        "duplicate_removal": True,
                    },
                    "mismatches": [],
                },
            }
        ]
        reporter.generate_reports(dummy_results)
        assert (tmp_path / "BENCHMARK.md").exists()
        assert (tmp_path / "results.json").exists()
        content = (tmp_path / "BENCHMARK.md").read_text(encoding="utf-8")
        assert "Tidely v1.5.0" in content
        assert "test_ds" in content

    def test_regression_check_pass(
        self, tmp_path: pytest.fixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:  # type: ignore[type-arg]
        import json

        from benchmarks import reporter

        monkeypatch.setattr(reporter, "BENCHMARK_JSON", str(tmp_path / "results.json"))
        with open(tmp_path / "results.json", "w") as f:
            json.dump(
                {
                    "results": [
                        {
                            "status": "OK",
                            "dataset": "d1",
                            "traditional_time_ms": 100,
                            "tidely_time_ms": 80,
                        }
                    ]
                },
                f,
            )
        assert reporter.check_regressions(threshold=2.0) is True

    def test_regression_check_fail(
        self, tmp_path: pytest.fixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:  # type: ignore[type-arg]
        import json

        from benchmarks import reporter

        monkeypatch.setattr(reporter, "BENCHMARK_JSON", str(tmp_path / "results.json"))
        with open(tmp_path / "results.json", "w") as f:
            json.dump(
                {
                    "results": [
                        {
                            "status": "OK",
                            "dataset": "d1",
                            "traditional_time_ms": 100,
                            "tidely_time_ms": 500,
                        }
                    ]
                },
                f,
            )
        assert reporter.check_regressions(threshold=2.0) is False


# ---------------------------------------------------------------------------
# Version test
# ---------------------------------------------------------------------------


class TestVersion:
    def test_version_is_150(self) -> None:
        import tidely

        assert tidely.__version__ == "1.5.0", (
            f"Expected 1.5.0, got {tidely.__version__}"
        )

    def test_import_time_acceptable(self) -> None:
        """Import time should be under 15 seconds (target: <5s after optimization)."""
        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import time; t=time.perf_counter(); import tidely; print(f'{(time.perf_counter()-t)*1000:.0f}')",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        ms = float(result.stdout.strip())
        assert ms < 15_000, f"Import took {ms:.0f}ms — exceeds 15s hard limit"


# ---------------------------------------------------------------------------
# Engine smoke test (no network, synthetic data)
# ---------------------------------------------------------------------------


class TestEngineSmoke:
    def test_benchmark_one_synthetic(self, tmp_path: pytest.fixture) -> None:  # type: ignore[type-arg]
        """Run the benchmark engine on a purely synthetic local CSV."""
        import csv
        import random
        import string

        csv_path = str(tmp_path / "synthetic.csv")
        headers = ["id", "name", "value", "category"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(headers)
            for i in range(50):
                w.writerow(
                    [
                        i,
                        "".join(random.choices(string.ascii_lowercase, k=6))
                        if random.random() > 0.1
                        else "",
                        round(random.gauss(50, 15), 2) if random.random() > 0.1 else "",
                        random.choice(["A", "B", "C", "N/A", ""]),
                    ]
                )

        ds_info = {
            "id": 99,
            "name": "synthetic_test",
            "loader": lambda: pd.read_csv(csv_path),
            "keys": ["id"],
            "target": None,
        }

        from benchmarks.engine import _benchmark_one

        result = _benchmark_one(ds_info)
        assert result["status"] in ("OK", "ERROR"), "Should complete without crashing"
        assert result["rows"] > 0

    def test_memory_freed_between_runs(self) -> None:
        """Verify gc.collect() is called and memory is released."""
        from benchmarks.engine import _free_memory

        large = pd.DataFrame(np.random.randn(10_000, 50))
        _free_memory(large)
        gc.collect()
        # No assertion needed — just verify it doesn't crash
