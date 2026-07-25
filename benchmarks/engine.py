"""Tidely v1.5.0 — Sequential Benchmark Engine.

Processes one dataset at a time:
  Load -> Traditional clean -> Tidely clean -> Validate -> Record -> Free memory

Never loads multiple large datasets simultaneously.
"""

from __future__ import annotations

import gc
import logging
import os
import tempfile
import time
import traceback
from typing import Any

import psutil

from benchmarks.registry import DATASETS, load_dataset_safe
from benchmarks.traditional import run_traditional_pipeline
from benchmarks.validator import validate_equivalence

logger = logging.getLogger("tidely.benchmark.engine")

# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------


def _peak_ram_mb() -> float:
    """Return current process RSS in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def _free_memory(*dfs: Any) -> None:
    """Explicitly delete dataframes and collect garbage."""
    for obj in dfs:
        try:
            del obj
        except Exception:
            pass
    gc.collect()


# ---------------------------------------------------------------------------
# Single-dataset benchmark
# ---------------------------------------------------------------------------


def _benchmark_one(ds_info: dict[str, Any], smoke_test: bool = False) -> dict[str, Any]:
    """Run the full benchmark pipeline for a single dataset.

    Args:
        ds_info: Dataset registry entry.
        smoke_test: If True, only runs first dataset and skips ML.

    Returns:
        Result dict with timing, memory, validation, and metadata.
    """
    import polars as pl

    import tidely as td

    ds_id = ds_info["id"]
    ds_name = ds_info["name"]
    keys = ds_info.get("keys", [])
    target = ds_info.get("target")

    result: dict[str, Any] = {
        "id": ds_id,
        "dataset": ds_name,
        "status": "ERROR",
        "rows": 0,
        "cols": 0,
        "traditional_time_ms": 0.0,
        "tidely_time_ms": 0.0,
        "peak_ram_mb": 0.0,
        "validation_passed": False,
        "validation": {},
        "error": "",
    }

    print(f"\n  [{ds_id:02d}] {ds_name}")

    # ── 1. Load ───────────────────────────────────────────────────────────
    raw_df, load_status = load_dataset_safe(ds_info)
    print(f"       Load: {load_status}")

    if raw_df is None:
        result["status"] = "SKIPPED"
        result["error"] = load_status
        return result

    result["rows"] = len(raw_df)
    result["cols"] = len(raw_df.columns)

    # Save to a temp CSV so Tidely can load it via file path
    tmp = tempfile.NamedTemporaryFile(
        suffix=".csv", delete=False, mode="w", encoding="utf-8"
    )
    try:
        raw_df.to_csv(tmp.name, index=False)
        tmp.close()
        tmp_path = tmp.name

        # ── 2. Traditional pipeline ───────────────────────────────────────
        ram_before = _peak_ram_mb()
        t0 = time.perf_counter()
        try:
            trad_df = run_traditional_pipeline(raw_df.copy(), keys=keys, target=target)
            trad_ms = (time.perf_counter() - t0) * 1000
        except Exception as exc:
            trad_ms = (time.perf_counter() - t0) * 1000
            logger.warning("Traditional pipeline failed for %s: %s", ds_name, exc)
            trad_df = raw_df.copy()

        ram_after_trad = _peak_ram_mb()
        print(f"       Traditional: {trad_ms:.1f}ms  ({len(trad_df)} rows)")

        # ── 3. Tidely pipeline ────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            tidely_result = td.clean(tmp_path)
            tide_ms = (time.perf_counter() - t0) * 1000

            tidely_df_raw = tidely_result.df
            if isinstance(tidely_df_raw, pl.DataFrame):
                tidely_df = tidely_df_raw.to_pandas()
            elif isinstance(tidely_df_raw, pl.LazyFrame):
                tidely_df = tidely_df_raw.collect().to_pandas()
            else:
                tidely_df = tidely_df_raw

        except Exception as exc:
            tide_ms = (time.perf_counter() - t0) * 1000
            logger.error("Tidely failed for %s: %s", ds_name, exc)
            result["status"] = "ERROR"
            result["error"] = str(exc)
            result["traditional_time_ms"] = trad_ms
            result["tidely_time_ms"] = tide_ms
            _free_memory(raw_df, trad_df)
            return result

        ram_peak = max(_peak_ram_mb(), ram_after_trad) - ram_before
        print(f"       Tidely:       {tide_ms:.1f}ms  ({len(tidely_df)} rows)")

        result["traditional_time_ms"] = round(trad_ms, 2)
        result["tidely_time_ms"] = round(tide_ms, 2)
        result["peak_ram_mb"] = round(max(0.0, ram_peak), 2)
        speedup = trad_ms / tide_ms if tide_ms > 0 else 0
        print(
            f"       Speedup: {speedup:.2f}x | RAM delta: {result['peak_ram_mb']:.1f} MB"
        )

        # ── 4. Validate equivalence ───────────────────────────────────────
        val = validate_equivalence(ds_name, raw_df, trad_df, tidely_df)
        result["validation_passed"] = val.passed
        result["validation"] = {
            "checks": val.checks,
            "mismatches": val.mismatches,
        }
        print(f"       Validation: {val.summary()}")

        result["status"] = "OK"

    except Exception as exc:
        result["status"] = "ERROR"
        result["error"] = str(exc)
        logger.error("Benchmark failed for %s:\n%s", ds_name, traceback.format_exc())

    finally:
        # Always free memory, always delete temp file
        _free_memory(
            raw_df,
            trad_df if "trad_df" in dir() else None,
            tidely_df if "tidely_df" in dir() else None,
        )
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        # Remove Tidely output file if created
        try:
            cleaned_path = tmp_path + ".cleaned.csv"
            if os.path.exists(cleaned_path):
                os.unlink(cleaned_path)
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------


def run_all(smoke_test: bool = False) -> list[dict[str, Any]]:
    """Run benchmarks for all registered datasets sequentially.

    Args:
        smoke_test: If True, only run the first 2 datasets (for CI smoke test).

    Returns:
        List of per-dataset result dicts.
    """
    print("\n" + "=" * 68)
    print("  Tidely v1.5.0 — Sequential Benchmark Engine")
    print("=" * 68)

    dataset_list = DATASETS[:2] if smoke_test else DATASETS
    all_results: list[dict[str, Any]] = []

    for ds_info in dataset_list:
        result = _benchmark_one(ds_info, smoke_test=smoke_test)
        all_results.append(result)

        # Aggressive memory cleanup between datasets
        gc.collect()

    print("\n" + "=" * 68)
    ok = sum(1 for r in all_results if r["status"] == "OK")
    skipped = sum(1 for r in all_results if r["status"] == "SKIPPED")
    errors = sum(1 for r in all_results if r["status"] == "ERROR")
    passed = sum(1 for r in all_results if r.get("validation_passed"))
    print(
        f"  Done: {ok} OK | {skipped} SKIPPED | {errors} ERRORS | {passed} VALIDATION PASSED"
    )
    print("=" * 68 + "\n")

    return all_results
