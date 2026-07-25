"""The Orchestrator for the Tidely Data Cleaning Pipeline."""

import os
from typing import Any

import polars as pl

from tidely.core.adapter import estimate_dataset_size, normalize_to_polars
from tidely.result import CleanResult


def run_pipeline(data: Any) -> CleanResult:
    """Runs the full intelligence layer pipeline and returns the CleanResult.

    Pipeline Steps:
    1. Determine dataset size and route backend
    2. Normalize input (Pandas/Polars/Arrow/Filepath) to Polars representation
    3. Profile dataset domain DNA and detect structure (out-of-core uses sampling)
    4. Run Semantic Engine to classify columns
    5. Compute initial Lighthouse trust score
    6. Generate the RepairPlan
    7. Execute the plan (natively or using DuckDB / out-of-core streaming)
    8. Profile the cleaned dataset to compute the final Lighthouse trust score
    9. Generate the CleanResult with an explainable report

    Args:
        data: The raw input DataFrame or filepath.

    Returns:
        CleanResult: The outcome object containing the cleaned DataFrame.
    """
    import time

    start_time = time.time()

    # 1. Deduplicate column names if pandas to prevent downstream crashes
    try:
        import pandas as pd
    except ImportError:
        pd = None

    from tidely.core.decision_engine import DecisionEngine
    from tidely.core.detector import DetectionEngine
    from tidely.core.dna import infer_dataset_dna
    from tidely.core.plan import plan
    from tidely.core.scorer import compute_trust_scores
    from tidely.core.semantic import SemanticEngine
    from tidely.core.streaming import StreamingEngine
    from tidely.core.summary import CleanSummary
    from tidely.core.tracker import OutcomeTracker

    if (
        pd is not None
        and isinstance(data, pd.DataFrame)
        and data.columns.has_duplicates
    ):
        data = data.copy()
        new_cols = []
        seen = set()
        for c in data.columns:
            new_c = str(c)
            counter = 1
            while new_c in seen:
                new_c = f"{c}_{counter}"
                counter += 1
            seen.add(new_c)
            new_cols.append(new_c)
        data.columns = new_cols

    # 2. Store original for revertibility
    if hasattr(data, "copy"):
        original_data = data.copy()
    else:
        original_data = data

    # 3. Estimate size and select engine routing
    _, format_name = normalize_to_polars(data)
    size_bytes = estimate_dataset_size(data)
    decision_engine = DecisionEngine()
    engine_name = decision_engine.route_backend(size_bytes, format_name)

    # 4. Execute Routing
    if engine_name in ("streaming", "duckdb") and isinstance(data, str):
        ext = os.path.splitext(data)[1].lower()

        # Load sample for profiling & planning
        if ext == ".csv":
            try:
                sample_df = pl.read_csv(data, n_rows=10000)
            except Exception:
                sample_df = pl.read_csv(data)
        elif ext == ".parquet":
            import pyarrow.parquet as pq

            pq_module: Any = pq
            pf = pq_module.ParquetFile(data)
            from typing import cast

            sample_df = cast(
                pl.DataFrame,
                pl.from_arrow(pf.read_row_group(0).slice(0, 10000)),
            )
        else:
            pl_lazy, _ = normalize_to_polars(data)
            if isinstance(pl_lazy, pl.LazyFrame):
                sample_df = pl_lazy.limit(10000).collect()
            else:
                sample_df = pl_lazy

        detector = DetectionEngine()
        metadata_initial = detector.analyze(sample_df)

        semantic_engine = SemanticEngine()
        semantics_initial = semantic_engine.infer(sample_df, metadata_initial)

        dna = infer_dataset_dna(sample_df.columns)
        trust_initial = compute_trust_scores(sample_df, semantics_initial, dna.domain)

        # Generate plan on the sample
        p = plan(sample_df)

        out_filepath = data + ".cleaned" + ext

        # Execute plan out-of-core
        if engine_name == "duckdb":
            cleaned_df_raw = StreamingEngine.clean_with_duckdb(
                p, data, sample_df.columns, format_name, output_filepath=out_filepath
            )
        else:
            cleaned_df_raw = StreamingEngine.clean_chunked_streaming(
                p, data, sample_df.columns, format_name
            )
            temp_out = data + ".cleaned.tmp"
            if os.path.exists(temp_out):
                if os.path.exists(out_filepath):
                    os.remove(out_filepath)
                os.rename(temp_out, out_filepath)

            if ext == ".parquet":
                cleaned_df_raw = pl.scan_parquet(out_filepath)
            else:
                cleaned_df_raw = pl.scan_csv(out_filepath)

        # Profile the cleaned sample for final metrics
        if isinstance(cleaned_df_raw, pl.LazyFrame):
            cleaned_sample = cleaned_df_raw.limit(10000).collect()
        else:
            cleaned_sample = cleaned_df_raw

        metadata_final = detector.analyze(cleaned_sample)
        semantics_final = semantic_engine.infer(cleaned_sample, metadata_final)
        trust_final = compute_trust_scores(cleaned_sample, semantics_final, dna.domain)

        df_initial_for_tracker = sample_df
        df_cleaned_for_tracker = cleaned_sample

    else:
        # Standard in-memory/lazy execution path
        pl_data, format_name = normalize_to_polars(data)
        if isinstance(pl_data, pl.LazyFrame):
            df_initial = pl_data.collect()
        else:
            df_initial = pl_data

        detector = DetectionEngine()
        metadata_initial = detector.analyze(df_initial)

        semantic_engine = SemanticEngine()
        semantics_initial = semantic_engine.infer(df_initial, metadata_initial)

        dna = infer_dataset_dna(df_initial.columns)
        trust_initial = compute_trust_scores(df_initial, semantics_initial, dna.domain)

        # Generate plan
        p = plan(df_initial)

        # Execute plan
        if engine_name == "duckdb":
            cleaned_df_raw = StreamingEngine.clean_with_duckdb(
                p, df_initial, df_initial.columns, format_name
            )
        else:
            cleaned_df_raw = p.execute()

        pl_cleaned, _ = normalize_to_polars(cleaned_df_raw)
        if isinstance(pl_cleaned, pl.LazyFrame):
            df_cleaned = pl_cleaned.collect()
        else:
            df_cleaned = pl_cleaned

        metadata_final = detector.analyze(df_cleaned)
        semantics_final = semantic_engine.infer(df_cleaned, metadata_final)
        trust_final = compute_trust_scores(df_cleaned, semantics_final, dna.domain)

        df_initial_for_tracker = df_initial
        df_cleaned_for_tracker = df_cleaned

    # 5. Build explainable report using the tracker
    tracker = OutcomeTracker(df_initial_for_tracker)
    tracker.initial_health = trust_initial.overall

    # Map RepairActions to the dict format expected by the tracker
    autofixes = []
    warnings = []

    # We retrieve the actions from the plan
    for action in p.actions:
        autofixes.append({
            "category": action.category,
            "column": getattr(action, "column", ""),
            "why": action.why_it_changed,
            "impact": action.what_changed,
            "confidence": int(action.confidence * 100),
        })

    # Generate warnings for anything left uncleaned
    for col, info in semantics_final.items():
        if (
            info["type"] == "Unknown"
            and metadata_final["columns"][col]["null_count"] > 0
        ):
            warnings.append({
                "category": "Missing Values",
                "column": col,
                "confidence": int(info["confidence"] * 100),
                "why": f"Column '{col}' contains un-imputed missing values with unknown semantic type.",
            })

    orig_h = (
        df_initial_for_tracker.height
        if hasattr(df_initial_for_tracker, "height")
        else 0
    )
    clean_h = (
        df_cleaned_for_tracker.height
        if hasattr(df_cleaned_for_tracker, "height")
        else 0
    )
    rows_removed = max(0, orig_h - clean_h)
    columns_modified = len({
        action.column for action in p.actions if getattr(action, "column", "")
    })

    missing_values_fixed = sum(
        action.rows_affected
        for action in p.actions
        if action.category == "Missing Values"
    )
    duplicates_removed_count = sum(
        action.rows_affected
        for action in p.actions
        if action.category in ("Duplicate IDs", "Duplicate Rows")
    )
    outliers_fixed = sum(
        action.rows_affected for action in p.actions if action.category == "Outliers"
    )
    datatypes_optimized = sum(
        1
        for action in p.actions
        if action.category
        in ("Datatype Optimization", "Categorical", "Smart Categorical")
    )
    semantic_corrections = sum(
        1
        for action in p.actions
        if action.category
        in (
            "Email",
            "Phone",
            "ZIP Code",
            "Coordinate Clip",
            "Smart String",
            "Smart Numeric",
            "Smart Date",
        )
    )

    outcome = tracker.track(df_cleaned_for_tracker, autofixes, warnings)
    outcome["final_health"] = trust_final.overall
    outcome["column_diagnostics"] = getattr(p, "column_diagnostics", {})
    outcome["engine_name"] = engine_name
    outcome["engine_reason"] = decision_engine.selected_reason
    outcome["execution_time"] = time.time() - start_time
    outcome["rows_removed"] = rows_removed
    outcome["columns_modified"] = columns_modified
    outcome["missing_values_fixed"] = missing_values_fixed
    outcome["duplicates_removed_count"] = duplicates_removed_count
    outcome["outliers_fixed"] = outliers_fixed
    outcome["datatypes_optimized"] = datatypes_optimized
    outcome["semantic_corrections"] = semantic_corrections
    outcome["actions"] = autofixes

    summary = CleanSummary(
        initial_health=outcome["initial_health"],
        final_health=outcome["final_health"],
        fixes=outcome["fixes"],
        warnings=outcome["warnings"],
        memory_before_mb=outcome["memory_before_mb"],
        memory_after_mb=outcome["memory_after_mb"],
        execution_time=outcome["execution_time"],
        backend=outcome["engine_name"],
        rows_removed=outcome["rows_removed"],
        missing_values_fixed=outcome["missing_values_fixed"],
        duplicates_removed=outcome["duplicates_removed_count"],
        outliers_fixed=outcome["outliers_fixed"],
        datatypes_optimized=outcome["datatypes_optimized"],
        semantic_corrections=outcome["semantic_corrections"],
    )

    # Convert raw cleaned dataframe to original incoming format
    if (
        pd is not None
        and format_name == "pandas"
        and not isinstance(cleaned_df_raw, pd.DataFrame)
    ):
        if isinstance(cleaned_df_raw, pl.LazyFrame):
            cleaned_df_out = cleaned_df_raw.collect().to_pandas()
        else:
            cleaned_df_out = cleaned_df_raw.to_pandas()
    elif format_name == "arrow" and hasattr(cleaned_df_raw, "to_arrow"):
        cleaned_df_out = cleaned_df_raw.to_arrow()
    else:
        cleaned_df_out = cleaned_df_raw

    return CleanResult(
        cleaned_df=cleaned_df_out,
        original_df=original_data,
        summary_text=str(summary),
        report_data=outcome,
    )
