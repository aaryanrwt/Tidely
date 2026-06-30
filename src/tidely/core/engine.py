"""The Orchestrator for the Tidely Data Cleaning Pipeline."""

from typing import Any
import polars as pl

from tidely.result import CleanResult
from tidely.core.adapter import normalize_to_polars
from tidely.core.errors import TidelyDataError


def run_pipeline(data: Any) -> CleanResult:
    """Runs the full intelligence layer pipeline and returns the CleanResult.
    
    Pipeline Steps:
    1. Normalize input (Pandas/Polars/Arrow/Filepath) to Polars representation
    2. Profile dataset domain DNA and detect structure
    3. Run Semantic Engine to classify columns
    4. Compute initial Lighthouse trust score
    5. Generate the RepairPlan
    6. Execute the plan natively using compiled Polars expressions
    7. Profile the cleaned dataset to compute the final Lighthouse trust score
    8. Generate the CleanResult with an explainable report
    
    Args:
        data: The raw input DataFrame or filepath.
        
    Returns:
        CleanResult: The outcome object containing the cleaned DataFrame.
    """
    from tidely.core.detector import DetectionEngine
    from tidely.core.semantic import SemanticEngine
    from tidely.core.scorer import compute_trust_scores
    from tidely.core.dna import infer_dataset_dna
    from tidely.core.plan import plan
    from tidely.core.tracker import OutcomeTracker
    from tidely.core.summary import CleanSummary

    # 1. Deduplicate column names if pandas to prevent downstream crashes
    import pandas as pd
    if isinstance(data, pd.DataFrame) and data.columns.has_duplicates:
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

    # 3. Normalize & Profile Initial
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

    # 4. Generate & Execute Plan
    p = plan(df_initial)
    cleaned_df_raw = p.execute()

    # Normalize cleaned to compute final metrics
    pl_cleaned, _ = normalize_to_polars(cleaned_df_raw)
    if isinstance(pl_cleaned, pl.LazyFrame):
        df_cleaned = pl_cleaned.collect()
    else:
        df_cleaned = pl_cleaned

    metadata_final = detector.analyze(df_cleaned)
    semantics_final = semantic_engine.infer(df_cleaned, metadata_final)
    trust_final = compute_trust_scores(df_cleaned, semantics_final, dna.domain)

    # 5. Build explainable report using the tracker
    tracker = OutcomeTracker(df_initial)
    # Set the real initial health score
    tracker.initial_health = trust_initial.overall

    # Map RepairActions to the dict format expected by the tracker
    autofixes = []
    warnings = []
    
    # We retrieve the actions from the plan
    for action in p.actions:
        # Since all actions in RepairPlan are applied in p.execute(), they are fixes
        autofixes.append({
            "category": action.category,
            "column": getattr(action, "column", ""),
            "why": action.why_it_changed,
            "impact": action.what_changed,
            "confidence": int(action.confidence * 100)
        })

    # Generate warnings for anything left uncleaned (e.g. Unknown/low confidence semantic columns)
    for col, info in semantics_final.items():
        if info["type"] == "Unknown" and metadata_final["columns"][col]["null_count"] > 0:
            warnings.append({
                "category": "Missing Values",
                "column": col,
                "confidence": int(info["confidence"] * 100),
                "why": f"Column '{col}' contains un-imputed missing values with unknown semantic type."
            })

    outcome = tracker.track(df_cleaned, autofixes, warnings)
    outcome["final_health"] = trust_final.overall

    summary = CleanSummary(
        initial_health=outcome["initial_health"],
        final_health=outcome["final_health"],
        fixes=outcome["fixes"],
        warnings=outcome["warnings"],
        memory_before_mb=outcome["memory_before_mb"],
        memory_after_mb=outcome["memory_after_mb"],
    )

    # Convert raw cleaned dataframe to original incoming format
    if format_name == "pandas" and not isinstance(cleaned_df_raw, pd.DataFrame):
        cleaned_df_out = df_cleaned.to_pandas()
    elif format_name == "arrow" and hasattr(cleaned_df_raw, "to_arrow"):
        cleaned_df_out = cleaned_df_raw.to_arrow()
    else:
        cleaned_df_out = cleaned_df_raw

    return CleanResult(
        cleaned_df=cleaned_df_out,
        original_df=original_data,
        summary_text=str(summary),
        report_data=outcome
    )
