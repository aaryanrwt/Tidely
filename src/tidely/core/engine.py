"""The Orchestrator for the Tidely Data Cleaning Pipeline."""

from typing import Any

from tidely.result import CleanResult
from tidely.core.adapter import normalize_to_polars


def run_pipeline(data: Any) -> CleanResult:
    """Runs the full intelligence layer pipeline and returns the CleanResult.
    
    Pipeline Steps:
    1. Normalize input (Pandas/Polars/Arrow)
    2. Detection Engine (Base types, missing values, cardinality)
    3. Semantic Engine (Infer business meaning)
    4. Trust Engine (Calculate confidence scores)
    5. Planner (Assign AutoFix vs Warnings)
    6. Executor (Apply vectorized Pandas/Polars operations)
    7. Tracker (Calculate memory savings and outcome deltas)
    
    Args:
        data: The raw input DataFrame.
        
    Returns:
        CleanResult: The outcome object.
    """
    from tidely.core.detector import DetectionEngine
    from tidely.core.semantic import SemanticEngine
    from tidely.core.trust import TrustEngine
    from tidely.core.planner import CleaningPlanner
    from tidely.core.executor import CleaningExecutor
    from tidely.core.tracker import OutcomeTracker
    from tidely.core.summary import CleanSummary
    
    # 1. Deduplicate column names to prevent ambiguity crashes
    if hasattr(data, "columns"):
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

    # 1.5 Store original
    if hasattr(data, "copy"):
        original_data = data.copy()
    else:
        original_data = data

    tracker = OutcomeTracker(original_data)
    
    # 2. Detection
    detector = DetectionEngine()
    metadata = detector.analyze(data)
    
    # 3. Semantic
    semantic_engine = SemanticEngine()
    semantics = semantic_engine.infer(data, metadata)
    
    # 4. Trust Engine
    trust_engine = TrustEngine(strictness="high")
    actions = trust_engine.evaluate(data, metadata, semantics)
    
    # 5. Planner
    planner = CleaningPlanner(threshold=95.0)
    autofixes, warnings = planner.plan(actions)
    
    # 6. Executor
    executor = CleaningExecutor()
    cleaned_df = executor.execute(data, autofixes)
    
    # 7. Tracker & Summary
    outcome = tracker.track(cleaned_df, autofixes, warnings)
    
    summary = CleanSummary(
        initial_health=outcome["initial_health"],
        final_health=outcome["final_health"],
        fixes=outcome["fixes"],
        warnings=outcome["warnings"],
        memory_before_mb=outcome["memory_before_mb"],
        memory_after_mb=outcome["memory_after_mb"],
    )
    
    return CleanResult(
        cleaned_df=cleaned_df,
        original_df=original_data,
        summary_text=str(summary),
        report_data=outcome
    )
