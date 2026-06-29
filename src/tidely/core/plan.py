"""Planning engine that inspects datasets and generates explainable cleaning plans."""

from typing import Any

import polars as pl

from tidely.core.clean_engine import RepairAction, RepairPlan

# Import inspect inside plan to avoid circular imports if needed, or import from tidely
from tidely.core.profile import DatasetProfile
from tidely.core.rules import (
    make_categorical_rule,
    make_date_rule,
    make_dedup_id_rule,
    make_dedup_rows_rule,
    make_downcast_rule,
    make_email_rule,
    make_impute_constant_rule,
    make_impute_median_rule,
    make_phone_rule,
)


def plan(data: Any) -> RepairPlan:
    """Generates an explainable cleaning plan for the given dataset.

    Args:
        data: Polars DataFrame, Pandas DataFrame, or PyArrow Table.

    Returns:
        RepairPlan: The execution plan outlining what will change and why.
    """
    # Import locally to avoid circular dependencies
    from tidely import inspect

    profile: DatasetProfile = inspect(data)
    df = profile._df_ref

    actions: list[RepairAction] = []
    points_recovered = 0.0

    # 1. Row-level Deduplication
    try:
        dup_rows = profile.row_count - df.n_unique()
        if dup_rows > 0:
            actions.append(
                RepairAction(
                    category="Duplicate Rows",
                    what_changed="Dropped exact duplicate rows.",
                    why_it_changed=f"Found {dup_rows} exact duplicates, which can bias downstream analysis.",
                    confidence=1.0,
                    expected_score_bump=10,
                    rule_fn=make_dedup_rows_rule(),
                )
            )
            points_recovered += 10.0
    except Exception:
        pass  # Ignore nested types failing uniqueness checks

    # 2. Schema / Semantic Deduplication & Type Enforcement
    for col, info in profile.semantic_types.items():
        stype = info["type"]
        conf = info["confidence"]

        dtype = df[col].dtype

        # ID Deduplication
        if stype == "ID/Key" and conf >= 0.9:
            id_dups = df.height - df.n_unique(subset=[col])
            if id_dups > 0:
                actions.append(
                    RepairAction(
                        category="Duplicate IDs",
                        what_changed=f"Deduplicated primary key '{col}'.",
                        why_it_changed=f"Found {id_dups} duplicate IDs. Dropped to enforce entity uniqueness.",
                        confidence=conf,
                        expected_score_bump=15,
                        rule_fn=make_dedup_id_rule(col),
                    )
                )
                points_recovered += 15.0

        # Semantic Normalizations
        if stype == "Email" and conf >= 0.7:
            actions.append(
                RepairAction(
                    category="Semantic Normalization",
                    what_changed=f"Normalized '{col}' as Email.",
                    why_it_changed="Lowercased and stripped whitespace to ensure reliable string matching.",
                    confidence=conf,
                    expected_score_bump=5,
                    rule_fn=make_email_rule(col),
                )
            )
            points_recovered += 5.0
        elif stype == "Phone" and conf >= 0.7:
            actions.append(
                RepairAction(
                    category="Semantic Normalization",
                    what_changed=f"Normalized '{col}' as Phone Number.",
                    why_it_changed="Stripped non-numeric formatting to standardize international/local formats.",
                    confidence=conf,
                    expected_score_bump=5,
                    rule_fn=make_phone_rule(col),
                )
            )
            points_recovered += 5.0
        elif stype in ("Date", "Datetime") and dtype == pl.String:
            actions.append(
                RepairAction(
                    category="Type Normalization",
                    what_changed=f"Converted '{col}' to Native Datetime.",
                    why_it_changed="Casting string dates enables temporal aggregations and time-series ML.",
                    confidence=conf,
                    expected_score_bump=10,
                    rule_fn=make_date_rule(col),
                )
            )
            points_recovered += 10.0

    # 3. Missing Value Imputation & Memory Optimization
    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].null_count()

        # Missing values
        if null_count > 0:
            if dtype.is_numeric():
                actions.append(
                    RepairAction(
                        category="Missing Values",
                        what_changed=f"Imputed missing '{col}' values with Median.",
                        why_it_changed=f"Column is missing {null_count} values. Median is robust to numeric outliers.",
                        confidence=0.8,
                        expected_score_bump=10,
                        rule_fn=make_impute_median_rule(col),
                    )
                )
                points_recovered += 10.0
            elif dtype == pl.String or dtype == pl.Categorical:
                actions.append(
                    RepairAction(
                        category="Missing Values",
                        what_changed=f"Imputed missing '{col}' values with 'Unknown'.",
                        why_it_changed=f"Column is missing {null_count} values. Constant fill prevents hallucinating cross-sectional data.",
                        confidence=0.7,
                        expected_score_bump=5,
                        rule_fn=make_impute_constant_rule(col, "Unknown"),
                    )
                )
                points_recovered += 5.0

        # Memory optimizations
        if dtype == pl.String:
            n_unique = df[col].n_unique()
            if n_unique < (df.height * 0.05) and df.height > 1000:
                actions.append(
                    RepairAction(
                        category="Memory Optimization",
                        what_changed=f"Converted '{col}' to Categorical.",
                        why_it_changed=f"Only {n_unique} unique values detected. Reduces memory footprint by up to 80%.",
                        confidence=0.95,
                        expected_score_bump=5,
                        rule_fn=make_categorical_rule(col),
                    )
                )
                points_recovered += 5.0
        elif dtype.is_integer():
            c_min, c_max = df[col].min(), df[col].max()
            if isinstance(c_min, (int, float)) and isinstance(c_max, (int, float)):
                if c_min >= -128 and c_max <= 127 and dtype != pl.Int8:
                    actions.append(
                        RepairAction(
                            category="Memory Optimization",
                            what_changed=f"Downcasted '{col}' to Int8.",
                            why_it_changed="Values strictly fall between -128 and 127. Saves massive memory overhead.",
                            confidence=1.0,
                            expected_score_bump=2,
                            rule_fn=make_downcast_rule(col, pl.Int8),
                        )
                    )
                    points_recovered += 2.0
                elif (
                    c_min >= -32768
                    and c_max <= 32767
                    and dtype not in (pl.Int8, pl.Int16)
                ):
                    actions.append(
                        RepairAction(
                            category="Memory Optimization",
                            what_changed=f"Downcasted '{col}' to Int16.",
                            why_it_changed="Values strictly fit in Int16 bounds. Saves massive memory overhead.",
                            confidence=1.0,
                            expected_score_bump=2,
                            rule_fn=make_downcast_rule(col, pl.Int16),
                        )
                    )
                    points_recovered += 2.0

    initial_score = profile.trust_score.overall
    target_score = min(int(initial_score + points_recovered), 100)

    return RepairPlan(
        original_data=data,
        actions=actions,
        initial_score=initial_score,
        target_score=target_score,
    )


def clean(data: Any, dry_run: bool = False) -> Any:
    """Convenience wrapper to automatically generate and execute a cleaning plan.

    Args:
        data: The uncleaned DataFrame.
        dry_run: If True, prints plan and audits without mutating return value.

    Returns:
        The cleaned DataFrame (in the original format).
    """
    p = plan(data)
    # Automatically show the plan if running in terminal (or always)
    p.show()
    return p.execute(dry_run=dry_run)
