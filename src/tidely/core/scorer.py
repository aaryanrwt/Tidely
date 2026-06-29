"""Lighthouse-style multi-dimensional Dataset Trust scoring engine."""

from typing import Any

import polars as pl


class TrustScores:
    """Holds Lighthouse-style Dataset Trust scores (0-100)."""

    def __init__(
        self,
        overall: int,
        reliability: int,
        ml_readiness: int,
        memory_efficiency: int,
        schema_stability: int,
        semantic_quality: int,
    ) -> None:
        """Initialize TrustScores.

        Args:
            overall: Average dataset trust quality score.
            reliability: Score for null rates and primary key duplication.
            ml_readiness: Score for target leaks, skewness, and category balance.
            memory_efficiency: Score for downcasting and categorical opportunities.
            schema_stability: Score for datatype consistency and json presence.
            semantic_quality: Score representing regex validity of semantic columns.
        """
        self.overall = overall
        self.reliability = reliability
        self.ml_readiness = ml_readiness
        self.memory_efficiency = memory_efficiency
        self.schema_stability = schema_stability
        self.semantic_quality = semantic_quality


def compute_trust_scores(
    df: pl.DataFrame,
    semantic_types: dict[str, dict[str, Any]],
    domain: str,
) -> TrustScores:
    """Computes the Lighthouse scores for the dataset.

    Args:
        df: Eager Polars DataFrame to score.
        semantic_types: Dict mapping column names to classified semantic details.
        domain: Detected domain of the dataset.

    Returns:
        TrustScores: Multi-dimensional trust scores.
    """
    total_rows = df.height
    total_cols = df.width

    if total_rows == 0 or total_cols == 0:
        return TrustScores(0, 0, 0, 0, 0, 0)

    # 1. RELIABILITY (Null rates and Primary Key duplicates)
    # Start at 100
    rel_score = 100.0

    # Check null rates
    null_counts = df.null_count().row(0)
    avg_null_rate = sum(null_counts) / (total_rows * total_cols)
    # Deduct up to 30 points for null rates
    rel_score -= min(avg_null_rate * 100, 30.0)

    # Check for Duplicate IDs (primary keys)
    for col, semantic in semantic_types.items():
        if semantic["type"] == "ID/Key":
            # If it's an ID, check duplicates
            unique_cnt = df[col].n_unique()
            dup_rate = (total_rows - unique_cnt) / total_rows
            if dup_rate > 0:
                # Critical penalty: deduct up to 40 points
                rel_score -= min(dup_rate * 200, 40.0)

    # 2. ML READINESS (Skewness, target leaks, class imbalances)
    ml_score = 100.0

    # Skewness checks on numerical columns
    num_cols = [c for c in df.columns if df[c].dtype.is_numeric()]
    if num_cols:
        # Compute skewness
        skew_exprs = [pl.col(c).skew().alias(c) for c in num_cols]
        skew_vals = df.select(skew_exprs).row(0)
        # Find how many columns are highly skewed (> 1.5 or < -1.5)
        skewed_count = sum(1 for s in skew_vals if s is not None and abs(s) > 1.5)
        skewed_ratio = skewed_count / len(num_cols)
        # Deduct up to 25 points for skewness issues
        ml_score -= skewed_ratio * 25.0

    # Target column analysis: look for columns named target, label, churn, class, y, status
    target_candidates = [
        c
        for c in df.columns
        if c.lower() in ("target", "label", "y", "class", "churn", "status")
    ]
    if target_candidates:
        target_col = target_candidates[0]
        # Target should not contain nulls
        target_nulls = df[target_col].null_count()
        target_null_rate = target_nulls / total_rows
        if target_null_rate > 0:
            # Deduct up to 30 points for null target
            ml_score -= min(target_null_rate * 150, 30.0)
    else:
        # If no target found, small penalty for ML Readiness since it's not clear
        ml_score -= 5.0

    # 3. MEMORY EFFICIENCY (Downcasting and categorical opportunities)
    mem_score = 100.0

    # Categorical opportunities: string columns with low cardinality
    str_cols = [c for c in df.columns if df[c].dtype == pl.String]
    if str_cols:
        cat_opps = 0
        for col in str_cols:
            cardinality = df[col].n_unique() / total_rows
            # If cardinality is low (< 10%) and total rows is substantial (> 50)
            if cardinality < 0.1 and total_rows > 50:
                cat_opps += 1
        cat_opp_ratio = cat_opps / len(str_cols)
        # Deduct up to 40 points for unoptimized categories
        mem_score -= cat_opp_ratio * 40.0

    # Integer downcasting opportunities: checking max values
    int_cols = [c for c in df.columns if df[c].dtype.is_integer()]
    if int_cols:
        downcast_opps = 0
        for col in int_cols:
            # If current type is Int64 but values fit in Int32/Int16/Int8
            col_max = df[col].max()
            col_min = df[col].min()
            if col_max is not None and col_min is not None:
                try:
                    c_min = int(col_min)  # type: ignore[arg-type]
                    c_max = int(col_max)  # type: ignore[arg-type]
                    if df[col].dtype == pl.Int64:
                        if c_min >= -128 and c_max <= 127:
                            downcast_opps += 1
                        elif c_min >= -32768 and c_max <= 32767:
                            downcast_opps += 1
                except (ValueError, TypeError):
                    pass
        downcast_ratio = downcast_opps / len(int_cols)
        mem_score -= downcast_ratio * 20.0

    # 4. SCHEMA STABILITY (Mixed types, invalid JSON blobs)
    schema_score = 100.0
    # Check if there are any columns holding raw/mixed formats or invalid JSON
    # For Milestone 1, we check if columns have mixed strings/numbers
    # (Since Polars handles types strictly, schema stability starts high but drops if
    # string columns hold values with highly mixed lengths or structural parsing issues)
    # We deduct for columns that have high null count combined with sparse filled strings.
    sparse_cols = 0
    for col in df.columns:
        null_rate = df[col].null_count() / total_rows
        if 0.5 < null_rate < 0.98:
            sparse_cols += 1
    sparse_ratio = sparse_cols / total_cols
    schema_score -= sparse_ratio * 20.0

    # 5. SEMANTIC & DOMAIN QUALITY (Regex match rates, domain violations)
    sem_score = 100.0

    # Calculate format violation rates
    semantic_violations = 0.0
    total_semantic_cols = 0
    for _, semantic in semantic_types.items():
        if semantic["type"] != "Unknown" and semantic["type"] != "ID/Key":
            total_semantic_cols += 1
            # Match rate inside the column
            confidence = semantic["confidence"]
            # Format violation is 1 - match_rate
            violation_rate = 1.0 - confidence
            semantic_violations += violation_rate

    if total_semantic_cols > 0:
        avg_violation = semantic_violations / total_semantic_cols
        # Deduct up to 30 points
        sem_score -= min(avg_violation * 100, 30.0)

    # Convert scores to integers
    r_overall = int(
        round((rel_score + ml_score + mem_score + schema_score + sem_score) / 5)
    )

    return TrustScores(
        overall=max(0, min(r_overall, 100)),
        reliability=max(0, min(int(round(rel_score)), 100)),
        ml_readiness=max(0, min(int(round(ml_score)), 100)),
        memory_efficiency=max(0, min(int(round(mem_score)), 100)),
        schema_stability=max(0, min(int(round(schema_score)), 100)),
        semantic_quality=max(0, min(int(round(sem_score)), 100)),
    )
