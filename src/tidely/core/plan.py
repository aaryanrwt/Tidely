"""Planning engine that inspects datasets and generates explainable cleaning plans."""

from typing import Any
import polars as pl

from tidely.core.clean_engine import RepairAction, RepairPlan
from tidely.core.profile import DatasetProfile
from tidely.core.decision_engine import DecisionEngine
from tidely.core.rules import (
    make_categorical_rule,
    make_date_rule,
    make_dedup_id_rule,
    make_dedup_rows_rule,
    make_downcast_rule,
    make_email_rule,
    make_impute_constant_rule,
    make_impute_median_rule,
    make_impute_mean_rule,
    make_impute_mode_rule,
    make_impute_ffill_rule,
    make_phone_rule,
    make_outlier_iqr_rule,
    make_outlier_zscore_rule,
    make_unicode_clean_rule,
    make_zip_code_rule,
    make_coordinate_clip_rule,
    make_replace_null_placeholders_rule,
)


def plan(data: Any) -> RepairPlan:
    """Generates an explainable cleaning plan for the given dataset.

    Args:
        data: Polars DataFrame, Pandas DataFrame, or PyArrow Table.

    Returns:
        RepairPlan: The execution plan outlining what will change and why.
    """
    from tidely import inspect

    profile: DatasetProfile = inspect(data)
    df = profile._df_ref

    actions: list[RepairAction] = []
    points_recovered = 0.0
    
    decision_engine = DecisionEngine()

    # Deduplication will run at the end of the pipeline to catch post-normalization/imputation duplicates

    # 2. Schema / Semantic Deduplication & Type Enforcement
    for col, info in profile.semantic_types.items():
        stype = info["type"]
        conf = info["confidence"]

        dtype = df[col].dtype

        # ID Deduplication
        if stype == "ID/Key" and conf >= 0.9:
            if any(kw in col.lower() for kw in ("name", "desc", "title", "text", "seq", "val")):
                continue
            try:
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
            except Exception:
                pass

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
        elif stype == "ZIP Code" and conf >= 0.7:
            actions.append(
                RepairAction(
                    category="Semantic Normalization",
                    what_changed=f"Normalized '{col}' as ZIP Code.",
                    why_it_changed="Padded ZIP code strings to exactly 5 digits for geographical consistency.",
                    confidence=conf,
                    expected_score_bump=5,
                    rule_fn=make_zip_code_rule(col),
                )
            )
            points_recovered += 5.0
        elif stype == "Latitude" and conf >= 0.8:
            actions.append(
                RepairAction(
                    category="Semantic Normalization",
                    what_changed=f"Clipped '{col}' Latitude coordinate.",
                    why_it_changed="Enforced latitude boundaries strictly inside [-90, 90].",
                    confidence=conf,
                    expected_score_bump=5,
                    rule_fn=make_coordinate_clip_rule(col, is_lat=True),
                )
            )
            points_recovered += 5.0
        elif stype == "Longitude" and conf >= 0.8:
            actions.append(
                RepairAction(
                    category="Semantic Normalization",
                    what_changed=f"Clipped '{col}' Longitude coordinate.",
                    why_it_changed="Enforced longitude boundaries strictly inside [-180, 180].",
                    confidence=conf,
                    expected_score_bump=5,
                    rule_fn=make_coordinate_clip_rule(col, is_lat=False),
                )
            )
            points_recovered += 5.0
        elif stype == "Text" and conf >= 0.8:
            actions.append(
                RepairAction(
                    category="String Normalization",
                    what_changed=f"Normalized Unicode and spacing in '{col}'.",
                    why_it_changed="Standardized unicode sequences and stripped extra whitespace.",
                    confidence=conf,
                    expected_score_bump=5,
                    rule_fn=make_unicode_clean_rule(col),
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

    # 3. Missing Value Imputation, Outliers, & Memory Optimization
    for col in df.columns:
        # Check if the column is a DNA Sequence to preserve it exactly
        stype = profile.semantic_types.get(col, {}).get("type", "Unknown")
        if stype == "DNA Sequence":
            continue

        dtype = df[col].dtype
        null_count = df[col].null_count()

        # Replaces custom null placeholders like '?'
        placeholder_count = 0
        if dtype == pl.String:
            try:
                placeholders = ["?", "N/A", "n/a", "null", "NULL", "NaN", "nan"]
                placeholder_count = df.select(pl.col(col).cast(pl.String).str.strip_chars().is_in(placeholders).sum()).item()
            except Exception:
                placeholder_count = 0
                
        if placeholder_count > 0:
            actions.append(
                RepairAction(
                    category="Missing Values",
                    what_changed=f"Replaced {placeholder_count} null placeholders ('?') in '{col}' with true nulls.",
                    why_it_changed="Standardized custom null representations to enable robust imputation.",
                    confidence=1.0,
                    expected_score_bump=5,
                    rule_fn=make_replace_null_placeholders_rule(col),
                )
            )
            points_recovered += 5.0
            null_count += placeholder_count

        # Missing values handling using DecisionEngine
        if null_count > 0:
            strategy, params = decision_engine.select_imputation_strategy(
                column_name=col,
                dtype_str=str(dtype),
                null_count=null_count,
                total_count=df.height,
                unique_count=df[col].n_unique()
            )
            
            if strategy == "impute_median":
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
            elif strategy == "impute_mean":
                actions.append(
                    RepairAction(
                        category="Missing Values",
                        what_changed=f"Imputed missing '{col}' values with Mean.",
                        why_it_changed=f"Column is missing {null_count} values. Mean works best for normally distributed values.",
                        confidence=0.8,
                        expected_score_bump=10,
                        rule_fn=make_impute_mean_rule(col),
                    )
                )
                points_recovered += 10.0
            elif strategy == "impute_mode":
                actions.append(
                    RepairAction(
                        category="Missing Values",
                        what_changed=f"Imputed missing '{col}' values with Mode.",
                        why_it_changed=f"Column is missing {null_count} values. Mode represents the most frequent value.",
                        confidence=0.75,
                        expected_score_bump=5,
                        rule_fn=make_impute_mode_rule(col),
                    )
                )
                points_recovered += 5.0
            elif strategy == "forward_fill":
                actions.append(
                    RepairAction(
                        category="Missing Values",
                        what_changed=f"Forward-filled missing '{col}' values.",
                        why_it_changed=f"Column is missing {null_count} values. Forward fill preserves time-series dependency.",
                        confidence=0.85,
                        expected_score_bump=10,
                        rule_fn=make_impute_ffill_rule(col),
                    )
                )
                points_recovered += 10.0
            else:
                fill_val = params.get("value", "Unknown")
                actions.append(
                    RepairAction(
                        category="Missing Values",
                        what_changed=f"Imputed missing '{col}' values with '{fill_val}'.",
                        why_it_changed=f"Column is missing {null_count} values. Constant fill prevents hallucinating cross-sectional data.",
                        confidence=0.7,
                        expected_score_bump=5,
                        rule_fn=make_impute_constant_rule(col, fill_val),
                    )
                )
                points_recovered += 5.0

        # Outliers clipping (if numerical and no nulls)
        if dtype.is_numeric() and null_count == 0:
            # Simple heuristic for skewness
            try:
                skew = df.select(pl.col(col).skew()).item()
                is_skewed = skew is not None and abs(skew) > 1.0
            except Exception:
                is_skewed = False
                
            outlier_strat, outlier_params = decision_engine.select_outlier_strategy(
                column_name=col,
                row_count=df.height,
                is_normal_dist=not is_skewed
            )
            
            if outlier_strat == "iqr":
                actions.append(
                    RepairAction(
                        category="Outlier Handling",
                        what_changed=f"Clipped outliers in '{col}' via IQR.",
                        why_it_changed="IQR-based clipping prevents extreme anomalies from skewing downstream training.",
                        confidence=0.9,
                        expected_score_bump=5,
                        rule_fn=make_outlier_iqr_rule(col, outlier_params.get("threshold", 1.5)),
                    )
                )
                points_recovered += 5.0
            elif outlier_strat == "z_score":
                actions.append(
                    RepairAction(
                        category="Outlier Handling",
                        what_changed=f"Clipped outliers in '{col}' via Z-Score.",
                        why_it_changed="Z-score clipping removes values beyond normal distribution threshold (3.0 std devs).",
                        confidence=0.9,
                        expected_score_bump=5,
                        rule_fn=make_outlier_zscore_rule(col, outlier_params.get("threshold", 3.0)),
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

    # Row-level Deduplication is run at the very end to catch pre-existing duplicates and any duplicates created during imputation/normalization
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
        pass

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
        data: The uncleaned DataFrame or filepath.
        dry_run: If True, prints plan and audits without mutating return value.

    Returns:
        The cleaned DataFrame (in the original format).
    """
    p = plan(data)
    p.show()
    return p.execute(dry_run=dry_run)
