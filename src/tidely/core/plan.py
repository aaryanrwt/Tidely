"""Planning engine that inspects datasets and generates explainable cleaning plans."""

import sys
from typing import Any

import polars as pl

from tidely.core.clean_engine import RepairAction, RepairPlan
from tidely.core.decision_engine import DecisionEngine
from tidely.core.profile import DatasetProfile
from tidely.core.rules import (
    make_categorical_rule,
    make_coordinate_clip_rule,
    make_dedup_id_rule,
    make_dedup_rows_rule,
    make_downcast_rule,
    make_email_rule,
    make_fuzzy_dedup_rule,
    make_impute_constant_rule,
    make_impute_ffill_rule,
    # New rules
    make_impute_group_median_rule,
    make_impute_group_mode_rule,
    make_impute_mean_rule,
    make_impute_median_rule,
    make_impute_mode_rule,
    make_outlier_iqr_rule,
    make_outlier_modified_zscore_rule,
    make_outlier_zscore_rule,
    make_phone_rule,
    make_replace_null_placeholders_rule,
    make_smart_categorical_rule,
    make_smart_date_rule,
    make_smart_numeric_clean_rule,
    make_smart_string_clean_rule,
    make_zip_code_rule,
)


def compute_column_quality(
    col_meta: dict[str, Any], semantic_info: dict[str, Any], has_outliers: bool = False
) -> float:
    """Computes a 0-100 data quality score for a single column."""
    score = 100.0
    null_pct = float(col_meta.get("null_percentage", 0.0))
    score -= null_pct * 50.0

    if has_outliers:
        score -= 15.0

    stype = semantic_info.get("type", "Unknown")
    if stype != "Unknown":
        match_rate = float(semantic_info.get("match_rate", 1.0))
        score -= (1.0 - match_rate) * 30.0

    return float(max(0.0, min(100.0, score)))


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
    metadata = profile.metadata

    # 1. Column Diagnostics and Quality Scoring Before Cleaning
    column_diagnostics: dict[str, dict[str, Any]] = {}
    for col in df.columns:
        col_meta = metadata["columns"][col]
        semantic_info = profile.semantic_types.get(
            col, {"type": "Unknown", "confidence": 0.0, "match_rate": 1.0}
        )
        stype = semantic_info["type"]

        # Simple initial outlier check for scoring
        has_outliers = False
        dtype = df[col].dtype
        if dtype.is_numeric() and col_meta.get("null_count", 0) == 0:
            try:
                skew = col_meta.get("skewness", 0.0) or 0.0
                if abs(skew) > 1.0:
                    q1 = df.select(pl.col(col).quantile(0.25)).item()
                    q3 = df.select(pl.col(col).quantile(0.75)).item()
                    if q1 is not None and q3 is not None:
                        iqr = q3 - q1
                        lower = q1 - 1.5 * iqr
                        upper = q3 + 1.5 * iqr
                        outliers = df.filter(
                            (pl.col(col) < lower) | (pl.col(col) > upper)
                        ).height
                        has_outliers = outliers > 0
            except Exception:
                pass

        q_before = compute_column_quality(col_meta, semantic_info, has_outliers)
        column_diagnostics[col] = {
            "quality_score_before": q_before,
            "quality_score_after": q_before,
            "confidence_score": float(col_meta.get("dtype_confidence", 1.0)),
            "semantic_score": float(semantic_info["confidence"]),
            "repair_score": 0.0,
            "algorithms_considered": [],
            "algorithm_chosen": "None",
            "reason": "Column is already clean.",
        }

    # 2. Schema / Semantic Deduplication & Type Enforcement
    for col, info in profile.semantic_types.items():
        stype = info["type"]
        conf = info["confidence"]
        dtype = df[col].dtype

        # ID Deduplication
        if stype == "ID/Key" and conf >= 0.9:
            if any(
                kw in col.lower()
                for kw in ("name", "desc", "title", "text", "seq", "val")
            ):
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
                            column=col,
                        )
                    )
                    points_recovered += 15.0
                    column_diagnostics[col]["algorithm_chosen"] = (
                        "Deduplicate Primary Key"
                    )
                    column_diagnostics[col]["algorithms_considered"].append(
                        "Deduplicate Primary Key"
                    )
                    column_diagnostics[col]["reason"] = (
                        "Dropped duplicate IDs to enforce entity uniqueness."
                    )
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
                    column=col,
                    sql_expr=f'LOWER(TRIM(CAST("{col}" AS VARCHAR)))',
                )
            )
            points_recovered += 5.0
            column_diagnostics[col]["algorithm_chosen"] = "Email Normalization"
            column_diagnostics[col]["reason"] = (
                "Lowercased and stripped whitespace to standardize format."
            )
        elif stype == "Phone" and conf >= 0.7:
            actions.append(
                RepairAction(
                    category="Semantic Normalization",
                    what_changed=f"Normalized '{col}' as Phone Number.",
                    why_it_changed="Stripped non-numeric formatting to standardize international/local formats.",
                    confidence=conf,
                    expected_score_bump=5,
                    rule_fn=make_phone_rule(col),
                    column=col,
                    sql_expr=f"REGEXP_REPLACE(CAST(\"{col}\" AS VARCHAR), '[^0-9]', '', 'g')",
                )
            )
            points_recovered += 5.0
            column_diagnostics[col]["algorithm_chosen"] = "Phone Normalization"
            column_diagnostics[col]["reason"] = "Removed non-digit formatting."
        elif stype == "ZIP Code" and conf >= 0.7:
            actions.append(
                RepairAction(
                    category="Semantic Normalization",
                    what_changed=f"Normalized '{col}' as ZIP Code.",
                    why_it_changed="Padded ZIP code strings to exactly 5 digits for geographical consistency.",
                    confidence=conf,
                    expected_score_bump=5,
                    rule_fn=make_zip_code_rule(col),
                    column=col,
                    sql_expr=f"LPAD(CAST(\"{col}\" AS VARCHAR), 5, '0')",
                )
            )
            points_recovered += 5.0
            column_diagnostics[col]["algorithm_chosen"] = "ZIP Code Padding"
            column_diagnostics[col]["reason"] = "Padded ZIP codes to exactly 5 digits."
        elif stype in ("Latitude", "Longitude") and conf >= 0.8:
            is_lat = stype == "Latitude"
            min_v, max_v = (-90.0, 90.0) if is_lat else (-180.0, 180.0)
            actions.append(
                RepairAction(
                    category="Semantic Normalization",
                    what_changed=f"Clipped '{col}' {stype} coordinate.",
                    why_it_changed=f"Enforced boundaries strictly inside [{min_v}, {max_v}].",
                    confidence=conf,
                    expected_score_bump=5,
                    rule_fn=make_coordinate_clip_rule(col, is_lat=is_lat),
                    column=col,
                    sql_expr=f'CASE WHEN "{col}" < {min_v} THEN {min_v} WHEN "{col}" > {max_v} THEN {max_v} ELSE "{col}" END',
                )
            )
            points_recovered += 5.0
            column_diagnostics[col]["algorithm_chosen"] = "Coordinate Boundary Clipping"
            column_diagnostics[col]["reason"] = (
                f"Clipped coordinate values to standard {stype} boundaries."
            )
        elif stype == "Text" and conf >= 0.8:
            actions.append(
                RepairAction(
                    category="String Normalization",
                    what_changed=f"Normalized Unicode and spacing in '{col}'.",
                    why_it_changed="Standardized unicode sequences and stripped extra whitespace.",
                    confidence=conf,
                    expected_score_bump=5,
                    rule_fn=make_smart_string_clean_rule(col),
                    column=col,
                    sql_expr=f"REGEXP_REPLACE(REGEXP_REPLACE(REGEXP_REPLACE(TRIM(CAST(\"{col}\" AS VARCHAR)), '[\\x{{200B}}-\\x{{200D}}\\x{{FEFF}}]', '', 'g'), '[\\x00-\\x1F\\x7F-\\x9F]', '', 'g'), '[\\s+]', ' ', 'g')",
                )
            )
            points_recovered += 5.0
            column_diagnostics[col]["algorithm_chosen"] = "Smart Unicode String Clean"
            column_diagnostics[col]["reason"] = (
                "Standardized Unicode NFKC, zero-width, smart quotes and extra whitespace."
            )
        elif stype in ("Date", "Datetime") and dtype == pl.String:
            actions.append(
                RepairAction(
                    category="Type Normalization",
                    what_changed=f"Converted '{col}' to Native Datetime.",
                    why_it_changed="Casting string dates enables temporal aggregations and time-series ML.",
                    confidence=conf,
                    expected_score_bump=10,
                    rule_fn=make_smart_date_rule(col),
                    column=col,
                    sql_expr=f'TRY_CAST("{col}" AS TIMESTAMP)',
                )
            )
            points_recovered += 10.0
            column_diagnostics[col]["algorithm_chosen"] = "Smart Date Parsing"
            column_diagnostics[col]["reason"] = (
                "Parsed mixed dates, Excel serial dates, and Unix timestamps."
            )
        elif stype in ("Gender", "US State") and conf >= 0.7:
            actions.append(
                RepairAction(
                    category="Semantic Normalization",
                    what_changed=f"Normalized categories in '{col}' ({stype}).",
                    why_it_changed="Standardized categorical casing and merged spelling variations.",
                    confidence=conf,
                    expected_score_bump=5,
                    rule_fn=make_smart_categorical_rule(col),
                    column=col,
                    sql_expr=f"CASE WHEN LOWER(TRIM(CAST(\"{col}\" AS VARCHAR))) IN ('yes', 'y', 'true', '1') THEN 'True' WHEN LOWER(TRIM(CAST(\"{col}\" AS VARCHAR))) IN ('no', 'n', 'false', '0') THEN 'False' ELSE INITCAP(LOWER(TRIM(CAST(\"{col}\" AS VARCHAR)))) END",
                )
            )
            points_recovered += 5.0
            column_diagnostics[col]["algorithm_chosen"] = "Smart Category Normalization"
            column_diagnostics[col]["reason"] = (
                "Standardized capitalization and merged truthy/falsy values."
            )
        elif stype in ("Salary", "Currency") and conf >= 0.7:
            actions.append(
                RepairAction(
                    category="Semantic Normalization",
                    what_changed=f"Normalized Currency/Salary in '{col}'.",
                    why_it_changed="Extracted numeric values and stripped currency symbols.",
                    confidence=conf,
                    expected_score_bump=7,
                    rule_fn=make_smart_numeric_clean_rule(col),
                    column=col,
                    sql_expr=f"TRY_CAST(REGEXP_REPLACE(REGEXP_REPLACE(TRIM(CAST(\"{col}\" AS VARCHAR)), '[\\$\\€\\£\\¥\\s,]', '', 'g'), '%', '', 'g') AS DOUBLE)",
                )
            )
            points_recovered += 7.0
            column_diagnostics[col]["algorithm_chosen"] = "Smart Numeric Clean"
            column_diagnostics[col]["reason"] = (
                "Extracted numeric amounts, stripped commas and currencies."
            )

    # Fuzzy duplicate deduplication (RapidFuzz if installed)
    if "rapidfuzz" in sys.modules:
        for col in df.columns:
            stype = profile.semantic_types.get(col, {}).get("type", "Unknown")
            conf = profile.semantic_types.get(col, {}).get("confidence", 0.0)
            if stype in ("Name", "Address") and conf >= 0.7:
                try:
                    import rapidfuzz

                    counts = df.group_by(col).count().sort("count", descending=True)
                    sorted_vals = counts[col].drop_nulls().to_list()
                    mapping = {}
                    seen_f: set[str] = set()
                    for val in sorted_vals:
                        val_str = str(val)
                        if val_str in seen_f:
                            continue
                        matched = False
                        for canonical in seen_f:
                            if rapidfuzz.fuzz.ratio(val_str, canonical) >= 90.0:
                                mapping[val] = canonical
                                matched = True
                                break
                        if not matched:
                            seen_f.add(val_str)
                            mapping[val] = val

                    case_parts = [
                        f"WHEN \"{col}\" = '{k}' THEN '{v}'"
                        for k, v in mapping.items()
                        if k != v
                    ]
                    sql_expr = (
                        f'CASE {" ".join(case_parts)} ELSE "{col}" END'
                        if case_parts
                        else f'"{col}"'
                    )

                    actions.append(
                        RepairAction(
                            category="Duplicate Rows",
                            what_changed=f"Fuzzy-merged spelling typos in '{col}'.",
                            why_it_changed="Grouped spelling variations of similar categories with similarity score >= 90%.",
                            confidence=0.85,
                            expected_score_bump=5,
                            rule_fn=make_fuzzy_dedup_rule(
                                col, threshold=90.0, mapping=mapping
                            ),
                            column=col,
                            sql_expr=sql_expr,
                        )
                    )
                    points_recovered += 5.0
                    column_diagnostics[col]["algorithm_chosen"] = (
                        "Fuzzy Duplicate Standardization"
                    )
                    column_diagnostics[col]["reason"] = (
                        "Merged categories using fuzzy string matching."
                    )
                except Exception:
                    pass

    # 3. Missing Value Imputation, Outliers, & Memory Optimization
    for col in df.columns:
        # Check if the column is protected to preserve it exactly
        stype = profile.semantic_types.get(col, {}).get("type", "Unknown")
        if stype in (
            "DNA Sequence",
            "ID/Key",
            "UUID",
            "SKU",
            "CustomerID",
            "InvoiceID",
        ) or stype.endswith("ID"):
            continue

        dtype = df[col].dtype
        col_meta = metadata["columns"][col]
        null_count = col_meta["null_count"]

        # Replaces custom null placeholders like '?'
        placeholder_count = 0
        if dtype == pl.String:
            try:
                placeholders = ["?", "N/A", "n/a", "null", "NULL", "NaN", "nan"]
                placeholder_count = df.select(
                    pl.col(col)
                    .cast(pl.String)
                    .str.strip_chars()
                    .is_in(placeholders)
                    .sum()
                ).item()
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
                    column=col,
                    sql_expr=f"CASE WHEN TRIM(CAST(\"{col}\" AS VARCHAR)) IN ('?', 'N/A', 'n/a', 'null', 'NULL', 'NaN', 'nan') THEN NULL ELSE \"{col}\" END",
                )
            )
            points_recovered += 5.0
            null_count += placeholder_count

        # Missing values handling using DecisionEngine
        if null_count > 0:
            skew_val = col_meta.get("skewness", 0.0) or 0.0
            is_skewed = abs(skew_val) > 1.0
            null_corrs = col_meta.get("null_correlations", {})

            strategy, params = decision_engine.select_imputation_strategy(
                column_name=col,
                dtype_str=str(dtype),
                null_count=null_count,
                total_count=df.height,
                unique_count=df[col].n_unique(),
                is_skewed=is_skewed,
                null_correlations=null_corrs,
            )

            column_diagnostics[col]["algorithms_considered"].extend(
                ["Mean Imputation", "Median Imputation", "Mode Imputation"]
            )

            if strategy == "impute_group_median":
                group_col = params["group_column"]
                try:
                    g_med = df.select(pl.col(col).median()).item()
                    g_med_val = float(g_med) if g_med is not None else 0.0
                    group_meds_df = df.group_by(group_col).agg(pl.col(col).median())
                    group_meds_map = {}
                    for r in group_meds_df.iter_rows():
                        if r[0] is not None and r[1] is not None:
                            group_meds_map[r[0]] = float(r[1])
                    case_parts = [
                        f"WHEN \"{group_col}\" = '{k}' THEN {v}"
                        for k, v in group_meds_map.items()
                    ]
                    sql_expr = (
                        f'COALESCE("{col}", CASE {" ".join(case_parts)} ELSE {g_med_val} END)'
                        if case_parts
                        else f'COALESCE("{col}", {g_med_val})'
                    )

                    actions.append(
                        RepairAction(
                            category="Missing Values",
                            what_changed=f"Imputed missing '{col}' values via Group Median by '{group_col}'.",
                            why_it_changed=f"Missingness is correlated with '{group_col}'. Group median handles MAR data.",
                            confidence=0.85,
                            expected_score_bump=12,
                            rule_fn=make_impute_group_median_rule(
                                col, group_col, global_median=g_med_val
                            ),
                            column=col,
                            sql_expr=sql_expr,
                        )
                    )
                    points_recovered += 12.0
                    column_diagnostics[col]["algorithm_chosen"] = "Group-by Median"
                    column_diagnostics[col]["reason"] = (
                        f"Imputed nulls using Group Median grouped by correlated column '{group_col}'."
                    )
                except Exception:
                    strategy = "impute_median"

            if strategy == "impute_group_mode":
                group_col = params["group_column"]
                try:
                    g_mode_series = df.select(pl.col(col).mode())
                    g_mode_val = (
                        g_mode_series.item(0, 0)
                        if g_mode_series.height > 0
                        else "Unknown"
                    )
                    if g_mode_val is None:
                        g_mode_val = "Unknown"
                    group_modes_df = df.group_by(group_col).agg(pl.col(col).mode())
                    group_modes_map = {}
                    for r in group_modes_df.iter_rows():
                        if r[0] is not None and r[1] is not None:
                            val = r[1]
                            if isinstance(val, pl.Series):
                                val = val.item(0) if val.len() > 0 else g_mode_val
                            group_modes_map[r[0]] = val
                    case_parts = [
                        f"WHEN \"{group_col}\" = '{k}' THEN '{v}'"
                        for k, v in group_modes_map.items()
                    ]
                    sql_expr = (
                        f"COALESCE(\"{col}\", CASE {' '.join(case_parts)} ELSE '{g_mode_val}' END)"
                        if case_parts
                        else f"COALESCE(\"{col}\", '{g_mode_val}')"
                    )

                    actions.append(
                        RepairAction(
                            category="Missing Values",
                            what_changed=f"Imputed missing '{col}' values via Group Mode by '{group_col}'.",
                            why_it_changed=f"Missingness is correlated with '{group_col}'. Group mode handles MAR data.",
                            confidence=0.8,
                            expected_score_bump=8,
                            rule_fn=make_impute_group_mode_rule(
                                col,
                                group_col,
                                global_mode_val=g_mode_val,
                                group_modes=group_modes_map,
                            ),
                            column=col,
                            sql_expr=sql_expr,
                        )
                    )
                    points_recovered += 8.0
                    column_diagnostics[col]["algorithm_chosen"] = "Group-by Mode"
                    column_diagnostics[col]["reason"] = (
                        f"Imputed nulls using Group Mode grouped by correlated column '{group_col}'."
                    )
                except Exception:
                    strategy = "impute_mode"

            if strategy == "impute_median":
                val = df.select(pl.col(col).median()).item()
                val_f = float(val) if val is not None else 0.0
                actions.append(
                    RepairAction(
                        category="Missing Values",
                        what_changed=f"Imputed missing '{col}' values with Median.",
                        why_it_changed=f"Column is missing {null_count} values. Median is robust to numeric outliers.",
                        confidence=0.8,
                        expected_score_bump=10,
                        rule_fn=make_impute_median_rule(col, value=val_f),
                        column=col,
                        sql_expr=f'COALESCE("{col}", {val_f})',
                    )
                )
                points_recovered += 10.0
                column_diagnostics[col]["algorithm_chosen"] = "Median Imputation"
                column_diagnostics[col]["reason"] = (
                    "Imputed using overall column Median (skewed distribution)."
                )
            elif strategy == "impute_mean":
                val = df.select(pl.col(col).mean()).item()
                val_f = float(val) if val is not None else 0.0
                actions.append(
                    RepairAction(
                        category="Missing Values",
                        what_changed=f"Imputed missing '{col}' values with Mean.",
                        why_it_changed=f"Column is missing {null_count} values. Mean works best for normally distributed values.",
                        confidence=0.8,
                        expected_score_bump=10,
                        rule_fn=make_impute_mean_rule(col, value=val_f),
                        column=col,
                        sql_expr=f'COALESCE("{col}", {val_f})',
                    )
                )
                points_recovered += 10.0
                column_diagnostics[col]["algorithm_chosen"] = "Mean Imputation"
                column_diagnostics[col]["reason"] = (
                    "Imputed using overall column Mean (normal distribution)."
                )
            elif strategy == "impute_mode":
                mode_series = df.select(pl.col(col).mode())
                val = mode_series.item(0, 0) if mode_series.height > 0 else "Unknown"
                if val is None:
                    val = "Unknown"
                sql_expr = (
                    f"COALESCE(\"{col}\", '{val}')"
                    if isinstance(val, str)
                    else f'COALESCE("{col}", {val})'
                )
                actions.append(
                    RepairAction(
                        category="Missing Values",
                        what_changed=f"Imputed missing '{col}' values with Mode.",
                        why_it_changed=f"Column is missing {null_count} values. Mode represents the most frequent value.",
                        confidence=0.75,
                        expected_score_bump=5,
                        rule_fn=make_impute_mode_rule(col, value=val),
                        column=col,
                        sql_expr=sql_expr,
                    )
                )
                points_recovered += 5.0
                column_diagnostics[col]["algorithm_chosen"] = "Mode Imputation"
                column_diagnostics[col]["reason"] = "Imputed using overall column Mode."
            elif strategy == "forward_fill":
                actions.append(
                    RepairAction(
                        category="Missing Values",
                        what_changed=f"Forward-filled missing '{col}' values.",
                        why_it_changed=f"Column is missing {null_count} values. Forward fill preserves time-series dependency.",
                        confidence=0.85,
                        expected_score_bump=10,
                        rule_fn=make_impute_ffill_rule(col),
                        column=col,
                        sql_expr=f'COALESCE("{col}", LAG("{col}") IGNORE NULLS OVER ())',
                    )
                )
                points_recovered += 10.0
                column_diagnostics[col]["algorithm_chosen"] = "Forward Fill"
                column_diagnostics[col]["reason"] = (
                    "Time-series/sequential columns forward filled."
                )
            elif strategy == "impute_constant":
                fill_val = params.get("value", "Unknown")
                actions.append(
                    RepairAction(
                        category="Missing Values",
                        what_changed=f"Imputed missing '{col}' values with '{fill_val}'.",
                        why_it_changed=f"Column is missing {null_count} values. Constant fill prevents hallucinating cross-sectional data.",
                        confidence=0.7,
                        expected_score_bump=5,
                        rule_fn=make_impute_constant_rule(col, fill_val),
                        column=col,
                        sql_expr=f"COALESCE(\"{col}\", '{fill_val}')",
                    )
                )
                points_recovered += 5.0
                column_diagnostics[col]["algorithm_chosen"] = "Constant Imputation"
                column_diagnostics[col]["reason"] = (
                    f"Imputed with constant '{fill_val}'."
                )

        # Outliers clipping (if numerical and no nulls)
        if dtype.is_numeric() and null_count == 0:
            skew_val = col_meta.get("skewness", 0.0) or 0.0
            kurt_val = col_meta.get("kurtosis", 0.0) or 0.0
            is_skewed = abs(skew_val) > 1.0

            outlier_strat, outlier_params = decision_engine.select_outlier_strategy(
                column_name=col,
                row_count=df.height,
                is_normal_dist=not is_skewed,
                skewness=skew_val,
                kurtosis=kurt_val,
            )

            column_diagnostics[col]["algorithms_considered"].extend(
                ["IQR Clipping", "Z-Score Clipping", "Modified Z-Score Clipping"]
            )

            if outlier_strat == "iqr":
                q1 = df.select(pl.col(col).quantile(0.25)).item()
                q3 = df.select(pl.col(col).quantile(0.75)).item()
                threshold = outlier_params.get("threshold", 1.5)
                if q1 is not None and q3 is not None:
                    iqr = q3 - q1
                    lower = float(q1 - threshold * iqr)
                    upper = float(q3 + threshold * iqr)
                else:
                    lower, upper = 0.0, 0.0

                actions.append(
                    RepairAction(
                        category="Outlier Handling",
                        what_changed=f"Clipped outliers in '{col}' via IQR.",
                        why_it_changed="IQR-based clipping prevents extreme anomalies from skewing downstream training.",
                        confidence=0.9,
                        expected_score_bump=5,
                        rule_fn=make_outlier_iqr_rule(
                            col,
                            threshold=threshold,
                            lower_bound=lower,
                            upper_bound=upper,
                        ),
                        column=col,
                        sql_expr=f'CASE WHEN "{col}" < {lower} THEN {lower} WHEN "{col}" > {upper} THEN {upper} ELSE "{col}" END',
                    )
                )
                points_recovered += 5.0
                column_diagnostics[col]["algorithm_chosen"] = "IQR Clipping"
                column_diagnostics[col]["reason"] = (
                    "Extreme heavy-tailed distribution, outliers clipped using IQR."
                )
            elif outlier_strat == "z_score":
                mean_v = df.select(pl.col(col).mean()).item()
                std_v = df.select(pl.col(col).std()).item()
                threshold = outlier_params.get("threshold", 3.0)
                if mean_v is not None and std_v is not None and std_v > 0:
                    lower = float(mean_v - threshold * std_v)
                    upper = float(mean_v + threshold * std_v)
                else:
                    lower, upper = 0.0, 0.0

                actions.append(
                    RepairAction(
                        category="Outlier Handling",
                        what_changed=f"Clipped outliers in '{col}' via Z-Score.",
                        why_it_changed="Z-score clipping removes values beyond normal distribution threshold (3.0 std devs).",
                        confidence=0.9,
                        expected_score_bump=5,
                        rule_fn=make_outlier_zscore_rule(
                            col, threshold=threshold, mean_val=mean_v, std_val=std_v
                        ),
                        column=col,
                        sql_expr=f'CASE WHEN "{col}" < {lower} THEN {lower} WHEN "{col}" > {upper} THEN {upper} ELSE "{col}" END',
                    )
                )
                points_recovered += 5.0
                column_diagnostics[col]["algorithm_chosen"] = "Z-Score Clipping"
                column_diagnostics[col]["reason"] = (
                    "Normally distributed column, Z-score clipping applied."
                )
            elif outlier_strat == "modified_zscore":
                median_v = df.select(pl.col(col).median()).item()
                threshold = outlier_params.get("threshold", 3.5)
                if median_v is not None:
                    mad_v = df.select((pl.col(col) - median_v).abs().median()).item()
                else:
                    mad_v = None
                if median_v is not None and mad_v is not None and mad_v > 0:
                    lower = float(median_v - (threshold * mad_v / 0.6745))
                    upper = float(median_v + (threshold * mad_v / 0.6745))
                else:
                    lower, upper = 0.0, 0.0

                actions.append(
                    RepairAction(
                        category="Outlier Handling",
                        what_changed=f"Clipped outliers in '{col}' via MAD Modified Z-Score.",
                        why_it_changed="Modified Z-score using Median Absolute Deviation (MAD) is robust for skewed distributions.",
                        confidence=0.9,
                        expected_score_bump=6,
                        rule_fn=make_outlier_modified_zscore_rule(
                            col, threshold=threshold, median_val=median_v, mad_val=mad_v
                        ),
                        column=col,
                        sql_expr=f'CASE WHEN "{col}" < {lower} THEN {lower} WHEN "{col}" > {upper} THEN {upper} ELSE "{col}" END',
                    )
                )
                points_recovered += 6.0
                column_diagnostics[col]["algorithm_chosen"] = "MAD Modified Z-Score"
                column_diagnostics[col]["reason"] = (
                    "Slightly skewed distribution, clipped using MAD-based Modified Z-Score."
                )

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
                        column=col,
                        sql_expr=f'CAST("{col}" AS VARCHAR)',
                    )
                )
                points_recovered += 5.0
                column_diagnostics[col]["algorithm_chosen"] = "Cast to Categorical"
                column_diagnostics[col]["reason"] = (
                    "String column converted to Categorical (cardinality ratio < 5%)."
                )
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
                            column=col,
                            sql_expr=f'CAST("{col}" AS TINYINT)',
                        )
                    )
                    points_recovered += 2.0
                    column_diagnostics[col]["algorithm_chosen"] = "Downcast to Int8"
                    column_diagnostics[col]["reason"] = (
                        "Integer column downcasted to Int8."
                    )
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
                            column=col,
                            sql_expr=f'CAST("{col}" AS SMALLINT)',
                        )
                    )
                    points_recovered += 2.0
                    column_diagnostics[col]["algorithm_chosen"] = "Downcast to Int16"
                    column_diagnostics[col]["reason"] = (
                        "Integer column downcasted to Int16."
                    )

    # Row-level Deduplication is run at the very end
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

    # 4. Compute quality scores after cleaning (approximate based on planned fixes)
    for col in df.columns:
        # If the column has missing values or outliers imputed, score goes up
        for action in actions:
            if getattr(action, "column", "") == col:
                column_diagnostics[col]["quality_score_after"] = min(
                    100.0,
                    column_diagnostics[col]["quality_score_before"]
                    + action.expected_score_bump,
                )
        column_diagnostics[col]["repair_score"] = (
            column_diagnostics[col]["quality_score_after"]
            - column_diagnostics[col]["quality_score_before"]
        )

    # Attach diagnostics to metadata for downstream HTML rendering
    plan_obj = RepairPlan(
        original_data=data,
        actions=actions,
        initial_score=initial_score,
        target_score=target_score,
    )
    plan_obj.column_diagnostics = column_diagnostics  # type: ignore[attr-defined]

    return plan_obj


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
