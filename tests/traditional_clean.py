import re
import numpy as np
import pandas as pd
from typing import Any

def parse_sql_expr_imputation(sql_expr: str) -> tuple[str | None, dict[str, Any] | None, Any]:
    """Parses sql_expr of imputation rule to extract group column, group mappings, and default values."""
    if not sql_expr:
        return None, None, None
    m = re.search(r'COALESCE\(\s*"[^"]+"\s*,\s*(.*)\s*\)', sql_expr, re.DOTALL)
    if not m:
        return None, None, None
    expr = m.group(1).strip()
    if expr.startswith("CASE"):
        group_col_match = re.search(r'WHEN\s*"([^"]+)"', expr)
        group_col = group_col_match.group(1) if group_col_match else None
        
        cases = re.findall(r"WHEN\s*\"[^\"]+\"\s*=\s*'([^']+)'\s*THEN\s*(-?[0-9.eE+-]+|'[^']+')", expr)
        mapping = {}
        for k, v in cases:
            if v.startswith("'") and v.endswith("'"):
                v = v[1:-1]
            else:
                try:
                    v = float(v)
                except ValueError:
                    pass
            mapping[k] = v
            
        else_match = re.search(r"ELSE\s*(-?[0-9.eE+-]+|'[^']+'|Unknown)", expr)
        else_val = None
        if else_match:
            else_val = else_match.group(1)
            if else_val.startswith("'") and else_val.endswith("'"):
                else_val = else_val[1:-1]
            else:
                try:
                    else_val = float(else_val)
                except ValueError:
                    pass
        return group_col, mapping, else_val
    else:
        val = expr
        if val.startswith("'") and val.endswith("'"):
            val = val[1:-1]
        else:
            try:
                val = float(val)
            except ValueError:
                pass
        return None, None, val

def parse_sql_expr_outliers(sql_expr: str) -> tuple[float, float] | None:
    """Parses outlier clipping bounds from sql_expr."""
    if not sql_expr:
        return None
    lower_match = re.search(r'<\s*(-?[0-9.eE+-]+)', sql_expr)
    upper_match = re.search(r'>\s*(-?[0-9.eE+-]+)', sql_expr)
    if lower_match and upper_match:
        return float(lower_match.group(1)), float(upper_match.group(1))
    return None

def clean_traditional(df_raw: pd.DataFrame, plan: Any) -> pd.DataFrame:
    """Applies equivalent traditional Pandas/NumPy cleaning transformations

    corresponding to each action in the Tidely RepairPlan.
    """
    df = df_raw.copy()

    for action in plan.actions:
        col = action.column
        cat = action.category
        msg = action.what_changed

        if cat == "Duplicate IDs":
            df = df.drop_duplicates(subset=[col], keep="first")

        elif cat == "Semantic Normalization":
            if "Email" in msg:
                df[col] = df[col].astype(str).str.strip().str.lower()
            elif "Phone" in msg:
                df[col] = df[col].astype(str).str.replace(r"[^\d+]", "", regex=True)
            elif "ZIP Code" in msg:
                df[col] = df[col].astype(str).str.strip().str.rjust(5, "0")
            elif "coordinate" in msg or "Latitude" in msg or "Longitude" in msg:
                bounds = parse_sql_expr_outliers(action.sql_expr)
                if bounds:
                    lower, upper = bounds
                else:
                    is_lat = "Latitude" in msg
                    lower, upper = (-90.0, 90.0) if is_lat else (-180.0, 180.0)
                df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower, upper)
            elif "Unicode and spacing" in msg:
                import unicodedata
                def clean_text(x):
                    if pd.isna(x):
                        return x
                    s = unicodedata.normalize("NFKC", str(x))
                    s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)
                    s = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", s)
                    s = re.sub(r"\s+", " ", s)
                    return s.strip()
                df[col] = df[col].apply(clean_text)
            elif "Converted" in msg and "Datetime" in msg:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            elif "Normalized categories" in msg:
                boolean_map = {
                    "yes": "True", "no": "False",
                    "y": "True", "n": "False",
                    "true": "True", "false": "False",
                    "t": "True", "f": "False",
                    "1": "True", "0": "False",
                }
                def clean_cat(x):
                    if pd.isna(x):
                        return x
                    s = str(x).strip().lower()
                    if s in boolean_map:
                        return boolean_map[s]
                    return s.capitalize()
                df[col] = df[col].apply(clean_cat)
            elif "Normalized Currency/Salary" in msg:
                def clean_currency(x):
                    if pd.isna(x):
                        return np.nan
                    s = str(x).strip().replace("$", "").replace("€", "").replace("£", "").replace("¥", "").replace(" ", "").replace(",", "")
                    if s.endswith("%"):
                        s = s.replace("%", "")
                        try:
                            return float(s) / 100.0
                        except ValueError:
                            return np.nan
                    try:
                        return float(s)
                    except ValueError:
                        return np.nan
                df[col] = df[col].apply(clean_currency)

        elif cat == "Duplicate Rows" and "Fuzzy-merged" in msg:
            try:
                import rapidfuzz
                counts = df[col].value_counts()
                sorted_vals = counts.index.dropna().tolist()
                mapping = {}
                seen_f = set()
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
                df[col] = df[col].map(mapping).fillna(df[col])
            except Exception:
                pass

        elif cat == "Missing Values" and "placeholder" in msg:
            placeholders = ["?", "N/A", "n/a", "null", "NULL", "NaN", "nan"]
            df[col] = df[col].astype(str).str.strip().replace(placeholders, np.nan)

        elif cat == "Missing Values":
            group_col, mapping, val = parse_sql_expr_imputation(action.sql_expr)
            if group_col and mapping:
                df[col] = df[col].fillna(df[group_col].map(mapping)).fillna(val)
            else:
                df[col] = df[col].fillna(val)

        elif cat == "Outlier Handling":
            bounds = parse_sql_expr_outliers(action.sql_expr)
            if bounds:
                lower, upper = bounds
                df[col] = df[col].clip(lower, upper)
            else:
                if "IQR" in msg:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    match = re.search(r"threshold=([0-9.]+)", msg)
                    threshold = float(match.group(1)) if match else 1.5
                    lower = q1 - threshold * iqr
                    upper = q3 + threshold * iqr
                    df[col] = df[col].clip(lower, upper)
                elif "Z-Score" in msg and "MAD" not in msg:
                    mean = df[col].mean()
                    std = df[col].std()
                    match = re.search(r"threshold=([0-9.]+)", msg)
                    threshold = float(match.group(1)) if match else 3.0
                    lower = mean - threshold * std
                    upper = mean + threshold * std
                    df[col] = df[col].clip(lower, upper)
                elif "MAD" in msg or "Modified Z-Score" in msg:
                    median = df[col].median()
                    mad = (df[col] - median).abs().median()
                    match = re.search(r"threshold=([0-9.]+)", msg)
                    threshold = float(match.group(1)) if match else 3.5
                    lower = median - threshold * mad / 0.6745
                    upper = median + threshold * mad / 0.6745
                    df[col] = df[col].clip(lower, upper)

        elif cat == "Memory Optimization":
            if "Categorical" in msg:
                df[col] = df[col].astype("category")
            elif "Downcasted" in msg:
                target_type = msg.split("to ")[-1].replace(".", "").lower()
                df[col] = df[col].astype(target_type)

        elif cat == "Duplicate Rows" and "Dropped exact duplicate rows" in msg:
            df = df.drop_duplicates()

    return df


def audit_parities(df_raw: pd.DataFrame, df_trad: pd.DataFrame, df_tidely: pd.DataFrame) -> dict[str, Any]:
    """Generates a detailed audit log comparing traditional clean and tidely clean results."""
    report = {
        "shapes": {
            "raw": df_raw.shape,
            "traditional": df_trad.shape,
            "tidely": df_tidely.shape,
        },
        "shape_parity": df_trad.shape == df_tidely.shape,
        "columns": {},
        "type_narrowing_overflows": [],
        "all_identical": True,
    }

    cols = list(df_tidely.columns)
    df_trad_aligned = df_trad[cols] if set(cols).issubset(df_trad.columns) else df_trad

    for col in cols:
        if col not in df_trad_aligned.columns:
            report["columns"][col] = {"error": "Missing in traditional clean"}
            report["all_identical"] = False
            continue

        s_trad = df_trad_aligned[col]
        s_tidely = df_tidely[col]

        dtype_trad = str(s_trad.dtype)
        dtype_tidely = str(s_tidely.dtype)

        try:
            if s_trad.dtype.name == 'category' or s_tidely.dtype.name == 'category':
                v_ident = (s_trad.astype(str) == s_tidely.astype(str)).all()
            elif np.issubdtype(s_trad.dtype, np.number) and np.issubdtype(s_tidely.dtype, np.number):
                v_ident = np.isclose(s_trad, s_tidely, equal_nan=True, rtol=1e-5, atol=1e-5).all()
            else:
                v_ident = (s_trad == s_tidely).fillna(True).all()
        except Exception:
            v_ident = False

        if not v_ident:
            report["all_identical"] = False

        stats = {}
        if np.issubdtype(s_trad.dtype, np.number) and np.issubdtype(s_tidely.dtype, np.number):
            stats = {
                "mean_diff": float(abs(s_trad.mean() - s_tidely.mean())),
                "std_diff": float(abs(s_trad.std() - s_tidely.std())),
                "min_diff": float(abs(s_trad.min() - s_tidely.min())),
                "max_diff": float(abs(s_trad.max() - s_tidely.max())),
            }
        else:
            stats = {
                "null_count_diff": int(abs(s_trad.isna().sum() - s_tidely.isna().sum())),
            }

        overflowed = False
        if "int" in dtype_tidely and "int" in str(df_raw[col].dtype):
            raw_min, raw_max = df_raw[col].min(), df_raw[col].max()
            tidely_min, tidely_max = s_tidely.min(), s_tidely.max()
            if (tidely_min != raw_min or tidely_max != raw_max) and not np.isclose(tidely_min, raw_min):
                clipped = False
                for c in stats:
                    if "diff" in c and stats[c] > 0.1:
                        clipped = True
                if not clipped:
                    overflowed = True
                    report["type_narrowing_overflows"].append(col)

        report["columns"][col] = {
            "dtype_traditional": dtype_trad,
            "dtype_tidely": dtype_tidely,
            "identical": bool(v_ident),
            "statistics": stats,
            "overflow_detected": overflowed,
        }

    return report
