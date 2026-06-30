"""Trust Engine for assigning confidence scores to potential cleaning actions."""

from typing import Any


class TrustEngine:
    """Calculates confidence scores (0-100%) for potential fixes.

    If confidence >= 95%, it is considered safe for AutoFix.
    Otherwise, it becomes a Warning/Suggestion.
    """

    def __init__(self, strictness: str = "high"):
        self.strictness = strictness
        # Mapping strictness to a confidence threshold
        self.threshold = {"conservative": 99.0, "high": 95.0, "moderate": 80.0}.get(
            strictness, 95.0
        )

    def evaluate(
        self, df: Any, metadata: dict[str, Any], semantics: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Evaluates what can be cleaned and assigns a confidence score.

        Args:
            df: The DataFrame.
            metadata: Structural metadata.
            semantics: Inferred semantic types.

        Returns:
            List of potential actions with their associated confidence scores.
        """
        actions = []
        columns = metadata.get("columns", {})

        # 1. Duplicate Rows
        if metadata.get("duplicate_rows", 0) > 0:
            actions.append(
                {
                    "type": "dedup_rows",
                    "category": "Duplicate Rows",
                    "confidence": 100.0,
                    "why": f"Found {metadata.get('duplicate_rows')} duplicate rows. Exact duplicates provide no new information and skew statistics.",
                    "impact": "Reduces dataset size and prevents model overfitting.",
                    "description": "Remove exact duplicate rows.",
                }
            )

        for col, col_meta in columns.items():
            sem = semantics.get(col, {})
            sem_type = sem.get("type", "Unknown")
            sem_rate = sem.get("match_rate", 0.0)

            # String Cleanup
            if sem_type == "Text":
                actions.append(
                    {
                        "type": "clean_string",
                        "column": col,
                        "category": "String Normalization",
                        "confidence": 100.0,
                        "why": f"Column '{col}' contains raw text.",
                        "impact": "Strips whitespace and invisible characters for consistency.",
                        "description": f"Normalize text in '{col}'.",
                    }
                )

            # Semantic specific rules
            if sem_type == "Email":
                conf = 99.0 if sem_rate > 0.9 else (sem_rate * 100)
                actions.append(
                    {
                        "type": "normalize_email",
                        "column": col,
                        "category": "Broken Emails",
                        "confidence": conf,
                        "why": f"Column '{col}' matches standard Email patterns.",
                        "impact": "Normalizes case and whitespace so identical emails match perfectly.",
                        "description": f"Normalize emails in '{col}'.",
                    }
                )

            if sem_type == "Date":
                actions.append(
                    {
                        "type": "normalize_date",
                        "column": col,
                        "category": "Date Normalization",
                        "confidence": 99.0,
                        "why": f"Column '{col}' contains temporal data.",
                        "impact": "Converts strings to standardized DateTime objects for time-series operations.",
                        "description": f"Parse dates in '{col}'.",
                    }
                )

            if sem_type == "Boolean":
                actions.append(
                    {
                        "type": "normalize_boolean",
                        "column": col,
                        "category": "Boolean Normalization",
                        "confidence": 98.0,
                        "why": f"Column '{col}' contains truthy/falsy values like 'yes/no'.",
                        "impact": "Standardizes to strict True/False primitives.",
                        "description": f"Standardize booleans in '{col}'.",
                    }
                )

            # Memory optimizations
            if sem_type == "Categorical" and col_meta["total_count"] > 1000:
                actions.append(
                    {
                        "type": "to_categorical",
                        "column": col,
                        "category": "Memory Optimization",
                        "confidence": 99.0,
                        "why": f"Column '{col}' has low cardinality.",
                        "impact": "Replaces repeated strings with integer pointers, drastically reducing memory footprint.",
                        "description": f"Convert '{col}' to categorical.",
                    }
                )

            if sem_type == "Numeric":
                actions.append(
                    {
                        "type": "downcast_numeric",
                        "column": col,
                        "category": "Memory Optimization",
                        "confidence": 100.0,
                        "why": f"Column '{col}' uses 64-bit precision but contains smaller values.",
                        "impact": "Downcasts precision (e.g., int64 -> int16) to save memory without data loss.",
                        "description": f"Downcast numeric column '{col}'.",
                    }
                )

            # Null imputation
            if col_meta["null_count"] > 0:
                if sem_type == "Numeric":
                    actions.append(
                        {
                            "type": "impute_median",
                            "column": col,
                            "category": "Missing Values",
                            "confidence": 85.0,
                            "why": f"Column '{col}' has missing values.",
                            "impact": "Imputes with the median to avoid outlier skewing.",
                            "description": f"Fill missing in '{col}' with median.",
                        }
                    )
                else:
                    actions.append(
                        {
                            "type": "impute_mode",
                            "column": col,
                            "category": "Missing Values",
                            "confidence": 75.0,
                            "why": f"Column '{col}' has missing values.",
                            "impact": "Imputes with the most frequent value (mode).",
                            "description": f"Fill missing in '{col}' with mode.",
                        }
                    )

        return actions
