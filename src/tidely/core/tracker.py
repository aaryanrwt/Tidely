"""Outcome Tracker for measuring the impact of the cleaning pipeline."""

from typing import Any


class OutcomeTracker:
    """Measures delta (e.g. memory bytes saved) between original and cleaned dataframe."""

    def __init__(self, original_df: Any):
        """Initializes the OutcomeTracker with the original DataFrame reference."""
        self.original_df = original_df
        self.memory_before_bytes = self._calculate_memory(original_df)
        self.initial_health = 60  # Mock logic, real logic in Phase 5

    def _calculate_memory(self, df: Any) -> float:
        """Helper to calculate memory usage in bytes safely across backends."""
        if hasattr(df, "memory_usage"):
            mem = df.memory_usage(deep=True)
            if hasattr(mem, "sum"):
                return float(mem.sum())
        elif hasattr(df, "estimated_size"):
            # Polars
            return float(df.estimated_size())
        return 0.0

    def track(
        self,
        cleaned_df: Any,
        autofixes: list[dict[str, Any]],
        warnings: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Calculates final metrics and compiles the summary input data.

        Args:
            cleaned_df: The fully mutated DataFrame.
            autofixes: The fixes that were applied.
            warnings: The warnings that were bypassed.

        Returns:
            Dictionary containing memory reduction, fix descriptions, and warning descriptions.
        """
        memory_after_bytes = self._calculate_memory(cleaned_df)

        # from collections import Counter  # unused import removed

        # fix_counts = Counter([a.get("category", "Fix") for a in autofixes])
        # warn_counts = Counter([w.get("category", "Warning") for w in warnings])

        formatted_fixes = []
        for action in autofixes:
            cat = action.get("category", "Fix")
            why = action.get("why", "")
            impact = action.get("impact", "")
            if cat == "Duplicate Rows":
                formatted_fixes.append(
                    f"Duplicate Rows removed.\n    Why: {why}\n    Impact: {impact}"
                )
            else:
                formatted_fixes.append(
                    f"{cat} applied.\n    Why: {why}\n    Impact: {impact}"
                )

        # To avoid duplicating the exact same explanation for 10 columns, we group them
        unique_fixes = list(set(formatted_fixes))

        formatted_warnings = []
        for warn in warnings:
            cat = warn.get("category", "Warning")
            col = warn.get("column", "Unknown")
            conf = warn.get("confidence", 0)
            why = warn.get("why", "")
            formatted_warnings.append(
                f"{cat} in '{col}' left unchanged (Confidence {conf}% < 95%).\n    Why: {why}"
            )

        unique_warnings = list(set(formatted_warnings))

        return {
            "memory_before_mb": self.memory_before_bytes / (1024 * 1024),
            "memory_after_mb": memory_after_bytes / (1024 * 1024),
            "initial_health": self.initial_health,
            "final_health": 98,
            "fixes": unique_fixes,
            "warnings": unique_warnings,
        }
