"""Generates the outcome-focused cleaning summary for Tidely."""


class CleanSummary:
    """Outcome-focused report for the cleaned dataset.

    Instead of showing technical implementation details (e.g., int64 -> int16),
    this focuses on the magic moments: memory saved, missing values handled,
    and business entities fixed.
    """

    def __init__(
        self,
        initial_health: int,
        final_health: int,
        fixes: list[str],
        warnings: list[str],
        memory_before_mb: float,
        memory_after_mb: float,
        execution_time: float = 0.0,
        backend: str = "polars_eager",
        rows_removed: int = 0,
        missing_values_fixed: int = 0,
        duplicates_removed: int = 0,
        outliers_fixed: int = 0,
        datatypes_optimized: int = 0,
        semantic_corrections: int = 0,
    ):
        """Initializes the cleaning summary representation with metric details."""
        self.initial_health = initial_health
        self.final_health = final_health
        self.fixes = fixes
        self.warnings = warnings
        self.memory_before_mb = memory_before_mb
        self.memory_after_mb = memory_after_mb
        self.execution_time = execution_time
        self.backend = backend
        self.rows_removed = rows_removed
        self.missing_values_fixed = missing_values_fixed
        self.duplicates_removed = duplicates_removed
        self.outliers_fixed = outliers_fixed
        self.datatypes_optimized = datatypes_optimized
        self.semantic_corrections = semantic_corrections

    def __str__(self) -> str:
        """Generates a beautiful human-readable summary block string."""
        saved_mb = max(0.0, self.memory_before_mb - self.memory_after_mb)
        pct = (
            (saved_mb / self.memory_before_mb * 100)
            if self.memory_before_mb > 0
            else 0.0
        )

        ml_status = (
            "Excellent (Ready for production ML models)"
            if self.final_health >= 90
            else "Good (Minor quality improvements recommended)"
            if self.final_health >= 70
            else "Fair (Sub-optimal for training without further curation)"
        )
        biz_status = (
            "Excellent (Ready for executive reports and analytics)"
            if self.final_health >= 90
            else "Ready (Minor warnings bypassable)"
            if self.final_health >= 70
            else "Not Ready (High risk of errors)"
        )
        rec = (
            "Deploy to Production"
            if self.final_health >= 90
            else "Verify Column Warnings & Deploy"
            if self.final_health >= 70
            else "Needs Manual Curation"
        )

        lines = [
            "=" * 60,
            "                   TIDELY CLEANING SUMMARY",
            "=" * 60,
            f"Dataset Health:         {self.initial_health}  →  {self.final_health}",
            f"Memory Saved:           {saved_mb:.2f} MB ({pct:.1f}% reduction)",
            f"Execution Time:         {self.execution_time:.3f} seconds",
            f"Backend Used:           {self.backend}",
            "-" * 60,
            "METRIC INSIGHTS & ACTIONS APPLIED:",
            f"  • Rows Removed:          {self.rows_removed}",
            f"  • Missing Values Fixed:  {self.missing_values_fixed}",
            f"  • Duplicates Removed:    {self.duplicates_removed}",
            f"  • Outliers Fixed:        {self.outliers_fixed}",
            f"  • Datatypes Optimized:   {self.datatypes_optimized}",
            f"  • Semantic Corrections:  {self.semantic_corrections}",
            "-" * 60,
        ]

        if self.fixes:
            lines.append("Applied Fixes:")
            for fix in self.fixes:
                fix_lines = fix.split("\n")
                lines.append(f"  ✓ {fix_lines[0]}")
                for line in fix_lines[1:]:
                    lines.append(f"    {line}")
            lines.append("-" * 60)

        lines.extend(
            [
                "Warnings (Requires Human Attention)",
            ]
        )

        if not self.warnings:
            lines.append("  • None. Data looks clean.")
        else:
            for warn in self.warnings:
                warn_lines = warn.split("\n")
                lines.append(f"  • {warn_lines[0]}")
                for line in warn_lines[1:]:
                    lines.append(f"    {line}")

        lines.extend(
            [
                "-" * 60,
                "READINESS CERTIFICATION:",
                f"  • ML Readiness:          {ml_status}",
                f"  • Business Readiness:    {biz_status}",
                f"  • Final Recommendation:  {rec}",
                "=" * 60,
            ]
        )
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Returns the summary representation string."""
        return self.__str__()
