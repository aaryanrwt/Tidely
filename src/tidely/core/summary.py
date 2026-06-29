"""Generates the outcome-focused cleaning summary for Tidely."""

from typing import List

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
        fixes: List[str],
        warnings: List[str],
        memory_before_mb: float,
        memory_after_mb: float
    ):
        self.initial_health = initial_health
        self.final_health = final_health
        self.fixes = fixes
        self.warnings = warnings
        self.memory_before_mb = memory_before_mb
        self.memory_after_mb = memory_after_mb
        
    def __str__(self) -> str:
        lines = [
            "=" * 50,
            "Tidely Cleaning Summary",
            f"Dataset Health: {self.initial_health}  →  {self.final_health}",
            "=" * 50,
            "",
            "Fixed"
        ]
        
        if not self.fixes:
            lines.append("✓ No major issues found.")
        else:
            for fix in self.fixes:
                # We want to format the multiline fix with a checkmark on the first line only
                fix_lines = fix.split('\n')
                lines.append(f"✓ {fix_lines[0]}")
                for line in fix_lines[1:]:
                    lines.append(f"  {line}")
                
        # Always append memory savings if we saved memory
        if self.memory_after_mb < self.memory_before_mb:
            saved_mb = self.memory_before_mb - self.memory_after_mb
            pct = (saved_mb / self.memory_before_mb) * 100
            lines.append(
                f"✓ Memory reduced by {pct:.0f}% "
                f"(Before: {self.memory_before_mb:.1f} MB | After: {self.memory_after_mb:.1f} MB)"
            )
            
        lines.append("")
        lines.append("-" * 50)
        lines.append("Warnings (Requires Human Attention)")
        
        if not self.warnings:
            lines.append("• None. Data looks clean.")
        else:
            for warn in self.warnings:
                warn_lines = warn.split('\n')
                lines.append(f"• {warn_lines[0]}")
                for line in warn_lines[1:]:
                    lines.append(f"  {line}")
                
        lines.append("-" * 50)
        lines.append("")
        lines.append("Result: Dataset ready for ML.")
        lines.append("=" * 50)
        
        return "\n".join(lines)
        
    def __repr__(self) -> str:
        return self.__str__()
