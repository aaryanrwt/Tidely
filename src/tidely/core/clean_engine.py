"""Core execution engine for explainable dataset cleaning."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import polars as pl
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tidely.core.adapter import normalize_to_polars


def safe_emoji(emoji: str, fallback: str = "") -> str:
    """Returns an emoji if supported by the terminal."""
    import sys

    try:
        encoding = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
        emoji.encode(encoding)
        return emoji
    except Exception:
        return fallback


@dataclass
class RepairAction:
    """A single, explainable dataset transformation."""

    category: str
    what_changed: str
    why_it_changed: str
    confidence: float
    expected_score_bump: int
    rule_fn: Callable[[pl.DataFrame], pl.DataFrame]
    rows_affected: int = 0


class RepairPlan:
    """The plan orchestrator containing actionable cleaning steps and their explanations."""

    def __init__(
        self,
        original_data: Any,
        actions: list[RepairAction],
        initial_score: int,
        target_score: int,
    ) -> None:
        """Initialize a RepairPlan."""
        self._original_data = original_data
        self.actions = actions
        self.initial_score = initial_score
        self.target_score = target_score
        self.audit_log: list[dict[str, Any]] = []

    def show(self) -> None:
        """Renders the execution plan in a beautiful terminal UI."""
        console = Console()
        console.print("")

        header = Panel.fit(
            f"[bold white]SPOTLESS CLEANING PLAN[/bold white]\n"
            f"[dim]Trust Score Improvement: {self.initial_score} -> {self.target_score}[/dim]",
            title=f"{safe_emoji('✨')} Explainable Cleaning",
            border_style="green",
            title_align="left",
        )
        console.print(header)

        if not self.actions:
            console.print(
                "[bold green]Dataset is perfectly clean! No actions required.[/bold green]"
            )
            return

        table = Table(
            show_header=True,
            header_style="bold magenta",
            border_style="dim white",
            box=None,
        )
        table.add_column("Category", style="cyan")
        table.add_column("What Changed", style="white")
        table.add_column("Why It Changed", style="dim")
        table.add_column("Confidence", justify="right")

        for action in self.actions:
            conf_color = (
                "green"
                if action.confidence > 0.9
                else "yellow"
                if action.confidence > 0.7
                else "red"
            )
            conf_str = f"[{conf_color}]{action.confidence:.0%}[/{conf_color}]"
            table.add_row(
                action.category, action.what_changed, action.why_it_changed, conf_str
            )

        console.print(
            Panel(
                table,
                title=f"{safe_emoji('🛠️')} Planned Transformations",
                border_style="cyan",
                title_align="left",
            )
        )
        console.print(
            "[dim]Run plan.execute() to apply these changes and view the audit log.[/dim]\n"
        )

    def execute(self, dry_run: bool = False) -> Any:
        """Applies the cleaning steps and returns the cleaned dataset.

        Args:
            dry_run: If True, executes the plan but discards the cleaned dataframe,
                     useful for evaluating `rows_affected` in the audit log.

        Returns:
            The cleaned DataFrame in the same format as the input (pandas, polars, or pyarrow).
        """
        pl_data, format_name = normalize_to_polars(self._original_data)
        if isinstance(pl_data, pl.LazyFrame):
            try:
                df = pl_data.collect(streaming=True)  # type: ignore[call-overload]
            except Exception:
                df = pl_data.collect()
        else:
            df = pl_data

        self.audit_log = []

        for action in self.actions:
            # Measure rows before
            rows_before = df.height

            # For column-level modifications, row count won't change, so we approximate
            # rows_affected by comparing nulls/uniques if applicable, or just state full column modified.
            # But true exact row tracking for arbitrary expressions is complex.
            # We will use simple heuristics.

            # Apply transformation
            try:
                new_df = action.rule_fn(df)
            except Exception as e:
                self.audit_log.append(
                    {
                        "category": action.category,
                        "action": action.what_changed,
                        "status": "FAILED",
                        "error": str(e),
                    }
                )
                continue

            rows_after = new_df.height
            if rows_before != rows_after:
                action.rows_affected = abs(rows_before - rows_after)
            else:
                # If rows are identical in count, assume it was a column level update.
                # Just mark rows_affected as height of dataframe for now, or track diffs.
                action.rows_affected = df.height

            df = new_df

            self.audit_log.append(
                {
                    "category": action.category,
                    "action": action.what_changed,
                    "reason": action.why_it_changed,
                    "confidence": action.confidence,
                    "rows_affected": action.rows_affected,
                    "status": "SUCCESS",
                }
            )

        if dry_run:
            print("Dry run completed. Original data remains unmodified.")
            return self._original_data

        # Convert back to original format
        if format_name == "pandas":
            import pandas as pd  # noqa: F401, F811

            return df.to_pandas()
        elif format_name == "pyarrow":
            return df.to_arrow()
        else:
            return df
