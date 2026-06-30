"""Dataset profile representation and visualization classes."""

import sys
from typing import Any

import polars as pl
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress
from rich.table import Table

from tidely.core.dna import DatasetDNA
from tidely.core.scorer import TrustScores


def safe_emoji(emoji: str, fallback: str = "") -> str:
    """Returns the emoji if sys.stdout supports its encoding, else the fallback."""
    try:
        encoding = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
        emoji.encode(encoding)
        return emoji
    except Exception:
        return fallback


class DatasetProfile:
    """Holds the full metadata diagnosis profile of a dataset.

    Provides programmatic inspection access and prints a premium terminal TUI.
    """

    def __init__(
        self,
        row_count: int,
        col_count: int,
        dna: DatasetDNA,
        trust_score: TrustScores,
        diagnoses: list[Any],
        semantic_types: dict[str, dict[str, Any]],
        format_name: str,
        _df_ref: pl.DataFrame,
    ) -> None:
        """Initialize DatasetProfile."""
        self.row_count = row_count
        self.col_count = col_count
        self.dna = dna
        self.trust_score = trust_score
        self.diagnoses = diagnoses
        self.semantic_types = semantic_types
        self.format_name = format_name
        self._df_ref = _df_ref

    def show(self) -> None:
        """Renders the stunning consult-grade Dataset Inspection terminal dashboard."""
        console = Console()

        # 1. Header Banner
        console.print("")
        console.print(
            Panel.fit(
                "[bold white]SPOTLESS DATASET INSPECTOR[/bold white]\n"
                f"[dim]Backend: {self.format_name.upper()} | "
                f"Rows: {self.row_count:,} | Columns: {self.col_count}[/dim]",
                title=f"{safe_emoji('🔍')} Dataset Intelligence Report",
                border_style="magenta",
                title_align="left",
            )
        )

        # Compute Core Health Metrics
        missing_count = sum(
            self._df_ref[col].null_count() for col in self._df_ref.columns
        )
        total_cells = self.row_count * self.col_count
        missing_pct = (missing_count / total_cells) * 100 if total_cells else 0

        duplicate_rows = 0
        try:
            duplicate_rows = self.row_count - self._df_ref.n_unique()
        except Exception:
            pass  # Ignore unique calculation failures for nested types

        # 2. Dataset DNA & Basic Health
        dna_table = Table.grid(padding=(0, 2))
        dna_table.add_column("Key", style="bold cyan", width=18)
        dna_table.add_column("Value", style="white")

        dna_table.add_row(
            "Detected Domain",
            f"{self.dna.domain} [dim](Confidence: {self.dna.confidence:.0%})[/dim]",
        )
        dna_table.add_row("Inferred Entities", ", ".join(self.dna.entities))
        dna_table.add_row(
            "Missing Cells", f"{missing_count:,} [dim]({missing_pct:.1f}%)[/dim]"
        )
        dna_table.add_row("Duplicate Rows", f"{duplicate_rows:,}")

        console.print(
            Panel(
                dna_table,
                title=f"{safe_emoji('🧬')} Dataset DNA & Health",
                border_style="cyan",
                title_align="left",
            )
        )

        # 3. Lighthouse-style Trust Scores
        def get_score_color(score: int) -> str:
            if score >= 90:
                return "green"
            if score >= 70:
                return "yellow"
            return "red"

        score_color = get_score_color(self.trust_score.overall)

        scores_table = Table.grid(padding=(0, 2))
        scores_table.add_column("Metric", style="bold white", width=22)
        scores_table.add_column("Score", style="white", justify="right", width=6)
        scores_table.add_column("Meter", width=35)

        metrics = [
            ("Reliability", self.trust_score.reliability),
            ("ML Readiness", self.trust_score.ml_readiness),
            ("Memory Efficiency", self.trust_score.memory_efficiency),
            ("Schema Stability", self.trust_score.schema_stability),
            ("Semantic Quality", self.trust_score.semantic_quality),
        ]

        for name, score in metrics:
            bar_color = get_score_color(score)
            prog = Progress(
                BarColumn(bar_width=30, style="dim white", complete_style=bar_color)
            )
            task_id = prog.add_task("", total=100)
            prog.update(task_id, completed=score)

            scores_table.add_row(name, f"[{bar_color}]{score}%[/{bar_color}]", prog)

        console.print(
            Panel(
                scores_table,
                title=f"{safe_emoji('🛡️')} Dataset Trust Score: [{score_color}]{self.trust_score.overall}/100[/{score_color}]",
                border_style=score_color,
                title_align="left",
            )
        )

        # 4. Semantic Intelligence
        sem_table = Table(
            show_header=True,
            header_style="bold magenta",
            border_style="dim white",
            box=None,
        )
        sem_table.add_column("Column", style="cyan")
        sem_table.add_column("Semantic Type", style="white")
        sem_table.add_column("Format Health", justify="right")
        sem_table.add_column("Business Impact / Insight", style="dim")

        for col, info in self.semantic_types.items():
            stype = info["type"]
            conf = info["confidence"]
            if stype != "Unknown":
                health_color = get_score_color(int(conf * 100))
                health_str = f"[{health_color}]{conf:.0%}[/{health_color}]"

                impact = ""
                if conf < 1.0:
                    impact = f"[yellow]Warning:[/yellow] {1 - conf:.0%} format violations. Downstream parsing may fail."
                elif stype in ("Email", "Phone", "PAN", "Aadhaar"):
                    impact = "[green]Valid format. Safe for PII extraction.[/green]"
                elif stype == "Currency":
                    impact = "Ready for financial aggregations."

                sem_table.add_row(col, stype, health_str, impact)

        if sem_table.row_count > 0:
            console.print(
                Panel(
                    sem_table,
                    title=f"{safe_emoji('🧠')} Semantic Intelligence",
                    border_style="magenta",
                    title_align="left",
                )
            )

        # 5. Memory & Type Insights
        mem_table = Table(
            show_header=True,
            header_style="bold yellow",
            border_style="dim white",
            box=None,
        )
        mem_table.add_column("Column", style="cyan")
        mem_table.add_column("Current Type", style="white")
        mem_table.add_column("Suggestion", style="green")

        for col in self._df_ref.columns:
            dtype = self._df_ref[col].dtype
            if dtype == pl.String:
                n_unique = self._df_ref[col].n_unique()
                if n_unique < (self.row_count * 0.05) and self.row_count > 1000:
                    mem_table.add_row(
                        col,
                        "String",
                        f"Convert to Categorical (Only {n_unique} unique values)",
                    )
            elif dtype.is_integer():
                c_min, c_max = self._df_ref[col].min(), self._df_ref[col].max()
                if isinstance(c_min, (int, float)) and isinstance(c_max, (int, float)):
                    if c_min >= -128 and c_max <= 127 and dtype != pl.Int8:
                        mem_table.add_row(
                            col,
                            str(dtype),
                            "Downcast to Int8 (Values range between -128 and 127)",
                        )
                    elif (
                        c_min >= -32768
                        and c_max <= 32767
                        and dtype not in (pl.Int8, pl.Int16)
                    ):
                        mem_table.add_row(col, str(dtype), "Downcast to Int16")

        if mem_table.row_count > 0:
            console.print(
                Panel(
                    mem_table,
                    title=f"{safe_emoji('⚡')} Optimization Opportunities",
                    border_style="yellow",
                    title_align="left",
                )
            )

        console.print("")
