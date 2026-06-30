"""The Result object returned by td.clean()."""

from typing import Any, cast


class CleanResult:
    """The outcome of a Tidely cleaning operation.

    Exposes the cleaned DataFrame directly (or via .df), along with
    methods to view the summary, export reports, or undo changes.
    """

    def __init__(
        self,
        cleaned_df: Any,
        original_df: Any,
        summary_text: str,
        report_data: dict[str, Any],
    ) -> None:
        """Initializes the Result object.

        Args:
            cleaned_df: The production-ready DataFrame.
            original_df: A copy of the original DataFrame before cleaning.
            summary_text: The beautiful terminal output string.
            report_data: Programmatic dictionary of what changed.
        """
        self.df = cleaned_df
        self._original_df = original_df
        self._summary_text = summary_text
        self.report = report_data
        """Initializes the Result object.

        Args:
            cleaned_df: The production-ready DataFrame.
            original_df: A copy of the original DataFrame before cleaning.
            summary_text: The beautiful terminal output string.
            report_data: Programmatic dictionary of what changed.
        """
        self.df = cleaned_df
        self._original_df = original_df
        self._summary_text = summary_text
        self.report = report_data

    def export(self, filepath: str) -> None:
        """Exports the cleaned dataset or report based on the file extension.

        Supported extensions: .csv, .parquet, .html, .pdf
        """
        ext = filepath.split(".")[-1].lower()
        if ext in ("csv", "parquet"):
            from tidely.api import save
            save(self.df, filepath)
        elif ext == "html":
            report = self.report
            col_diag = report.get("column_diagnostics", {})
            fixes = report.get("fixes", [])
            warnings = report.get("warnings", [])
            mem_before = report.get("memory_before_mb", 0.0)
            mem_after = report.get("memory_after_mb", 0.0)
            health_before = report.get("initial_health", 0.0)
            health_after = report.get("final_health", 0.0)
            engine_name = report.get("engine_name", "polars_eager")
            engine_reason = report.get("engine_reason", "Low-latency default in-memory execution.")

            # preview rows
            try:
                import pandas as pd
                if isinstance(self.df, pd.DataFrame):
                    preview_df = self.df.head(10)
                else:
                    preview_df = self.df.head(10)
                preview_html = preview_df._repr_html_() if hasattr(preview_df, "_repr_html_") else ""
                if preview_html is None:
                    preview_html = "<p>Data preview unavailable</p>"
            except Exception:
                preview_html = "<p>Data preview unavailable</p>"

            # Render column diagnostic rows
            diag_rows = ""
            for col, diag in col_diag.items():
                algs_considered = ", ".join(diag.get("algorithms_considered", [])) or "None"
                alg_chosen = diag.get("algorithm_chosen", "None")
                reason = diag.get("reason", "Column already clean.")
                score_before = diag.get("quality_score_before", 100.0)
                score_after = diag.get("quality_score_after", 100.0)
                conf = diag.get("confidence_score", 1.0)
                sem_conf = diag.get("semantic_score", 0.0)

                color_b = "green" if score_before >= 90 else "orange" if score_before >= 70 else "red"
                color_a = "green" if score_after >= 90 else "orange" if score_after >= 70 else "red"

                diag_rows += f"""
                <tr>
                    <td><strong>{col}</strong></td>
                    <td class="score-{color_b}">{score_before:.0f}</td>
                    <td class="score-{color_a}">{score_after:.0f}</td>
                    <td>{conf:.0%}</td>
                    <td>{sem_conf:.0%}</td>
                    <td><code>{alg_chosen}</code></td>
                    <td><span class="considered">{algs_considered}</span></td>
                    <td class="reason">{reason}</td>
                </tr>
                """

            # Render Audit Log
            audit_rows = ""
            for fix in fixes:
                audit_rows += f"<li>{fix.replace(chr(10), '<br>')}</li>"
            for warn in warnings:
                audit_rows += f"<li class='warning-item'>{warn.replace(chr(10), '<br>')}</li>"

            html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Tidely Data Quality Report</title>
    <style>
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #0b0f19;
            color: #f1f5f9;
            margin: 0;
            padding: 2rem;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            background: linear-gradient(135deg, #1e1b4b 0%, #311042 100%);
            border-radius: 12px;
            padding: 2.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
            border: 1px solid #312e81;
        }}
        .header h1 {{
            margin: 0 0 0.5rem 0;
            font-size: 2.5rem;
            background: linear-gradient(to right, #6366f1, #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .header p {{
            color: #94a3b8;
            margin: 0;
            font-size: 1.1rem;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        .card {{
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .card h3 {{
            margin: 0 0 0.5rem 0;
            color: #94a3b8;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .card .value {{
            font-size: 2rem;
            font-weight: 700;
            color: #f8fafc;
        }}
        .tabs {{
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
            border-bottom: 2px solid #334155;
            padding-bottom: 0.5rem;
        }}
        .tab {{
            background: none;
            border: none;
            color: #94a3b8;
            font-size: 1.1rem;
            padding: 0.75rem 1.5rem;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.2s;
            border-radius: 4px;
        }}
        .tab.active, .tab:hover {{
            color: #6366f1;
            background: #1e293b;
        }}
        .tab-content {{
            background: #111827;
            border: 1px solid #1f2937;
            border-radius: 8px;
            padding: 2rem;
            min-height: 300px;
        }}
        .tab-panel {{
            display: none;
        }}
        .tab-panel.active {{
            display: block;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}
        th, td {{
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid #1f2937;
        }}
        th {{
            background-color: #1f2937;
            color: #94a3b8;
            font-weight: 600;
        }}
        .score-green {{ color: #10b981; font-weight: bold; }}
        .score-orange {{ color: #f59e0b; font-weight: bold; }}
        .score-red {{ color: #ef4444; font-weight: bold; }}
        .considered {{ color: #64748b; font-size: 0.85rem; }}
        .reason {{ color: #94a3b8; font-style: italic; }}
        ul {{
            padding-left: 1.5rem;
        }}
        li {{
            margin-bottom: 1rem;
            line-height: 1.5;
        }}
        .warning-item {{
            color: #fbbf24;
        }}
        pre {{
            background: #030712;
            padding: 1.5rem;
            border-radius: 6px;
            overflow-x: auto;
            border: 1px solid #1f2937;
            color: #10b981;
            font-family: 'Fira Code', monospace;
        }}
    </style>
    <script>
        function showTab(tabId, e) {{
            document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            e.currentTarget.classList.add('active');
        }}
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Tidely Spotless Clean Report</h1>
            <p>Dataset Trust Score improved from <strong>{health_before:.0f}%</strong> to <strong>{health_after:.0f}%</strong>. Routing: <em>{engine_reason}</em></p>
        </div>

        <div class="metrics-grid">
            <div class="card">
                <h3>Initial Health Score</h3>
                <div class="value" style="color: #ef4444;">{health_before:.0f}%</div>
            </div>
            <div class="card">
                <h3>Final Health Score</h3>
                <div class="value" style="color: #10b981;">{health_after:.0f}%</div>
            </div>
            <div class="card">
                <h3>Execution Engine</h3>
                <div class="value" style="color: #6366f1; font-size: 1.6rem; padding-top: 0.4rem;">{engine_name}</div>
            </div>
            <div class="card">
                <h3>Initial Memory Usage</h3>
                <div class="value">{mem_before:.2f} MB</div>
            </div>
            <div class="card">
                <h3>Final Memory Usage</h3>
                <div class="value">{mem_after:.2f} MB</div>
            </div>
        </div>

        <div class="tabs">
            <button class="tab active" onclick="showTab('diagnostics', event)">Column Diagnostics</button>
            <button class="tab" onclick="showTab('audit', event)">Audit Logs & Decisions</button>
            <button class="tab" onclick="showTab('preview', event)">Cleaned Data Preview</button>
            <button class="tab" onclick="showTab('summary', event)">Execution Summary</button>
        </div>

        <div class="tab-content">
            <div id="diagnostics" class="tab-panel active">
                <h2>Column-Level Quality Diagnostics</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Column Name</th>
                            <th>Quality Before</th>
                            <th>Quality After</th>
                            <th>Type Confidence</th>
                            <th>Semantic Match</th>
                            <th>Chosen Action</th>
                            <th>Alternatives Rejected</th>
                            <th>Cleaning Heuristics / Decision Reason</th>
                        </tr>
                    </thead>
                    <tbody>
                        {diag_rows}
                    </tbody>
                </table>
            </div>

            <div id="audit" class="tab-panel">
                <h2>Applied Transformations & Decision Logic</h2>
                <ul>
                    {audit_rows}
                </ul>
            </div>

            <div id="preview" class="tab-panel">
                <h2>Cleaned Dataset Preview (First 10 Rows)</h2>
                {preview_html}
            </div>

            <div id="summary" class="tab-panel">
                <h2>Detailed Summary</h2>
                <pre>{self._summary_text}</pre>
            </div>
        </div>
    </div>
</body>
</html>"""
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html_content)
        else:
            raise ValueError(f"Unsupported export format: {ext}")

    def summary(self) -> str:
        """Returns the outcome-focused cleaning summary."""
        return self._summary_text

    def undo(self) -> Any:
        """Reverts the cleaning operation, returning the original DataFrame."""
        if hasattr(self._original_df, "copy"):
            return self._original_df.copy()
        return self._original_df

    # Proxy all other attributes to the underlying DataFrame
    def __getattr__(self, name: str) -> Any:
        """Proxies attribute access to the underlying DataFrame."""
        return getattr(self.df, name)

    def __getitem__(self, key: Any) -> Any:
        """Proxies indexing to the underlying DataFrame."""
        return self.df[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        """Proxies index assignment to the underlying DataFrame."""
        self.df[key] = value

    def __repr__(self) -> str:
        """Returns a string representation of the underlying DataFrame."""
        return repr(self.df)

    def _repr_html_(self) -> str | None:
        if hasattr(self.df, "_repr_html_"):
            return cast(str | None, self.df._repr_html_())
        return None
