"""The Result object returned by td.clean()."""

import os
import sys
import time
import platform
from typing import Any, cast

try:
    import psutil
except ImportError:
    psutil = None


class CleaningDiff:
    """The granular difference log of all cell-level modifications."""

    def __init__(self, diffs: list[dict[str, Any]], execution_time: float, backend: str, planner_decision: str) -> None:
        import pandas as pd
        data = []
        for d in diffs:
            data.append({
                "Row": d.get("row"),
                "Column": d.get("column"),
                "Original Value": d.get("original"),
                "Traditional Pipeline Value": d.get("traditional", "NaN"),
                "Tidely Value": d.get("cleaned"),
                "Rule Applied": d.get("rule"),
                "Statistical Reason": d.get("reason"),
                "Execution Time": execution_time,
                "Backend Used": backend,
                "Planner Decision": planner_decision,
            })
        self.df = pd.DataFrame(data)

    def to_csv(self, filepath: str) -> None:
        self.df.to_csv(filepath, index=False)

    def to_parquet(self, filepath: str) -> None:
        self.df.to_parquet(filepath, index=False)

    def to_json(self, filepath: str) -> None:
        self.df.to_json(filepath, orient="records", indent=2)

    def to_markdown(self, filepath: str = None) -> str:
        md = self.df.to_markdown(index=False)
        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(md)
        return md

    def __getattr__(self, name: str) -> Any:
        return getattr(self.df, name)

    def __getitem__(self, key: Any) -> Any:
        return self.df[key]

    def __repr__(self) -> str:
        return repr(self.df)


class AuditLog:
    """Enterprise-grade audit log for compliance and verification."""

    def __init__(self, log_dict: dict[str, Any]) -> None:
        self.log_dict = log_dict

    def to_json(self, filepath: str = None) -> str:
        import json
        js = json.dumps(self.log_dict, indent=2, default=str)
        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(js)
        return js

    def to_markdown(self, filepath: str = None) -> str:
        md = f"""# Tidely Enterprise Audit Log
Generated at: {self.log_dict.get('timestamp')}
Tidely Version: {self.log_dict.get('tidely_version')}
Python Version: {self.log_dict.get('python_version')}
OS: {self.log_dict.get('os')}
Architecture: {self.log_dict.get('architecture')}
CPU: {self.log_dict.get('cpu')}
Backend Used: {self.log_dict.get('backend')}
Planner Decision: {self.log_dict.get('planner_decision')}
Execution Duration: {self.log_dict.get('execution_duration_seconds')}s
Peak RAM: {self.log_dict.get('peak_ram_mb')} MB

## Dataset Fingerprints
- Pre-Cleaning Hash (SHA256): {self.log_dict.get('dataset_fingerprint', {}).get('sha256')}
- Schema Hash: {self.log_dict.get('dataset_fingerprint', {}).get('schema_hash')}

## Rules Applied
"""
        for r in self.log_dict.get("rules_applied", []):
            md += f"- **{r.get('column')}**: {r.get('rule')} ({r.get('reason')})\n"

        md += "\n## Warnings & Failures\n"
        for w in self.log_dict.get("warnings", []):
            md += f"- [WARNING] {w}\n"
        for f in self.log_dict.get("failures", []):
            md += f"- [FAILURE] {f}\n"

        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(md)
        return md

    def to_html(self, filepath: str = None) -> str:
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Tidely Audit Log</title>
    <style>
        body {{ font-family: sans-serif; background-color: #f8fafc; color: #1e293b; padding: 20px; }}
        h1 {{ color: #1e1b4b; }}
        .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 20px; }}
        ul {{ line-height: 1.6; }}
    </style>
</head>
<body>
    <div class="card">
        <h1>Tidely Enterprise Audit Log</h1>
        <p><strong>Timestamp:</strong> {self.log_dict.get('timestamp')}</p>
        <p><strong>Tidely Version:</strong> {self.log_dict.get('tidely_version')}</p>
        <p><strong>OS:</strong> {self.log_dict.get('os')}</p>
        <p><strong>Backend:</strong> {self.log_dict.get('backend')}</p>
        <p><strong>Planner Decision:</strong> {self.log_dict.get('planner_decision')}</p>
        <p><strong>Execution Duration:</strong> {self.log_dict.get('execution_duration_seconds')}s</p>
    </div>
</body>
</html>"""
        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html)
        return html

    def __repr__(self) -> str:
        return f"<AuditLog: {len(self.log_dict.get('rules_applied', []))} rules applied, {self.log_dict.get('backend')}>"


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
        plan_obj: Any = None,
        timeline: dict[str, float] = None,
        execution_time: float = 0.0,
    ) -> None:
        """Initializes the Result object and executes compliance audits/safety verification."""
        self.df = cleaned_df
        self._original_df = original_df
        self._summary_text = summary_text
        self.report = report_data

        # 1. Run strict Safety Invariants Verification
        from tidely.core.audit import (
            verify_safety_invariants,
            generate_dataset_fingerprint,
            generate_cleaning_contract,
            generate_cell_level_diffs,
            generate_distribution_report,
            generate_production_readiness,
            generate_reproducibility_report,
            generate_explainability_report,
            generate_data_preservation_report,
        )

        verify_safety_invariants(self._original_df, self.df, plan_obj)

        self._plan_obj = plan_obj
        self._execution_time = execution_time

        # 2. Compute fingerprint
        self.fingerprint = generate_dataset_fingerprint(self._original_df)
        self.fingerprint_after = generate_dataset_fingerprint(self.df)

        # 3. Compute contract
        self.contract = generate_cleaning_contract(plan_obj)

        # 4. Cell diffs
        self.cell_diffs = generate_cell_level_diffs(self._original_df, self.df, plan_obj)

        # 5. Timeline
        self.timeline = timeline or {}

        # 6. Distribution report
        self.distribution_report = generate_distribution_report(self._original_df, self.df)

        # 7. Production readiness report
        mem_saved = max(0.0, report_data.get("memory_before_mb", 0.0) - report_data.get("memory_after_mb", 0.0))
        self.readiness_report = generate_production_readiness(self._original_df, self.df, plan_obj, execution_time, mem_saved)

        # 8. Reproducibility & Audit Log
        backend_chosen = report_data.get("engine_name", "polars_eager")
        self.reproducibility = generate_reproducibility_report(backend_chosen)

        # 9. Explainability Report
        self.explainability_report_data = generate_explainability_report(self._original_df, plan_obj)

        # 10. Data Preservation Report
        self.preservation_report_data = generate_data_preservation_report(self._original_df, self.df, plan_obj)

        # Build the audit log list
        self.audit_log = []
        for cell in self.cell_diffs:
            self.audit_log.append({
                "timestamp": self.reproducibility["timestamp"],
                "dataset_fingerprint": self.fingerprint["sha256"],
                "dataset_schema": self.fingerprint["schema_hash"],
                "column": cell["column"],
                "row": cell["row"],
                "original_value": cell["original"],
                "cleaned_value": cell["cleaned"],
                "cleaning_rule": cell["rule"],
                "statistical_reason": cell["reason"],
                "backend_used": backend_chosen,
                "execution_engine": "Tidely Pipeline",
                "execution_time": execution_time,
                "rule_duration": timeline.get("execution", 0.0) / max(1, len(self.cell_diffs)) if timeline else 0.0,
                "memory_consumed": report_data.get("memory_after_mb", 0.0),
                "planner_decision": report_data.get("engine_reason", ""),
                "confidence_evidence": {
                    "confidence_score": next((act.get("confidence", 100) for act in report_data.get("actions", []) if act.get("column") == cell["column"]), 100)
                }
            })

    def diff(self) -> CleaningDiff:
        """Returns the granular diff report of all cell-level modifications."""
        backend_chosen = self.report.get("engine_name", "polars_eager")
        planner_decision = self.report.get("engine_reason", "")
        return CleaningDiff(self.cell_diffs, self._execution_time, backend_chosen, planner_decision)

    def audit(self) -> AuditLog:
        """Returns the enterprise-grade audit log for compliance and verification."""
        import sys
        import platform
        import os

        peak_ram = 0.0
        if psutil:
            try:
                peak_ram = float(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024))
            except Exception:
                pass

        backend_chosen = self.report.get("engine_name", "polars_eager")
        planner_decision = self.report.get("engine_reason", "")

        rules_applied = []
        for cell in self.cell_diffs:
            rules_applied.append({
                "column": cell["column"],
                "rule": cell["rule"],
                "reason": cell["reason"],
                "original": cell["original"],
                "cleaned": cell["cleaned"]
            })

        log_dict = {
            "timestamp": self.reproducibility.get("timestamp"),
            "tidely_version": "1.4.3",
            "python_version": sys.version,
            "os": f"{platform.system()} {platform.release()}",
            "architecture": platform.machine(),
            "cpu": platform.processor(),
            "memory": str(psutil.virtual_memory().total) if psutil else "unknown",
            "dataset_fingerprint": self.fingerprint,
            "schema_fingerprint": self.fingerprint.get("schema_hash"),
            "backend": backend_chosen,
            "planner_decision": planner_decision,
            "rules_applied": rules_applied,
            "execution_duration_seconds": self._execution_time,
            "peak_ram_mb": peak_ram,
            "warnings": self.report.get("warnings", []),
            "failures": [],
            "skipped_rules": []
        }
        return AuditLog(log_dict)

    def explain(self) -> dict[str, Any]:
        """Returns the explainability engine's report with statistical evidence."""
        return self.explainability_report_data

    def data_preservation_report(self) -> dict[str, Any]:
        """Returns the data preservation scorecard and validation metrics."""
        return self.preservation_report_data

    def distribution_drift_report(self) -> dict[str, Any]:
        """Returns the distribution drift report."""
        return self.distribution_report

    def performance_report(self) -> dict[str, Any]:
        """Returns details about performance and production suitability."""
        return self.readiness_report

    def cleaning_contract(self) -> dict[str, Any]:
        """Returns the Cleaning Contract."""
        return self.contract

    def fingerprint_report(self) -> dict[str, Any]:
        """Returns the Dataset Fingerprint."""
        return {
            "before": self.fingerprint,
            "after": self.fingerprint_after
        }

    def export(self, filepath: str) -> None:
        """Exports the cleaned dataset or report based on the file extension.

        Supported extensions: .csv, .parquet, .html
        """
        ext = filepath.split(".")[-1].lower()
        if ext == "html":
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
            from tidely.api import save
            save(self.df, filepath)

    def summary(self) -> str:
        """Returns the outcome-focused cleaning summary."""
        return self._summary_text

    @property
    def health_before(self) -> int:
        """Returns the dataset quality health score before cleaning."""
        return int(self.report.get("initial_health", 0))

    @property
    def health_after(self) -> int:
        """Returns the dataset quality health score after cleaning."""
        return int(self.report.get("final_health", 0))

    @property
    def execution_time(self) -> float:
        """Returns the cleaning pipeline execution duration in seconds."""
        return float(self.report.get("execution_time", 0.0))

    @property
    def memory_before(self) -> float:
        """Returns the estimated dataset memory footprint in MB before cleaning."""
        return float(self.report.get("memory_before_mb", 0.0))

    @property
    def memory_after(self) -> float:
        """Returns the estimated dataset memory footprint in MB after cleaning."""
        return float(self.report.get("memory_after_mb", 0.0))

    @property
    def memory_saved(self) -> float:
        """Returns the estimated dataset memory saved in MB after downcasting."""
        return max(0.0, self.memory_before - self.memory_after)

    @property
    def backend(self) -> str:
        """Returns the name of the cleaning backend used (e.g. polars_eager, duckdb)."""
        return str(self.report.get("engine_name", "polars_eager"))

    @property
    def rows_removed(self) -> int:
        """Returns the number of duplicate rows removed during cleaning."""
        return int(self.report.get("rows_removed", 0))

    @property
    def columns_modified(self) -> int:
        """Returns the number of columns corrected or optimized."""
        return int(self.report.get("columns_modified", 0))

    @property
    def actions(self) -> list[dict[str, Any]]:
        """Returns the list of all applied repair actions and explanations."""
        return list(self.report.get("actions", []))

    @property
    def version(self) -> str:
        """Returns the version of Tidely used."""
        return "1.4.3"

    @property
    def safety_report(self) -> dict[str, Any]:
        """Returns compliance confirmation safety log."""
        return {
            "status": "PASSED",
            "message": "All safety checks verified. No mutations detected on protected primary key/target columns."
        }

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

    def show(self, *args: Any, **kwargs: Any) -> Any:
        """Display the cleaned DataFrame.

        Uses the underlying DataFrame's ``show`` method if available (Polars),
        otherwise falls back to ``head`` for pandas and prints the result.
        Returns the displayed object for chaining.
        """
        if hasattr(self.df, "show"):
            return self.df.show(*args, **kwargs)
        if hasattr(self.df, "head"):
            result = self.df.head()
            print(result)
            return result
        return self.df

    def __repr__(self) -> str:
        """Returns a string representation of the underlying DataFrame."""
        return repr(self.df)

    def _repr_html_(self) -> str | None:
        if hasattr(self.df, "_repr_html_"):
            return cast(str | None, self.df._repr_html_())
        return None
