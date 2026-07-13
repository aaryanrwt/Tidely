"""Enterprise Compliance and Auditing regression test suite."""

import pytest
import pandas as pd
import polars as pl
import tidely as td
from tidely.core.audit import SafetyValidationError


def test_enterprise_audit_log_and_fingerprints():
    # 1. Setup sample dirty dataset
    data = {
        "id": [1, 2, 3, 4, 5],
        "name": ["John", "Jane  ", "Bob", "Alice", "Charlie"],
        "age": [25.0, 30.0, None, 45.0, 22.0],
        "salary": ["$1,000", "$2,500", "invalid", "$4,000", "$1,500"],
        "is_active": ["true", "false", "yes", "no", None],
        "Survived": [1, 0, 1, 0, 1]  # Target column
    }
    df = pd.DataFrame(data)

    # 2. Run clean
    res = td.clean(df)

    # 3. Check fingerprint
    fp = res.fingerprint
    assert "sha256" in fp
    assert fp["row_count"] == 5
    assert fp["column_count"] == 6
    assert "schema_hash" in fp
    assert fp["null_profile"]["age"] == 1

    # 4. Check timeline
    tl = res.timeline
    assert "backend_selection" in tl
    assert "profiler" in tl
    assert "planner" in tl
    assert "rule_generation" in tl
    assert "execution" in tl
    assert "optimization" in tl
    assert "summary" in tl

    # 5. Check contract
    contract = res.contract
    assert "allowed_mutations" in contract
    assert "forbidden_mutations" in contract
    assert any(col["column"] == "id" for col in contract["protected_columns"])
    assert any(col["column"] == "Survived" for col in contract["protected_columns"])

    # 6. Check cell diffs
    diffs = res.cell_diffs
    assert len(diffs) > 0
    # age column was imputed, salary normalized
    assert any(d["column"] == "age" for d in diffs)
    assert any(d["column"] == "salary" for d in diffs)

    # 7. Check audit log
    log = res.audit_log
    assert len(log) == len(diffs)
    for entry in log:
        assert "timestamp" in entry
        assert "dataset_fingerprint" in entry
        assert "column" in entry
        assert "original_value" in entry
        assert "cleaned_value" in entry
        assert "cleaning_rule" in entry
        assert "statistical_reason" in entry

    # 8. Check distribution report
    dist = res.distribution_report
    # Age is continuous numeric
    assert "age" in dist
    assert dist["age"]["before"]["mean"] == 30.5
    assert dist["age"]["after"]["mean"] == 30.5  # Mean imputed

    # 9. Check safety report
    safety = res.safety_report
    assert safety["status"] == "PASSED"


def test_safety_invariants_fails_on_id_or_target_mutation():
    # Attempt to bypass protection or mock a pipeline run that modifies target/ID column
    # We will verify that verify_safety_invariants raises SafetyValidationError
    from tidely.core.audit import verify_safety_invariants
    
    class MockAction:
        def __init__(self, col, category):
            self.column = col
            self.category = category
            self.why_it_changed = "Testing"
            self.what_changed = "Testing"
            self.confidence = 1.0

    class MockPlan:
        def __init__(self):
            self.actions = []
            self.column_diagnostics = {
                "id": {"role": "Primary Key", "algorithm_chosen": "Keep Raw"},
                "Survived": {"role": "Target", "algorithm_chosen": "Keep Raw"}
            }

    p = MockPlan()

    df_orig = pd.DataFrame({"id": [1, 2, 3], "Survived": [0, 1, 0]})
    
    # Clean matches orig -> should pass
    df_clean_ok = pd.DataFrame({"id": [1, 2, 3], "Survived": [0, 1, 0]})
    verify_safety_invariants(df_orig, df_clean_ok, p)

    # Clean mutates ID -> should fail
    df_clean_bad_id = pd.DataFrame({"id": [1, 9, 3], "Survived": [0, 1, 0]})
    with pytest.raises(SafetyValidationError, match="ID Corruption"):
        verify_safety_invariants(df_orig, df_clean_bad_id, p)

    # Clean mutates Target -> should fail
    df_clean_bad_target = pd.DataFrame({"id": [1, 2, 3], "Survived": [0, 1, 1]})
    with pytest.raises(SafetyValidationError, match="Target Corruption"):
        verify_safety_invariants(df_orig, df_clean_bad_target, p)


def test_result_diff_and_audit_methods(tmp_path):
    data = {
        "id": [1, 2, 3, 4, 5],
        "name": ["John", "Jane  ", "Bob", "Alice", "Charlie"],
        "age": [25.0, 30.0, None, 45.0, 22.0],
        "salary": ["$1,000", "$2,500", "invalid", "$4,000", "$1,500"],
        "is_active": ["true", "false", "yes", "no", None],
        "Survived": [1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    res = td.clean(df)

    # 1. Test diff
    diff_report = res.diff()
    assert isinstance(diff_report.df, pd.DataFrame)
    # Check expected columns
    for col in [
        "Row", "Column", "Original Value", "Traditional Pipeline Value",
        "Tidely Value", "Rule Applied", "Statistical Reason",
        "Execution Time", "Backend Used", "Planner Decision"
    ]:
        assert col in diff_report.columns

    # Test diff exporting
    csv_file = tmp_path / "diff.csv"
    pq_file = tmp_path / "diff.parquet"
    json_file = tmp_path / "diff.json"
    md_file = tmp_path / "diff.md"

    diff_report.to_csv(str(csv_file))
    diff_report.to_parquet(str(pq_file))
    diff_report.to_json(str(json_file))
    diff_report.to_markdown(str(md_file))

    assert csv_file.exists()
    assert pq_file.exists()
    assert json_file.exists()
    assert md_file.exists()

    # 2. Test audit
    audit_log = res.audit()
    assert "timestamp" in audit_log.log_dict
    assert "tidely_version" in audit_log.log_dict
    assert "python_version" in audit_log.log_dict
    assert "os" in audit_log.log_dict
    assert "architecture" in audit_log.log_dict
    assert "cpu" in audit_log.log_dict
    assert "memory" in audit_log.log_dict
    assert "dataset_fingerprint" in audit_log.log_dict
    assert "schema_fingerprint" in audit_log.log_dict
    assert "backend" in audit_log.log_dict
    assert "planner_decision" in audit_log.log_dict
    assert "rules_applied" in audit_log.log_dict
    assert "execution_duration_seconds" in audit_log.log_dict
    assert "peak_ram_mb" in audit_log.log_dict
    assert "warnings" in audit_log.log_dict
    assert "failures" in audit_log.log_dict
    assert "skipped_rules" in audit_log.log_dict

    # Test audit exporting
    audit_json = tmp_path / "audit.json"
    audit_md = tmp_path / "audit.md"
    audit_html = tmp_path / "audit.html"

    audit_log.to_json(str(audit_json))
    audit_log.to_markdown(str(audit_md))
    audit_log.to_html(str(audit_html))

    assert audit_json.exists()
    assert audit_md.exists()
    assert audit_html.exists()

    # 3. Test other reports
    exp = res.explain()
    assert "age" in exp
    assert "role" in exp["age"]
    assert "evidence" in exp["age"]

    pres = res.data_preservation_report()
    assert pres["rows_preserved"] == 5
    assert pres["overall_preservation_score"] == 100.0

    drift = res.distribution_drift_report()
    assert "age" in drift
    assert "wasserstein_distance" in drift["age"]["metrics"]

    perf = res.performance_report()
    assert perf["dataset_size_rows"] == 5
    assert perf["dataset_size_cols"] == 6

    contract = res.cleaning_contract()
    assert "allowed_mutations" in contract

    fp_report = res.fingerprint_report()
    assert "before" in fp_report
    assert "after" in fp_report

