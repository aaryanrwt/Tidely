"""Regression and feature tests for Tidely v1.4.0."""

import os
import tempfile

import polars as pl

from tidely.core.clean_engine import RepairAction, RepairPlan
from tidely.core.decision_engine import DecisionEngine
from tidely.core.detector import DetectionEngine
from tidely.core.rules import make_impute_median_rule
from tidely.core.streaming import StreamingEngine


def test_profiler_extended_metrics() -> None:
    """Verifies that the DetectionEngine calculates density, memory, and cost correctly."""
    df = pl.DataFrame({
        "a": [1, 2, None, 4],
        "b": ["x", "y", "z", "w"],
    })
    detector = DetectionEngine()
    meta = detector.analyze(df)

    assert "density" in meta["columns"]["a"]
    assert meta["columns"]["a"]["density"] == 0.75
    assert "memory_footprint_bytes" in meta["columns"]["a"]
    assert meta["columns"]["a"]["memory_footprint_bytes"] > 0
    assert "memory_footprint_bytes" in meta
    assert meta["memory_footprint_bytes"] > 0
    assert "estimated_execution_cost_ms" in meta
    assert meta["estimated_execution_cost_ms"] == 4 * 2 * 0.01


def test_decision_engine_routing() -> None:
    """Verifies backend routing based on dataset size and available memory."""
    engine = DecisionEngine()

    # Tiny dataset -> polars_eager
    route = engine.route_backend(100, "csv")
    assert route == "polars_eager"
    assert "comfort" in engine.selected_reason.lower()

    # Medium dataset -> polars_lazy
    route = engine.route_backend(20 * 1024 * 1024, "csv")
    assert route == "polars_lazy"
    assert "lazy" in engine.selected_reason.lower()

    # Large file -> duckdb
    route = engine.route_backend(100 * 1024 * 1024, "csv")
    assert route == "duckdb"
    assert "duckdb" in engine.selected_reason.lower()

    # Humongous dataset -> streaming
    route = engine.route_backend(100 * 1024 * 1024 * 1024, "csv")
    assert route == "streaming"
    assert "streaming" in engine.selected_reason.lower()


def test_sql_compilation() -> None:
    """Verifies translation of functional rules into SQL statements."""

    def dummy_rule(df: pl.DataFrame) -> pl.DataFrame:
        return df

    # Create plan with actions
    action1 = RepairAction(
        category="Missing Values",
        what_changed="Imputed col1 with 1.0",
        why_it_changed="Standard mean imputation",
        confidence=1.0,
        expected_score_bump=10,
        rule_fn=dummy_rule,
        column="col1",
        sql_expr='COALESCE("col1", 1.0)',
    )

    action2 = RepairAction(
        category="Outlier Handling",
        what_changed="Clipped col2 to [0, 100]",
        why_it_changed="Outlier threshold boundary clip",
        confidence=0.9,
        expected_score_bump=5,
        rule_fn=dummy_rule,
        column="col2",
        sql_expr='CASE WHEN "col2" < 0 THEN 0 WHEN "col2" > 100 THEN 100 ELSE "col2" END',
    )

    plan_obj = RepairPlan(
        original_data=None,
        actions=[action1, action2],
        initial_score=80,
        target_score=95,
    )

    sql = plan_obj.compile_to_sql("my_table", ["col1", "col2", "col3"])

    assert "WITH" in sql
    assert (
        'raw_source AS (SELECT "col1" AS "col1", "col2" AS "col2", "col3" AS "col3" FROM "my_table")'
        in sql
    )
    assert 'COALESCE("col1", 1.0) AS "col1"' in sql
    assert 'CASE WHEN "col2" < 0' in sql
    assert "FROM step_2" in sql


def test_duckdb_execution() -> None:
    """Verifies DuckDB execution of the compiled plan on in-memory data."""
    df = pl.DataFrame({
        "col1": [1.0, None, 3.0],
        "col2": [-10, 50, 150],
        "col3": ["a", "b", "c"],
    })

    def dummy_rule(df: pl.DataFrame) -> pl.DataFrame:
        return df

    action1 = RepairAction(
        category="Missing Values",
        what_changed="Imputed col1",
        why_it_changed="Mean imputation",
        confidence=1.0,
        expected_score_bump=10,
        rule_fn=dummy_rule,
        column="col1",
        sql_expr='COALESCE("col1", 2.0)',
    )

    action2 = RepairAction(
        category="Outlier Handling",
        what_changed="Clipped col2",
        why_it_changed="Clip outliers",
        confidence=1.0,
        expected_score_bump=5,
        rule_fn=dummy_rule,
        column="col2",
        sql_expr='CASE WHEN "col2" < 0 THEN 0 WHEN "col2" > 100 THEN 100 ELSE "col2" END',
    )

    plan_obj = RepairPlan(
        original_data=df,
        actions=[action1, action2],
        initial_score=80,
        target_score=95,
    )

    res = StreamingEngine.clean_with_duckdb(plan_obj, df, df.columns, "pandas")

    assert isinstance(res, pl.DataFrame)
    # col1 missing imputed to 2.0
    assert res["col1"].to_list() == [1.0, 2.0, 3.0]
    # col2 outliers clipped to [0, 100]
    assert res["col2"].to_list() == [0, 50, 100]


def test_chunked_streaming() -> None:
    """Verifies chunk-by-chunk streaming execution for CSV files."""
    # Write a dirty file
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test.csv")
        df = pl.DataFrame({
            "col1": [1.0, None, 3.0, None, 5.0],
            "col2": ["foo", "bar", "foo", "baz", "foo"],
        })
        df.write_csv(csv_path)

        # Impute missing values with median (3.0)
        action = RepairAction(
            category="Missing Values",
            what_changed="Impute median",
            why_it_changed="Consistent statistics",
            confidence=1.0,
            expected_score_bump=10,
            rule_fn=make_impute_median_rule("col1", value=3.0),
            column="col1",
            sql_expr='COALESCE("col1", 3.0)',
        )

        plan_obj = RepairPlan(
            original_data=None,
            actions=[action],
            initial_score=80,
            target_score=90,
        )

        # Run streaming chunked cleaning (using small chunk size 2)
        res_lazy = StreamingEngine.clean_chunked_streaming(
            plan_obj, csv_path, df.columns, "csv_lazy", chunk_size=2
        )

        assert isinstance(res_lazy, pl.LazyFrame)
        cleaned_df = res_lazy.collect()
        assert cleaned_df["col1"].to_list() == [1.0, 3.0, 3.0, 3.0, 5.0]
