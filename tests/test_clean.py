"""Tests for the Tidely V0.2 Explainable Cleaning Engine."""

from typing import Any

import pandas as pd
import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

import tidely as td
from tidely.core.clean_engine import RepairPlan


def test_clean_basic_deduplication() -> None:
    """Verifies that exact duplicate rows are dropped."""
    df = pl.DataFrame({"id": [1, 2, 2, 3], "val": ["A", "B", "B", "C"]})

    plan = td.plan(df)
    assert isinstance(plan, RepairPlan)

    # Execute dry run first
    dry_df = plan.execute(dry_run=True)
    assert dry_df.height == 4  # Unmodified

    # Execute actual clean
    clean_df = plan.execute(dry_run=False)
    assert clean_df.height == 3

    # Check audit log
    assert len(plan.audit_log) > 0
    assert any(log["category"] == "Duplicate Rows" for log in plan.audit_log)


def test_clean_semantic_normalization() -> None:
    """Verifies that semantic types like Email and Phone are normalized."""
    df = pl.DataFrame(
        {
            "email": [
                " TEST@example.com ",
                "valid@test.org",
                "another@domain.com",
                "INVALID",
            ],
            "phone": ["(555) 123-4567", "555-987-6543", "+1234567890", "12345"],
        }
    )

    clean_df = td.clean(df)

    # Check email normalization
    assert clean_df["email"][0] == "test@example.com"

    # Check phone normalization
    assert clean_df["phone"][0] == "5551234567"
    assert clean_df["phone"][1] == "5559876543"


def test_clean_missing_imputation() -> None:
    """Verifies that missing values are imputed correctly."""
    df = pl.DataFrame(
        {"numeric_col": [1.0, 10.0, None, 10.0], "string_col": ["A", None, "C", "D"]}
    )

    plan = td.plan(df)
    clean_df = plan.execute()

    # Numeric -> Median (10.0)
    assert clean_df["numeric_col"][2] == 10.0
    # String -> Unknown
    assert clean_df["string_col"][1] == "Unknown"

    assert any(log["category"] == "Missing Values" for log in plan.audit_log)


def test_clean_memory_optimization() -> None:
    """Verifies that columns are downcasted and cast to categorical."""
    # Add a low cardinality string column manually to guarantee length > 1000
    long_df = pl.DataFrame({"small_ints": [1] * 2000, "cat_str": ["A", "B"] * 1000})

    clean_df = td.clean(long_df)

    assert clean_df["small_ints"].dtype == pl.Int8
    assert clean_df["cat_str"].dtype == pl.Categorical


@settings(max_examples=20, deadline=None)
@given(
    st.lists(
        st.dictionaries(
            keys=st.sampled_from(["id", "email", "age", "score", "date"]),
            values=st.one_of(
                st.integers(min_value=-1000, max_value=1000),
                st.floats(allow_nan=True, allow_infinity=True),
                st.text(max_size=10),
                st.none(),
            ),
        ),
        min_size=1,
        max_size=50,
    )
)
def test_clean_property_fuzzing(data: list[dict[str, Any]]) -> None:
    """Fuzz testing to ensure clean() never crashes."""
    df = pd.DataFrame(data)
    try:
        td.clean(df)
    except td.TidelyError:
        pass  # Expected for invalid pandas data types


def test_empty_dataframe_trust_score_division_by_zero() -> None:
    """Verifies that an empty DataFrame does not crash the engine."""
    df = pl.DataFrame({"a": [], "b": []})
    plan = td.plan(df)
    assert plan.initial_score == 0
    assert plan.target_score == 0
    clean_df = plan.execute()
    assert clean_df.height == 0


def test_one_row_dataframe_variance_checks() -> None:
    """Verifies that a single-row dataset processes successfully."""
    df = pl.DataFrame({"id": [1], "name": ["Aaryan"], "val": [None]})
    clean_df = td.clean(df)
    assert clean_df.height == 1


def test_categorical_imputation_default_is_constant() -> None:
    """Verifies that missing strings/categoricals use 'Unknown' constant."""
    df = pl.DataFrame({"cat": ["A", None, "B"]})
    clean_df = td.clean(df)
    assert clean_df["cat"].to_list() == ["A", "Unknown", "B"]


def test_edge_case_emojis_and_utf8() -> None:
    """Verifies that Emojis and Massive strings don't crash the engine."""
    df = pl.DataFrame({
        "emoji": ["🚀", "🔥", "A" * 5000],
        "utf8": [b"\\xff".decode("utf-16", errors="ignore"), "normal", None]
    })
    clean_df = td.clean(df)
    assert clean_df.height == 3
