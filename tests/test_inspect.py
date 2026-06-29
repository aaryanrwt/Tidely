from typing import Any

import pandas as pd
import polars as pl
import pyarrow as pa
from hypothesis import given, settings
from hypothesis import strategies as st

import tidely as td
from tidely.core.profile import DatasetProfile


def test_inspect_basic_polars() -> None:
    """Test basic inspection on a Polars DataFrame."""
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 1],  # 1 duplicate row
            "email": [
                "user@example.com",
                "test@test.com",
                "invalid",
                "admin@domain.org",
                "user@example.com",
            ],
            "age": [25, 30, None, 45, 25],  # 1 null
        }
    )

    profile = td.inspect(df)

    assert isinstance(profile, DatasetProfile)
    assert profile.row_count == 5
    assert profile.col_count == 3
    assert profile.format_name == "polars_eager"
    assert "email" in profile.semantic_types

    # Test UI rendering doesn't crash
    profile.show()


def test_inspect_basic_pandas() -> None:
    """Test basic inspection on a Pandas DataFrame."""
    df = pd.DataFrame(
        {
            "uuid": ["123e4567-e89b-12d3-a456-426614174000", "invalid-uuid", None],
            "salary": ["$1,000.00", "500", "€100"],
        }
    )

    profile = td.inspect(df)

    assert isinstance(profile, DatasetProfile)
    assert profile.row_count == 3
    assert profile.format_name == "pandas"

    # Should identify partial semantics
    assert profile.semantic_types["uuid"]["type"] in ("ID/Key", "Unknown")


def test_inspect_basic_pyarrow() -> None:
    """Test basic inspection on a PyArrow Table."""
    table = pa.table(
        {"status": ["active", "inactive", "active"], "score": [1.2, 3.4, 5.6]}
    )

    profile = td.inspect(table)

    assert isinstance(profile, DatasetProfile)
    assert profile.row_count == 3
    assert profile.format_name == "arrow"


@settings(max_examples=50, deadline=None)
@given(
    st.lists(
        st.dictionaries(
            keys=st.sampled_from(["col_a", "col_b", "col_c"]),
            values=st.one_of(
                st.integers(),
                st.floats(allow_nan=True, allow_infinity=True),
                st.text(max_size=20),
                st.none(),
            ),
        ),
        min_size=1,
        max_size=100,
    )
)
def test_inspect_property_fuzzing(data: list[dict[str, Any]]) -> None:
    """Fuzz testing to ensure inspect() never crashes on diverse data shapes and types."""
    # Convert list of dicts to DataFrame. Pandas is used to handle mixed types cleanly before passing.
    df = pd.DataFrame(data)

    try:
        profile = td.inspect(df)
        assert isinstance(profile, DatasetProfile)
        assert profile.row_count == len(df)
        assert profile.col_count == len(df.columns)
    except td.TidelyError:
        pass  # Expected for un-convertible pandas data (e.g. huge ints)
