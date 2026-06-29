import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.extra.pandas import data_frames, column
import pandas as pd
import numpy as np
import tidely as td

# Generate valid but weird column names
col_names = st.text(min_size=1, max_size=10).filter(lambda x: "\x00" not in x)

# Strategy for generic random columns (numeric, text, bool)
generic_columns = [
    column(name="col_int", dtype=int, elements=st.integers(min_value=-9223372036854775808, max_value=9223372036854775807)),
    column(name="col_float", dtype=float, elements=st.floats(allow_nan=True, allow_infinity=True)),
    column(name="col_bool", dtype=bool, elements=st.booleans()),
    column(name="col_str", dtype=object, elements=st.text())
]

@settings(max_examples=100, deadline=None)
@given(df=data_frames(columns=generic_columns))
def test_hypothesis_generic_dataframe(df):
    """
    Test that Tidely can process a completely randomized DataFrame.
    Invariants:
    1. Should never raise an exception (unless it's an expected library validation error).
    2. Column count should remain exactly the same.
    3. Row count should be less than or equal to original (due to dedup).
    """
    try:
        result = td.clean(df)
        assert len(result.df.columns) == len(df.columns)
        assert len(result.df) <= len(df)
    except Exception as e:
        pytest.fail(f"Hypothesis found a crashing DataFrame layout: {e}")
