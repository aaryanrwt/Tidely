import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def empty_df():
    """Returns a completely empty DataFrame."""
    return pd.DataFrame()

@pytest.fixture
def single_row_df():
    """Returns a DataFrame with exactly one row."""
    return pd.DataFrame({
        "id": [1],
        "name": ["Alice"],
        "age": [25]
    })

@pytest.fixture
def dirty_df():
    """Returns a DataFrame with common dirty patterns."""
    return pd.DataFrame({
        "id": [1, 2, 2, 3],
        "name": ["Alice  ", "Bob", "Bob", "Charlie\u200b"],
        "email": ["ALICE@GMAIL.COM", "bob@yahoo.com", "bob@yahoo.com", "charlie_no_domain"],
        "is_active": ["Yes", "f", "f", "1"],
        "age": [25, np.nan, np.nan, 30]
    })
