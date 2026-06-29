import pytest
import pandas as pd
import tidely as td

def test_ds3_false_positive_boolean_corruption():
    """Ensure that columns with mixed text and booleans (e.g. 0, 1, 2, 3+) are not falsely cast to strict Booleans."""
    df = pd.DataFrame({
        "Dependents": ["0", "1", "0", "0", "1", "2", "3+", "0", "1"]
    })
    
    result = td.clean(df)
    
    # It should NOT be a boolean dtype (e.g., should be category or object)
    assert result.df["Dependents"].dtype != "bool"
