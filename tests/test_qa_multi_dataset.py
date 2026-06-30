import pandas as pd

import tidely as td


def test_ds1_false_positive_duplicate_reporting():
    """Ensure that we do not report 'dedup_rows' if duplicates do not exist."""
    df = pd.DataFrame({"id": [1, 2, 3], "val": ["A", "B", "C"]})

    # 0 duplicates
    result = td.clean(df)

    assert "Duplicate Rows removed" not in result.summary()


def test_ds1_false_negative_id_detection():
    """Ensure that integer columns with 100% unique values are classified as ID/Key."""
    df = pd.DataFrame({"id": [101, 102, 103], "val": [1.5, 2.5, 3.5]})

    profile = td.inspect(df)

    assert profile.semantic_types["id"]["type"] == "ID/Key"
    assert profile.semantic_types["val"]["type"] == "Numeric"
