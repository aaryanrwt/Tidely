import pandas as pd

import tidely as td


def test_ds2_memory_optimization_categoricals():
    """Ensure that Tidely automatically downcasts low-cardinality string columns to category."""
    df = pd.DataFrame(
        {"model": ["Fiesta", "Focus", "Fiesta", "Mustang", "Focus"] * 1000}
    )

    result = td.clean(df)

    assert result.df["model"].dtype == "category"
    assert "Memory Optimization applied" in result.summary()
