import inspect
from typing import Any

import tidely as td


def test_api_stability_clean():
    """Ensure td.clean signature never changes unexpectedly."""
    sig = inspect.signature(td.clean)
    assert "data" in sig.parameters
    assert len(sig.parameters) == 1, (
        "td.clean should only take 'data' to maintain API simplicity."
    )
    assert (
        sig.return_annotation.__name__ == "CleanResult"
        if hasattr(sig.return_annotation, "__name__")
        else str(sig.return_annotation).endswith("CleanResult")
    )


def test_api_stability_inspect():
    """Ensure td.inspect signature never changes unexpectedly."""
    sig = inspect.signature(td.inspect)
    assert "data" in sig.parameters
    assert len(sig.parameters) == 1, "td.inspect should only take 'data'."
    assert (
        sig.return_annotation is Any
        or sig.return_annotation == "Any"
        or sig.return_annotation == inspect.Parameter.empty
        or str(sig.return_annotation) == "typing.Any"
    )


def test_api_stability_validate():
    """Ensure td.validate signature never changes unexpectedly."""
    sig = inspect.signature(td.validate)
    assert "data" in sig.parameters
    assert "schema" in sig.parameters
    assert len(sig.parameters) == 2, "td.validate should take 'data' and 'schema'."
    assert sig.return_annotation is bool or str(sig.return_annotation) == "bool"


def test_result_structure():
    """Ensure CleanResult structure remains stable."""
    import pandas as pd

    from tidely.result import CleanResult

    # Mock result to check structure
    res = CleanResult(
        cleaned_df=pd.DataFrame(),
        original_df=pd.DataFrame(),
        summary_text="mock summary",
        report_data={"mock": "data"},
    )

    # These properties must always exist
    assert hasattr(res, "df")
    assert hasattr(res, "summary")
    assert callable(res.summary)
