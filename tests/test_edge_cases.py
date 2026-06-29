"""Edge cases and boundary value test suite for Tidely."""

import datetime

import polars as pl
import pytest

from tidely import inspect
from tidely.core.errors import DatasetError


def test_empty_dataset() -> None:
    """Verifies inspect handles 0-row datasets gracefully."""
    df = pl.DataFrame({"id": pl.Series([], dtype=pl.Int32)})
    profile = inspect(df)
    assert profile.row_count == 0
    assert profile.trust_score.overall == 0


def test_single_row_dataset() -> None:
    """Verifies inspect handles 1-row datasets."""
    df = pl.DataFrame({"id": [1], "email": ["test@domain.com"]})
    profile = inspect(df)
    assert profile.row_count == 1
    assert profile.trust_score.overall > 0


def test_single_column_dataset() -> None:
    """Verifies inspect handles 1-column datasets."""
    df = pl.DataFrame({"email": ["test@domain.com", "test2@domain.com"]})
    profile = inspect(df)
    assert profile.col_count == 1
    assert profile.row_count == 2
    assert profile.semantic_types["email"]["type"] == "Email"


def test_wide_dataset() -> None:
    """Verifies inspect scales to 1000 columns."""
    data = {f"col_{i}": [1, 2, 3] for i in range(1000)}
    df = pl.DataFrame(data)
    profile = inspect(df)
    assert profile.col_count == 1000
    assert profile.row_count == 3


def test_tall_dataset() -> None:
    """Verifies inspect scales to tall datasets."""
    df = pl.DataFrame(
        {
            "id": pl.Series(range(10000), dtype=pl.Int32),
            "val": [10.0] * 10000,
        }
    )
    profile = inspect(df)
    assert profile.row_count == 10000


def test_unicode_and_emojis() -> None:
    """Verifies inspect handles unicode text and emojis cleanly."""
    df = pl.DataFrame(
        {
            "id": [1, 2],
            "text": ["こんにちは", "Hello 😊 Spark"],
        }
    )
    profile = inspect(df)
    assert profile.row_count == 2
    # Ensure TUI prints without throwing encoding errors
    profile.show()


def test_null_heavy_dataset() -> None:
    """Verifies trust score deductions on heavily null-sparse datasets."""
    # 90% nulls
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "val": [1.0, None, None, None, None, None, None, None, None, None],
        }
    )
    profile = inspect(df)
    assert profile.trust_score.reliability < 90


def test_duplicate_heavy_dataset() -> None:
    """Verifies trust score deductions on heavily duplicated ID fields."""
    from tidely.core.scorer import compute_trust_scores

    df = pl.DataFrame(
        {
            "id": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "val": [10.0] * 10,
        }
    )
    semantic_types = {
        "id": {"type": "ID/Key", "confidence": 1.0},
        "val": {"type": "Unknown", "confidence": 0.0},
    }
    scores = compute_trust_scores(df, semantic_types, "Generic")
    assert scores.reliability == 60


def test_invalid_semantic_values() -> None:
    """Verifies detection profiles containing invalid semantic formats."""
    df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "email": ["valid@domain.com", "not_an_email", "another_invalid"],
        }
    )
    profile = inspect(df)
    # Email match rate is 33.3%, which is low, meaning semantic quality should drop
    assert profile.semantic_types["email"]["type"] == "Email"
    assert profile.trust_score.semantic_quality < 100


def test_floating_point_edges() -> None:
    """Verifies floating point edge cases (inf, -inf, nan) are handled gracefully."""
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "val": [1.0, float("nan"), float("inf"), float("-inf")],
        }
    )
    profile = inspect(df)
    assert profile.row_count == 4
    # Ensure it doesn't crash during skew calculation or scoring
    assert profile.trust_score.overall >= 0


def test_integer_overflow_boundaries() -> None:
    """Verifies integer overflow boundaries (64-bit boundaries) do not cause crashes."""
    df = pl.DataFrame(
        {
            "id": [1, 2],
            "val": [9223372036854775807, -9223372036854775808],  # Int64 max and min
        }
    )
    profile = inspect(df)
    assert profile.row_count == 2
    # Ensure it doesn't crash during downcasting check
    assert profile.trust_score.overall >= 0


def test_timezone_aware_datetimes() -> None:
    """Verifies timezone-aware datetimes are processed cleanly."""
    tz_aware_data = [
        datetime.datetime(2026, 1, 1, 12, 0, tzinfo=datetime.UTC),
        datetime.datetime(2026, 1, 2, 12, 0, tzinfo=datetime.UTC),
    ]
    df = pl.DataFrame(
        {
            "id": [1, 2],
            "timestamp": tz_aware_data,
        }
    )
    profile = inspect(df)
    assert profile.row_count == 2


def test_leap_years_and_dst() -> None:
    """Verifies leap year dates and DST transition dates are processed cleanly."""
    dates = [
        datetime.date(2024, 2, 29),  # Leap year day
        datetime.date(2026, 3, 29),  # Potential DST transition day
    ]
    df = pl.DataFrame(
        {
            "id": [1, 2],
            "date": dates,
        }
    )
    profile = inspect(df)
    assert profile.row_count == 2


def test_unsupported_type_raises_error() -> None:
    """Verifies passing unsupported type raises DatasetError."""
    with pytest.raises(DatasetError):
        inspect("not_a_dataframe")
