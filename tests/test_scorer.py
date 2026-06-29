"""Tests for the Lighthouse trust scoring engine."""

import polars as pl

from tidely.core.scorer import compute_trust_scores


def test_trust_scoring_perfect() -> None:
    """Perfect dataset should score 100/100."""
    df = pl.DataFrame(
        {
            "id": pl.Series([1, 2, 3], dtype=pl.Int8),
            "target": [10.0, 11.0, 12.0],
        }
    )
    semantic_types = {
        "id": {"type": "ID/Key", "confidence": 1.0},
        "target": {"type": "Unknown", "confidence": 0.0},
    }
    scores = compute_trust_scores(df, semantic_types, "Generic")
    assert scores.overall == 100
    assert scores.reliability == 100
    assert scores.ml_readiness == 100
    assert scores.memory_efficiency == 100


def test_trust_scoring_messy() -> None:
    """Messy dataset should drop scores across dimensions."""
    # Data has duplicate IDs, null values, highly skewed numbers, and unoptimized strings
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 9],
            "val": [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                None,
                1000.0,
            ],  # Highly skewed (> 1.5)
            "status": ["active"] * 10,  # Categorical candidate
        }
    )
    semantic_types = {
        "id": {"type": "ID/Key", "confidence": 0.90},
        "val": {"type": "Unknown", "confidence": 0.0},
        "status": {"type": "Unknown", "confidence": 0.0},
    }

    scores = compute_trust_scores(df, semantic_types, "Generic")
    # Duplicate IDs and nulls should drop Reliability
    assert scores.reliability < 100
    # Skewness should drop ML Readiness
    assert scores.ml_readiness < 100
    # Low-cardinality status string column should drop Memory Efficiency
    assert scores.memory_efficiency < 100
    # Overall is calculated as average of dimensions, so it must also be < 100
    assert scores.overall < 100
