"""Tests for the Deep Semantic Engine and validators."""

from tidely.core.semantic import (
    classify_series,
    classify_value,
    validate_luhn,
    validate_verhoeff,
)


def test_verhoeff_validation() -> None:
    """Verhoeff check digits validation."""
    # Valid Aadhaar numbers
    assert validate_verhoeff("361234567890")
    # Invalid Aadhaar numbers
    assert not validate_verhoeff("361234567891")
    assert not validate_verhoeff("123")


def test_luhn_validation() -> None:
    """Luhn validation for credit cards."""
    # Valid card numbers (standard 16-digit test cards)
    assert validate_luhn("4012888888881881")
    # Invalid
    assert not validate_luhn("4012888888881882")


def test_classify_individual_values() -> None:
    """Check individual value class detection."""
    assert classify_value("test@domain.com") == "Email"
    assert classify_value("+1-555-019-2834") == "Phone"
    assert classify_value("ABCDE1234F") == "PAN"
    assert classify_value("361234567890") == "Aadhaar"
    assert classify_value("18AABCU9603R1ZM") == "GSTIN"
    assert classify_value("4012888888881881") == "CreditCard"
    assert classify_value("192.168.1.1") == "IPv4"
    assert classify_value("https://google.com/search?q=test") == "URL"
    assert classify_value("$10500.50") == "Currency"
    assert classify_value("₹500") == "Currency"


def test_classify_series() -> None:
    """Check series-level type and confidence profiling."""
    emails = [
        "test1@domain.com",
        "test2@domain.com",
        None,
        "invalid_email",
        "test3@domain.com",
    ]
    result = classify_series(emails, "user_email")
    assert result["type"] == "Email"
    # 3 emails out of 4 non-nulls match the pattern (75% match rate)
    assert result["confidence"] == 0.75
    assert result["evidence"] == 3

    # Primary key ID column detection
    ids = ["123", "456", "789", "1011"]
    id_result = classify_series(ids, "user_id")
    assert id_result["type"] == "ID/Key"
    assert id_result["confidence"] == 1.0
