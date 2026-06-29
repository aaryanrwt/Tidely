"""Deep semantic type classification and format validation engine."""

import re
from typing import Any

# Verhoeff algorithm multiplication table for Aadhaar validation
VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]

VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]

VERHOEFF_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def validate_verhoeff(num_str: str) -> bool:
    """Validate a number string using the Verhoeff checksum algorithm."""
    try:
        digits = [int(c) for c in num_str if c.isdigit()]
        if not digits:
            return False
        checksum = 0
        for i, digit in enumerate(reversed(digits)):
            checksum = VERHOEFF_D[checksum][VERHOEFF_P[i % 8][digit]]
        return checksum == 0
    except Exception:
        return False


def validate_luhn(card_str: str) -> bool:
    """Validate a credit card number using Luhn algorithm."""
    digits = [int(c) for c in card_str if c.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    odd_digits = digits[-1::-2]
    even_digits = digits[-2::-2]
    checksum = sum(odd_digits)
    for d in even_digits:
        checksum += sum(divmod(d * 2, 10))
    return checksum % 10 == 0


# Semantic Type Regexes
REGEX_PATTERNS = {
    "Email": re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"),
    "Phone": re.compile(r"^\+?[0-9\-\s\(\)\.]{7,22}$"),
    "PAN": re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$"),
    "GSTIN": re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]$"),
    "Aadhaar": re.compile(r"^[2-9][0-9]{3}\s?[0-9]{4}\s?[0-9]{4}$"),
    "UUID": re.compile(
        r"^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}$"
    ),
    "CreditCard": re.compile(
        r"^(?:4[0-9]{12}(?:[0-9]{3})?|[52][1-5][0-9]{14}|6(?:011|5[0-9][0-9])[0-9]{12}|3[47][0-9]{13})$"
    ),
    "IPv4": re.compile(r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$"),
    "IPv6": re.compile(r"^(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}$"),
    "URL": re.compile(
        r"^https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)$"
    ),
    "Currency": re.compile(
        r"^[\$€£¥₹\u20aa]\s?\-?\d+(?:[\.,]\d+)?$|^[A-Z]{3}\s?\-?\d+(?:[\.,]\d+)?$"
    ),
    "Date": re.compile(
        r"^\d{4}-\d{2}-\d{2}$|^\d{2}/\d{2}/\d{4}$|^\d{4}/\d{2}/\d{2}$|^\d{2}-\d{2}-\d{4}$"
    ),
}


def is_valid_ip(val: str) -> bool:
    """Validate IP octets (0-255) for IPv4."""
    if REGEX_PATTERNS["IPv4"].match(val):
        parts = val.split(".")
        return all(0 <= int(p) <= 255 for p in parts)
    return False


def classify_value(val: Any) -> str | None:
    """Identifies the semantic type of a single string value.

    Args:
        val: Any python value, usually string.

    Returns:
        Optional[str]: Semantic type identifier if matched, else None.
    """
    if not isinstance(val, str) or len(val) == 0:
        return None

    val_stripped = val.strip()

    # 1. Email check
    if REGEX_PATTERNS["Email"].match(val_stripped):
        return "Email"

    # 2. PAN Check
    if REGEX_PATTERNS["PAN"].match(val_stripped):
        return "PAN"

    # 3. GSTIN Check
    if REGEX_PATTERNS["GSTIN"].match(val_stripped):
        return "GSTIN"

    # 4. Aadhaar Check (with Verhoeff checksum validation)
    if REGEX_PATTERNS["Aadhaar"].match(val_stripped):
        digits_only = "".join(c for c in val_stripped if c.isdigit())
        if validate_verhoeff(digits_only):
            return "Aadhaar"

    # 5. UUID Check
    if REGEX_PATTERNS["UUID"].match(val_stripped):
        return "UUID"

    # 6. CreditCard Check (with Luhn validation)
    digits_only = "".join(c for c in val_stripped if c.isdigit())
    if len(digits_only) >= 13 and len(digits_only) <= 19:
        if (
            REGEX_PATTERNS["CreditCard"].match(digits_only)
            or digits_only.startswith("4")
            or digits_only.startswith("5")
        ):
            if validate_luhn(digits_only):
                return "CreditCard"

    # 7. IPv4 & IPv6 checks
    if is_valid_ip(val_stripped):
        return "IPv4"
    if REGEX_PATTERNS["IPv6"].match(val_stripped):
        return "IPv6"

    # 8. URL check
    if REGEX_PATTERNS["URL"].match(val_stripped):
        return "URL"

    # 9. Currency check
    if REGEX_PATTERNS["Currency"].match(val_stripped):
        return "Currency"

    # 10. Date check
    if REGEX_PATTERNS["Date"].match(val_stripped):
        return "Date"

    # 11. Phone check
    # Lower priority, matches many numbers, run last
    if REGEX_PATTERNS["Phone"].match(val_stripped):
        digits_only = "".join(c for c in val_stripped if c.isdigit())
        if len(digits_only) >= 7:
            return "Phone"

    return None


def classify_series(
    values: list[Any],
    col_name: str,
) -> dict[str, Any]:
    """Classifies a series of values, returning type and confidence.

    Args:
        values: Sample list of values.
        col_name: Name of the column.

    Returns:
        Dict: Classification details (type, confidence, evidence, reason).
    """
    non_nulls = [v for v in values if v is not None and v != ""]
    if not non_nulls:
        return {
            "type": "Unknown",
            "confidence": 0.0,
            "evidence": 0,
            "reason": "All sampled values are null or empty.",
        }

    # If it's a numeric sample, check if it fits ID/Key indicators
    # e.g., column named 'id', 'key', or '_id'
    col_lower = col_name.lower()
    if col_lower in ("id", "key", "pk", "uuid") or col_lower.endswith("_id"):
        # Check if values are mostly unique
        unique_cnt = len(set(non_nulls))
        ratio = unique_cnt / len(non_nulls)
        if ratio > 0.9:
            return {
                "type": "ID/Key",
                "confidence": ratio,
                "evidence": len(non_nulls),
                "reason": f"Column name '{col_name}' indicator and {ratio:.1%} uniqueness ratio.",
            }

    matches: dict[str, int] = {}
    for val in non_nulls:
        t = classify_value(val)
        if t:
            matches[t] = matches.get(t, 0) + 1

    if not matches:
        return {
            "type": "Unknown",
            "confidence": 0.0,
            "evidence": 0,
            "reason": "No matches against semantic regex templates.",
        }

    # Get the best match type
    best_type = max(matches, key=matches.get)  # type: ignore
    match_count = matches[best_type]
    confidence = match_count / len(non_nulls)

    # The confidence score returned represents the format match rate (cleanliness)
    # of the data itself, which scorers and diagnostics rely on to flag issues.
    return {
        "type": best_type,
        "confidence": confidence,
        "evidence": match_count,
        "reason": f"{match_count}/{len(non_nulls)} ({confidence:.1%}) sampled non-nulls matched {best_type} patterns.",
    }
