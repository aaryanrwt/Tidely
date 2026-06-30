"""Semantic Understanding Engine for inferring business meaning and recommendations."""

import re
from typing import Any


class SemanticEngine:
    """Infers business meaning (e.g. Emails, Dates, IDs, locations) from raw columns.

    Assigns confidence and risk scores to each column.
    """

    def __init__(self):
        # Compile regexes once to optimize startup and search speed
        self.patterns = {
            "Email": re.compile(r"^[\w\.-]+\s*@\s*[\w\.-]+\.\w+$"),
            "URL": re.compile(
                r"^(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?$"
            ),
            "Phone": re.compile(r"^\+?[\d\s\-\(\)]{7,20}$"),
            "Currency": re.compile(r"^[\$\€\£\¥]\s*\d+([,\.]\d+)?$"),
            "Boolean": re.compile(r"^(yes|no|true|false|t|f|y|n|0|1)$", re.IGNORECASE),
            "UUID": re.compile(
                r"^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[1-5][a-fA-F0-9]{3}-[89abAB][a-fA-F0-9]{3}-[a-fA-F0-9]{12}$"
            ),
            "IP Address": re.compile(
                r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
            ),
            "ZIP Code": re.compile(r"^\d{5}(-\d{4})?$|^[A-Z]\d[A-Z]\s*\d[A-Z]\d$"),
            "Latitude": re.compile(r"^[-+]?([1-8]?\d(\.\d+)?|90(\.0+)?)$"),
            "Longitude": re.compile(
                r"^[-+]?(180(\.0+)?|((1[0-7]\d|[1-9]?\d)(\.\d+)?))$"
            ),
            "Percentage": re.compile(r"^\d+(\.\d+)?\s*\%$"),
            "JSON": re.compile(r"^\s*\{.*\}\s*$|^\s*\[.*\]\s*$"),
            "DNA Sequence": re.compile(r"^[ACGTNacgtn\s]{10,}$"),
        }

    def infer(self, df: Any, metadata: dict[str, Any]) -> dict[str, Any]:
        """Infers semantic meaning of columns based on random samples.

        Args:
            df: The DataFrame.
            metadata: Structural metadata from DetectionEngine.

        Returns:
            Dict mapping column names to:
            {
                "type": inferred_type,
                "confidence": float (0.0 to 1.0),
                "match_rate": float (0.0 to 1.0),
                "risk_score": float (0.0 to 1.0),
                "recommended_cleaning": str
            }
        """
        semantics = {}
        samples = metadata.get("samples", {})

        for col, col_meta in metadata.get("columns", {}).items():
            sample_list = samples.get(col, [])
            dtype = col_meta["dtype"].lower()

            # Default fallback
            inferred_type = "Unknown"
            confidence = 0.0
            match_rate = 0.0
            risk_score = 0.0
            recommended_cleaning = "None"

            if not sample_list:
                semantics[col] = {
                    "type": inferred_type,
                    "confidence": confidence,
                    "match_rate": match_rate,
                    "risk_score": risk_score,
                    "recommended_cleaning": recommended_cleaning,
                }
                continue

            col_lower = str(col).lower()

            # String parsing rules
            if any(t in dtype for t in ("object", "string", "str")):
                # Check for high-cardinality ID keys
                total = col_meta.get("total_count", 0)
                unique = col_meta.get("unique_count", 0)
                unique_ratio = unique / max(total, 1)

                # Check UUID pattern match rate
                uuid_matches = sum(
                    1
                    for val in sample_list
                    if isinstance(val, str) and self.patterns["UUID"].match(val.strip())
                )
                uuid_rate = uuid_matches / len(sample_list)

                if uuid_rate > 0.8:
                    inferred_type = "UUID"
                    match_rate = uuid_rate
                    confidence = uuid_rate
                    recommended_cleaning = "Enforce lowercase standard UUID format"
                elif unique_ratio >= 0.99 and total > 10:
                    if "customer" in col_lower:
                        inferred_type = "CustomerID"
                        recommended_cleaning = "Enforce uppercase ID formatting"
                    elif "invoice" in col_lower:
                        inferred_type = "InvoiceID"
                        recommended_cleaning = "Enforce uppercase ID formatting"
                    elif "sku" in col_lower:
                        inferred_type = "SKU"
                        recommended_cleaning = "Enforce SKU alphanumeric pattern"
                    else:
                        inferred_type = "ID/Key"
                        recommended_cleaning = "Strip whitespace"
                    confidence = 1.0
                    match_rate = 1.0
                else:
                    # General Regex patterns matching
                    pattern_rates = {}
                    for p_name, pattern in self.patterns.items():
                        matches = sum(
                            1
                            for val in sample_list
                            if isinstance(val, str) and pattern.match(str(val).strip())
                        )
                        pattern_rates[p_name] = matches / len(sample_list)

                    best_match = max(pattern_rates, key=pattern_rates.get)
                    best_rate = pattern_rates[best_match]

                    if best_rate >= 0.5:
                        inferred_type = best_match
                        match_rate = best_rate
                        confidence = best_rate

                        if best_match == "Email":
                            recommended_cleaning = (
                                "Normalize to lowercase, strip whitespaces"
                            )
                            risk_score = 0.1
                        elif best_match == "Phone":
                            recommended_cleaning = (
                                "Strip non-digits, format international"
                            )
                            risk_score = 0.2
                        elif best_match == "ZIP Code":
                            recommended_cleaning = "Format to 5-digit ZIP"
                            risk_score = 0.1
                        elif best_match == "Boolean":
                            recommended_cleaning = (
                                "Cast truthy/falsy to bool primitives"
                            )
                            risk_score = 0.0
                        elif best_match == "Currency":
                            recommended_cleaning = (
                                "Extract numeric amount, normalize symbol"
                            )
                            risk_score = 0.1
                        elif best_match == "JSON":
                            recommended_cleaning = (
                                "Standardize JSON quotes and indentations"
                            )
                            risk_score = 0.3
                        elif best_match == "DNA Sequence":
                            recommended_cleaning = (
                                "None (DNA sequence preserved exactly)"
                            )
                            risk_score = 0.0
                    else:
                        # Fallback heuristic rules
                        if "address" in col_lower or any(
                            kw in col_lower
                            for kw in (
                                "street",
                                "road",
                                "ave",
                                "drive",
                                "st",
                                "lane",
                                "ln",
                            )
                        ):
                            inferred_type = "Address"
                            confidence = 0.7
                            recommended_cleaning = "Standardize street abbreviations"
                        elif (
                            "name" in col_lower
                            or "first" in col_lower
                            or "last" in col_lower
                        ):
                            inferred_type = "Name"
                            confidence = 0.7
                            recommended_cleaning = "Titlecase names"
                        elif "country" in col_lower:
                            inferred_type = "Country"
                            confidence = 0.8
                            recommended_cleaning = "Convert to ISO country names/codes"
                        elif "city" in col_lower:
                            inferred_type = "City"
                            confidence = 0.8
                            recommended_cleaning = "Normalize city capitalization"
                        elif unique_ratio < 0.05:
                            inferred_type = "Categorical"
                            confidence = 0.9
                            recommended_cleaning = "Convert to Categorical dtype"
                        else:
                            inferred_type = "Text"
                            confidence = 1.0
                            recommended_cleaning = "Unicode normalization (NFKC)"

                        match_rate = confidence

            # Numeric or Date parsing rules
            elif any(t in dtype for t in ("int", "float", "double")):
                # Check if Latitude/Longitude bounds match
                vals = [float(v) for v in sample_list if isinstance(v, (int, float))]
                if vals:
                    min_val = min(vals)
                    max_val = max(vals)
                    if (
                        "lat" in col_lower
                        and -90 <= min_val <= 90
                        and -90 <= max_val <= 90
                    ):
                        inferred_type = "Latitude"
                        confidence = 0.95
                        recommended_cleaning = "Clip to range [-90, 90]"
                    elif (
                        "lon" in col_lower
                        and -180 <= min_val <= 180
                        and -180 <= max_val <= 180
                    ):
                        inferred_type = "Longitude"
                        confidence = 0.95
                        recommended_cleaning = "Clip to range [-180, 180]"
                    elif "int" in dtype and col_meta.get(
                        "unique_count", 0
                    ) == col_meta.get("total_count", 0):
                        inferred_type = "ID/Key"
                        confidence = 1.0
                        recommended_cleaning = "None"
                    elif (
                        "price" in col_lower
                        or "amount" in col_lower
                        or "cost" in col_lower
                    ):
                        inferred_type = "Price"
                        confidence = 0.9
                        recommended_cleaning = (
                            "Standardize decimals, treat negative values"
                        )
                    else:
                        inferred_type = "Numeric"
                        confidence = 1.0
                        recommended_cleaning = "Automatic numeric scaling/downcasting"
                else:
                    inferred_type = "Numeric"
                    confidence = 1.0
                match_rate = confidence

            elif any(t in dtype for t in ("date", "time")):
                inferred_type = "Date"
                confidence = 1.0
                match_rate = 1.0
                recommended_cleaning = "Convert to standard UTC ISO8601 Datetime"

            elif "bool" in dtype:
                inferred_type = "Boolean"
                confidence = 1.0
                match_rate = 1.0
                recommended_cleaning = "Cast to true boolean"

            semantics[col] = {
                "type": inferred_type,
                "confidence": confidence,
                "match_rate": match_rate,
                "risk_score": risk_score,
                "recommended_cleaning": recommended_cleaning,
            }

        return semantics
