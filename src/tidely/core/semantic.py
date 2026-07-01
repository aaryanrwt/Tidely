"""Semantic Understanding Engine for inferring business meaning and recommendations."""

import re
from typing import Any


class SemanticEngine:
    """Infers business meaning (e.g. Emails, Dates, IDs, locations) from raw columns.

    Assigns confidence and risk scores to each column.
    """

    def __init__(self) -> None:
        """Initializes the SemanticEngine and compiles regex patterns."""
        # Compile regexes once to optimize startup and search speed
        self.patterns = {
            "Email": re.compile(r"^[\w\.-]+\s*@\s*[\w\.-]+\.\w+$"),
            "URL": re.compile(
                r"^(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?$"
            ),
            "Phone": re.compile(r"^\+?1?\s*\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}$"),
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
            "SSN": re.compile(r"^\d{3}-\d{2}-\d{4}$"),
            "Credit Card": re.compile(r"^(?:\d{4}[-\s]?){3}\d{4}$"),
            "IBAN": re.compile(r"^[A-Z]{2}\d{2}[A-Z0-9]{11,30}$"),
            "US State": re.compile(r"^(?:A[LKSZRAEP]|C[AOT]|D[EC]|F[LM]|G[AU]|HI|I[ADLN]|K[SY]|LA|M[ADEHINOPST]|N[CDEHJMVY]|O[HKR]|P[ARW]|RI|S[CD]|T[NX]|UT|V[AIT]|W[AIVY])$", re.IGNORECASE),
            "Gender": re.compile(r"^(male|female|m|f|other|unknown|non-binary)$", re.IGNORECASE),
            "Age": re.compile(r"^(?:100|[1-9]?\d)$"),
            "Salary": re.compile(r"^[\$\€\£\¥]?\s*\d{3,10}(?:[,\.]\d{2})?$"),
            "Hash": re.compile(r"^[a-fA-F0-9]{32}$|^[a-fA-F0-9]{40}$|^[a-fA-F0-9]{64}$"),
            "HTML": re.compile(r"<[a-z/][\s\S]*>", re.IGNORECASE),
            "Date": re.compile(r"^\d{4}[-\/]\d{2}[-\/]\d{2}(?:\s+\d{2}:\d{2}:\d{2})?$|^\d{2}[-\/]\d{2}[-\/]\d{4}$"),
            "Vehicle ID": re.compile(r"^[A-HJ-NPR-Z0-9]{17}$", re.IGNORECASE),
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
                    # If Date and Phone both match, prefer Date to avoid YYYY-MM-DD to Phone mapping
                    if pattern_rates.get("Date", 0.0) >= 0.5 and pattern_rates.get("Phone", 0.0) == pattern_rates.get("Date", 0.0):
                        best_match = "Date"
                    else:
                        best_match = max(pattern_rates, key=lambda k: pattern_rates[k])
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
                        elif best_match == "SSN":
                            recommended_cleaning = "Mask/Tokenize or validate SSN digits"
                            risk_score = 0.8
                        elif best_match == "Credit Card":
                            recommended_cleaning = "Mask/Tokenize PCI-DSS card digits"
                            risk_score = 0.9
                        elif best_match == "IBAN":
                            recommended_cleaning = "Validate bank account format"
                            risk_score = 0.7
                        elif best_match == "US State":
                            recommended_cleaning = "Convert to standard uppercase state code"
                            risk_score = 0.0
                        elif best_match == "Gender":
                            recommended_cleaning = "Standardize gender categories"
                            risk_score = 0.0
                        elif best_match == "Age":
                            recommended_cleaning = "Enforce integer range [0, 120]"
                            risk_score = 0.0
                        elif best_match == "Salary":
                            recommended_cleaning = "Extract numeric salary value"
                            risk_score = 0.1
                        elif best_match == "Hash":
                            recommended_cleaning = "Format hash standard lowercase representation"
                            risk_score = 0.0
                        elif best_match == "HTML":
                            recommended_cleaning = "Strip markup/HTML tags to plain text"
                            risk_score = 0.2
                        elif best_match == "Date":
                            recommended_cleaning = "Convert to standard UTC ISO8601 Datetime"
                            risk_score = 0.0
                        elif best_match == "Vehicle ID":
                            recommended_cleaning = "Standardize VIN characters"
                            risk_score = 0.0
                    else:
                        # Fallback heuristic rules
                        import re
                        col_words = set(re.split(r"[_ \-]", col_lower))

                        if "address" in col_lower or any(
                            (kw in col_words if len(kw) <= 3 else kw in col_lower)
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
                            or "owner" in col_lower
                            or "contact" in col_lower
                        ):
                            inferred_type = "Name"
                            confidence = 0.8
                            recommended_cleaning = "Titlecase names"
                        elif "country" in col_lower or "nation" in col_lower:
                            inferred_type = "Country"
                            confidence = 0.8
                            recommended_cleaning = "Convert to ISO country names/codes"
                        elif "city" in col_lower or "town" in col_lower or "municipality" in col_lower:
                            inferred_type = "City"
                            confidence = 0.8
                            recommended_cleaning = "Normalize city capitalization"
                        elif "customer" in col_lower and ("id" in col_words or "key" in col_words or "code" in col_words):
                            inferred_type = "CustomerID"
                            confidence = 0.9
                            recommended_cleaning = "Enforce uppercase ID formatting"
                        elif "invoice" in col_lower and ("id" in col_words or "key" in col_words or "code" in col_words):
                            inferred_type = "InvoiceID"
                            confidence = 0.9
                            recommended_cleaning = "Enforce uppercase ID formatting"
                        elif ("product" in col_lower or "item" in col_lower or "sku" in col_lower) and ("id" in col_words or "key" in col_words or "code" in col_words or "sku" in col_lower):
                            inferred_type = "ProductID"
                            confidence = 0.9
                            recommended_cleaning = "Enforce SKU alphanumeric pattern"
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
