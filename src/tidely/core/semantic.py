"""Semantic Understanding Engine for inferring business meaning from data."""

from typing import Any, Dict


class SemanticEngine:
    """Infers business meaning (e.g. Emails, Dates, IDs) from raw data columns."""
    
    def __init__(self):
        pass
        
    def infer(self, df: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Infers the semantic meaning of columns based on random samples.
        
        Args:
            df: The DataFrame.
            metadata: Structural metadata from the DetectionEngine.
            
        Returns:
            Dictionary mapping column names to inferred Semantic Types.
        """
        import re
        
        semantics = {}
        samples = metadata.get("samples", {})
        
        patterns = {
            "Email": re.compile(r"^[\w\.-]+\s*@\s*[\w\.-]+\.\w+$"),
            "URL": re.compile(r"^(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?$"),
            "Phone": re.compile(r"^\+?[\d\s\-\(\)]{7,}$"),
            "Currency": re.compile(r"^[\$\€\£\¥]\s*\d+([,\.]\d+)?$"),
            "Date": re.compile(r"^(?:(?:19|20)\d\d[- /.](?:0?[1-9]|1[012])[- /.](?:0?[1-9]|[12][0-9]|3[01])|(?:0?[1-9]|1[012])[- /.](?:0?[1-9]|[12][0-9]|3[01])[- /.](?:19|20)\d\d|(?:0?[1-9]|[12][0-9]|3[01])[- /.](?:0?[1-9]|1[012])[- /.](?:19|20)\d\d)(?:[T\s]\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?$"),
            "Boolean": re.compile(r"^(yes|no|true|false|t|f|y|n|0|1)$", re.IGNORECASE),
            "SSN": re.compile(r"^\d{3}-\d{2}-\d{4}$"),
            "IPv4": re.compile(r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$"),
            "Coordinates": re.compile(r"^-?\d{1,3}\.\d+,\s*-?\d{1,3}\.\d+$"),
            "CreditCard": re.compile(r"^(?:\d{4}[-\s]?){3}\d{4}$"),
        }
        
        for col, col_meta in metadata.get("columns", {}).items():
            sample_list = samples.get(col, [])
            if not sample_list:
                semantics[col] = {"type": "Unknown", "match_rate": 0.0}
                continue
                
            # If it's heavily unique string, check if it's an ID
            dtype = col_meta["dtype"].lower()
            if "object" in dtype or "string" in dtype or "str" in dtype:
                # If almost 100% unique, it's an ID (skip regex)
                if col_meta.get("unique_count", 0) >= col_meta.get("total_count", 1) * 0.99 and col_meta.get("total_count", 0) > 0:
                    semantics[col] = {"type": "ID/Key", "match_rate": 1.0}
                    continue

                total_samples = len(sample_list)
                best_match = "String"
                highest_rate = 0.0
                
                # Check regex patterns
                for sem_type, pattern in patterns.items():
                    matches = sum(1 for val in sample_list if isinstance(val, str) and pattern.match(str(val).strip()))
                    rate = matches / total_samples
                    
                    # Boolean requires very high confidence (>= 0.95) to prevent corrupting categoricals like 0,1,2,3+
                    if sem_type == "Boolean" and rate < 0.95:
                        continue
                        
                    if rate > highest_rate:
                        highest_rate = rate
                        best_match = sem_type
                
                if highest_rate >= 0.5:
                    semantics[col] = {"type": best_match, "match_rate": highest_rate}
                else:
                    # Check for ID (high cardinality string)
                    col_lower = str(col).lower()
                    if col_meta["unique_count"] == col_meta["total_count"] and col_meta["total_count"] > 0:
                        if "customer" in col_lower:
                            semantics[col] = {"type": "CustomerID", "match_rate": 1.0}
                        elif "invoice" in col_lower:
                            semantics[col] = {"type": "InvoiceID", "match_rate": 1.0}
                        elif "product" in col_lower:
                            semantics[col] = {"type": "ProductID", "match_rate": 1.0}
                        else:
                            semantics[col] = {"type": "ID", "match_rate": 1.0}
                    # Check for categorical (low cardinality string)
                    elif col_meta["unique_count"] / max(col_meta["total_count"], 1) < 0.05:
                        semantics[col] = {"type": "Categorical", "match_rate": 1.0}
                    else:
                        semantics[col] = {"type": "Text", "match_rate": 1.0}
                        
            elif "int" in dtype or "float" in dtype:
                if "int" in dtype and col_meta["unique_count"] == col_meta["total_count"] and col_meta["total_count"] > 0:
                    semantics[col] = {"type": "ID/Key", "match_rate": 1.0}
                else:
                    semantics[col] = {"type": "Numeric", "match_rate": 1.0}
            elif "datetime" in dtype or "date" in dtype:
                semantics[col] = {"type": "Date", "match_rate": 1.0}
            elif "bool" in dtype:
                semantics[col] = {"type": "Boolean", "match_rate": 1.0}
            else:
                semantics[col] = {"type": "Unknown", "match_rate": 0.0}
                
        return semantics
