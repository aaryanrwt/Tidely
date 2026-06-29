"""Schema Validation Engine."""

from typing import Any, Dict
from tidely.core.errors import TidelyError

def validate_schema(df: Any, schema: Dict[str, Any]) -> bool:
    """Validates the DataFrame against the expected schema.
    
    Args:
        df: The pandas/polars DataFrame.
        schema: A dictionary where keys are column names and values are 
                expected types (e.g. 'int', 'float', 'str', 'datetime', 'bool').
                
    Raises:
        TidelyError: If the schema validation fails.
    """
    import pandas as pd
    
    if not isinstance(df, pd.DataFrame):
        # Support for Polars or other structures can be added
        return True 

    errors = []
    
    for col, expected_type in schema.items():
        if col not in df.columns:
            errors.append(f"Missing required column: '{col}'")
            continue
            
        actual_dtype = str(df[col].dtype).lower()
        
        # Simple dtype mapping
        if expected_type in ("int", "integer") and "int" not in actual_dtype:
            errors.append(f"Column '{col}' expected integer, got {actual_dtype}")
        elif expected_type == "float" and "float" not in actual_dtype:
            errors.append(f"Column '{col}' expected float, got {actual_dtype}")
        elif expected_type in ("str", "string") and "object" not in actual_dtype and "string" not in actual_dtype and "str" not in actual_dtype:
            errors.append(f"Column '{col}' expected string, got {actual_dtype}")
        elif expected_type == "bool" and "bool" not in actual_dtype:
            errors.append(f"Column '{col}' expected boolean, got {actual_dtype}")
        elif expected_type in ("datetime", "date") and "datetime" not in actual_dtype:
            errors.append(f"Column '{col}' expected datetime, got {actual_dtype}")

    if errors:
        error_msg = "Schema validation failed:\n" + "\n".join(f"- {e}" for e in errors)
        raise TidelyError(error_msg)
        
    return True
