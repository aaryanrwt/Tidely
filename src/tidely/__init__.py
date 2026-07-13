"""Tidely: The Operating System for Data Quality.

This library provides production-grade data quality scoring, semantic type inference,
and automated data cleaning for Pandas and Polars.
"""

from typing import Any

_LAZY_EXPORTS = {
    "clean": "tidely.api",
    "inspect": "tidely.api",
    "load": "tidely.api",
    "save": "tidely.api",
    "validate": "tidely.api",
    "DatasetProfile": "tidely.core.profile",
    "RepairPlan": "tidely.core.clean_engine",
    "TidelyError": "tidely.core.errors",
}

__version__ = "1.4.3"

def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        import importlib
        module = importlib.import_module(_LAZY_EXPORTS[name])
        val = getattr(module, name)
        # Cache the imported attribute at module level to avoid future lookup overhead
        globals()[name] = val
        return val
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def __dir__() -> list[str]:
    return list(_LAZY_EXPORTS.keys()) + ["__version__"]

__all__ = list(_LAZY_EXPORTS.keys())
