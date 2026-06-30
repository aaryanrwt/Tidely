"""Tidely: The Operating System for Data Quality.

This library provides production-grade data quality scoring, semantic type inference,
and automated data cleaning for Pandas and Polars.
"""

from tidely.api import clean, inspect, load, save, validate
from tidely.core.clean_engine import RepairPlan
from tidely.core.errors import TidelyError
from tidely.core.profile import DatasetProfile

__version__ = "1.4.0"

__all__ = [
    "clean",
    "inspect",
    "load",
    "save",
    "validate",
    "DatasetProfile",
    "RepairPlan",
    "TidelyError",
]
