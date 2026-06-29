"""Tidely: The Operating System for Data Quality.

This library provides production-grade data quality scoring, semantic type inference,
and automated data cleaning for Pandas and Polars.
"""

from tidely.api import clean, inspect, load, save, validate
from tidely.core.errors import TidelyError
from tidely.core.profile import DatasetProfile
from tidely.core.clean_engine import RepairPlan

__version__ = "0.7.0-dev"

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
