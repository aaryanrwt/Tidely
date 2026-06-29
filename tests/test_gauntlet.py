"""Gauntlet tests to run Tidely across multiple real-world domains."""

import os
import urllib.request

import polars as pl
import pytest

import tidely as td

# Collection of diverse datasets spanning different domains
DATASETS = {
    "Titanic_Classification": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
    "California_Housing_Regression": "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv",
    "Heart_Disease_Healthcare": "https://raw.githubusercontent.com/plotly/datasets/master/heart.csv",
    "Iris_Biology": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
    "Tips_Retail": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv",
    "Flights_Time_Series": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/flights.csv",
    "Diamonds_Retail": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv",
    "Penguins_Biology": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv",
    "Car_Crashes_Gov": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/car_crashes.csv",
    "Planets_Astronomy": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/planets.csv"
}

def download_dataset(url: str, name: str) -> str:
    """Helper to download dataset."""
    cache_dir = ".pytest_cache/datasets"
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{name}.csv")
    if not os.path.exists(path):
        try:
            urllib.request.urlretrieve(url, path)
        except Exception:
            pytest.skip(f"Could not download {url}")
    return path

@pytest.mark.parametrize("name,url", DATASETS.items())
def test_gauntlet_dataset(name: str, url: str) -> None:
    """Runs the full Tidely gauntlet on diverse datasets."""
    path = download_dataset(url, name)

    try:
        df = pl.read_csv(path, ignore_errors=True, infer_schema_length=1000)
    except Exception:
        pytest.skip("Failed to parse CSV")

    # 1. Inspect
    try:
        profile = td.inspect(df)
        assert profile is not None
    except Exception as e:
        pytest.fail(f"[{name}] td.inspect() failed: {e}")

    # 2. Plan
    try:
        plan = td.plan(df)
        assert plan is not None
        assert hasattr(plan, "initial_score")
        assert hasattr(plan, "target_score")
    except Exception as e:
        pytest.fail(f"[{name}] td.plan() failed: {e}")

    # 3. Clean
    try:
        clean_df = plan.execute()
        assert clean_df is not None
        assert clean_df.height <= df.height  # Row count might drop if dedup happens
    except Exception as e:
        pytest.fail(f"[{name}] plan.execute() failed: {e}")
