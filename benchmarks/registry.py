"""Tidely v1.5.0 — Dataset Loader Registry.

All 12 benchmark datasets are loaded here sequentially using
streaming (200-row subsets) to avoid OOM on multi-TB sources.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger("tidely.benchmark.registry")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HF_API = "https://datasets-server.huggingface.co"
_NROWS = 200


def _hf_streaming(
    repo: str, config: str | None = None, split: str = "train", nrows: int = _NROWS
) -> pd.DataFrame | None:
    """Load a small streaming subset from HuggingFace datasets."""
    try:
        from datasets import load_dataset  # type: ignore[import]

        kwargs: dict[str, Any] = {"split": split, "streaming": True}
        if config:
            ds = load_dataset(repo, config, **kwargs)
        else:
            ds = load_dataset(repo, **kwargs)

        rows = []
        for i, row in enumerate(ds):
            if i >= nrows:
                break
            rows.append(row)

        if not rows:
            logger.warning("No rows returned for %s", repo)
            return None

        return pd.DataFrame(rows)
    except Exception as exc:
        logger.warning("Failed to load %s via load_dataset: %s", repo, exc)
        return None


def _hf_api_first_rows(
    dataset: str, config: str = "default", split: str = "train"
) -> pd.DataFrame | None:
    """Load rows using the HuggingFace datasets-server first-rows API."""
    url = f"{_HF_API}/first-rows"
    try:
        resp = requests.get(
            url,
            params={"dataset": dataset, "config": config, "split": split},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        rows = [row["row"] for row in data.get("rows", [])]
        if not rows:
            return None
        return pd.DataFrame(rows)
    except Exception as exc:
        logger.warning("API first-rows failed for %s: %s", dataset, exc)
        return None


def _hf_api_rows(
    dataset: str, config: str, split: str, offset: int = 0, length: int = 100
) -> pd.DataFrame | None:
    """Load rows using the HuggingFace datasets-server rows API."""
    url = f"{_HF_API}/rows"
    try:
        resp = requests.get(
            url,
            params={
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": length,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        rows = [row["row"] for row in data.get("rows", [])]
        if not rows:
            return None
        return pd.DataFrame(rows)
    except Exception as exc:
        logger.warning("API rows failed for %s: %s", dataset, exc)
        return None


def _hf_api_splits(dataset: str) -> pd.DataFrame | None:
    """Load split metadata using the HuggingFace datasets-server splits API."""
    url = f"{_HF_API}/splits"
    try:
        resp = requests.get(url, params={"dataset": dataset}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        splits = data.get("splits", [])
        if not splits:
            return None
        return pd.DataFrame(splits)
    except Exception as exc:
        logger.warning("API splits failed for %s: %s", dataset, exc)
        return None


def _hf_parquet_meta(dataset: str, config: str, split: str) -> pd.DataFrame | None:
    """Fetch parquet file metadata from HuggingFace API as a single-row DataFrame."""
    url = f"https://huggingface.co/api/datasets/{dataset}/parquet/{config}/{split}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return pd.DataFrame(data)
        return pd.DataFrame([data])
    except Exception as exc:
        logger.warning("Parquet meta failed for %s: %s", dataset, exc)
        return None


# ---------------------------------------------------------------------------
# Dataset Registry
# ---------------------------------------------------------------------------

DATASETS: list[dict[str, Any]] = [
    {
        "id": 1,
        "name": "anisoleai/fineweb-tokenized",
        "description": "FineWeb tokenized corpus",
        "loader": lambda: _hf_streaming("anisoleai/fineweb-tokenized"),
        "target": None,
        "keys": [],
    },
    {
        "id": 2,
        "name": "huggingface/documentation-images",
        "description": "HuggingFace documentation image metadata",
        "loader": lambda: _hf_streaming("huggingface/documentation-images"),
        "target": None,
        "keys": [],
    },
    {
        "id": 3,
        "name": "openai/gsm8k (main)",
        "description": "GSM8K math word problems – main split",
        "loader": lambda: _hf_streaming("openai/gsm8k", config="main"),
        "target": None,
        "keys": [],
    },
    {
        "id": 4,
        "name": "openai/gsm8k (socratic)",
        "description": "GSM8K math word problems – socratic split",
        "loader": lambda: _hf_streaming("openai/gsm8k", config="socratic"),
        "target": None,
        "keys": [],
    },
    {
        "id": 5,
        "name": "mteb/results",
        "description": "MTEB benchmark results",
        "loader": lambda: _hf_streaming("mteb/results"),
        "target": None,
        "keys": [],
    },
    {
        "id": 6,
        "name": "apple/DFNDR-2B",
        "description": "Apple DFNDR-2B dataset (API first-rows)",
        "loader": lambda: _hf_api_first_rows("apple/DFNDR-2B"),
        "target": None,
        "keys": [],
    },
    {
        "id": 7,
        "name": "mvp-lab/LLaVA-OneVision",
        "description": "LLaVA OneVision split metadata",
        "loader": lambda: _hf_api_splits(
            "mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M"
        ),
        "target": None,
        "keys": [],
    },
    {
        "id": 8,
        "name": "LiLabUNC/Variant-Foundation-Embeddings",
        "description": "Variant Foundation Embeddings (API first-rows)",
        "loader": lambda: _hf_api_first_rows("LiLabUNC/Variant-Foundation-Embeddings"),
        "target": None,
        "keys": [],
    },
    {
        "id": 9,
        "name": "Spawning/pd-extended",
        "description": "Spawning PD-extended dataset (API first-rows)",
        "loader": lambda: _hf_api_first_rows("Spawning/pd-extended"),
        "target": None,
        "keys": [],
    },
    {
        "id": 10,
        "name": "InternRobotics/OmniWorld",
        "description": "OmniWorld robotics dataset",
        "loader": lambda: _hf_streaming("InternRobotics/OmniWorld"),
        "target": None,
        "keys": [],
    },
    {
        "id": 11,
        "name": "HPLT/HPLT2.0_cleaned (ace_Arab rows)",
        "description": "HPLT 2.0 cleaned Arabic rows (API rows endpoint)",
        "loader": lambda: _hf_api_rows(
            "HPLT/HPLT2.0_cleaned", "ace_Arab", "train", offset=0, length=100
        ),
        "target": None,
        "keys": [],
    },
    {
        "id": 12,
        "name": "HPLT/HPLT2.0_cleaned (ace_Arab parquet meta)",
        "description": "HPLT 2.0 parquet file metadata (HuggingFace API)",
        "loader": lambda: _hf_parquet_meta("HPLT/HPLT2.0_cleaned", "ace_Arab", "train"),
        "target": None,
        "keys": [],
    },
]


def load_dataset_safe(ds_info: dict[str, Any]) -> tuple[pd.DataFrame | None, str]:
    """Load a dataset safely, returning (df, status_message)."""
    try:
        df = ds_info["loader"]()
        if df is None or df.empty:
            return None, "SKIPPED (no data returned)"
        return df, f"OK ({len(df)} rows x {len(df.columns)} cols)"
    except Exception as exc:
        return None, f"ERROR: {exc}"
