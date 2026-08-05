"""Shared helpers for dataset/build_dataset.py and
dataset/build_instruction_dataset.py -- both build a train/validation/test
DatasetDict from a flat list of Pydantic records the same way; this used to
be copy-pasted in both files."""

from __future__ import annotations

from typing import Any

from datasets import Dataset, DatasetDict
from pydantic import BaseModel


def records_to_split_dataset(
    records: list[BaseModel], cfg: dict[str, Any], shuffle: bool = False
) -> DatasetDict:
    """Turn a list of Pydantic model instances into a train/validation/test
    HF DatasetDict using the ratios in cfg['splits']."""
    full = Dataset.from_list([r.model_dump() for r in records])
    if shuffle:
        full = full.shuffle(seed=cfg["splits"]["seed"])

    n = len(full)
    n_train = int(n * cfg["splits"]["train"])
    n_val = int(n * cfg["splits"]["validation"])

    return DatasetDict(
        {
            "train": full.select(range(0, n_train)),
            "validation": full.select(range(n_train, n_train + n_val)),
            "test": full.select(range(n_train + n_val, n)),
        }
    )
