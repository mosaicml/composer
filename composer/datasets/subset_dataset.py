# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

from typing import Any

from composer.core.types import Dataset


class SubsetDataset(Dataset):

    def __init__(self, dataset: Dataset, batch_size: int, num_total_batches: int):
        self.dataset = dataset
        self.size = batch_size * num_total_batches

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Any:
        return self.dataset[idx]
