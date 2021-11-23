# Copyright 2021 MosaicML. All Rights Reserved.

from typing import cast

import torch

from composer.core.types import Batch, BatchPair, Tensor, Tensors
from composer.utils.iter_helpers import map_collection


def tensors_to_device(x: Tensors, device: torch.device) -> Tensors:
    return map_collection(x, lambda t: cast(Tensor, t).to(device, non_blocking=True))


def move_batch_to_device(batch: Batch, device: torch.device) -> Batch:
    """Move data to the specified device.

    Args:
        batch (Batch): The data to move the device.
    """
    if isinstance(batch, Tensor):
        return cast(Tensor, tensors_to_device(batch, device))
    if isinstance(batch, (tuple, list)):  # BatchPair
        return cast(BatchPair, tuple(tensors_to_device(x, device) for x in batch))
    if isinstance(batch, dict):  # BatchDict
        return {k: cast(Tensor, tensors_to_device(v, device)) for k, v in batch.items()}
    raise TypeError(f"Unsupported type for batch: {type(batch)}")
