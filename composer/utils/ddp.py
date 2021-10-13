# Copyright 2021 MosaicML. All Rights Reserved.

import warnings

import torch.distributed as dist


def get_global_rank() -> int:
    if not dist.is_available():
        warnings.warn("Torch distributed is not available; returning 0 for global rank")
        return 0
    return dist.get_rank()  # raises a runtime error if not yet initialized


def is_rank_zero() -> bool:
    return get_global_rank() == 0


def is_rank_set() -> bool:
    try:
        get_global_rank()
        return True
    except RuntimeError:
        return False
