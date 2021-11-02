# Copyright 2021 MosaicML. All Rights Reserved.

import os
import warnings

import torch.distributed as dist


def get_world_size() -> int:
    if not dist.is_available():
        warnings.warn("Torch distributed is not available; returning 1 for world size.")
        return 1

    try:
        return dist.get_world_size()
    except RuntimeError:
        pass

    if not "WORLD_SIZE" in os.environ:
        warnings.warn("WORLD_SIZE env var not set and process group not initialized; returning 1 for world size.")
        return 1
    return int(os.environ["WORLD_SIZE"])


def get_global_rank() -> int:
    if not dist.is_available():
        warnings.warn("Torch distributed is not available; returning 0 for global rank.")
        return 0

    try:
        return dist.get_rank()
    except RuntimeError:
        pass

    if not "RANK" in os.environ:
        warnings.warn("RANK env var not set and process group not initialized; returning 0 for global rank.")
        return 0
    return int(os.environ["RANK"])


def get_local_world_size() -> int:
    if not dist.is_available():
        warnings.warn("Torch distributed is not available; returning 1 for local world size.")

    if not "LOCAL_WORLD_SIZE" in os.environ:
        warnings.warn("LOCAL_WORLD_SIZE env var not set; returning 1 for local world size.")
        return 1
    return int(os.environ["LOCAL_WORLD_SIZE"])


def get_local_rank() -> int:
    if not dist.is_available():
        warnings.warn("Torch distributed is not available; returning 0 for local rank.")

    if not "LOCAL_RANK" in os.environ:
        warnings.warn("LOCAL_RANK env var not set; returning 0 for local rank.")
        return 0
    return int(os.environ["LOCAL_RANK"])


def is_rank_zero() -> bool:
    return get_global_rank() == 0


def is_rank_set() -> bool:
    try:
        get_global_rank()
        return True
    except RuntimeError:
        return False
