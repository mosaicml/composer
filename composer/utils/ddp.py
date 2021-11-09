# Copyright 2021 MosaicML. All Rights Reserved.

import os
import warnings
from typing import Callable, Optional

import torch.distributed as dist


def _get_distributed_config_var(env_var: str,
                                human_name: str,
                                default: int,
                                fetch_fn: Optional[Callable[[], int]] = None) -> int:
    if not dist.is_available():
        warnings.warn(f"Torch distributed is not available; returning {default} for {human_name}")
        return default

    if fetch_fn is not None:
        try:
            return fetch_fn()
        except RuntimeError:
            pass

    if not env_var in os.environ:
        warnings.warn(f"{env_var} env var not set"
                      f"{' and process group not initialized' if fetch_fn is not None else ''}; "
                      f"returning {default} for {human_name}.")
        return default

    return int(os.environ[env_var])


def get_world_size() -> int:
    return _get_distributed_config_var(env_var="WORLD_SIZE",
                                       human_name="world size",
                                       default=1,
                                       fetch_fn=dist.get_world_size)


def get_global_rank() -> int:
    return _get_distributed_config_var(env_var="RANK", human_name="global rank", default=0, fetch_fn=dist.get_rank)


def get_local_world_size() -> int:
    return _get_distributed_config_var(env_var="LOCAL_WORLD_SIZE", human_name="local world size", default=1)


def get_local_rank() -> int:
    return _get_distributed_config_var(env_var="LOCAL_RANK", human_name="local rank", default=0)


def is_rank_zero() -> bool:
    return get_global_rank() == 0


def is_rank_set() -> bool:
    try:
        get_global_rank()
        return True
    except RuntimeError:
        return False
