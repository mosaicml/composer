# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Useful functions for generating state dicts and manipulating them."""

import fnmatch
import logging
from typing import Any, Dict, Optional, Sequence, Union

import torch
from packaging import version
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel

from composer.models import ComposerModel
from composer.utils import dist

log = logging.getLogger(__name__)


def get_model_state_dict(
    model: Union[ComposerModel, nn.Module],
    sharded: bool,
    precision: Union[str, torch.dtype] = 'fp32',
    include_keys: Optional[Union[str, Sequence[str]]] = None,
    ignore_keys: Optional[Union[str, Sequence[str]]] = None,
    cpu_offload: Optional[bool] = None,
) -> Dict[str, Any]:
    """Generate the state dict of the model.

    Args:
        model: The model to get the state dict from.
        sharded: Whether the model state dict should be sharded or not. If True, every rank returns the state dict of its shards.
            If False, then rank 0 returns the state dict of the entire model.
        precision: The precision of the model.
        include_keys: The list of keys to exclusively include in the state dict. If None, all keys are included. Both include_keys and ignore_keys cannot be non-None.
        ignore_keys: The list of keys to ignore in the state dict. If None, no keys are ignored. Both include_keys and ignore_keys cannot be non-None.

    Returns:
        The state dict of the model.
    """
    if include_keys is not None and ignore_keys is not None:
        raise ValueError('Both include_keys and ignore_keys cannot be non-None.')

    is_fsdp = _is_model_fsdp(model)
    cpu_offload = cpu_offload if cpu_offload is not None else is_fsdp

    log.debug('Extracting model state dict')
    if version.parse(torch.__version__) >= version.parse('2.3.0') and dist.is_initialized():
        from torch.distributed.checkpoint.state_dict import StateDictOptions
        from torch.distributed.checkpoint.state_dict import get_model_state_dict as dcp_get_model_state_dict
        get_nonsharded_state_dict = not sharded

        model_state_dict = dcp_get_model_state_dict(
            model=model,
            submodules=None,
            options=StateDictOptions(
                full_state_dict=get_nonsharded_state_dict,
                cpu_offload=cpu_offload,
            ),
        )
    else:
        if is_fsdp:
            model_state_dict = _get_model_state_dict_with_fsdp_context_manager(model, sharded)
        else:
            model_state_dict = model.state_dict()
        if isinstance(model, DistributedDataParallel):
            nn.modules.utils.consume_prefix_in_state_dict_if_present(model_state_dict, 'module.')

    if include_keys is not None:
        model_state_dict = _extract_keys_from_state_dict(model_state_dict, include_keys)

    if ignore_keys is not None:
        model_state_dict = _remove_keys_from_state_dict(model_state_dict, ignore_keys)

    model_state_dict = _convert_to_dict_to_precision(state_dict=model_state_dict, precision=precision)

    log.debug('Finished extracting model state dict')
    return model_state_dict


STR_TO_DTYPE = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}


def _convert_to_dict_to_precision(state_dict: Dict[str, Any], precision: Union[str, torch.dtype]):
    if isinstance(precision, str):
        precision = STR_TO_DTYPE[precision]

    new_state_dict = {k: v.to(precision) for k, v in state_dict.items()}
    return new_state_dict


def _extract_keys_from_state_dict(state_dict: Dict[str, Any], include_keys: Union[str, Sequence[str]]):
    if isinstance(include_keys, str):
        include_keys = [include_keys]
    new_state_dict = {k: v for k, v in state_dict.items() if any(fnmatch.fnmatch(k, key) for key in include_keys)}

    return new_state_dict


def _remove_keys_from_state_dict(state_dict: Dict[str, Any], ignore_keys: Union[str, Sequence[str]]):
    if isinstance(ignore_keys, str):
        ignore_keys = [ignore_keys]
    new_state_dict = {k: v for k, v in state_dict.items() if not any(fnmatch.fnmatch(k, key) for key in ignore_keys)}
    return new_state_dict


def _is_model_fsdp(model) -> bool:
    """Indicates if FSDP is enabled.

    Args:
        model: The model to check if FSDP is enabled.

    Returns:
        True if FSDP is enabled, False otherwise.

    """
    for module in model.modules():
        if isinstance(module, FSDP):
            return True
    return False


def _get_model_state_dict_with_fsdp_context_manager(model: nn.Module, sharded: bool) -> Dict[str, Any]:
    """Get the model state dict with the FSDP context manager.

    Args:
        model: The model to get the state dict from.
        sharded: Whether the model state dict should be sharded or not. If True, every rank returns the state dict of its shards.
            If False, then rank 0 returns the state dict of the entire model.

    Returns:
        The state dict of the model.
    """
    from torch.distributed.fsdp.fully_sharded_data_parallel import (
        FullStateDictConfig,
        ShardedStateDictConfig,
        StateDictType,
    )
    state_dict_type = StateDictType.SHARDED_STATE_DICT if sharded else StateDictType.FULL_STATE_DICT
    state_dict_config = ShardedStateDictConfig(offload_to_cpu=True,) if sharded else FullStateDictConfig(
        rank0_only=True,
        offload_to_cpu=True,
    )
    with FSDP.state_dict_type(model, state_dict_type=state_dict_type, state_dict_config=state_dict_config):
        model_state_dict = model.state_dict()
    return model_state_dict
