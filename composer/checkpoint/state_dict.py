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
from composer.utils import STR_TO_DTYPE, dist

log = logging.getLogger(__name__)

__all__ = ['get_model_state_dict', 'get_optim_state_dict']


def get_model_state_dict(
    model: Union[ComposerModel, nn.Module],
    sharded_state_dict: bool = False,
    precision: Union[str, torch.dtype] = 'fp32',
    include_keys: Optional[Union[str, Sequence[str]]] = None,
    ignore_keys: Optional[Union[str, Sequence[str]]] = None,
    cpu_offload: Optional[bool] = None,
) -> Dict[str, Any]:
    """Generate the state dict of the model.

    Args:
        model: The model to get the state dict from.
        sharded_state_dict: Whether the model state dict should be sharded or not. If True, every rank returns the state dict of its shards.
            If False, then rank 0 returns the state dict of the entire model and the other ranks return an empty dict. Default is False.
        precision: The precision of the model. Can be specified as a string ('fp32', 'fp16', 'bf16') or a torch.dtype.
        include_keys: The list of keys to exclusively include in the state dict. If None, all keys are included. Both include_keys and ignore_keys cannot be non-None.
        ignore_keys: The list of keys to ignore in the state dict. If None, no keys are ignored. Both include_keys and ignore_keys cannot be non-None.
        cpu_offload: Whether to offload the state dict to CPU. If None, it is set to True if FSDP is enabled, False otherwise.

    Returns:
        The state dict of the model.
    """
    if include_keys is not None and ignore_keys is not None:
        raise ValueError(f'Both {include_keys=} and {ignore_keys=} cannot be non-None.')

    is_fsdp = _is_model_fsdp(model)
    if not is_fsdp and sharded_state_dict:
        raise ValueError('Sharded state dict can only be generated for FSDP models.')
    cpu_offload = cpu_offload if cpu_offload is not None else (is_fsdp and not sharded_state_dict)

    log.debug('Extracting model state dict')
    if version.parse(torch.__version__) >= version.parse('2.2.0') and dist.is_initialized():
        from torch.distributed.checkpoint import state_dict as DCPSD  # Distributed Checkpoint State Dict
        from torch.distributed.checkpoint.state_dict import StateDictOptions

        use_unsharded_state_dict = not sharded_state_dict

        log.debug('Calling torch get_model_state_dict...')
        model_state_dict = DCPSD.get_model_state_dict(
            model=model,
            submodules=None,  # We extract submodules below
            options=StateDictOptions(
                full_state_dict=use_unsharded_state_dict,
                cpu_offload=cpu_offload,
            ),
        )
    else:
        if is_fsdp:
            log.debug('Calling legacy FSDP context manager to get model state dict...')
            model_state_dict = _get_model_state_dict_with_fsdp_context_manager(model, sharded_state_dict, cpu_offload)
        else:
            log.debug('Calling model.state_dict() for non-FSDP model...')
            model_state_dict = model.state_dict()
        if isinstance(model, DistributedDataParallel):
            nn.modules.utils.consume_prefix_in_state_dict_if_present(model_state_dict, 'module.')

    if include_keys is not None:
        model_state_dict = _extract_keys_from_state_dict(model_state_dict, include_keys)

    if ignore_keys is not None:
        model_state_dict = _remove_keys_from_state_dict(model_state_dict, ignore_keys)

    model_state_dict = _cast_state_dict_to_precision(state_dict=model_state_dict, precision=precision)

    log.debug('Finished extracting model state dict')
    return model_state_dict


def _cast_state_dict_to_precision(state_dict: Dict[str, Any], precision: Union[str, torch.dtype]) -> Dict[str, Any]:
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


def _get_model_state_dict_with_fsdp_context_manager(model: nn.Module, sharded_state_dict: bool,
                                                    cpu_offload: bool) -> Dict[str, Any]:
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
    state_dict_type = StateDictType.SHARDED_STATE_DICT if sharded_state_dict else StateDictType.FULL_STATE_DICT
    state_dict_config = ShardedStateDictConfig(offload_to_cpu=cpu_offload,
                                              ) if sharded_state_dict else FullStateDictConfig(
                                                  rank0_only=True,
                                                  offload_to_cpu=cpu_offload,
                                              )
    with FSDP.state_dict_type(model, state_dict_type=state_dict_type, state_dict_config=state_dict_config):
        model_state_dict = model.state_dict()
    return model_state_dict


def _get_optim_state_dict_with_fsdp_context_manager(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    sharded_state_dict: bool,
    cpu_offload: bool,
) -> Dict[str, Any]:
    """Get the optimizer state dict with the FSDP context manager.

    Args:
        model: The model containing the parameters that the optimizer is optimizing.
        optimizer: The optimizer to get the state dict from.
        sharded_state_dict: Whether the optimizer state dict should be sharded or not. If True, every rank returns the state dict of its shards.
            If False, then rank 0 returns the state dict of the entire optimizer.
        cpu_offload: Whether to offload the state dict to CPU.

    Returns:
        The state dict of the optimizer.

    """
    from torch.distributed.fsdp.fully_sharded_data_parallel import (
        FullOptimStateDictConfig,
        FullStateDictConfig,
        ShardedOptimStateDictConfig,
        ShardedStateDictConfig,
        StateDictType,
    )
    state_dict_type = StateDictType.SHARDED_STATE_DICT if sharded_state_dict else StateDictType.FULL_STATE_DICT

    state_dict_config = ShardedStateDictConfig(offload_to_cpu=cpu_offload,
                                              ) if sharded_state_dict else FullStateDictConfig(
                                                  rank0_only=True,
                                                  offload_to_cpu=cpu_offload,
                                              )
    optim_state_dict_config = ShardedOptimStateDictConfig(
        offload_to_cpu=cpu_offload,
    ) if sharded_state_dict else FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=cpu_offload)
    with FSDP.state_dict_type(
        model,
        state_dict_type=state_dict_type,
        state_dict_config=state_dict_config,
        optim_state_dict_config=optim_state_dict_config,
    ):
        optim_state_dict = FSDP.optim_state_dict(model, optimizer)
    return optim_state_dict


def get_optim_state_dict(
    model: Union[ComposerModel, nn.Module],
    optimizer: torch.optim.Optimizer,
    sharded_state_dict: bool = False,
    precision: str = 'fp32',
    include_keys: Optional[Union[str, Sequence[str]]] = None,
    ignore_keys: Optional[Union[str, Sequence[str]]] = None,
    cpu_offload: Optional[bool] = None,
) -> Dict[str, Any]:
    """Generate the state dict of the optimizer.

    Args:
        model: The model containing the parameters that the optimizer is optimizing.
        optimizer: The optimizer to get the state dict from.
        sharded: Whether the optimizer is sharded or not. If True, every rank returns the state dict of its shards.
            If False, then rank 0 returns the state dict of the entire optimizer and all other ranks return an empty dict.
        precision: The precision of the optimizer.
        include_keys: The list of keys to exclusively include in the state dict. If None, all keys are included. Both include_keys and ignore_keys cannot be non-None.
        ignore_keys: The list of keys to ignore in the state dict. If None, no keys are ignored. Both include_keys and ignore_keys cannot be non-None.
        cpu_offload: Whether to offload the state dict to CPU. If None, it is set to True if FSDP is enabled with non-sharded state dict and False otherwise.

    Returns:
        The state dict of the optimizer.
    """
    if include_keys is not None and ignore_keys is not None:
        raise ValueError(f'Both {include_keys=} and {ignore_keys=} cannot be non-None.')

    is_fsdp = _is_model_fsdp(model)
    if not is_fsdp and sharded_state_dict:
        raise ValueError('Sharded optim state dict can only be generated for FSDP models.')

    cpu_offload = cpu_offload if cpu_offload is not None else (is_fsdp and not sharded_state_dict)
    log.debug('Extracting optim state dict')
    if version.parse(torch.__version__) >= version.parse('2.2.0') and dist.is_initialized():
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict
        log.debug('Calling torch get_optimizer_state_dict...')
        optim_state_dict: Dict[str, Any] = get_optimizer_state_dict(
                model=model,
                optimizers=optimizer,
                submodules=None, # We extract submodules below
                options=StateDictOptions(
                    full_state_dict=not sharded_state_dict,
                    cpu_offload=cpu_offload,
                ),
            )
    else:
        if is_fsdp:
            log.debug('Calling legacy FSDP context manager to get optim state dict...')
            optim_state_dict = _get_optim_state_dict_with_fsdp_context_manager(
                model,
                optimizer,
                sharded_state_dict,
                cpu_offload,
            )
        else:
            optim_state_dict = optimizer.state_dict()

    # For sharded models with non-sharded state dicts, only rank 0 has the full state dict including all the keys
    target_state_dict_on_this_rank = (not sharded_state_dict and dist.get_global_rank() == 0) or sharded_state_dict
    
    if target_state_dict_on_this_rank:
        if ignore_keys is not None:
            optim_state_dict = _remove_keys_from_optim_state_dict(optim_state_dict, model, ignore_keys)
        if include_keys is not None:
            optim_state_dict = _extract_keys_from_optim_state_dict(optim_state_dict, model, include_keys)

        # param_key := index (0,1,2,..., len(model.parameters())-1) for unsharded models.
        # param_key := fqn for sharded models.
        for param_key, param_state_dict in optim_state_dict['state'].items():
            optim_state_dict['state'][param_key] = _cast_state_dict_to_precision(param_state_dict, precision)
    return optim_state_dict


def _remove_keys_from_optim_state_dict(
    optim_state_dict: Dict[str, Any],
    model: Union[ComposerModel, nn.Module],
    ignore_keys: Union[str, Sequence[str]],
):
    if isinstance(ignore_keys, str):
        ignore_keys = [ignore_keys]

    # optim_state_dict['state'] is a dictionary mapping the param_key
    # to the optimizer state ( e.g. 'step', 'exp_avg', 'exp_avg_sq') for that parameter.
    # For sharded models the param_key is just the fqn for the underlying model parameter,
    # but for unsharded models the param_key is an index (0,1,2,..., len(model.parameters())-1)
    if _is_model_fsdp(model):
        for param_fqn in optim_state_dict['state'].keys():
            if any([fnmatch.fnmatch(param_fqn, ignore_key) for ignore_key in ignore_keys]):
                optim_state_dict['state'].pop(param_fqn)

    # The param index ordering is determined by passing model.parameters()
    # to the optimizer. The underlying generator for model.parameters() is model.named_parameters()
    # so we need to use model.named_parameters() instead of model.state_dict().keys() to match fqn to ind correctly.
    else:
        param_inds = list(optim_state_dict['state'].keys())
        for param_ind, (param_fqn, _) in zip(param_inds, model.named_parameters()):
            if any([fnmatch.fnmatch(param_fqn, ignore_key) for ignore_key in ignore_keys]):
                optim_state_dict['state'].pop(param_ind)
 

    return optim_state_dict


def _extract_keys_from_optim_state_dict(
    optim_state_dict: Dict[str, Any],
    model: Union[ComposerModel, nn.Module],
    include_keys: Union[str, Sequence[str]],
):
    if isinstance(include_keys, str):
        include_keys = [include_keys]
    param_inds = list(optim_state_dict['state'].keys())
    # See comment in _remove_keys_from_optim_state_dict.
    for param_ind, (param_fqn, _) in zip(param_inds, model.named_parameters()):
        if not any([fnmatch.fnmatch(param_fqn, include_key) for include_key in include_keys]):
            optim_state_dict['state'].pop(param_ind)

    return optim_state_dict
