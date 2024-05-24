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
from composer.utils import STR_TO_DTYPE, dist, get_composer_env_dict
import sys
from composer.devices import Device
from composer.models import HuggingFaceModel
import contextlib

log = logging.getLogger(__name__)


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
            If False, then rank 0 returns the state dict of the entire model and the other ranks return a dict of their shards. Default is False.
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


def _cast_state_dict_to_precision(state_dict: Dict[str, Any], precision: Union[str, torch.dtype]):
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


def get_metadata_state_dict(model: Optional[Union[ComposerModel, nn.Module]]=None,
                            sharded_state_dict: Optional[bool]=None,
                            precision: Optional[Union[str, torch.dtype]]=None,
                            device: Optional[Device] = None,
                            device_train_microbatch_size: Optional[int] = None,
                            generate_parameter_info: Optional[bool] = False,
                            ) -> Dict[str, Any]:
    """Generate the metadata and integrations for a training run.

    Args:
        model: The model to get the metadata state dict from. Only applicable if model is HuggingFaceModel
        sharded_state_dict: Whether the checkpoint this metadata state dict is associated with is sharded or not.
            Optional argument because this function may not be called in the context of making a full checkpoint.
        precision: The precision of the model. Can be specified as a string ('fp32', 'fp16', 'bf16') or a torch.dtype.
        device: The device the model is on.
        device_train_microbatch_size: The microbatch size used for training on the device.
        generate_parameter_info: Whether to generate parameter information for the model. Default is False.

    This state dict includes:
        * composer version 
        * composer commit hash
        * gpu model
        * num nodes
        * num gpus 
        * num gpus per node
        * cpu core count
        * cpu model
        * torch version
        * python version
        * optionally
            * dist/communication backend
            * precision
            * hf model metadata
            * device_train_microbatch_size
            * sharded vs unsharded state dict
            * model name
            * param fqns, shapes, and requires_grad
            * huggingface metadata

    Returns:
        The state dict containing the metadata and any integrations for a training run.
    """

    composer_env_dict = get_composer_env_dict()
    ced = composer_env_dict
    
    python_version = '.'.join([str(getattr(sys.version_info, k)) for k in ['major', 'minor', 'micro']])
    
    metadata_state_dict = {
        'composer_version': ced['composer_version'],
        'composer_commit_hash': ced['composer_commit_hash'],
        'torch_version': torch.__version__,
        'python_version': python_version,
        'num_nodes': ced['node_world_size'],
        'num_gpus_per_node': ced['local_world_size'],
        'num_gpus': dist.get_world_size(),
        'gpu_model':ced['accelerator_model_name'],
        'cpu_model': ced['host_processor_model_name'], 
        'cpu_core_count': ced['host_processor_core_count'],
    }
    if sharded_state_dict is not None:
        metadata_state_dict['sharded_state_dict'] = sharded_state_dict

    if model is not None:
        if isinstance(model, HuggingFaceModel):
            metadata_state_dict['huggingface'] = model.get_metadata()
        elif isinstance(model, DistributedDataParallel) and isinstance(model.module, HuggingFaceModel):
            metadata_state_dict['huggingface'] = model.module.get_metadata()
        
        metadata_state_dict['model_name'] = model.__class__.__name__

        if generate_parameter_info:
            for name, param in model.named_parameters():
                metadata_state_dict['param_info'][name] = {
                    'shape': param.shape,
                    'requires_grad': param.requires_grad,
                }

    if device is not None:
        metadata_state_dict['dist_backend'] = device.dist_backend

    if device_train_microbatch_size:
        metadata_state_dict['device_train_microbatch_size'] = device_train_microbatch_size

    if precision is not None:
        if isinstance(precision, str):
            metadata_state_dict['precision'] = precision
        else:
            dtype_to_str = {v:k for k,v in STR_TO_DTYPE.items()}
            metadata_state_dict['precision'] = dtype_to_str[precision]

    if generate_parameter_info:
        if model is None:
            raise ValueError('model must be provided to generate parameter information')
        if _is_model_fsdp(model):
            ctxt_manager = FSDP.summon_full_params(model)
        else:
            ctxt_manager = contextlib.nullcontext()
        with ctxt_manager:
            metadata_state_dict['param_info'] = {param_name: param.shape for param_name, param in model.named_parameters()}
            
    return metadata_state_dict