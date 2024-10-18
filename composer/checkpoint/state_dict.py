# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Useful functions for generating state dicts and manipulating them."""

import fnmatch
import logging
import sys
from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence, Union

from torch.utils.data import DataLoader, Dataset

from composer.core.data_spec import DataSpec

if TYPE_CHECKING:
    from composer.core.evaluator import Evaluator

import torch
from packaging import version
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from composer.core.evaluator import Evaluator
from composer.core.state import State
from composer.core.time import Timestamp
from composer.devices import Device
from composer.models import ComposerModel, HuggingFaceModel
from composer.utils import STR_TO_DTYPE, dist, get_composer_env_dict, reproducibility

log = logging.getLogger(__name__)

__all__ = ['get_model_state_dict', 'get_optim_state_dict', 'get_metadata_state_dict', 'get_resumption_state_dict']


def get_model_state_dict(
    model: Union[ComposerModel, nn.Module],
    sharded_state_dict: bool = False,
    precision: Union[str, torch.dtype] = 'fp32',
    include_keys: Optional[Union[str, Sequence[str]]] = None,
    ignore_keys: Optional[Union[str, Sequence[str]]] = None,
    cpu_offload: Optional[bool] = None,
) -> dict[str, Any]:
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
            nn.modules.utils.consume_prefix_in_state_dict_if_present(model_state_dict, 'module.')  # type: ignore

    if include_keys is not None:
        model_state_dict = _extract_keys_from_state_dict(model_state_dict, include_keys)

    if ignore_keys is not None:
        model_state_dict = _remove_keys_from_state_dict(model_state_dict, ignore_keys)

    model_state_dict = _cast_state_dict_to_precision(state_dict=model_state_dict, precision=precision)

    log.debug('Finished extracting model state dict')
    return model_state_dict


def _cast_state_dict_to_precision(state_dict: dict[str, Any], precision: Union[str, torch.dtype]) -> dict[str, Any]:
    if isinstance(precision, str):
        precision = STR_TO_DTYPE[precision]

    new_state_dict = {k: v.to(precision) for k, v in state_dict.items()}
    return new_state_dict


def _extract_keys_from_state_dict(state_dict: dict[str, Any], include_keys: Union[str, Sequence[str]]):
    if isinstance(include_keys, str):
        include_keys = [include_keys]
    new_state_dict = {k: v for k, v in state_dict.items() if any(fnmatch.fnmatch(k, key) for key in include_keys)}

    return new_state_dict


def _remove_keys_from_state_dict(state_dict: dict[str, Any], ignore_keys: Union[str, Sequence[str]]):
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
                                                    cpu_offload: bool) -> dict[str, Any]:
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


def get_resumption_state_dict(state: State) -> dict[str, Any]:
    """Generate the state dict for any objects needed for resumption.

    This includes:
        * timestamp
        * scheduler
        * dataset_state
        * scaler
        * rank_zero_seed
        * callbacks
        * algorithms

    Returns:
        The state dict containing the objects needed for resumption.
    """
    resumption_state_dict = {}
    resumption_state_dict['dataset_state'] = get_dataset_state_dict(
        state.train_dataloader,
        state.timestamp,
    )
    resumption_state_dict['timestamp'] = state.timestamp.state_dict()

    scheduler_state_dict = _make_state_dict_for_list_of_objects(state.schedulers)
    if scheduler_state_dict != {}:
        resumption_state_dict['schedulers'] = scheduler_state_dict

    # Use list of tuples to account for duplicates
    callbacks_state_dict = _make_state_dict_for_list_of_objects(state.callbacks, use_list_of_tuples=True)
    if callbacks_state_dict != {}:
        resumption_state_dict['callbacks'] = callbacks_state_dict

    # Use list of tuples to preserve order.
    algorithms_state_dict = _make_state_dict_for_list_of_objects(state.algorithms, use_list_of_tuples=True)
    if algorithms_state_dict != {}:
        resumption_state_dict['algorithms'] = algorithms_state_dict

    if state.scaler is not None:
        scaler_sd = _make_state_dict_for_list_of_objects(state.scaler)
        if scaler_sd != {}:
            resumption_state_dict['scaler'] = scaler_sd

    resumption_state_dict['rank_zero_seed'] = state.rank_zero_seed
    resumption_state_dict['run_name'] = state.run_name
    resumption_state_dict['rng'] = reproducibility.get_rng_state()

    return resumption_state_dict


def _make_state_dict_for_list_of_objects(objects: Union[Sequence[Any], Any],
                                         use_list_of_tuples=False) -> Union[dict[str, Any], list]:
    object_list = []
    object_dict = {}
    if not isinstance(objects, Sequence):
        objects = [objects]
    for obj in objects:
        if not hasattr(obj, 'state_dict') or obj.state_dict() == {}:
            continue
        if use_list_of_tuples:
            object_list.append((type(obj).__qualname__, obj.state_dict()))
        else:
            object_dict[type(obj).__qualname__] = obj.state_dict()
    if use_list_of_tuples:
        return object_list
    else:
        return object_dict


def get_dataset_state_dict(
    train_dataloader: Optional[Union[DataLoader, Iterable]],
    timestamp: Timestamp,
) -> dict[str, Any]:
    """Collect the state dict(s) of our train and eval dataset(s).

    Returns:
        Dict[str, Any]: The state dict(s).
    """
    dataset_state_dict = {
        'train': None,
    }
    dataset = _dataset_of(train_dataloader)
    if hasattr(dataset, 'state_dict'):
        num_samples = int(timestamp.sample_in_epoch.value)
        dataset_state_dict['train'] = dataset.state_dict(num_samples, True)  # pyright: ignore

    return dataset_state_dict


def _get_optim_state_dict_with_fsdp_context_manager(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    sharded_state_dict: bool,
    cpu_offload: bool,
) -> dict[str, Any]:
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
) -> dict[str, Any]:
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
        optim_state_dict: dict[str, Any] = get_optimizer_state_dict(
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
            raise NotImplementedError('Ignoring keys in the optimizer state dict is not supported yet.')
        if include_keys is not None:
            raise NotImplementedError('Ignoring keys in the optimizer state dict is not supported yet.')

        # param_key := index (0,1,2,..., len(model.parameters())-1) for unsharded models.
        # param_key := fqn for sharded models.
        for param_key, param_state_dict in optim_state_dict['state'].items():
            optim_state_dict['state'][param_key] = _cast_state_dict_to_precision(param_state_dict, precision)
    return optim_state_dict


def get_metadata_state_dict(
    model: Optional[Union[ComposerModel, nn.Module]] = None,
    sharded_state_dict: Optional[bool] = None,
    precision: Optional[Union[str, torch.dtype]] = None,
    device: Optional[Device] = None,
    device_train_microbatch_size: Optional[Union[int, float]] = None,
) -> dict[str, Any]:
    """Generate the metadata and integrations for a training run.

    Args:
        model: The model to get the metadata state dict from. Only applicable if model is HuggingFaceModel
        sharded_state_dict: Whether the checkpoint this metadata state dict is associated with is sharded or not.
            Optional argument because this function may not be called in the context of making a full checkpoint.
        precision: The precision of the model. Can be specified as a string ('fp32', 'fp16', 'bf16') or a torch.dtype.
        device: The device the model is on.
        device_train_microbatch_size: The microbatch size used for training on the device.

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
    metadata_state_dict = get_composer_env_dict()

    python_version = '.'.join([str(getattr(sys.version_info, k)) for k in ['major', 'minor', 'micro']])

    metadata_state_dict['torch_version'] = torch.__version__
    metadata_state_dict['python_version'] = python_version
    if sharded_state_dict is not None:
        metadata_state_dict['sharded_state_dict'] = sharded_state_dict

    if model is not None:
        if isinstance(model, HuggingFaceModel):
            metadata_state_dict['huggingface'] = model.get_metadata()
            metadata_state_dict['model_name'] = model.model.__class__.__name__
        elif (isinstance(model, DistributedDataParallel) or
              isinstance(model, FSDP)) and isinstance(model.module, HuggingFaceModel):
            metadata_state_dict['huggingface'] = model.module.get_metadata()
            metadata_state_dict['model_name'] = model.module.model.__class__.__name__
        elif isinstance(model, FSDP) or isinstance(model, DistributedDataParallel):
            metadata_state_dict['model_name'] = model.module.__class__.__name__
        else:
            metadata_state_dict['model_name'] = model.__class__.__name__

    if device is not None:
        metadata_state_dict['dist_backend'] = device.dist_backend

    if device_train_microbatch_size:
        metadata_state_dict['device_train_microbatch_size'] = device_train_microbatch_size

    if precision is not None:
        if isinstance(precision, str):
            metadata_state_dict['precision'] = precision
        else:
            dtype_to_str = {v: k for k, v in STR_TO_DTYPE.items()}
            metadata_state_dict['precision'] = dtype_to_str[precision]
    else:
        metadata_state_dict['precision'] = 'fp32'

    return metadata_state_dict


def _dataset_of(dataloader: Optional[Union[Evaluator, DataSpec, DataLoader, Iterable]]) -> Optional[Dataset]:
    """Get the dataset contained by the given dataloader-like object.

    Args:
        dataloader (Evaluator | DataSpec | DataLoader | Iterable, optional): The dataloader, wrapped dataloader, or
            generic python iterable to get the dataset of, if applicable.

    Returns:
        Dataset: Its dataset, if there is one.
    """
    from composer.core.evaluator import Evaluator

    # If it's None, no dataset for you.
    if dataloader is None:
        return None

    # An Evaluator is a dataloader wrapped with metrics. Unwrap its dataloader.
    if isinstance(dataloader, Evaluator):
        dataloader = dataloader.dataloader

    # A DataSpec is a dataloader wrapped with an on-device transform. Unwrap its dataloader.
    if isinstance(dataloader, DataSpec):
        dataloader = dataloader.dataloader

    # If what we now have is an actual DataLoader, return its dataset. If not, return None.
    if isinstance(dataloader, DataLoader):
        return dataloader.dataset
    else:
        return None
