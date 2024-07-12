from typing import List, Optional, Sequence, Union
from composer.core.state import State
import logging
from composer.models import ComposerModel
from composer.utils.checkpoint import safe_torch_load, _torch_load_with_validation, FileSystemReaderWithValidation, DistCPObjectStoreReader, _ensure_valid_checkpoint
from composer.checkpoint.state_dict import _is_model_fsdp, get_model_state_dict, _extract_keys_from_state_dict, _remove_keys_from_state_dict, get_optim_state_dict
import torch.distributed.checkpoint as DCP
from composer.utils import dist
from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions
from packaging import version
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch
import os
import tempfile
from torch.distributed.checkpoint.planner import LoadPlanner
import contextlib
from composer.utils.file_helpers import is_uri
from torch.distributed._tensor import DeviceMesh
import textwrap
from torch import nn
from composer.utils import reproducibility, FSDPConfig
from dataclasses import dataclass
from composer.checkpoint.state_dict import _cast_state_dict_to_precision
from composer.distributed import prepare_fsdp_module
import pickle
from composer.checkpoint.download import download_monolithic_checkpoint
from pathlib import Path


log = logging.getLogger(__name__)

@dataclass
class CheckpointLoadOptions:
    load_path: str # Can be local path, uri, or hf name
    load_model: bool = True
    load_optimizer: bool = False
    load_resumption_keys: bool = False # dataset_state, timestamp,
    # Specific key-level loading configs
        # e.g.'model.layers.transformer.0.w1.weight'
    include_keys: Optional[Union[str, Sequence[str]]] = None
    ignore_keys: Optional[Union[str, Sequence[str]]] = None
    sharded_checkpoint: bool = False # TODO: Auto-detect sharded
    shard_as_needed_during_load: bool
    strict: bool = False
    # Load precision.
    precision: str = 'fp32'


def load_checkpoint(
    state: State,
    load_options: CheckpointLoadOptions
    ):
    """
    Optionally download and load  a checkpoint according to the options into specified state.

    Args:
        state (State): The State object containing the model, optim, timestamp, scheduler, etc.
        load_options (CheckpointLoadOptions): The options to use for loading the checkpoint.
    """
    load_model_and_optim_together = load_options.load_optimizer and (load_options.sharded_checkpoint or load_options.shard_as_needed_during_load)
    if load_model_and_optim_together:
        load_model_checkpoint(state.model,
                              load_path=load_options.load_path,
                              optimizer=state.optimizers[0],
                              include_keys=load_options.include_keys, 
                              ignore_keys=load_options.ignore_keys,
                              strict=load_options.strict,
                              sharded_checkpoint=load_options.sharded_checkpoint,)


    else:
        load_model_checkpoint(state.model, load_options.load_path, include_keys=load_options.include_keys, 
                             ignore_keys=load_options.ignore_keys, strict=load_options.strict,)
        if load_options.load_optimizer:
            load_optim_checkpoint(state.model, state.optimizers[0], load_options.load_path, sharded_checkpoint=load_options.sharded_checkpoint)

    if load_options.load_resumption_keys:
        load_resumption_checkpoint(state)


def load_model_checkpoint(
    model: ComposerModel,
    load_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    include_keys: Optional[Union[str, Sequence[str]]] = None,
    ignore_keys: Optional[Union[str, Sequence[str]]] = None,
    precision: str = 'fp32',
    fsdp_config: Optional[FSDPConfig] = None,
    strict: bool = False,
    device_mesh: Optional[DeviceMesh]=None,
    sharded_checkpoint: bool = False,
    shard_as_needed_during_load: bool = False,
    planner: Optional[LoadPlanner] = None,
    seed: Optional[int] = 42
):
    """
    Load a a model checkpoint from the specified path into the model.

    Args:
        model (ComposerModel): The model to load the checkpoint into.
        load_path (str): The local path to the checkpoint.
        include_keys (Optional[Union[str, Sequence[str]]]): The keys to include from the model checkpoint. Note that if ignore_keys is specified, then this argument will be ignored.
        ignore_keys (Optional[Union[str, Sequence[str]]]): The keys to ignore from the model checkpoint. Note that if include_keys is specified, then this argument will be
        precision (str): The precision to cast the model checkpoint to.
        device_mesh (Optional[DeviceMesh]): The device mesh to use for loading the checkpoint.
        sharded_checkpoint (bool): If the checkpoint is sharded or not.
        planner (Optional[LoadPlanner]): The planner to use for loading the checkpoint.
    """
    if include_keys is not None and ignore_keys is not None:
        raise ValueError("Only one of include_keys or ignore_keys can be specified.")
    
    if include_keys is not None or ignore_keys is not None:
        strict = True
    if sharded_checkpoint:
        if not _is_model_fsdp(model):
            if shard_as_needed_during_load:
                _shard_model_and_optimizer(model, optimizer, fsdp_config, precision, seed)
            else:
                raise ValueError("Model is not sharded but checkpoint is sharded. Please pass in a model wrapped with FSDP or set shard_model_as_needed_during_load to True.")   
        _load_sharded_model_checkpoint(model, load_path, include_keys=include_keys, 
                                      ignore_keys=ignore_keys, strict=strict, 
                                      device_mesh=device_mesh, planner=planner, precision=precision)
        if optimizer is not None:
            _load_sharded_optim_checkpoint(model, optimizer, load_path, precision=precision)
    else:
        _load_unsharded_model_checkpoint(model, load_path, include_keys=include_keys, ignore_keys=ignore_keys, strict=strict, precision=precision)
        if optimizer is not None:
            _load_unsharded_optim_checkpoint(model, optimizer, load_path, precision=precision)
        if shard_as_needed_during_load:
            _shard_model_and_optimizer(model, optimizer, fsdp_config, precision, seed)


def _shard_model_and_optimizer(model, optimizer, fsdp_config, precision, seed):
    with reproducibility.seed_context(seed):
        prepare_fsdp_module(
            model,
            optimizers=[optimizer] if optimizer is not None else None,
            fsdp_config=fsdp_config,
            precision=precision,
        )

def _load_sharded_model_checkpoint(
    model: ComposerModel,
    load_path: str,
    include_keys: Optional[Union[str, Sequence[str]]] = None,
    ignore_keys: Optional[Union[str, Sequence[str]]] = None,
    precision: str = 'fp32',
    strict: bool = False,
    device_mesh: Optional[DeviceMesh]=None,
    planner: Optional[LoadPlanner] = None,
):
 
    model_state_dict = get_model_state_dict(model, sharded_state_dict=True, include_keys=include_keys, ignore_keys=ignore_keys)
    model_state_dict = download_and_load_sharded_state_dict(load_path, device_mesh, model_state_dict, planner)
    model_state_dict = _cast_state_dict_to_precision(model_state_dict, precision)
    # TODO: raise warning for unknown or missing keys.
    log.debug(f'Loading sharded state dict into model.')
    if version.parse(torch.__version__) >= version.parse('2.3.0') and dist.is_initialized():
        torch_set_model_state_dict(model, model_state_dict, strict=strict, options=StateDictOptions(cpu_offload=True))
    else:
        _load_model_state_dict_with_fsdp_context_manager(model, model_state_dict, sharded_state_dict=True, strict=strict)


def _load_unsharded_model_checkpoint(model: ComposerModel,
                                    load_path: str,
                                    include_keys: Optional[Union[str, Sequence[str]]] = None,
                                    ignore_keys: Optional[Union[str, Sequence[str]]] = None,
                                    precision: str = 'fp32',
                                    strict: bool = False):
    
    if not _is_model_fsdp(model):
        raise NotImplementedError("Model is sharded but checkpoint is not sharded. Please pass in an unwrapped model!")

    load_path_is_remote = is_uri(load_path)
    download_dir_context = tempfile.TemporaryDirectory if load_path_is_remote else contextlib.nullcontext
    with download_dir_context() as download_dir:
        file_path = load_path
        if load_path_is_remote:
            filename = Path(load_path).name
            file_path = os.path.join(download_dir, filename)
            download_monolithic_checkpoint(load_path, download_dir)

        if dist.get_global_rank() == 0:
            model_state_dict = _torch_load_with_validation(file_path)
            if include_keys is not None:
                model_state_dict = _extract_keys_from_state_dict(model_state_dict, include_keys)
            if ignore_keys is not None:
                model_state_dict = _remove_keys_from_state_dict(model_state_dict, ignore_keys)
            # TODO: raise warning for unknown or missing keys.
            if version.parse(torch.__version__) >= version.parse('2.3.0') and dist.is_initialized():
                torch_set_model_state_dict(model, model_state_dict, strict=strict, options=StateDictOptions(cpu_offload=False, full_state_dict=True))
            else:
                if _is_model_fsdp(model):
                    _load_model_state_dict_with_fsdp_context_manager(model, model_state_dict, sharded_state_dict=False, strict=strict)
                else:
                    model.load_state_dict(model_state_dict, strict=strict)


def load_optim_checkpoint(
    model: ComposerModel,
    optim: torch.optim.Optimizer,
    load_path: str,
    sharded_checkpoint: bool,
    precision: str = 'fp32',
):
    """
    Load an optimizer checkpoint from the specified path into the optimizer.

    Args:
        model (ComposerModel): The model to load the optimizer checkpoint into.        
        optim (torch.optim.Optimizer): The optimizer to load the checkpoint into.
        load_path (str): The path or uri to the checkpoint to load or symlink to the path/uri. If URI specified then files will be downloaded first.
        sharded_checkpoint (bool): If the optimizer state checkpoint is sharded or not.
    """
    if sharded_checkpoint:
        _load_sharded_optim_checkpoint(model, optim, load_path, precision=precision)
    else:
        _load_unsharded_optim_checkpoint(model, optim, load_path, precision=precision)


def _load_sharded_optim_checkpoint(
    model: ComposerModel,
    optim: torch.optim.Optimizer,
    load_path: str,
    precision: str = 'fp32',
    strict: bool = False,
    device_mesh: Optional[DeviceMesh]=None,
    planner: Optional[LoadPlanner] = None,
):
    optim_state_dict = get_optim_state_dict(model, optim, sharded_state_dict=True)
    optim_state_dict = download_and_load_sharded_state_dict(load_path, device_mesh, optim_state_dict, planner)
    for param_key, param_state_dict in optim_state_dict['state'].items():
        optim_state_dict['state'][param_key] = _cast_state_dict_to_precision(param_state_dict, precision)


def _load_unsharded_optim_checkpoint(
    model: ComposerModel,
    optim: torch.optim.Optimizer,
    load_path: str,
    precision: str = 'fp32',
    strict: bool = False
):
    load_path_is_remote = is_uri(load_path)
    download_dir_context = tempfile.TemporaryDirectory if load_path_is_remote else contextlib.nullcontext
    with download_dir_context() as download_dir:
        file_path = load_path
        if load_path_is_remote:
            filename = Path(load_path).name
            file_path = os.path.join(download_dir, filename)
            download_monolithic_checkpoint(load_path, file_path)

        if dist.get_global_rank() == 0:
            optim_state_dict = _torch_load_with_validation(file_path)
            for param_key, param_state_dict in optim_state_dict['state'].items():
                optim_state_dict['state'][param_key] = _cast_state_dict_to_precision(param_state_dict, precision)
            optim.load_state_dict(optim_state_dict, strict=strict)


def _preprocess_local_load_path(load_path: str):
    if os.path.islink(load_path):
        load_path = os.path.join(os.path.dirname(load_path), os.readlink(load_path))
    if os.path.exists(load_path):
        if not os.path.isdir(load_path):
            raise ValueError(f'load_path must be a directory when using sharded state dict. Got {load_path}')
    else:
        raise FileNotFoundError(f'{load_path} not found!')
    return load_path


def _load_model_state_dict_with_fsdp_context_manager(model: nn.Module,
                                                     model_state_dict: dict,
                                                     sharded_state_dict: bool,
                                                     strict: bool):
    from torch.distributed.fsdp.fully_sharded_data_parallel import ShardedStateDictConfig, StateDictType, FullStateDictConfig
    state_dict_type = StateDictType.SHARDED_STATE_DICT if sharded_state_dict else StateDictType.FULL_STATE_DICT
    state_dict_config = ShardedStateDictConfig(offload_to_cpu=True) if sharded_state_dict else FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, state_dict_type=state_dict_type, 
                                state_dict_config=state_dict_config):
        missing_keys, unexpected_keys = model.load_state_dict(
                            model_state_dict,
                            strict=strict,
                        )
    
    if len(missing_keys) > 0:
        log.warning(f"Found these missing keys in the checkpoint: {', '.join(missing_keys)}")
    if len(unexpected_keys) > 0:
        log.warning(f"Found these unexpected keys in the checkpoint: {', '.join(unexpected_keys)}")

def torch_set_model_state_dict(model: torch.nn.Module, model_state_dict: dict, strict: bool, options: StateDictOptions):
    try:
        set_model_state_dict(model, model_state_dict, options=StateDictOptions(strict=strict, cpu_offload=True))
    except AttributeError as e:
        # Issue: https://github.com/pytorch/pytorch/issues/127351
        if "ShardedTensor' object has no attribute 'placements'" in str(e):
            raise RuntimeError(
                textwrap.dedent(
                    'PyTorch DTensor broke backwards compatibility in older checkpoints '
                    'with ShardedTensor, which is now deprecated. To load old checkpoints, '
                    'either downgrade to PyTorch <2.3.0 or explicitly pass process groups '
                    'in the Trainer constructor via '
                    "`parallelism_config = {'fsdp': {'process_group': 'mod1'}}`. We can "
                    'provide assistance at https://github.com/mosaicml/composer/issues.',
                ),
            ) from e
        else:
            raise e

def download_and_load_sharded_state_dict(load_path: str, device_mesh: Optional[DeviceMesh], state_dict: dict, planner: Optional[LoadPlanner] = None):
    load_path_is_remote = is_uri(load_path)
    download_dir_context = tempfile.TemporaryDirectory if load_path_is_remote else contextlib.nullcontext
    with download_dir_context() as download_dir:
        if load_path_is_remote:
            storage_reader = DistCPObjectStoreReader(source_path=load_path,
                                                     destination_path=download_dir,
                                                     device_mesh=device_mesh)
        else:
            load_path = _preprocess_local_load_path(load_path)
            storage_reader = FileSystemReaderWithValidation(load_path)

            try:
                storage_reader.read_metadata()
            except AttributeError as e:
                if '_MEM_FORMAT_ENCODING' in str(e):
                    raise ValueError(
                        'Unable to read checkpoint metadata. The checkpoint was likely saved with a '
                        'newer version of torch. Upgrade your torch version to load this checkpoint.',
                    )
                else:
                    raise

        log.debug(f'Loading sharded state dict from {load_path} into memory.')
        if version.parse(torch.__version__) < version.parse('2.2.0'):
            DCP.load_state_dict(state_dict=state_dict,
                                storage_reader=storage_reader,
                                planner=planner)
        else:
            DCP.load(state_dict=state_dict,
                     storage_reader=storage_reader,
                     planner=planner)
    return state_dict


def load_resumption_checkpoint(state: State, load_path: str):
    resumption_state_dict = pickle.load(_ensure_valid_checkpoint(load_path))
    if 'dataset_state' in resumption_state_dict:
        state._load_dataset_state(resumption_state_dict['dataset_state'])
    if 'timestamp' in resumption_state_dict:
        state.timestamp.load_state_dict(resumption_state_dict['timestamp'])
    
    
    if 'schedulers' in resumption_state_dict:
        for scheduler in state.schedulers:
            fqn = type(scheduler).__qualname__
            scheduler.load_state_dict(resumption_state_dict['schedulers'][fqn])
    
    if 'algorithms' in resumption_state_dict:
        for algorithm in state.algorithms:
            fqn = type(algorithm).__qualname__
            algorithm.load_state_dict(resumption_state_dict['algorithms'][fqn])

    if 'callbacks' in resumption_state_dict:
        for callback in state.callbacks:
            fqn = type(callback).__qualname__
            callback.load_state_dict(resumption_state_dict['callbacks'][fqn])

    if 'scaler' in resumption_state_dict:
        fqn = type(state.scaler).__qualname__
        state.scaler.load_state_dict(resumption_state_dict['scaler'][fqn])
    

