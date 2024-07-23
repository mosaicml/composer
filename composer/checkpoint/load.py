from typing import List, Optional, Sequence, Union, Dict
from composer.core.state import State
import logging
from composer.models import ComposerModel
from composer.utils.checkpoint import safe_torch_load, _torch_load_with_validation, FileSystemReaderWithValidation, DistCPObjectStoreReader, _ensure_valid_checkpoint
from composer.checkpoint.state_dict import _is_model_fsdp, get_model_state_dict, _extract_keys_from_state_dict, _remove_keys_from_state_dict, get_optim_state_dict
import torch.distributed.checkpoint as DCP
from composer.utils import dist
from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions, set_optimizer_state_dict
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
from torch.distributed.fsdp import FlatParameter
from torch.optim import Optimizer
from torch.distributed._tensor import DTensor
from torch.distributed.fsdp import FlatParameter
from torch.distributed._shard.sharded_tensor import ShardedTensor


log = logging.getLogger(__name__)

@dataclass
class CheckpointLoadOptions:
    load_model: bool = True
    load_optimizer: bool = False
    load_resumption_state: bool = False # dataset_state, timestamp,
    # Specific key-level loading configs
        # e.g.'model.layers.transformer.0.w1.weight'
    include_keys: Optional[Union[str, Sequence[str]]] = None
    ignore_keys: Optional[Union[str, Sequence[str]]] = None
    sharded_checkpoint: bool = False # TODO: Auto-detect sharded
    shard_as_needed_during_load: bool = False
    strict: bool = False
    precision: str = 'fp32'
    cpu_offload: bool = True
    load_planner: Optional[LoadPlanner] = None
    fsdp_config: Optional[Union[FSDPConfig, dict]] = None
    device_mesh: Optional[DeviceMesh]=None
    seed: Optional[int] = 42
    """
    Options for loading a checkpoint.

    Args:
        load_model (bool): Whether to load the model checkpoint.
        load_optimizer (bool): Whether to load the optimizer checkpoint.
        load_resumption_state (bool): Whether to load the resumption state.
        include_keys (Optional[Union[str, Sequence[str]]]): Specific keys to include in the state dict.
        ignore_keys (Optional[Union[str, Sequence[str]]]): Specific keys to ignore in the state dict.
        sharded_checkpoint (bool): Whether the checkpoint is sharded or not.
        shard_as_needed_during_load (bool): Whether to shard the model and optimizer during load. If model is unsharded and checkpoint is sharded, model will be sharded before.
            If model is unsharded and checkpoint is unsharded, model will be sharded after loading the checkpoint.
        strict (bool): Whether to load the checkpoint strictly.
        precision (str): The precision to cast the model/optim to after loading
        cpu_offload (bool): Whether to offload the state dict to CPU before loading.
        load_planner (Optional[LoadPlanner]): The load planner to use for loading the checkpoint.
        fsdp_config (Optional[Union[FSDPConfig, dict]]): The FSDP config to use for sharding the model and optimizer.
        device_mesh (Optional[DeviceMesh]): The device mesh to use for loading the checkpoint.
        seed (Optional[int]): The seed to use for sharding the model and optimizer.

    """


def load_checkpoint(
    load_path: str,
    load_options: Optional[Union[CheckpointLoadOptions, Dict]],
    model: Optional[ComposerModel]=None,
    optim: Optional[Optimizer]=None,
    state: Optional[State]=None,
    ):
    """
    Optionally download and load a checkpoint according to the options into specified state.

    Args:
        load_path (str): The path or uri to the checkpoint to load or symlink to the path/uri. If URI specified then files will be downloaded first.
        load_options (CheckpointLoadOptions): The options for loading the checkpoint.
        model (Optional[ComposerModel]): The model to load the checkpoint into. If not provided, the model from the state will be used.
        optim (Optional[Optimizer]): The optimizer to load the checkpoint into. If not provided, the optimizer from the state will be used.
        state (Optional[State]): The state to load the checkpoint. If not provided, the model and optimizer must be provided.
    """
    if isinstance(load_options, dict):
        load_options = CheckpointLoadOptions(**load_options)
    if load_options.load_model:
        if model is None:
            if state is None:
                raise ValueError("Model or state must be provided if loading model checkpoint.")
            model = state.model

    if load_options.load_optimizer:
        if optim is None:
            if state is None:
                raise ValueError("Optimizer or state must be provided if loading optimizer checkpoint.")
            optim = state.optimizers[0]
    
    if load_options.load_resumption_state and state is None:
        raise ValueError("State must be provided if loading resumption state.")
    
    load_model_and_optim_together = (load_options.load_optimizer and load_options.load_model) and (load_options.sharded_checkpoint or load_options.shard_as_needed_during_load)
    if load_model_and_optim_together:
        load_model_checkpoint(model,
                              load_path=load_path,
                              optimizer=optim,
                              load_options=load_options)


    else:
        load_model_checkpoint(model, load_path=load_path, load_options=load_options)
        if load_options.load_optimizer:
            load_optim_checkpoint(state.model, state.optimizers[0], load_path, sharded_checkpoint=load_options.sharded_checkpoint)

    if load_options.load_resumption_state:
        load_resumption_checkpoint(state)

def load_model_checkpoint(
    model: ComposerModel,
    load_path: str,
    load_options: Optional[Union[CheckpointLoadOptions, Dict]]=None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    optim_load_path: Optional[str] = None,
    seed: Optional[int] = 42,
):
    """
    Load a a model checkpoint from the specified path into the model.

    Args:
    """
    if load_options is None:
        load_options = CheckpointLoadOptions()
    elif isinstance(load_options, dict):
            load_options = CheckpointLoadOptions(**load_options)
    if load_options.include_keys is not None and load_options.ignore_keys is not None:
        raise ValueError("Only one of include_keys or ignore_keys can be specified.")
    
    if load_options.include_keys is not None or load_options.ignore_keys is not None:
        load_options.strict = True
    #assert seed == 1, f"{sharded_checkpoint=}, sharded_model={_is_model_fsdp(model)}, {shard_as_needed_during_load=}"
    if load_options.sharded_checkpoint:
        if not _is_model_fsdp(model):
            if load_options.shard_as_needed_during_load:
                _shard_model(model, 
                             fsdp_config=load_options.fsdp_config,
                             precision=load_options.precision, seed=seed)
            else:
                raise ValueError("Model is not sharded but checkpoint is sharded. Please pass in a model wrapped with FSDP or set shard_model_as_needed_during_load to True.")   
        _load_sharded_model_checkpoint(model, load_path=load_path, load_options=load_options)
    else:
        _load_unsharded_model_checkpoint(model, load_path=load_path, load_options=load_options)

        if load_options.shard_as_needed_during_load and not _is_model_fsdp(model):
            if load_options.fsdp_config is None:
                load_options.fsdp_config = {'sync_module_states': True}
            elif isinstance(load_options.fsdp_config, dict):
                load_options.fsdp_config.update({'sync_module_states': True})
            else:
                load_options.fsdp_config.sync_module_states = True
            _shard_model(model, 
                         fsdp_config=load_options.fsdp_config,
                         precision=load_options.precision, seed=seed)


def _shard_model(model: ComposerModel,
                               fsdp_config: Optional[Union[FSDPConfig, dict]] = None,
                               precision: Optional[str]=None,
                               seed: Optional[int] = 42):
    if fsdp_config is None:
        fsdp_config = FSDPConfig()
    if isinstance(fsdp_config, dict):
        fsdp_config = FSDPConfig(**fsdp_config)
    with reproducibility.seed_context(seed):
        prepare_fsdp_module(
            model,
            optimizers=None,
            fsdp_config=fsdp_config,
            precision=precision,
        )

def _load_sharded_model_checkpoint(
    model: ComposerModel,
    load_path: str,
    load_options: Optional[Union[CheckpointLoadOptions, Dict]],
):
    if load_options is None:
        load_options = CheckpointLoadOptions()
    elif isinstance(load_options, dict):
            load_options = CheckpointLoadOptions(**load_options)
    model_state_dict = get_model_state_dict(model, sharded_state_dict=True, include_keys=load_options.include_keys, ignore_keys=load_options.ignore_keys)
    model_state_dict = download_and_load_sharded_state_dict(load_path, load_options.device_mesh, model_state_dict, load_options.load_planner)
    model_state_dict = _cast_state_dict_to_precision(model_state_dict, load_options.precision)
    # TODO: raise warning for unknown or missing keys.
    log.debug(f'Loading sharded state dict into model.')
    if version.parse(torch.__version__) >= version.parse('2.3.0') and dist.is_initialized():
        torch_set_model_state_dict(model,
                                   model_state_dict,
                                   strict=load_options.strict,
                                   cpu_offload=load_options.cpu_offload,
                                   sharded_state_dict=True)
    else:
        _load_model_state_dict_with_fsdp_context_manager(model, model_state_dict, sharded_state_dict=True, strict=load_options.strict)


def _load_unsharded_model_checkpoint(model: ComposerModel,
                                    load_path: str,
                                     load_options: Optional[Union[CheckpointLoadOptions, Dict]] = None):
    if dist.get_global_rank() != 0:
        return
    if load_options is None:
        load_options = CheckpointLoadOptions()
    elif isinstance(load_options, dict):
            load_options = CheckpointLoadOptions(**load_options)
    if _is_model_fsdp(model):
        raise ValueError("Model is sharded but checkpoint is not sharded. Please pass in a model not wrapped with FSDP.")
    load_path_is_remote = is_uri(load_path)
    download_dir_context = tempfile.TemporaryDirectory if load_path_is_remote else contextlib.nullcontext
    with download_dir_context() as download_dir:
        file_path = load_path
        if load_path_is_remote:
            filename = Path(load_path).name
            file_path = os.path.join(download_dir, filename)
            download_monolithic_checkpoint(load_path, download_dir)

        model_state_dict = _torch_load_with_validation(file_path, map_location='cpu')
        model_state_dict = _cast_state_dict_to_precision(model_state_dict, load_options.precision)
        if load_options.include_keys is not None:
            model_state_dict = _extract_keys_from_state_dict(model_state_dict, load_options.include_keys)
        if load_options.ignore_keys is not None:
            model_state_dict = _remove_keys_from_state_dict(model_state_dict, load_options.ignore_keys)
        model.load_state_dict(model_state_dict, strict=load_options.strict)


def load_optim_checkpoint(
    model: ComposerModel,
    optim: torch.optim.Optimizer,
    load_path: Optional[str] = None,
    load_options: Optional[Union[CheckpointLoadOptions, Dict]]=None,
):
    """
    Load an optimizer checkpoint from the specified path into the optimizer.

    Args:
        model (ComposerModel): The model to load the optimizer checkpoint into.        
        optim (torch.optim.Optimizer): The optimizer to load the checkpoint into.
        load_path (Optional[str]): The path to the checkpoint to load.
        load_options (Optional[Union[CheckpointLoadOptions, Dict]]): The options for loading the checkpoint.
    """
    if load_options is None:
        load_options = CheckpointLoadOptions()
    elif isinstance(load_options, dict):
            load_options = CheckpointLoadOptions(**load_options)
    if load_options.sharded_checkpoint and not _is_model_fsdp(model):
        raise ValueError(textwrap.dedent("""Model/Optim is not sharded but checkpoint is sharded. 
                                         Please  pass in a model/optim wrapped with FSDP."""))
    if not load_options.sharded_checkpoint and not _is_model_fsdp(model) and load_options.shard_as_needed_during_load:
        raise ValueError(textwrap.dedent("""Neither model nor optim nor checkpoint is sharded, but shard_as_needed_during_load is set.
                                         Sharding the optim after load is not supported. please set shard_as_needed_during_load to False.
                                         """))
    if load_options.sharded_checkpoint:
        _load_sharded_optim_checkpoint(model=model, optim=optim, load_path=load_path, load_options=load_options)
    else:
        _load_unsharded_optim_checkpoint(model=model, optim=optim, load_path=load_path, precision=load_options.precision)


def _load_sharded_optim_checkpoint(
    model: ComposerModel,
    optim: torch.optim.Optimizer,
    load_path: str,
    load_options: Optional[Union[CheckpointLoadOptions, Dict]] =None,
):
    if not _is_model_fsdp(model):
        raise ValueError("Model is not sharded but checkpoint is sharded. Please either use load_model_checkpoint(model, load_path, optimizer, shard_as_needed_during_load=True) or pass in a model wrapped with FSDP.")
    # if not _is_optimizer_sharded(optim):
    #     raise ValueError("Optimizer is not sharded but checkpoint is sharded. Please pass in a sharded optimizer by passing a sharded model's parameters to an optimizer constructor.")
    optim_state_dict = get_optim_state_dict(model, optim, sharded_state_dict=True)
    optim_state_dict = download_and_load_sharded_state_dict(load_path=load_path,
                                                            device_mesh=load_options.device_mesh,
                                                            state_dict=optim_state_dict, load_planner=load_options.load_planner)
    for param_key, param_state_dict in optim_state_dict['state'].items():
        optim_state_dict['state'][param_key] = _cast_state_dict_to_precision(param_state_dict, load_options.precision)
    if version.parse(torch.__version__) >= version.parse('2.3.0') and dist.is_initialized():
        torch_set_optimizer_state_dict(model, optim, optim_state_dict, strict=load_options.strict, cpu_offload=load_options.cpu_offload, sharded_state_dict=True)
    else:
        _load_optim_state_dict_with_fsdp_context_manager(model, optim, optim_state_dict, sharded_state_dict=True, strict=load_options.strict)


def _load_unsharded_optim_checkpoint(
    model: ComposerModel,
    optim: torch.optim.Optimizer,
    load_path: str,
    precision: str = 'fp32',
):
    if _is_model_fsdp(model):
        raise ValueError("Model is sharded, but checkpoint is not sharded. Please pass in a model unwrapped from FSDP.")
    # if _is_optimizer_sharded(optim):
    #     raise ValueError("Optimizer is sharded, but checkpoint is not sharded. Please pass in an unsharded optimizer by passing an unsharded model's parameters to an optimizer constructor.")
    load_path_is_remote = is_uri(load_path)
    download_dir_context = tempfile.TemporaryDirectory if load_path_is_remote else contextlib.nullcontext
    with download_dir_context() as download_dir:
        file_path = load_path
        if load_path_is_remote:
            filename = Path(load_path).name
            file_path = os.path.join(download_dir, filename)
            download_monolithic_checkpoint(load_path, file_path)

        if dist.get_global_rank() == 0:
            optim_state_dict = _torch_load_with_validation(file_path, map_location='cpu')
            for param_key, param_state_dict in optim_state_dict['state'].items():
                optim_state_dict['state'][param_key] = _cast_state_dict_to_precision(param_state_dict, precision)
            optim.load_state_dict(optim_state_dict)


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


def _load_optim_state_dict_with_fsdp_context_manager(model: nn.Module,
                                                     optim,
                                                     optim_state_dict: dict,
                                                     sharded_state_dict: bool,
                                                     strict: bool):
    from torch.distributed.fsdp.fully_sharded_data_parallel import ShardedOptimStateDictConfig, StateDictType, FullOptimStateDictConfig
    state_dict_type = StateDictType.SHARDED_STATE_DICT if sharded_state_dict else StateDictType.FULL_STATE_DICT
    optim_state_dict_config = optim_state_dict_config = ShardedOptimStateDictConfig() if sharded_state_dict else FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, state_dict_type=state_dict_type, 
                                optim_state_dict_config=optim_state_dict_config):
        optim_state_dict = FSDP.optim_state_dict_to_load(  #  type: ignore
                                optim_state_dict=optim_state_dict, model=model, optim=optim,
                            )
    assert optim_state_dict is not None
    optim.load_state_dict(optim_state_dict)
        

def torch_set_model_state_dict(model: torch.nn.Module,
                               model_state_dict: dict,
                               strict: bool,
                               cpu_offload: bool,
                               sharded_state_dict: bool = True):
    try:
        set_model_state_dict(model, model_state_dict, options=StateDictOptions(strict=strict,
                                                                               cpu_offload=cpu_offload,
                                                                               full_state_dict=not sharded_state_dict))
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
        
def torch_set_optimizer_state_dict(model: torch.nn.Module,
                                   optim: torch.optim.Optimizer,
                                   optim_state_dict: dict,
                                   strict: bool,
                                   cpu_offload: bool,
                                   sharded_state_dict: bool = True):
        if version.parse(torch.__version__) >= version.parse('2.3.0') and dist.is_initialized():
            from torch.distributed.checkpoint.state_dict import StateDictOptions, set_optimizer_state_dict
            set_optimizer_state_dict(
                model=model,
                optimizers=optim,
                optim_state_dict=optim_state_dict,
                options=StateDictOptions(
                    full_state_dict=not sharded_state_dict,
                    strict=strict,
                    cpu_offload=cpu_offload,
                ),
            )
    

def download_and_load_sharded_state_dict(load_path: str, device_mesh: Optional[DeviceMesh], state_dict: dict, load_planner: Optional[LoadPlanner] = None):
    load_path_is_remote = is_uri(load_path)
    download_dir_context = tempfile.TemporaryDirectory if load_path_is_remote else contextlib.nullcontext
    with download_dir_context() as download_dir:
        if load_path_is_remote:
            storage_reader = DistCPObjectStoreReader(source_path=load_path,
                                                     destination_path=download_dir,
                                                     device_mesh=device_mesh,
                                                     )
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
                                planner=load_planner)
        else:
            DCP.load(state_dict=state_dict,
                     storage_reader=storage_reader,
                     planner=load_planner)
    return state_dict


def load_resumption_checkpoint(state: State, load_path: str):
    load_path = _ensure_valid_checkpoint(load_path)
    resumption_state_dict = pickle.load(open(load_path, 'rb'))
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
            algorithm.load_state_dict(dict(resumption_state_dict['algorithms'])[fqn])

    if 'callbacks' in resumption_state_dict:
        for callback in state.callbacks:
            fqn = type(callback).__qualname__
            callback.load_state_dict(dict(resumption_state_dict['callbacks'])[fqn])

    if 'scaler' in resumption_state_dict:
        fqn = type(state.scaler).__qualname__
        state.scaler.load_state_dict(resumption_state_dict['scaler'][fqn])
    
