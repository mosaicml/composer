from typing import List, Optional, Sequence, Union
from composer.core.state import State
import logging
from composer.model.model import ComposerModel
from composer.utils.checkpoint import safe_torch_load, _torch_load_with_validation, FileSystemReaderWithValidation, DistCPObjectStoreReader
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
from composer.utils import reproducibility


log = logging.getLogger(__name__)

@dataclass
class CheckpointLoadOptions:
    load_path: str # Can be local path, uri, or hf name
    load_model: bool = True
    load_optimizer: bool = False
    load_resumption_keys: bool = False # dataset_state, timestamp,
    # Specific key-level loading configs
        # e.g.'model.layers.transformer.0.w1.weight'
    do_not_load_keys: Optional[List[str]] = None
    only_load_keys: Optional[List[str]] = None
    sharded_checkpoint: bool = False # TODO: Auto-detect sharded
    sharded_model: bool = False
    strict: bool = False



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
    if load_options.load_model:
        load_model_checkpoint(state.model, load_options.load_path, include_keys=load_options.only_load_keys, 
                             ignore_keys=load_options.do_not_load_keys, strict=load_options.strict)
    if load_options.load_optimizer:
        load_optim_checkpoint(state.model, state.optimizers[0], load_options.load_path, sharded_checkpoint=load_options.sharded_checkpoint)

    if load_options.load_resumption_keys:
        load_resumption_checkpoint(state)

    if load_options.sharded_model and not load_options.sharded_checkpoint:
        assert state.fsdp_config is not None
        log.info('Wrapping model with FSDP after loading model_state.')
        with reproducibility.seed_context(state.rank_zero_seed):
            from composer.distributed import prepare_fsdp_module
            prepare_fsdp_module(
                state.model,
                state.optimizers,
                state.fsdp_config,
                state.precision,
                state.device,
                state.auto_microbatching,
            )



def load_model_checkpoint(
    model: ComposerModel,
    load_path: str,
    include_keys: Optional[Union[str, Sequence[str]]] = None,
    ignore_keys: Optional[Union[str, Sequence[str]]] = None,
    strict: bool = False,
    device_mesh: Optional[DeviceMesh]=None,
    sharded_checkpoint: bool = False,
    planner: Optional[LoadPlanner] = None
):
    """
    Load a a model checkpoint from the specified path into the model.

    Args:
        model (ComposerModel): The model to load the checkpoint into.
        load_path (str): The local path to the checkpoint.
        include_keys (Optional[Union[str, Sequence[str]]]): The keys to include from the model checkpoint. Note that if ignore_keys is specified, then this argument will be ignored.
        ignore_keys (Optional[Union[str, Sequence[str]]]): The keys to ignore from the model checkpoint. Note that if include_keys is specified, then this argument will be
        device_mesh (Optional[DeviceMesh]): The device mesh to use for loading the checkpoint.
        sharded_checkpoint (bool): If the checkpoint is sharded or not.
        planner (Optional[LoadPlanner]): The planner to use for loading the checkpoint.
    """
    
    if include_keys is not None or ignore_keys is not None:
        strict = True
    if sharded_checkpoint:
        load_sharded_model_checkpoint(model, load_path, include_keys=include_keys, 
                                      ignore_keys=ignore_keys, strict=strict, 
                                      device_mesh=device_mesh, planner=planner)
    else:
        load_unsharded_model_checkpoint(model, load_path, include_keys=include_keys, ignore_keys=ignore_keys, strict=strict)
            
  
def load_unsharded_model_checkpoint(model: ComposerModel,
                                    load_path: str,
                                    include_keys: Optional[Union[str, Sequence[str]]] = None,
                                    ignore_keys: Optional[Union[str, Sequence[str]]] = None,
                                    strict: bool = False):
    if is_uri(load_path):
        pass # Download the checkpoint first.

    if include_keys is not None and ignore_keys is not None:
        raise ValueError("Only one of include_keys or ignore_keys can be specified.")
    
    # TODO: Implement loading unsharded model checkpoint into sharded model using FSDP.summon.
    if _is_model_fsdp(model):
        raise NotImplementedError("Model is sharded but checkpoint is not sharded. Please pass in a model not wrapped with FSDP.")
    
    if dist.get_global_rank() == 0:
        model_state_dict = _torch_load_with_validation(load_path)
        if include_keys is not None:
            model_state_dict = _extract_keys_from_state_dict(model_state_dict, include_keys)
        if ignore_keys is not None:
            model_state_dict = _remove_keys_from_state_dict(model_state_dict, ignore_keys)
        # TODO: raise warning for unknown or missing keys.
        model.load_state_dict(model_state_dict, strict=strict)
    

def _preprocess_local_load_path(load_path: str):
    if os.path.islink(load_path):
        load_path = os.path.join(os.path.dirname(load_path), os.readlink(load_path))
    if os.path.exists(load_path):
        if not os.path.isdir(load_path):
            raise ValueError(f'load_path must be a directory when using sharded state dict. Got {load_path}')
    else:
        raise FileNotFoundError(f'{load_path} not found!')
    return load_path


def load_sharded_model_checkpoint(
    model: ComposerModel,
    load_path: str,
    include_keys: Optional[Union[str, Sequence[str]]] = None,
    ignore_keys: Optional[Union[str, Sequence[str]]] = None,
    strict: bool = False,
    device_mesh: Optional[DeviceMesh]=None,
    planner: Optional[LoadPlanner] = None,
):
    # TODO: Implement loading sharded model checkpoint into unsharded model.
    if not _is_model_fsdp(model):
        raise NotImplementedError("Model is not sharded but checkpoint is sharded. Please pass in a model wrapped with FSDP.")
    
    if include_keys is not None and ignore_keys is not None:
        raise ValueError("Only one of include_keys or ignore_keys can be specified.")
    
    model_state_dict = get_model_state_dict(model, sharded_state_dict=True, include_keys=include_keys, ignore_keys=ignore_keys)

    model_state_dict = download_and_load_sharded_state_dict(load_path, device_mesh, model_state_dict, planner)

    # TODO: raise warning for unknown or missing keys.
    log.debug(f'Loading sharded state dict into model.')
    if version.parse(torch.__version__) >= version.parse('2.3.0') and dist.is_initialized():
        torch_set_model_state_dict(model, model_state_dict, strict=strict, options=StateDictOptions(cpu_offload=True))
    else:
        _load_model_state_dict_with_fsdp_context_manager(model, model_state_dict, strict)
 

def _load_model_state_dict_with_fsdp_context_manager(model: nn.Module,
                                                     model_state_dict: dict,
                                                     strict: bool):
    """Get the model state dict with the FSDP context manager.

    Args:
        model: The model to get the state dict from.
        sharded: Whether the model state dict should be sharded or not. If True, every rank returns the state dict of its shards.
            If False, then rank 0 returns the state dict of the entire model.

    Returns:
        The state dict of the model.
    """
    from torch.distributed.fsdp.fully_sharded_data_parallel import ShardedStateDictConfig, StateDictType
    with FSDP.state_dict_type(model, state_dict_type=StateDictType.SHARDED_STATE_DICT, 
                                state_dict_config=ShardedStateDictConfig(offload_to_cpu=True)):
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

def load_optim_checkpoint(
    model: ComposerModel,
    optim: torch.optim.Optimizer,
    load_path: str,
    sharded_checkpoint: bool
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
        load_sharded_optim_checkpoint(model, optim, load_path)
    else:
        load_unsharded_optim_checkpoint(model, optim, load_path)

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

def load_sharded_optim_checkpoint(
    model: ComposerModel,
    optim: torch.optim.Optimizer,
    load_path: str,
    strict: bool = False,
    device_mesh: Optional[DeviceMesh]=None,
    planner: Optional[LoadPlanner] = None,
):
    optim_state_dict = get_optim_state_dict(model, optim, sharded_state_dict=True)
    optim_state_dict = download_and_load_sharded_state_dict(load_path, device_mesh, optim_state_dict, planner)


def load_unsharded_optim_checkpoint(
    model: ComposerModel,
    optim: torch.optim.Optimizer,
    load_path: str,
    strict: bool = False
):
    if is_uri(load_path):
        pass # Download the checkpoint first.

    if dist.get_global_rank() == 0:
        optim_state_dict = _torch_load_with_validation(load_path)
        optim.load_state_dict(optim_state_dict, strict=strict)



def load_resumption_checkpoint():
    pass
