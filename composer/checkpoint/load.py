# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""API for loading checkpoints."""

import contextlib
import logging
import os
import pickle
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import torch
import torch.distributed.checkpoint as DCP
from packaging import version
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import Optimizer

from composer.checkpoint.download import download_and_extract_symlink, download_monolithic_checkpoint
from composer.checkpoint.state_dict import (
    _cast_state_dict_to_precision,
    _extract_keys_from_state_dict,
    _is_model_fsdp,
    _remove_keys_from_state_dict,
    get_model_state_dict,
    get_optim_state_dict,
)
from composer.core.state import State
from composer.distributed import prepare_fsdp_module
from composer.models import ComposerModel
from composer.utils import FSDPConfig, dist, reproducibility
from composer.utils.checkpoint import (
    DistCPObjectStoreReader,
    FileSystemReaderWithValidation,
    _ensure_valid_checkpoint,
    _torch_load_with_validation,
)
from composer.utils.file_helpers import is_uri

log = logging.getLogger(__name__)


@dataclass
class CheckpointLoadOptions:
    """Options for loading a checkpoint.

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
    load_model: bool = True
    load_optimizer: bool = False
    load_resumption_state: bool = False  # dataset_state, timestamp,
    # Specific key-level loading configs
    # e.g.'model.layers.transformer.0.w1.weight'
    include_keys: Optional[Union[str, Sequence[str]]] = None
    ignore_keys: Optional[Union[str, Sequence[str]]] = None
    sharded_checkpoint: bool = False  # TODO: Auto-detect sharded
    shard_as_needed_during_load: bool = False
    strict: bool = True
    precision: str = 'fp32'
    cpu_offload: bool = True
    load_planner: Optional[Any] = None
    fsdp_config: Optional[Union[FSDPConfig, dict]] = None
    device_mesh: Optional[Any] = None
    seed: Optional[int] = 42


def load_checkpoint(
    load_path: str,
    load_options: Optional[Union[CheckpointLoadOptions, dict]],
    model: Optional[Union[ComposerModel, nn.Module]] = None,
    optim: Optional[Optimizer] = None,
    state: Optional[State] = None,
    model_child_path: Optional[str] = None,
    optim_child_path: Optional[str] = None,
    resumption_filename: str = 'resumption.pkl',
):
    """Optionally download and load a checkpoint according to the options into specified state.

    Args:
        load_path (str): The path or uri to the checkpoint to load or symlink to the path/uri. If URI specified then files will be downloaded first.
        load_options (CheckpointLoadOptions): The options for loading the checkpoint.
        model (Optional[ComposerModel]): The model to load the checkpoint into. If not provided, the model from the state will be used.
        optim (Optional[Optimizer]): The optimizer to load the checkpoint into. If not provided, the optimizer from the state will be used.
        state (Optional[State]): The state to load the checkpoint. If not provided, the model and optimizer must be provided.
        model_child_path (Optional[str]): The path to the model checkpoint within the load_path.
            E.g. if your checkpoints are unsharded and saved in load_path/my_cool_model/my_model.pt, set model_child_path='my_cool_model/my_model.pt'. If not specified, 'model/model.pt' is used.
            if your checkpoints are sharded and saved in load_path/my_sharded_model/ then set model_child_path='my_sharded_model'. If not specified, 'model' is used.
        optim_child_path (Optional[str]): The path to the optimizer checkpoint within the load_path.
            e.g. if your checkpoints are unsharded and saved in load_path/my_cool_optim/my_optim.pt, set optim_child_path='my_cool_optim/my_optim.pt'. If not specified, 'optim/optim.pt' is used.
            if your checkpoints are sharded and saved in load_path/my_sharded_optim/ then set optim_child_path='my_sharded_optim'. If not specified, 'optim' is used.
        resumption_filename (Optional[str]): The filename of the resumption state file. Default is 'resumption.pkl'.
    """
    if load_options is None:
        load_options = CheckpointLoadOptions()
    if isinstance(load_options, dict):
        load_options = CheckpointLoadOptions(**load_options)

    if load_options.load_model:
        if model is None:
            if state is None:
                raise ValueError('Model or state must be provided if loading model checkpoint.')
            model = state.model
        if model_child_path is None:
            model_child_path = 'model' if load_options.sharded_checkpoint else 'model/model.pt'

    if load_options.load_optimizer:
        if optim is None:
            if state is None:
                raise ValueError('Optimizer or state must be provided if loading optimizer checkpoint.')
            optim = state.optimizers[0]
        if optim_child_path is None:
            optim_child_path = 'optim' if load_options.sharded_checkpoint else 'optim/optim.pt'

    if load_options.load_resumption_state:
        if state is None:
            raise ValueError('State must be provided if loading resumption state.')
        load_resumption_checkpoint(state, os.path.join(load_path, resumption_filename))

    if load_options.load_model:
        assert model is not None
        assert model_child_path is not None
        model_load_path = os.path.join(load_path, model_child_path)
        if state is not None:
            state.automicrobatch_fsdp_hook_handles, state.fsdp_modules = load_model_checkpoint(
                model,
                load_path=model_load_path,
                load_options=load_options,
            )
        else:
            load_model_checkpoint(
                model,
                load_path=model_load_path,
                load_options=load_options,
            )

    if load_options.load_optimizer:
        assert optim_child_path is not None
        optim_load_path = os.path.join(load_path, optim_child_path)
        assert model is not None
        assert optim is not None
        load_optim_checkpoint(
            model,
            optim,
            optim_load_path,
            load_options=load_options,
        )


def load_model_checkpoint(
    model: Union[ComposerModel, nn.Module],
    load_path: Optional[str] = None,
    load_options: Optional[Union[CheckpointLoadOptions, dict]] = None,
    seed: int = 42,
) -> tuple[list, dict]:
    """Load a a model checkpoint from the specified path into the model.

    Args:
        model (Union[ComposerModel, nn.Module]): The model to load the checkpoint into.
        load_path (Optional[str]): The path to the checkpoint to load.
        load_options (Optional[Union[CheckpointLoadOptions, Dict]]): The options for loading the checkpoint.
        seed (int): The seed to use for sharding the model and optimizer.
    """
    if load_options is None:
        load_options = CheckpointLoadOptions()
    elif isinstance(load_options, dict):
        load_options = CheckpointLoadOptions(**load_options)
    if load_options.include_keys is not None and load_options.ignore_keys is not None:
        raise ValueError('Only one of include_keys or ignore_keys can be specified.')

    if load_options.include_keys is not None or load_options.ignore_keys is not None:
        load_options.strict = False

    automicrobatch_fsdp_hook_handles = []
    fsdp_modules = {}

    if load_options.sharded_checkpoint:
        if not _is_model_fsdp(model):
            if load_options.shard_as_needed_during_load:
                automicrobatch_fsdp_hook_handles, fsdp_modules = _shard_with_fsdp(
                    model,
                    fsdp_config=load_options.fsdp_config,
                    precision=load_options.precision,
                    seed=seed,
                )
            else:
                raise ValueError(
                    'Model is not sharded but checkpoint is sharded. Please pass in a model wrapped with FSDP or set shard_model_as_needed_during_load to True.',
                )
        assert load_path is not None
        _load_sharded_model_checkpoint(model, load_path=load_path, load_options=load_options)
    else:
        if dist.get_global_rank() == 0:
            assert load_path is not None
            _load_unsharded_model_checkpoint(model, load_path=load_path, load_options=load_options)

        if load_options.shard_as_needed_during_load and not _is_model_fsdp(model):
            if load_options.fsdp_config is None:
                load_options.fsdp_config = {'sync_module_states': True}
            elif isinstance(load_options.fsdp_config, dict):
                load_options.fsdp_config.update({'sync_module_states': True})
            else:
                load_options.fsdp_config.sync_module_states = True
            automicrobatch_fsdp_hook_handles, fsdp_modules = _shard_with_fsdp(
                model,
                fsdp_config=load_options.fsdp_config,
                precision=load_options.precision,
                seed=seed,
            )
    return automicrobatch_fsdp_hook_handles, fsdp_modules


def _shard_with_fsdp(
    model: Union[ComposerModel, nn.Module],
    optimizer: Optional[Optimizer] = None,
    fsdp_config: Optional[Union[FSDPConfig, dict]] = None,
    precision: Optional[str] = None,
    seed: int = 42,
) -> tuple[list, dict]:
    if fsdp_config is None:
        fsdp_config = FSDPConfig()
    if isinstance(fsdp_config, dict):
        fsdp_config = FSDPConfig(**fsdp_config)
    with reproducibility.seed_context(seed):
        automicrobatch_fsdp_hook_handles, fsdp_modules = prepare_fsdp_module(
            model,
            optimizers=optimizer,
            fsdp_config=fsdp_config,
            precision=precision,
        )
    return automicrobatch_fsdp_hook_handles, fsdp_modules


def _load_sharded_model_checkpoint(
    model: Union[ComposerModel, nn.Module],
    load_path: str,
    load_options: Optional[Union[CheckpointLoadOptions, dict]],
):
    if load_options is None:
        load_options = CheckpointLoadOptions()
    elif isinstance(load_options, dict):
        load_options = CheckpointLoadOptions(**load_options)
    model_state_dict = get_model_state_dict(
        model,
        sharded_state_dict=True,
        include_keys=load_options.include_keys,
        ignore_keys=load_options.ignore_keys,
    )
    model_state_dict = download_and_load_sharded_state_dict(
        load_path,
        load_options.device_mesh,
        model_state_dict,
        load_options.load_planner,
    )
    model_state_dict = _cast_state_dict_to_precision(model_state_dict, load_options.precision)
    # TODO: raise warning for unknown or missing keys.
    log.debug(f'Loading sharded state dict into model.')
    if version.parse(torch.__version__) >= version.parse('2.3.0') and dist.is_initialized():
        torch_set_model_state_dict(
            model,
            model_state_dict,
            strict=load_options.strict,
            cpu_offload=load_options.cpu_offload,
            sharded_state_dict=True,
        )
    else:
        _load_model_state_dict_with_fsdp_context_manager(
            model,
            model_state_dict,
            sharded_state_dict=True,
            strict=load_options.strict,
        )


def _load_unsharded_model_checkpoint(
    model: Union[ComposerModel, nn.Module],
    load_path: str,
    load_options: Optional[Union[CheckpointLoadOptions, dict]] = None,
):
    if dist.get_global_rank() != 0:
        return
    if load_options is None:
        load_options = CheckpointLoadOptions()
    elif isinstance(load_options, dict):
        load_options = CheckpointLoadOptions(**load_options)
    if _is_model_fsdp(model):
        raise ValueError(
            'Model is sharded but checkpoint is not sharded. Please pass in a model not wrapped with FSDP.',
        )
    load_path_is_remote = is_uri(load_path)
    download_dir_context = tempfile.TemporaryDirectory if load_path_is_remote else contextlib.nullcontext
    with download_dir_context() as download_dir:
        file_path = load_path
        if load_path_is_remote:
            filename = Path(load_path).name
            assert download_dir is not None
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
    model: Union[ComposerModel, nn.Module],
    optim: torch.optim.Optimizer,
    load_path: str,
    load_options: Optional[Union[CheckpointLoadOptions, dict]] = None,
):
    """Load an optimizer checkpoint from the specified path into the optimizer.

    Args:
        model (Union[ComposerModel, nn.Module]): The model to load the optimizer checkpoint into.
        optim (torch.optim.Optimizer): The optimizer to load the checkpoint into.
        load_path (Optional[str]): The path to the checkpoint to load.
        load_options (Optional[Union[CheckpointLoadOptions, Dict]]): The options for loading the checkpoint.
    """
    if load_options is None:
        load_options = CheckpointLoadOptions()
    elif isinstance(load_options, dict):
        load_options = CheckpointLoadOptions(**load_options)
    if load_options.sharded_checkpoint and not _is_model_fsdp(model):
        raise ValueError(
            textwrap.dedent(
                """Model/Optim is not sharded but checkpoint is sharded.
                                         Please  pass in a model/optim wrapped with FSDP.""",
            ),
        )
    if not load_options.sharded_checkpoint and not _is_model_fsdp(model) and load_options.shard_as_needed_during_load:
        raise ValueError(
            textwrap.dedent(
                """Neither model nor optim nor checkpoint is sharded, but shard_as_needed_during_load is set.
                                         Sharding the optim after load is not supported. please set shard_as_needed_during_load to False.
                                         """,
            ),
        )

    if load_options.shard_as_needed_during_load and _is_model_fsdp(model):
        raise NotImplementedError(
            'Loading optimizers for models that have been sharded during load (either before or after loading in the model checkpoint) is not currently supported.',
        )

    if load_options.sharded_checkpoint:
        _load_sharded_optim_checkpoint(model=model, optim=optim, load_path=load_path, load_options=load_options)
    else:
        _load_unsharded_optim_checkpoint(
            model=model,
            optim=optim,
            load_path=load_path,
            precision=load_options.precision,
        )


def _load_sharded_optim_checkpoint(
    model: Union[ComposerModel, nn.Module],
    optim: torch.optim.Optimizer,
    load_path: str,
    load_options: CheckpointLoadOptions,
):
    if not _is_model_fsdp(model):
        raise ValueError(
            'Model is not sharded but checkpoint is sharded. Please either use load_model_checkpoint(model, load_path, optimizer, shard_as_needed_during_load=True) or pass in a model wrapped with FSDP.',
        )
    # if not _is_optimizer_sharded(optim):
    #     raise ValueError("Optimizer is not sharded but checkpoint is sharded. Please pass in a sharded optimizer by passing a sharded model's parameters to an optimizer constructor.")
    optim_state_dict = get_optim_state_dict(model, optim, sharded_state_dict=True)
    optim_state_dict = download_and_load_sharded_state_dict(
        load_path=load_path,
        device_mesh=load_options.device_mesh,
        state_dict=optim_state_dict,
        load_planner=load_options.load_planner,
    )
    for param_key, param_state_dict in optim_state_dict['state'].items():
        optim_state_dict['state'][param_key] = _cast_state_dict_to_precision(param_state_dict, load_options.precision)
    if version.parse(torch.__version__) >= version.parse('2.3.0') and dist.is_initialized():
        torch_set_optimizer_state_dict(
            model,
            optim,
            optim_state_dict,
            strict=load_options.strict,
            cpu_offload=load_options.cpu_offload,
            sharded_state_dict=True,
        )
    else:
        _load_optim_state_dict_with_fsdp_context_manager(
            model,
            optim,
            optim_state_dict,
            sharded_state_dict=True,
            strict=load_options.strict,
        )


def _load_unsharded_optim_checkpoint(
    model: Union[ComposerModel, nn.Module],
    optim: torch.optim.Optimizer,
    load_path: str,
    precision: str = 'fp32',
):
    if dist.get_global_rank() != 0:
        return
    if _is_model_fsdp(model):
        raise ValueError('Model is sharded, but checkpoint is not sharded. Please pass in a model unwrapped from FSDP.')
    # if _is_optimizer_sharded(optim):
    #     raise ValueError("Optimizer is sharded, but checkpoint is not sharded. Please pass in an unsharded optimizer by passing an unsharded model's parameters to an optimizer constructor.")
    load_path_is_remote = is_uri(load_path)
    download_dir_context = tempfile.TemporaryDirectory if load_path_is_remote else contextlib.nullcontext
    with download_dir_context() as download_dir:
        file_path = load_path
        if load_path_is_remote:
            filename = Path(load_path).name
            assert download_dir is not None
            file_path = os.path.join(download_dir, filename)
            download_monolithic_checkpoint(load_path, file_path)

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


def _load_model_state_dict_with_fsdp_context_manager(
    model: nn.Module,
    model_state_dict: dict,
    sharded_state_dict: bool,
    strict: bool,
):
    from torch.distributed.fsdp.fully_sharded_data_parallel import (
        FullStateDictConfig,
        ShardedStateDictConfig,
        StateDictType,
    )
    state_dict_type = StateDictType.SHARDED_STATE_DICT if sharded_state_dict else StateDictType.FULL_STATE_DICT
    state_dict_config = ShardedStateDictConfig(offload_to_cpu=True,) if sharded_state_dict else FullStateDictConfig(
        offload_to_cpu=True,
        rank0_only=True,
    )
    with FSDP.state_dict_type(model, state_dict_type=state_dict_type, state_dict_config=state_dict_config):
        missing_keys, unexpected_keys = model.load_state_dict(
            model_state_dict,
            strict=strict,
        )

    if len(missing_keys) > 0:
        log.warning(f"Found these missing keys in the checkpoint: {', '.join(missing_keys)}")
    if len(unexpected_keys) > 0:
        log.warning(f"Found these unexpected keys in the checkpoint: {', '.join(unexpected_keys)}")


def _load_optim_state_dict_with_fsdp_context_manager(
    model: nn.Module,
    optim,
    optim_state_dict: dict,
    sharded_state_dict: bool,
    strict: bool,
):
    from torch.distributed.fsdp.fully_sharded_data_parallel import (
        FullOptimStateDictConfig,
        ShardedOptimStateDictConfig,
        StateDictType,
    )
    state_dict_type = StateDictType.SHARDED_STATE_DICT if sharded_state_dict else StateDictType.FULL_STATE_DICT
    optim_state_dict_config = optim_state_dict_config = ShardedOptimStateDictConfig(
    ) if sharded_state_dict else FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, state_dict_type=state_dict_type, optim_state_dict_config=optim_state_dict_config):
        optim_state_dict = FSDP.optim_state_dict_to_load(  #  type: ignore
                                optim_state_dict=optim_state_dict, model=model, optim=optim,
                            )
    assert optim_state_dict is not None
    optim.load_state_dict(optim_state_dict)


def torch_set_model_state_dict(
    model: torch.nn.Module,
    model_state_dict: dict,
    strict: bool,
    cpu_offload: bool,
    sharded_state_dict: bool = True,
):
    """Set the model state dict with the given options.

    Args:
        model (torch.nn.Module): The model to set the state dict to.
        model_state_dict (dict): The model state dict to set.
        strict (bool): Whether to set the state dict strictly.
        cpu_offload (bool): Whether to offload the state dict to CPU before setting.
        sharded_state_dict (bool): Whether the state dict is sharded or not.
    """
    if version.parse(torch.__version__) >= version.parse('2.3.0') and dist.is_initialized():
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
        try:
            set_model_state_dict(
                model,
                model_state_dict,
                options=StateDictOptions(
                    strict=strict,
                    cpu_offload=cpu_offload,
                    full_state_dict=not sharded_state_dict,
                ),
            )
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


def torch_set_optimizer_state_dict(
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    optim_state_dict: dict,
    strict: bool,
    cpu_offload: bool,
    sharded_state_dict: bool = True,
):
    """Set the optimizer state dict with the given options.

    Args:
        model (torch.nn.Module): The model to set the state dict to.
        optim (torch.optim.Optimizer): The optimizer to set the state dict to.
        optim_state_dict (dict): The optimizer state dict to set.
        strict (bool): Whether to set the state dict strictly.
        cpu_offload (bool): Whether to offload the state dict to CPU before setting.
        sharded_state_dict (bool): Whether the state dict is sharded or not.
    """
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


def download_and_load_sharded_state_dict(
    load_path: str,
    device_mesh: Optional[Any],
    state_dict: dict,
    load_planner: Optional[Any] = None,
):
    """A helper function to download and load a sharded state dict.

    Args:
        load_path (str): The path to the checkpoint to load.
        device_mesh (Optional[Any]): The device mesh to use for loading the checkpoint.
        state_dict (dict): The state dict to load.
        load_planner (Optional[Any]): The load planner to use for loading the checkpoint.
    """
    load_path_is_remote = is_uri(load_path)
    download_dir_context = tempfile.TemporaryDirectory if load_path_is_remote else contextlib.nullcontext
    with download_dir_context() as download_dir:
        if load_path_is_remote:
            local_rank0_index = dist.get_global_rank() - dist.get_local_rank()
            rank0_download_tempdir = str(dist.all_gather_object(download_dir)[local_rank0_index])
            if load_path.endswith('.symlink'):
                load_path = download_and_extract_symlink(load_path)
            storage_reader = DistCPObjectStoreReader(
                source_path=load_path,
                destination_path=str(Path(rank0_download_tempdir) / Path('checkpoints')),
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
            DCP.load_state_dict(state_dict=state_dict, storage_reader=storage_reader, planner=load_planner)
        else:
            DCP.load(state_dict=state_dict, storage_reader=storage_reader, planner=load_planner)
    return state_dict


def load_resumption_checkpoint(state: State, load_path: str):
    """Load the resumption state from the specified path into the state.

    Args:
        state (State): The state to load the resumption state into.
        load_path (str): The path to the resumption state to load.

    """
    load_path = str(_ensure_valid_checkpoint(load_path))
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
        assert state.scaler is not None
        state.scaler.load_state_dict(resumption_state_dict['scaler'][fqn])

    state.rank_zero_seed = resumption_state_dict['rank_zero_seed']
    return resumption_state_dict['rng']
