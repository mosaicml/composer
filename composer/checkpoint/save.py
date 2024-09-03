# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Useful functions for saving state dicts to disk."""

import json
import logging
import os
import pickle
import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import torch
import torch.distributed.checkpoint as DCP
from packaging import version
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._tensor import DTensor

from composer.checkpoint.state_dict import (
    get_metadata_state_dict,
    get_model_state_dict,
    get_optim_state_dict,
    get_resumption_state_dict,
)
from composer.core import State, Time
from composer.devices import Device
from composer.models import ComposerModel
from composer.utils import dist
from composer.utils.checkpoint import _TORCH_DISTRIBUTED_CHECKPOINTS_FILENAME, _write_checkpoint_file
from composer.utils.file_helpers import format_name_with_dist_and_time

log = logging.getLogger(__name__)

MODEL_CHECKPOINT_DIRECTORY_NAME = 'model'
MONOLITHIC_MODEL_CHECKPOINT_FILENAME = 'model.pt'
OPTIM_CHECKPOINT_DIRECTORY_NAME = 'optim'
OPTIM_MONO_CHECKPOINT_FILENAME = 'optim.pt'
METADATA_CHECKPOINT_FILENAME = 'composer_metadata.json'
RESUMPTION_CHECKPOINT_FILENAME = 'resumption.pkl'


@dataclass
class CheckpointSaveOptions:
    """Options for saving a checkpoint to disk.

    Args:
        destination_dir (str): The directory to save the checkpoint to.
        save_frequency (Union[str, int, Time]): The frequency to save the checkpoint.
            If '1ep', the checkpoint will be saved after each epoch.
            If '1ba', the checkpoint will be saved after each batch.
            If an int, the checkpoint will be saved after that many epochs.
        dir_prefix (str): The prefix to use for the directory name. Can include {epoch} and {batch}.
        overwrite (bool): Whether to overwrite the checkpoint if it already exists.
        save_model (bool): Whether to save the model.
        save_optimizer (bool): Whether to save the optimizer.
        save_resumption_state (bool): Whether to save the resumption state.
        num_checkpoints_to_keep (int): The number of checkpoints to keep.
            If -1, all checkpoints will be kept.
        save_format (str): The format to save the model in. 'pt', which is the standard pytorch serializarion, is the only option for now.
        sharded_checkpoint (bool): Whether to save the model as a sharded checkpoint.
        precision (str): The precision to save the model in. One of 'bf16', 'fp32', 'fp16', 'fp64'.
        include_keys (Optional[Union[str, Sequence[str]]]): Keys to include in the saved model.
        ignore_keys (Optional[Union[str, Sequence[str]]]): Keys to ignore in the saved model.
    """
    save_frequency: Union[str, int, Time] = '1ep'
    dir_prefix: str = 'ep{epoch}-ba{batch}'
    overwrite: bool = False
    save_model: bool = True
    save_optimizer: bool = True
    save_resumption_state: bool = True
    num_checkpoints_to_keep: int = -1
    save_format: str = 'pt'
    sharded_checkpoint: bool = False
    precision: str = 'fp32'
    include_keys: Optional[Union[str, Sequence[str]]] = None
    ignore_keys: Optional[Union[str, Sequence[str]]] = None


def save_checkpoint_to_disk(
    destination_dir: str,
    state: State,
    options: Optional[Union[CheckpointSaveOptions, dict]] = None,
):
    """Saves a checkpoint to disk.

    Args:
        state (State): The state to save.
        options (Optional[Union[CheckpointSaveOptions, Dict]]): The options for saving the checkpoint.
            If None, destination_dir must be provided.
        destination_dir (Optional[str]): The directory to save the checkpoint to.
            If options is provided, this will overwrite save_path.
    """
    if options is None:
        options = CheckpointSaveOptions()
    else:
        if isinstance(options, dict):
            options = CheckpointSaveOptions(**options)

    save_path = os.path.join(destination_dir, options.dir_prefix)
    save_path = format_name_with_dist_and_time(save_path, state.run_name, state.timestamp)
    os.makedirs(save_path, exist_ok=True)
    if options.save_model:
        model_save_path = (
            os.path.join(save_path, MODEL_CHECKPOINT_DIRECTORY_NAME) if options.sharded_checkpoint else
            os.path.join(save_path, MODEL_CHECKPOINT_DIRECTORY_NAME, MONOLITHIC_MODEL_CHECKPOINT_FILENAME)
        )
        save_model_to_disk(
            state.model,
            model_save_path,
            options.sharded_checkpoint,
            options.precision,
            options.include_keys,
            options.ignore_keys,
            options.overwrite,
            options.save_format,
        )
    if options.save_optimizer:
        optim_save_path = os.path.join(save_path,
                                       OPTIM_CHECKPOINT_DIRECTORY_NAME) if options.sharded_checkpoint else os.path.join(
                                           save_path,
                                           OPTIM_CHECKPOINT_DIRECTORY_NAME,
                                           OPTIM_MONO_CHECKPOINT_FILENAME,
                                       )
        optimizer = state.optimizers[0]
        save_optim_to_disk(
            state.model,
            optimizer,
            optim_save_path,
            options.sharded_checkpoint,
            options.precision,
            options.overwrite,
            options.save_format,
        )
    if options.save_resumption_state:
        resumption_save_path = os.path.join(save_path, RESUMPTION_CHECKPOINT_FILENAME)
        save_resumption_state_to_disk(state, resumption_save_path)

    md_save_path = os.path.join(save_path, METADATA_CHECKPOINT_FILENAME)
    save_composer_metadata_to_disk(
        md_save_path,
        state.model,
        options.sharded_checkpoint,
        options.precision,
        state.device,
        state.device_train_microbatch_size,
    )
    return save_path


def save_model_to_disk(
    model: Union[ComposerModel, torch.nn.Module],
    destination_dir: str,
    sharded_checkpoint: bool = False,
    precision: str = 'fp32',
    include_keys: Optional[Union[str, Sequence[str]]] = None,
    ignore_keys: Optional[Union[str, Sequence[str]]] = None,
    overwrite: bool = False,
    save_format: str = 'pt',  # or hf, safetensor
) -> Optional[str]:
    """Saves a model to disk.

    Args:
        model (Union[ComposerModel, torch.nn.Module]): The model to save.
        destination_dir (str): The directory to save the model to.
            Model will be saved as distination_dir/model.pt if sharded_checkpoint is False,
            otherwise all shards will be saved as destination_dir/__<rank>_0.distcp along with a metadata file (destination_dir/.metadata).
        sharded_checkpoint (bool): Whether to save the model as a sharded checkpoint.
        precision (str): The precision to save the model in. One of 'bf16', 'fp32', 'fp16', 'fp64'.
        include_keys (Optional[Union[str, Sequence[str]]]): Keys to include in the saved model.
        ignore_keys (Optional[Union[str, Sequence[str]]]): Keys to ignore in the saved model.
        overwrite (bool): If True, the file will be overwritten if it exists.
        save_format (str): The format to save the model in. One of 'pt', 'hf', or 'safetensor'.

    Returns:
        str: The full path to the saved model.
    """
    if save_format != 'pt':
        raise NotImplementedError(
            f"Saving checkpoint in format {save_format} is not supported. Please choose from ['pt'].",
        )
    model_state_dict = get_model_state_dict(
        model,
        sharded_checkpoint,
        precision,
        include_keys,
        ignore_keys,
    )
    saved_path = save_state_dict_to_disk(
        state_dict=model_state_dict,
        destination_file_path=destination_dir,
        overwrite=overwrite,
        save_format=save_format,
    )
    return saved_path


def save_optim_to_disk(
    model: Union[ComposerModel, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    destination_dir: str,
    sharded_checkpoint: bool = False,
    precision: str = 'fp32',
    overwrite: bool = False,
    save_format: str = 'pt',
) -> Optional[str]:
    """Saves an optimizer to disk.

    Args:
        model (Union[ComposerModel, torch.nn.Module]): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        destination_dir (str): The directory to save the optimizer to.
            Optimizer will be saved as destination_dir if sharded_checkpoint is False,
            otherwise all shards will be saved as destination_dir/__<rank>_0.distcp along with a metadata file (destination_dir/.metadata).
        sharded_checkpoint (bool): Whether to save the optimizer as a sharded checkpoint.
        precision (str): The precision to save the optimizer in. One of 'bf16', 'fp32', 'fp16', 'fp64'.
        overwrite (bool): If True, the file will be overwritten if it exists.
        save_format (str): The format to save the optimizer in. One of 'pt'.
    """
    optim_state_dict = get_optim_state_dict(
        model,
        optimizer,
        sharded_state_dict=sharded_checkpoint,
        precision=precision,
    )
    saved_path = save_state_dict_to_disk(
        state_dict=optim_state_dict,
        destination_file_path=destination_dir,
        overwrite=overwrite,
        save_format=save_format,
    )

    return saved_path


def save_composer_metadata_to_disk(
    destination_file_path: str,
    model: Optional[Union[ComposerModel, torch.nn.Module]] = None,
    sharded_state_dict: Optional[bool] = None,
    precision: Optional[Union[str, torch.dtype]] = None,
    device: Optional[Device] = None,
    device_train_microbatch_size: Optional[Union[int, float]] = None,
):
    """Saves metadata about the model to disk.

    Args:
        destination_file_path (str): The path to save the metadata to.
        model (Optional[Union[ComposerModel, torch.nn.Module]]): The model to save metadata about.
        sharded_state_dict (Optional[bool]): Whether the model is sharded.
        precision (Optional[Union[str, torch.dtype]]): The precision of the model.
        device (Optional[Device]): The device the model is on.
        device_train_microbatch_size (Optional[Union[int, float]]): The device train microbatch size.
    """
    md_dict = get_metadata_state_dict(
        model,
        sharded_state_dict,
        precision,
        device,
        device_train_microbatch_size,
    )
    os.makedirs(str(Path(destination_file_path).parent), exist_ok=True)

    if dist.get_global_rank() == 0:
        with open(destination_file_path, 'w') as f:
            json.dump(md_dict, f, indent=4)
    return destination_file_path


def save_resumption_state_to_disk(
    state: State,
    destination_file_path: str,
):
    """Saves the resumption state to disk.

    Args:
        state (State): The state to save.
        destination_file_path (str): The path to save the resumption state to.
    """
    resumption_state_dict = get_resumption_state_dict(state)
    os.makedirs(Path(destination_file_path).parent, exist_ok=True)
    with open(destination_file_path, 'wb') as f:
        pickle.dump(resumption_state_dict, f)
    return destination_file_path


from composer.utils import dist
from composer.utils.checkpoint import _TORCH_DISTRIBUTED_CHECKPOINTS_FILENAME, _write_checkpoint_file

log = logging.getLogger(__name__)


def save_state_dict_to_disk(
    state_dict: dict[str, Any],
    destination_file_path: str,
    overwrite: bool = False,
    save_format: str = 'pt',  # or hf, safetensor
) -> Optional[str]:
    """Saves a state dict to local disk.

    Args:
        state_dict (Dict[str,Any]): The state dict to save.
        destination_file_path (str): The path to save the state dict to. If sharded,
          this should be the pth to a directory. Otherwise, it should be a path to a file.
        overwrite (bool): If True, the file will be overwritten if it exists.
        save_format (str): The format to save the state dict in. One of 'pt', 'hf', or 'safetensor'.

    Returns:
        str: The full path to the saved state dict if (sharded is false and rank 0) or if sharded is true, otherwise None.
    """
    if state_dict == {}:
        return None
    if is_state_dict_sharded(state_dict):
        path_saved = _save_sharded_state_dict_to_disk(state_dict, destination_file_path, overwrite, save_format)
    else:
        if dist.get_global_rank() == 0:
            path_saved = _save_full_state_dict_to_disk(state_dict, destination_file_path, overwrite, save_format)
        else:
            path_saved = None

    return path_saved


def _save_sharded_state_dict_to_disk(
    state_dict: dict[str, Any],
    destination_file_path: str,
    overwrite: bool = False,
    save_format: str = 'pt',
) -> Optional[str]:

    if save_format != 'pt':
        raise NotImplementedError(
            f"Saving sharded state dict to disk in format {save_format} is not supported. Please choose from ['pt'].",
        )

    if state_dict == {}:
        return None

    # If user specifies filename instead of directory suffixes, strip them and warn
    if len(Path(destination_file_path).suffixes) > 0:
        stripped_path = _strip_suffixes(destination_file_path)
        warnings.warn(
            textwrap.dedent(
                f"""Sharded checkpoints require a directory path not a file path:
            {destination_file_path} will have its extensions stripped and checkpoints will be saved in {stripped_path}
            as {stripped_path}/{_TORCH_DISTRIBUTED_CHECKPOINTS_FILENAME}""",
            ),
        )
        destination_file_path = stripped_path

    # Wait for all ranks to get here before checking if the directory exists.
    dist.barrier()
    if dist.get_global_rank() == 0 and not overwrite and os.path.exists(destination_file_path):
        raise ValueError(f'Directory {destination_file_path} already exists. Set overwrite=True to overwrite it.')

    log.debug(
        f'Starting saving of sharded state dict to {destination_file_path}/{_TORCH_DISTRIBUTED_CHECKPOINTS_FILENAME}',
    )

    # For 2.3.0 and above you can use checkpoint_id, but this version works the best for all versions
    # of torch (and makes pyright happier) that we support, so we use it for now.
    if version.parse(torch.__version__) < version.parse('2.2.0'):
        DCP.save_state_dict(state_dict=state_dict, storage_writer=DCP.FileSystemWriter(destination_file_path))
    else:
        DCP.save(state_dict=state_dict, storage_writer=DCP.FileSystemWriter(destination_file_path))

    log.debug(
        f'Finished saving of sharded state dict to {destination_file_path}/{_TORCH_DISTRIBUTED_CHECKPOINTS_FILENAME}',
    )
    return destination_file_path + '/' + _TORCH_DISTRIBUTED_CHECKPOINTS_FILENAME


def _save_full_state_dict_to_disk(
    state_dict: dict[str, Any],
    destination_file_path: str,
    overwrite: bool = False,
    save_format: str = 'pt',  # or hf, safetensor
) -> Optional[str]:

    if save_format != 'pt':
        raise NotImplementedError(
            f"Saving full state dict to disk in format {save_format} is not supported. Please choose from ['pt'].",
        )

    if not overwrite and os.path.exists(destination_file_path):
        raise ValueError(f'File {destination_file_path} already exists. Set overwrite=True to overwrite it.')

    if dist.get_global_rank() == 0:
        os.makedirs(os.path.dirname(destination_file_path), exist_ok=True)
        _write_checkpoint_file(state_dict=state_dict, filename=destination_file_path)
        return destination_file_path
    return None


def is_state_dict_sharded(state_dict: dict[str, Any]) -> bool:
    """Determines if the state dict is sharded.

    Args:
        state_dict (Dict[str, Any]): The state dict to check.

    Returns:
        bool: Whether the state dict is sharded.
    """
    for value in state_dict.values():
        if isinstance(value, ShardedTensor) or isinstance(value, DTensor):
            return True
        elif isinstance(value, dict):
            is_sharded = is_state_dict_sharded(value)
            if is_sharded:
                return True
    return False


def _strip_suffixes(path: Union[str, Path]) -> str:
    path = Path(path)
    for _ in path.suffixes:
        path = path.with_suffix('')

    return str(path)
