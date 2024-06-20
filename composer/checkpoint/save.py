# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Useful functions for saving state dicts to disk."""

import logging
import os
import textwrap
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.distributed.checkpoint as DCP
from packaging import version
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._tensor import DTensor

from composer.utils import dist
from composer.utils.checkpoint import _TORCH_DISTRIBUTED_CHECKPOINTS_FILENAME, _write_checkpoint_file

log = logging.getLogger(__name__)


def save_state_dict_to_disk(
    state_dict: Dict[str, Any],
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
    state_dict: Dict[str, Any],
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

    return destination_file_path + '/' + _TORCH_DISTRIBUTED_CHECKPOINTS_FILENAME


def _save_full_state_dict_to_disk(
    state_dict: Dict[str, Any],
    destination_file_path: str,
    overwrite: bool = False,
    save_format: str = 'pt',  # or hf, safetensor
) -> Optional[str]:

    if save_format != 'pt':
        raise NotImplementedError(
            f"Saving sharded state dict to disk in format {save_format} is not supported. Please choose from ['pt'].",
        )

    if not overwrite and os.path.exists(destination_file_path):
        raise ValueError(f'File {destination_file_path} already exists. Set overwrite=True to overwrite it.')

    if dist.get_global_rank() == 0:
        _write_checkpoint_file(state_dict=state_dict, filename=destination_file_path)
        return destination_file_path
    return None


def is_state_dict_sharded(state_dict: Dict[str, Any]) -> bool:
    """Determines if the state dict is sharded.

    Args:
        state_dict (Dict[str, Any]): The state dict to check.

    Returns:
        bool: Whether the state dict is sharded.
    """
    for value in state_dict.values():
        if isinstance(value, ShardedTensor) or isinstance(value, DTensor):
            return True
        if isinstance(value, Dict):
            is_sharded = is_state_dict_sharded(value)
            if is_sharded:
                return True
    return False


def _strip_suffixes(path: Union[str, Path]) -> str:
    path = Path(path)
    for _ in path.suffixes:
        path = path.with_suffix('')

    return str(path)
