from typing import List, Optional, Sequence, Union
from composer.core.state import State
from composer.model.model import ComposerModel
from composer.checkpoint.state_dict import _is_model_fsdp

import torch
# def load_checkpoint(
#     state: State,
#     load_options: CheckpointLoadOptions
#     ):
#     """
#     Optionally download and load  a checkpoint according to the options into specified state.

#     Args:
#         state (State): The State object containing the model, optim, timestamp, scheduler, etc.
#         load_options (CheckpointLoadOptions): The options to use for loading the checkpoint.
#     """

def load_model_checkpoint(
    model: ComposerModel,
    load_path: str,
    include_keys: Optional[Union[str, Sequence[str]]] = None,
    ignore_keys: Optional[Union[str, Sequence[str]]] = None, 
    sharded: bool=False
):
    """
    Load a a model checkpoint from the specified path into the model.

    Args:
        model (ComposerModel): The model to load the checkpoint into.
        load_path (str): The path or uri to the checkpoint to load or symlink to the path/uri. If URI specified then files will be downloaded first.
        include_keys (Optional[Union[str, Sequence[str]]]): The keys to include from the model checkpoint. Note that if ignore_keys is specified, then this argument will be ignored.
        ignore_keys (Optional[Union[str, Sequence[str]]]): The keys to ignore from the model checkpoint. Note that if include_keys is specified, then this argument will be
        sharded (bool): If the checkpoint is sharded or not.
    """
    is_model_sharded = _is_model_fsdp(model)
    is_checkpoint_sharded =  _is_checkpoint_sharded(load_path)

def _is_checkpoint_sharded(load_path: str) -> bool:
    """
    Check if the checkpoint at the specified path is sharded or not.

    Args:
        load_path (str): The path to the checkpoint to check.

    Returns:
        bool: True if the checkpoint is sharded, False otherwise.
    """
    pass

def load_optim_checkpoint(
    optim: torch.optim.Optimizer,
    load_path: str,
    do_not_load_keys: Optional[List[str]]=None,
    only_load_keys: Optional[List[str]]=None, 
    sharded: bool
):
    """
    Load an optimizer checkpoint from the specified path into the optimizer.

    Args:
        optim (torch.optim.Optimizer): The optimizer to load the checkpoint into.
        load_path (str): The path or uri to the checkpoint to load or symlink to the path/uri. If URI specified then files will be downloaded first.
        do_not_load_keys (List[str]): The keys to not load from the optimizer checkpoint. Note that if only_load_keys is specified, then this argument will be ignored.
        only_load_keys (List[str]): The keys to only load from the optimizer checkpoint. Note that if do_not_load_keys is specified, then this argument will be ignored.
        sharded (bool): If the optimizer state checkpoint is sharded or not.
    """


def load_rng_checkpoint():

def load_schedulers():

def load_timestamp():