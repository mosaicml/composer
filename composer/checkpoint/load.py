from typing import List, Optional
from composer.core.state import State
from composer.model.model import ComposerModel


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
    do_not_load_keys: Optional[List[str]]=None,
    only_load_keys: Optional[List[str]]=None, 
    sharded: bool=False
):
    """
    Load a a model checkpoint from the specified path into the model.

    Args:
        model (ComposerModel): The model to load the checkpoint into.
        load_path (str): The path or uri to the checkpoint to load or symlink to the path/uri. If URI specified then files will be downloaded first.
        load_ignore_keys (List[str]): The keys to not load from the model checkpoint. Note that if only_load_keys is specified, then this argument will be ignored.
        only_load_keys (List[str]): The keys to only load from the model checkpoint. Note that if load_ignore_keys is specified, then this argument will be ignored.
        sharded (bool): If the checkpoint is sharded or not.
    """

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