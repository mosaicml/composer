import torch
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._tensor import DTensor
from typing import Any, Dict, Optional
from composer.utils.checkpoint import _write_checkpoint_file
from composer.utils.file_helpers import format_name_with_dist, format_name_with_dist_and_time

def save_state_dict_to_disk(
        state_dict: Dict[str, Any],
        destination_file_path: str = None,
        overwrite: bool = False,
        save_format: str = 'pt', # or hf, safetensor
        ) -> str:
    """Saves a state dict to local disk.

       If sharded is true, then every rank saves, otherwise just rank 0 does.
       If sharded is True, calls save_sharded_state_dict_to_disk else calls    save_full_state_dict_to_disk

    Args:
        state_dict (Dict[str,Any]): The state dict to save.
        destination_dir (str): The directory to save the state dict to.
        filename (str): The name of the file to save the state dict to.
        overwrite (bool): If True, the file will be overwritten if it exists.
        save_format (str): The format to save the state dict in. One of 'pt', 'hf', or 'safetensor'.
        async_save (bool): If True, the save will be done asynchronously and the function will return with the path of where it was going to be saved
    
    Returns:
        str: The full path to the saved state dict if sharded is false and rank 0 or if sharded is true, otherwise None.
    """
    sharded_state_dict = is_state_dict_sharded(state_dict)
        
    if sharded_state_dict:
        _save_sharded_state_dict_to_disk(state_dict, destination_file_path, overwrite, save_format)
    else:
        _save_full_state_dict_to_disk(state_dict, destination_file_path, overwrite, save_format)


def _save_sharded_state_dict_to_disk(
        state_dict: Dict[str,Any],
        destination_file_path: str = None,
        overwrite: bool = False, 
        save_format: str = 'pt', # or safetensor 
        hybrid_sharding: bool) -> str:
    pass


def _save_full_state_dict_to_disk(
    state_dict: Dict[str,Any],
    destination_file_path: str = None,
    overwrite: bool = False, 
    save_format: str = 'pt', # or hf, safetensor 
    ) -> Optional[str]:

    # fill in placeholders
    _write_checkpoint_file(state_dict=state_dict,
                           filename=destination_file_path)




def is_state_dict_sharded(state_dict: Dict[str, Any]) -> bool:
    """Determines if the state dict is sharded.

    Args:
        state_dict (Dict[str, Any]): The state dict to check.

    Returns:
        bool: Whether the state dict is sharded.
    """
    sample_value = next(iter(state_dict.values()))
    return isinstance(sample_value, ShardedTensor) or isinstance(sample_value, DTensor)