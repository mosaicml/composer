import torch
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._tensor import DTensor
from typing import Any, Dict, Optional
from composer.utils.checkpoint import _write_checkpoint_file
from composer.utils.file_helpers import format_name_with_dist, format_name_with_dist_and_time
import os 
from composer.utils import dist

def save_state_dict_to_disk(
        state_dict: Dict[str, Any],
        destination_file_path: str = None,
        overwrite: bool = False,
        save_format: str = 'pt', # or hf, safetensor
        ) -> str:
    """Saves a state dict to local disk.

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
    if state_dict == {}:
        return None
    sharded_state_dict = is_state_dict_sharded(state_dict)
        
    if sharded_state_dict:
        path_saved = _save_sharded_state_dict_to_disk(state_dict,
                                                      destination_file_path,
                                                      overwrite,
                                                      save_format)
    else:
        if dist.get_global_rank() == 0:
            path_saved = _save_full_state_dict_to_disk(state_dict,
                                                    destination_file_path,
                                                    overwrite,
                                                    save_format)
        else:
            path_saved = None
            
    return path_saved


def _save_sharded_state_dict_to_disk(
        state_dict: Dict[str,Any],
        destination_file_path: str = None,
        overwrite: bool = False, 
        save_format: str = 'pt', # or safetensor 
        hybrid_sharding: bool = False) -> str:
    pass


def _save_full_state_dict_to_disk(
    state_dict: Dict[str,Any],
    destination_file_path: str = None,
    overwrite: bool = False, 
    save_format: str = 'pt', # or hf, safetensor 
    ) -> Optional[str]:

    if save_format != 'pt':
        raise NotImplementedError(f"Saving full state dict to disk in format {save_format} is not supported.")
    
    if not overwrite and os.path.exists(destination_file_path):
        raise ValueError(f"File {destination_file_path} already exists. Set overwrite=True to overwrite it.")
    
    if dist.get_global_rank() == 0:
        _write_checkpoint_file(state_dict=state_dict,
                            filename=destination_file_path)
        return destination_file_path
    return None




def is_state_dict_sharded(state_dict: Dict[str, Any]) -> bool:
    """Determines if the state dict is sharded.

    Args:
        state_dict (Dict[str, Any]): The state dict to check.

    Returns:
        bool: Whether the state dict is sharded.
    """
    sample_value = next(iter(state_dict.values()))
    return isinstance(sample_value, ShardedTensor) or isinstance(sample_value, DTensor)
