def save_state_dict_to_disk(
        state_dict: Dict[str, Any],
        destination_dir: str = None,
        filename: str,
        overwrite: bool = False,
        save_format: str = 'pt', # or hf, safetensor
        async_save: bool = False,
        sharded: bool = False) -> str:
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
        sharded (bool): If True, the state dict is sharded across ranks and each rank saves their shard.

    Returns:
        str: The full path to the saved state dict if sharded is false and rank 0 or if sharded is true, otherwise None.
    """

def save_sharded_state_dict_to_disk(
        state_dict; Dict[str,Any],
        destination_dir: str = None,
        filename: Optional[str] = None,
        overwrite: bool = False, 
        save_format: str = 'pt', # or safetensor 
        async_save: bool = False,
 hybrid_sharding: bool) -> str:


def save_full_state_dict_to_disk(
    state_dict; Dict[str,Any],
    destination_dir: str = None, 
    filename: str,
    overwrite: bool = False, 
    save_format: str = 'pt', # or hf, safetensor 
    async_save: bool = False)) -> Optional[str]:
	if save_format == 'safetensor':
	  save_state_dict_to_safe_tensor(...)
