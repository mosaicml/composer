# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

def get_model_state_dict(model: ComposerModel, sharded: bool, precision: str) -> Dict[str, Any]:
    """Generate the state dict of the model.

    Args:
        model: The model to get the state dict from.
        sharded: Whether the model is sharded or not. If True, every rank returns the state dict of its shards.
            If False, then rank 0 returns the state dict of the entire model.
        precision: The precision of the model.

    Returns:
        The state dict of the model.
    """


def get_optim_state_dict(optimizer: torch.optim.Optimizer, sharded: bool, precision: str) -> Dict[str, Any]:
    """Generate the state dict of the optimizer.

    Args:
        optimizer: The optimizer to get the state dict from.
        sharded: Whether the optimizer is sharded or not. If True, every rank returns the state dict of its shards.
            If False, then rank 0 returns the state dict of the entire optimizer.
        precision: The precision of the optimizer.

    Returns:
        The state dict of the optimizer.
    """


def get_resumption_state_dict() -> Dict[str, Any]:
    """Generate the state dict for any objects needed for resumption.

    This includes:
        * timestamp
        * scheduler
        * dataset_state
        * scaler
        * rank_zero_seed
        * callbacks
        * algorithms?

    Returns:
        The state dict containing the objects needed for resumption.
    """

def get_checkpoint_metadata() -> Dict[str, Any]:
    """Generate the metadata and integrations for a training run.

    This includes:
        * composer version
        * torch version
        * device
        * precision
        * hf model metadata

    Returns:
        The state dict containing the metadata and integrations for a training run.
    """


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

    Args:
        state_dict: The state dict to save.
        destination_dir: The directory to save the state dict to. If None, then the current working directory is used.
        filename: The name of the file to save the state dict to.
        overwrite: Whether to overwrite the file if it already exists.
        save_format: The format to save the state dict in. Can be 'pt', 'hf', or 'safetensor'.
        async_save: Whether to save the state dict asynchronously.
        sharded: Whether the state dict is sharded.

    Returns:
        The path to the dir where checkpoints that was saved.
    """

def _save_sharded_state_dict_to_disk(
        state_dict: Dict[str, Any],
        destination_dir: str = None,
        overwrite: bool = False,
        save_format: str = 'pt', # or hf, safetensor
        async_save: bool = False,
):
    """Saves a sharded state dict to local disk by calling torch.distributed.checkpoint.save

    No filename needed for sharded as torch hardcodes the filename based on rank.
    If async_save is True, then the save is done asynchronously either using
    torch.distributed.checkpoint.async_save or a separate process.

    """

def _save_full_state_dict_to_disk():
