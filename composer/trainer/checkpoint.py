# Copyright 2021 MosaicML. All Rights Reserved.

import logging
import os
import random
import warnings
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

from composer.core import Event, State
from composer.core.types import StateDict
from composer.trainer.devices.device import Device
from composer.utils import ddp, seed_all

log = logging.getLogger(__name__)


class CheckpointLoader:
    """Manager for initializing state and restoring RNG state from existing checkpoints.

    Args:
        checkpoint_filepath (str): The path to an existing checkpoint file.
    """

    def __init__(self, checkpoint_filepath: str):
        self.state_dict = torch.load(checkpoint_filepath, map_location='cpu')

    def load_checkpoint(self, state: State):
        """Initialize state from the loaded checkpoint's data.
        """

        state.load_state_dict(self.state_dict["state"])
        self.checkpoint_rng_state = self._get_checkpoint_rng_state(state, self.state_dict["rng"])

        if "seed" in self.state_dict:
            seed_all(self.state_dict["seed"])

    def restore_checkpoint_rng_state(self, state: State, device: Device):
        """Restore the state of all RNG objects in this context from the loaded checkpoint's data.
        """

        if self.checkpoint_rng_state is None:
            return

        assert ddp.get_world_size() == len(
            self.checkpoint_rng_state['torch']
        ), f"invariant violation: if the rng state is being restored, then" \
            "the world size should be the same as in the checkpoint."

        torch.set_rng_state(self.checkpoint_rng_state['torch'][ddp.get_global_rank()])
        device.load_state_dict(self.checkpoint_rng_state['device'][ddp.get_global_rank()])
        random.setstate(self.checkpoint_rng_state['python'][ddp.get_global_rank()])
        np.random.set_state(self.checkpoint_rng_state['numpy'][ddp.get_global_rank()])

        self.checkpoint_rng_state = None

    def _get_checkpoint_rng_state(self, state: State, checkpoint_rng_state: StateDict) -> Optional[StateDict]:
        original_world_size = len(checkpoint_rng_state["torch"])
        if original_world_size == ddp.get_world_size():
            return checkpoint_rng_state
        else:
            warnings.warn(f"The checkpoint was created with world_size({original_world_size}), "
                          f"which differs from the current world_size({ddp.get_world_size()})."
                          f"RNG state will not be restored.")


class Checkpointer:
    """Manager for saving state to checkpoint files.

    Args:
        checkpoint_folder (str): The path to the folder to store checkpoints in.
        checkpoint_interval (int): The amount of time units to wait between checkpoints.
        checkpoint_interval_unit (str): The unit (`"ep"` or `"it"`) that
            `checkpoint_interval` should be measured in.
    """

    def __init__(self, checkpoint_folder: str, checkpoint_interval: int, checkpoint_interval_unit: str):
        if checkpoint_interval_unit.lower() == "ep":
            self.save_event = Event.EPOCH_END
        elif checkpoint_interval_unit.lower() == "it":
            self.save_event = Event.BATCH_END
        else:
            raise ValueError(f"Unknown checkpointing interval: {checkpoint_interval_unit}")
        self.checkpoint_folder = checkpoint_folder
        self.save_interval = checkpoint_interval

    def should_checkpoint(self, state: State, event: Event) -> bool:
        """Given the current state and event, determine whether a checkpoint needs to be created.

        Args:
            state (State): The current State of the trainer.
            event (Event): The current Event being executed.
        """

        if event != self.save_event:
            return False
        if self.save_event == Event.EPOCH_END:
            return state.epoch % self.save_interval == 0
        if self.save_event == Event.BATCH_END:
            return state.step % self.save_interval == 0
        return False

    def save_checkpoint(self, state: State, seed: int, device: Device, config: Optional[Dict[str, Any]] = None) -> None:
        """Save the current state to a a new checkpoint file.

        Args:
            state (State): The current State of the trainer.
            device (Device): The Device in use by this process.
            ddp (DDP): The DDP engine in use by this trainer.
            config (Optional[Dict[str, Any]]): The hparams used to initialize this trainer, if any.
        """

        # Store the rank0 seed, if the seed was provided on trainer init
        # then this is the same seed on all processes
        # If the seed was not provided, then the rank0 seed will be copied
        # to all processes on checkpoint resume.
        # This will be fixed by: https://github.com/mosaicml/composer/issues/12
        state_dict = {
            'state': state.state_dict(),  # should be the same across all ranks. per-rank state not stored
            'rng': self._get_rng_state(device=device),  # stored across all ranks
            'seed': seed,
        }
        if ddp.get_global_rank() != 0:
            # only rank 0 saves checkpoints
            # Need the check down here so all the DDP syncs will work for generating the checkpoint
            return

        if config:
            hparams_path = os.path.join(self.checkpoint_folder, "hparams.yaml")
            os.makedirs(self.checkpoint_folder, mode=0o775, exist_ok=True)
            config_yaml_str = yaml.dump(config)
            try:
                with open(hparams_path, "x") as f:
                    # Storing the config (ex. hparams) in a separate file so they can be modified before resuming
                    f.write(config_yaml_str)
            except FileExistsError as e:
                with open(hparams_path, "r") as f:
                    # comparing the parsed hparams to ignore whitespace and formatting differences
                    if yaml.safe_load(config_yaml_str) != yaml.safe_load(f):
                        raise RuntimeError(f"The hparams in the existing checkpoint folder {self.checkpoint_folder} "
                                           "differ from those being used in the current training run. "
                                           "Please specify a new checkpoint folder.") from e
        if self.save_event == Event.EPOCH_END:
            filename = f"ep{state.epoch}.pt"
        elif self.save_event == Event.BATCH_END:
            filename = f"it{state.step}.pt"
        else:
            raise ValueError(f"Invalid checkpoint event: {self.save_event}")
        save_file = os.path.join(self.checkpoint_folder, filename)
        with open(save_file, 'xb') as f:
            torch.save(state_dict, f)
        log.info(f'Trainer checkpoint saved to {save_file}')

    def _get_rng_state(self, device: Device) -> StateDict:
        rng_state = {
            "python": ddp.all_gather_object(random.getstate()),
            "numpy": ddp.all_gather_object(np.random.get_state()),
            "torch": ddp.all_gather_object(torch.random.get_rng_state()),
            "device": ddp.all_gather_object(device.state_dict()),
        }
        # casting the state dict as on non-rank-0, entries will be None-like
        return rng_state
