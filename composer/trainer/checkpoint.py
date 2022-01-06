# Copyright 2021 MosaicML. All Rights Reserved.

import logging
import os
import random
import shutil
import tarfile
import tempfile
import warnings
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

from composer.core import Event, State
from composer.core.types import StateDict
from composer.trainer.devices.device import Device
from composer.utils import dist, reproducibility

log = logging.getLogger(__name__)


def get_mosaic_checkpoint_filepath(checkpoint_folder: str, checkpoint_tag: str):
    return os.path.join(checkpoint_folder, checkpoint_tag, "mosaic_states.pt")


class CheckpointLoader:
    """Manager for initializing state and restoring RNG state from existing checkpoints.

    Args:
        checkpoint_filepath (str): The path to an existing checkpoint file.
        load_weights_only (bool): Whether to only restore the weights from the checkpoint without restoring the associated state.
        strict_model_weights (bool): Whether to force that the checkpointed weights must exactly match the model weights.
    """

    def __init__(self, checkpoint_filepath: str, load_weights_only: bool = False, strict_model_weights: bool = False):
        self.checkpoint_filepath = checkpoint_filepath
        self.load_weights_only = load_weights_only
        self.strict_model_weights = strict_model_weights
        self.checkpoint_rng_state = None

    def load_checkpoint(self, state: State):
        """Initialize state from the loaded checkpoint's data.

        Args:
            state (`~composer.core.State`): The state to load the checkpoint into.

        Returns:
            The seed that was loaded from the checkpoint if it exists otherwise `None`.
        """
        seed_to_restore = None

        with tempfile.TemporaryDirectory() as checkpoint_folder:
            with tarfile.open(self.checkpoint_filepath) as tarball:
                tarball.extractall(checkpoint_folder)

            checkpoint_tag = os.listdir(checkpoint_folder)[0]
            mosaic_checkpoint_filepath = get_mosaic_checkpoint_filepath(checkpoint_folder, checkpoint_tag)

            state_dict = torch.load(mosaic_checkpoint_filepath, map_location='cpu')

            if self.load_weights_only:
                state.load_model_state(state_dict['state'], strict=self.strict_model_weights)
            else:
                state.load_state_dict(state_dict["state"])
                self.checkpoint_rng_state = self._get_checkpoint_rng_state(state_dict["rng"])

                if "seed" in state_dict:
                    world_size = dist.get_world_size()
                    checkpointed_world_size = len(state_dict["seed"])
                    if world_size != checkpointed_world_size:
                        warnings.warn(f"Current world size {world_size} does not match the checkpointed world size "
                                      f"{checkpointed_world_size}. The seed will not be restored.")
                    else:
                        seed_to_restore = state_dict["seed"][dist.get_global_rank()]
                        reproducibility.seed_all(seed_to_restore)

            try:
                import deepspeed
                if isinstance(state.model, deepspeed.DeepSpeedEngine):
                    load_path, _ = state.model.load_checkpoint(checkpoint_folder, checkpoint_tag)  # type: ignore
                    if load_path is None:
                        raise RuntimeError(f"Failed to load DeepSpeed checkpoint from {self.checkpoint_filepath}")
            except ImportError:
                pass

        return seed_to_restore

    def restore_checkpoint_rng_state(self, device: Device):
        """Restore the state of all RNG objects in this context from the loaded checkpoint's data.
        """

        if self.checkpoint_rng_state is None:
            return

        assert dist.get_world_size() == len(
            self.checkpoint_rng_state['torch']
        ), f"invariant violation: if the rng state is being restored, then" \
            "the world size should be the same as in the checkpoint."

        torch.set_rng_state(self.checkpoint_rng_state['torch'][dist.get_global_rank()])
        device.load_state_dict(self.checkpoint_rng_state['device'][dist.get_global_rank()])
        random.setstate(self.checkpoint_rng_state['python'][dist.get_global_rank()])
        np.random.set_state(self.checkpoint_rng_state['numpy'][dist.get_global_rank()])

        self.checkpoint_rng_state = None

    def _get_checkpoint_rng_state(self, checkpoint_rng_state: StateDict) -> Optional[StateDict]:
        original_world_size = len(checkpoint_rng_state["torch"])
        if original_world_size == dist.get_world_size():
            return checkpoint_rng_state
        else:
            warnings.warn(f"The checkpoint was created with world_size({original_world_size}), "
                          f"which differs from the current world_size({dist.get_world_size()})."
                          f"RNG state will not be restored.")


class CheckpointSaver:
    """Manager for saving state to checkpoint files.

    Args:
        checkpoint_folder (str): The path to store checkpoints in.
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
            seed (int): The seed used for random number generation.
            device (Device): The Device in use by this process.
            ddp (DDP): The DDP engine in use by this trainer.
            config (Optional[Dict[str, Any]]): The hparams used to initialize this trainer, if any.
        """
        state_dict = {
            'rng': self._get_rng_state(device=device),  # stored across all ranks
            'seed': dist.all_gather_object(seed),
        }

        if self.save_event == Event.EPOCH_END:
            tag = f"ep{state.epoch}"
        elif self.save_event == Event.BATCH_END:
            tag = f"it{state.step}"
        else:
            raise ValueError(f"Invalid checkpoint event: {self.save_event}")

        try:
            import deepspeed
            if isinstance(state.model, deepspeed.DeepSpeedEngine):
                state.model.save_checkpoint(self.checkpoint_folder, tag)  # type: ignore
        except ImportError:
            pass

        if dist.get_global_rank() == 0:
            # only rank 0 saves checkpoints

            # we add the state only on rank 0 since other processes don't have loggers to serialize
            state_dict['state'] = state.state_dict()  # should be the same across all ranks. per-rank state not stored

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
                            raise RuntimeError(
                                f"The hparams in the existing checkpoint folder {self.checkpoint_folder} "
                                "differ from those being used in the current training run. "
                                "Please specify a new checkpoint folder.") from e
            checkpoint_filepath = os.path.join(self.checkpoint_folder, tag)
            mosaic_states_filepath = get_mosaic_checkpoint_filepath(self.checkpoint_folder, tag)
            if not os.path.exists(checkpoint_filepath):
                os.makedirs(checkpoint_filepath)
            with open(mosaic_states_filepath, 'xb') as f:
                torch.save(state_dict, f)

            checkpoint_archive_filepath = os.path.join(self.checkpoint_folder, f'{tag}.tz')
            with tarfile.open(checkpoint_archive_filepath, "w") as tarball:
                tarball.add(checkpoint_filepath, arcname=tag)

            shutil.rmtree(checkpoint_filepath)

            log.info(f'Trainer checkpoint saved to {checkpoint_archive_filepath}')

        # Ensure that the non-rank 0 processes don't exit before the checkpoint is saved.
        dist.barrier()

    def _get_rng_state(self, device: Device) -> StateDict:
        rng_state = {
            "python": dist.all_gather_object(random.getstate()),
            "numpy": dist.all_gather_object(np.random.get_state()),
            "torch": dist.all_gather_object(torch.random.get_rng_state()),
            "device": dist.all_gather_object(device.state_dict()),
        }
        # casting the state dict as on non-rank-0, entries will be None-like
        return rng_state
