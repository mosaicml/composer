# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import contextlib
import logging
import os
import random
import tarfile
import tempfile
import textwrap
import urllib.parse
import warnings
from typing import TYPE_CHECKING, Iterator, Optional, Tuple, Union, cast

import numpy as np
import requests
import torch
import tqdm

from composer.core import Event, State
from composer.core.time import Time, TimeUnit
from composer.core.types import StateDict
from composer.trainer.deepspeed import is_module_deepspeed
from composer.trainer.devices.device import Device
from composer.utils import ObjectStoreProvider, dist, iterate_with_pbar, reproducibility, run_directory

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    import deepspeed

_COMPOSER_STATES_FILENAME = "composer_states.pt"
_DEEPSPEED_TAG = "deepspeed"  # always tag with the same, deterministic name. We'll rename the tarball to the appropriate name.


def _format_path_with_rank(path: str, rank: int):
    """Returns the path with ``{{RANK}}`` substituted with the ``rank`` argument. See the :class:`CheckpointLoader` docs
    for a description of how this is used.

    Args:
        path (str): Path to format
        rank (int): The rank
    """
    return path.format(RANK=rank)


class CheckpointLoader:
    """Manager for initializing state and restoring RNG state from existing checkpoints.

    Args:
        path (str): The template path to an existing checkpoint file.
            It can be a path to a file on local disk, a URL, or if ``object_store_hparams`` is set, the object name
            for a checkpoint in a cloud bucket.

            When using Deepspeed zero, the :class:`CheckpointSaver` shards checkpoints by rank. To load deepspeed checkpoints,
            specify ``{RANK}`` in in the ``checkpoint`` parameter, and this variable will be substituted with the global rank.
            For example, suppose that checkpoints are stored in the following structure:

            .. code-block::

                my_model/rank_0/ep1.tar
                my_model/rank_1/ep1.tar
                my_model/rank_2/ep1.tar
                ...

            Then, ``checkpoint`` should be set to ``my_model/rank_{RANK}/ep1.tar``, and all ranks will load the correct
            data.

        object_store (ObjectStoreProvider, optional): If the ``checkpoint`` is in an object store
            (i.e. AWS S3 or Google Cloud Storage), an instance of :class:`ObjectStoreProvider` which will be used
            to retreive the checkpoint. Otherwise, if the checkpoint is a local filepath, set to ``None``.
            (default: ``None``).
        load_weights_only (bool): Whether to only restore the weights from the checkpoint without
            restoring the associated state.
        strict_model_weights (bool): Whether to force that the checkpointed weights must exactly match the model
            weights.
        chunk_size (int, optional): Chunk size (in bytes) to use when downloading checkpoints. Ignored if the checkpoint
            is a local file path. (default: ``1_048_576`` bytes (1 MB))
        progress_bar (bool): Whether to show a progress bar when downloading checkpoints. Ignored if the checkpoint
            is a local file path. (default: ``True``)
    """

    def __init__(
        self,
        path: str,
        object_store: Optional[ObjectStoreProvider] = None,
        load_weights_only: bool = False,
        strict_model_weights: bool = False,
        chunk_size: int = 1_048_576,
        progress_bar: bool = True,
    ):
        checkpoint_uri_parsed = urllib.parse.urlparse(path)
        if checkpoint_uri_parsed.scheme != "":
            if object_store is not None:
                raise ValueError(
                    textwrap.dedent("""\
                        When specifying `object_store`,
                        the `checkpoint` parameter must be the key for the checkpoint in the bucket, NOT a uri."""))

        self.path = path
        self.object_store = object_store
        self.load_weights_only = load_weights_only
        self.strict_model_weights = strict_model_weights
        self.chunk_size = chunk_size
        self.progress_bar = progress_bar
        self.checkpoint_rng_state = None

    def _retrieve_checkpoint(self, rank: int, destination_filepath: str, ignore_not_found_errors: bool):
        checkpoint_name = _format_path_with_rank(self.path, rank)
        if self.object_store is not None:
            try:
                total_size_in_bytes = self.object_store.get_object_size(checkpoint_name)
            except Exception as e:
                if "ObjectDoesNotExistError" in str(e) and ignore_not_found_errors:
                    return
                raise
            self._write_to_file_with_pbar(
                destination_filepath=destination_filepath,
                total_size=total_size_in_bytes,
                iterator=self.object_store.download_object_as_stream(checkpoint_name, chunk_size=self.chunk_size),
            )
            return
        checkpoint_uri_parsed = urllib.parse.urlparse(checkpoint_name)

        if checkpoint_uri_parsed.scheme == "":
            # assume it's a local file
            os.symlink(os.path.abspath(checkpoint_name), destination_filepath)
            return
        # it's a url
        with requests.get(checkpoint_name, stream=True) as r:
            r.raise_for_status()
            total_size_in_bytes = r.headers.get('content-length')
            if total_size_in_bytes is not None:
                total_size_in_bytes = int(total_size_in_bytes)
            self._write_to_file_with_pbar(destination_filepath,
                                          total_size=total_size_in_bytes,
                                          iterator=r.iter_content(self.chunk_size))

    def _write_to_file_with_pbar(self, destination_filepath: str, total_size: Optional[int], iterator: Iterator[bytes]):
        if self.progress_bar:
            desc = f"Downloading {self.path}"
            if len(desc) > 60:
                desc = desc[:42] + "..." + desc[-15:]
            pbar = tqdm.tqdm(desc=desc, total=total_size, unit='iB', unit_scale=True)
        else:
            pbar = None
        with open(destination_filepath, "wb") as fp:
            for chunk in iterate_with_pbar(iterator, pbar):
                fp.write(chunk)

    def _get_node_checkpoint_download_folder(self, path: Optional[str]) -> str:
        """Broadcasts the path from the local rank zero to all ranks."""
        local_rank_zero = dist.get_local_world_size() * dist.get_node_rank()
        paths = dist.all_gather_object(path)
        local_rank_zero_path = paths[local_rank_zero]
        assert local_rank_zero_path is not None, "local rank zero provides the path"
        return local_rank_zero_path

    def _download_checkpoint(self, node_checkpoint_folder: str) -> Tuple[str, Optional[str]]:
        """Download the checkpoint to ``node_checkpoint_folder``

        Args:
            node_checkpoint_folder (str): The folder to which to download the checkpoint

        Returns:
            Tuple[str, Optional[str]]: A tuple of ``composer_checkpoint_filepath``, ``extracted_checkpoint_folder``
                The ``composer_checkpoint_filepath``, is the path to the composer states, which can be passed into
                :meth:`torch.load`.

                The ``extracted_checkpoint_folder`` is the path to the checkpoint folder, which can be passed into
                :meth:`deepspeed.DeepSpeedEngine.load_checkpoint`.
        """
        checkpoint_archive_name = self.path.split(os.path.sep)[-1]
        rank_zero_checkpoint_archive_name = "rank_0." + _format_path_with_rank(checkpoint_archive_name, 0)
        rank_n_checkpoint_archive_name = f"rank_{dist.get_global_rank()}." + _format_path_with_rank(
            checkpoint_archive_name, dist.get_global_rank())
        rank_zero_checkpoint_archive_filepath = os.path.join(node_checkpoint_folder, rank_zero_checkpoint_archive_name)
        rank_n_checkpoint_archive_filepath = os.path.join(node_checkpoint_folder, rank_n_checkpoint_archive_name)
        extracted_checkpoint_folder = None
        if rank_zero_checkpoint_archive_filepath.endswith(".pt"):
            # it's not an archive; it's just the composer state dict
            # and only rank zero has this file
            extracted_checkpoint_folder = None
            composer_checkpoint_filepath = rank_zero_checkpoint_archive_filepath
        else:
            extracted_checkpoint_folder = os.path.join(node_checkpoint_folder, "checkpoint")
            composer_checkpoint_filepath = os.path.join(extracted_checkpoint_folder, _COMPOSER_STATES_FILENAME)

        try:
            if dist.get_local_rank() == 0:
                # every NODE needs the GLOBAL rank zero checkpoint
                self._retrieve_checkpoint(destination_filepath=rank_zero_checkpoint_archive_filepath,
                                          rank=dist.get_global_rank(),
                                          ignore_not_found_errors=False)
                if extracted_checkpoint_folder is not None:
                    try:
                        with tarfile.open(rank_zero_checkpoint_archive_filepath) as tarball:
                            tarball.extractall(extracted_checkpoint_folder)
                    except FileNotFoundError:
                        checkpoint_name = _format_path_with_rank(self.path, dist.get_global_rank())
                        # Not re-raising the file-not-found error as that is irrelevant;
                        # the underlying issue is that the checkpoint file does not exist on the disk
                        # or could not be downloaded
                        raise RuntimeError(f"Checkpoint {checkpoint_name} does not exist")

            if rank_zero_checkpoint_archive_filepath != rank_n_checkpoint_archive_filepath:
                # every RANK needs ITS OWN checkpoint.
                # But, the  global rank zero is a special case -- these files are the same!
                assert dist.get_global_rank() != 0, "invariant violation"

                # Allowing not-found errors to be ignored as sometimes there won't be rank-local checkpoints
                # (e.g. when not using deepspeed)
                self._retrieve_checkpoint(destination_filepath=rank_n_checkpoint_archive_filepath,
                                          rank=dist.get_global_rank(),
                                          ignore_not_found_errors=True)

                if extracted_checkpoint_folder is not None:
                    # it's an archive and needs to be extracted
                    with tarfile.open(rank_n_checkpoint_archive_filepath) as tarball:
                        tarball.extractall(extracted_checkpoint_folder)
        finally:
            # Wait for all checkpoints on the node to finish downloading
            # Putting the barrier in a finally so the rank will always block on the barrier,
            # even if it has an exception.
            # Any exception will be re-raised after the barrier passes. The launcher script
            # will detect the process crash and terminate the other ranks
            dist.barrier()

        return composer_checkpoint_filepath, extracted_checkpoint_folder

    def _restore_checkpoint(self, state: State, composer_checkpoint_filepath: str,
                            extracted_checkpoint_folder: Optional[str]) -> Optional[int]:
        """Restore a checkpoint into ``state``.

        Args:
            state (State): The state to load the checkpoint into
            composer_checkpoint_filepath (str): The filepath to the moasic states, which is passed into
                :meth:`torch.load`
            extracted_checkpoint_folder (Optional[str]): The path to the checkpoint folder, which is passed into
                :meth:`deepspeed.DeepSpeedEngine.load_checkpoint`.

        Returns:
            Optional[int]: The seed that was loaded from the checkpoint if it exists otherwise `None`.
        """
        # Now, all ranks load the checkpoint that local rank zero downloaded
        state_dict = torch.load(composer_checkpoint_filepath, map_location='cpu')
        log.debug(f"Loaded checkpoint with keys {state_dict.keys()} and state with keys {state_dict['state'].keys()}")
        seed_to_restore = None

        if is_module_deepspeed(state.model):
            if extracted_checkpoint_folder is None:
                raise RuntimeError("Deepspeed checkpoints require a tarball, not a weights file.")

            load_path, _ = cast("deepspeed.DeepSpeedEngine", state.model).load_checkpoint(
                extracted_checkpoint_folder,
                tag=_DEEPSPEED_TAG,
                load_module_only=self.load_weights_only,
                load_module_strict=self.strict_model_weights,
            )
            if load_path is None:
                raise RuntimeError(f"Failed to load DeepSpeed checkpoint from {self.path}")
        elif self.load_weights_only:
            state.load_model_state(state_dict['state'], strict=self.strict_model_weights)

        if not self.load_weights_only:
            state.load_state_dict(state_dict["state"])
            self.checkpoint_rng_state = self._get_checkpoint_rng_state(state_dict["rng"])

            if "seed" in state_dict:
                world_size = dist.get_world_size()
                checkpointed_world_size = len(state_dict["seed"])
                if world_size != checkpointed_world_size:
                    warnings.warn(
                        textwrap.dedent(f"""\
                            Current world size {world_size} does not match the checkpointed
                            world size {checkpointed_world_size}. The seed will not be restored."""))
                else:
                    seed_to_restore = state_dict["seed"][dist.get_global_rank()]
                    reproducibility.seed_all(seed_to_restore)

        return seed_to_restore

    def load_checkpoint(self, state: State):
        """Initialize state from the loaded checkpoint's data.

        Args:
            state (State): The state to load the checkpoint into.

        Returns:
            The seed that was loaded from the checkpoint if it exists otherwise `None`.
        """

        # download the checkpoint to the node-local folder
        tempdir_ctx = tempfile.TemporaryDirectory() if dist.get_local_rank() == 0 else contextlib.nullcontext(None)
        with tempdir_ctx as tempdir:
            node_checkpoint_folder = self._get_node_checkpoint_download_folder(tempdir)
            composer_checkpoint_filepath, extracted_checkpoint_folder = self._download_checkpoint(
                node_checkpoint_folder)
            seed_to_restore = self._restore_checkpoint(state, composer_checkpoint_filepath, extracted_checkpoint_folder)

        log.info(f'{"Model weights" if self.load_weights_only else "Trainer checkpoint"}'
                 f' loaded from {self.path}.')

        return seed_to_restore

    def restore_checkpoint_rng_state(self, device: Device):
        """Restore the state of all RNG objects in this context from the loaded checkpoint's data."""

        if self.checkpoint_rng_state is None:
            return

        assert dist.get_world_size() == len(self.checkpoint_rng_state['torch']), textwrap.dedent("""\
            invariant violation: if the rng state is being restored, then
            the world size should be the same as in the checkpoint.""")

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
            warnings.warn(
                textwrap.dedent(f"""\
                    The checkpoint was created with world_size({original_world_size}),
                    which differs from the current world_size({dist.get_world_size()}).
                    RNG state will not be restored."""))


class CheckpointSaver:
    """Manager for saving state to checkpoint files.

    Args:
        save_folder (str): The path to store checkpoints in.
        interval (Time or str): The amount of time units to wait between checkpoints.
        compression (str): Compression algorithm to run on checkpoints. Can be `gzip`, `bzip2`,
            `lzma`, or left blank for no compression.  (default: ``""`` for no compression).
    """

    def __init__(self, save_folder: str, interval: Union[Time, str], compression: str = ""):
        if not isinstance(interval, Time):
            interval = Time.from_timestring(interval)
        if interval.unit == TimeUnit.EPOCH:
            self.save_event = Event.EPOCH_END
        elif interval.unit == TimeUnit.BATCH:
            self.save_event = Event.BATCH_END
        else:
            raise ValueError(f"Unknown checkpointing interval: {interval.unit}. Must be epochs or batches.")
        self.checkpoint_folder = os.path.join(run_directory.get_run_directory(), save_folder)
        os.makedirs(self.checkpoint_folder, mode=0o775, exist_ok=True)
        self.save_interval = interval
        if compression == "":
            self.write_mode = "w"
            self.file_extension = ".tar"
        elif compression == "gzip":
            self.write_mode = "w:gz"
            self.file_extension = ".tar.gz"
        elif compression == "bzip2":
            self.write_mode = "w:bz2"
            self.file_extension = ".tar.bz2"
        elif compression == "lzma":
            self.write_mode = "w:xz"
            self.file_extension = ".tar.lzma"
        else:
            raise ValueError(f"Unknown encryption mode: {compression}")

    def should_checkpoint(self, state: State, event: Event) -> bool:
        """Given the current state and event, determine whether a checkpoint needs to be created.

        Args:
            state (State): The current State of the trainer.
            event (Event): The current Event being executed.
        """

        # if we're at the end of training, ensure that we checkpoint regardless of save_event frequency
        if state.get_elapsed_duration() >= 1.0:
            return True

        if event != self.save_event:
            return False
        if self.save_event == Event.EPOCH_END:
            return int(state.timer.epoch) % int(self.save_interval) == 0
        if self.save_event == Event.BATCH_END:
            return int(state.timer.batch) % int(self.save_interval) == 0

        return False

    def save_checkpoint(self, state: State, seed: int, device: Device) -> None:
        """Save the current state to a a new checkpoint file.

        Args:
            state (State): The current State of the trainer.
            seed (int): The seed used for random number generation.
            device (Device): The Device in use by this process.
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

        with tempfile.TemporaryDirectory() as tmpdir:
            if is_module_deepspeed(state.model):
                model = cast("deepspeed.DeepSpeedEngine", state.model)
                model.save_checkpoint(tmpdir, _DEEPSPEED_TAG)

            if dist.get_global_rank() == 0:
                # only rank 0 saves checkpoints

                # we add the state only on rank 0 since other processes don't have loggers to serialize
                # it should be the same across all ranks. per-rank state not stored
                state_dict['state'] = state.state_dict()

                composer_states_filepath = os.path.join(tmpdir, _COMPOSER_STATES_FILENAME)
                with open(composer_states_filepath, 'xb') as f:
                    torch.save(state_dict, f)

            checkpoint_archive_filepath = os.path.join(self.checkpoint_folder, f'{tag}{self.file_extension}')
            with tarfile.open(checkpoint_archive_filepath, self.write_mode) as tarball:
                tarball.add(tmpdir, arcname="")  # add files flat to the tarball

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
