# Copyright 2021 MosaicML. All Rights Reserved.

"""Load and save checkpoints during training."""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import tarfile
import tempfile
from typing import List, Optional, Tuple, Union

import torch

from composer.core import Event, State, types
from composer.core.time import Time, TimeUnit
from composer.trainer.devices.device import Device
from composer.utils import ObjectStoreProvider, dist, reproducibility, run_directory
from composer.utils.file_retriever import GetFileNotFoundException, get_file

log = logging.getLogger(__name__)

__all__ = ["load_checkpoint", "CheckpointSaver"]

_COMPOSER_STATES_FILENAME = "composer_states.pt"
_DEEPSPEED_TAG = "deepspeed"  # always tag with the same, deterministic name. We'll rename the tarball to the appropriate name.


def _format_path_with_rank_zero(path: str) -> str:
    """Formats ``path`` with the rank zero values."""
    return path.format(
        rank=0,
        local_rank=0,
        node_rank=0,
    )


def _format_path_with_current_rank(path: str) -> str:
    """Formats ``path`` formatted with the current rank values."""
    return path.format(
        rank=dist.get_global_rank(),
        local_rank=dist.get_local_rank(),
        node_rank=dist.get_node_rank(),
    )

def _is_archive(path: str) -> bool:
    """Returns whether the path is a tar archive."""
    return any(path.endswith(x) for x in (".tar", ".tgz", ".tar.gz", ".tar.bz2", ".tar.lzma"))


def load_checkpoint(
    path_format: str,
    state: State,
    object_store: Optional[ObjectStoreProvider] = None,
    load_weights_only: bool = False,
    strict_model_weights: bool = False,
    chunk_size: int = 1_048_576,
    progress_bar: bool = True,
):
    """Load a checkpoint from a local file, URI, or cloud object store into ``state``.

    Args:
        path_format (str): The path format string to an existing checkpoint file.

            It can be a path to a file on the local disk, a URL, or if ``object_store`` is set, the object name
            for a checkpoint in a cloud bucket.

            When using `Deepspeed ZeRO <https://www.deepspeed.ai/tutorials/zero/>`_, checkpoints are shareded by rank.
            Instead of hard-coding the rank in the ``path_format``, use the following format variables:

            +------------------------+-------------------------------------------------------+
            | Variable               | Description                                           |
            +========================+=======================================================+
            | ``{rank}``             | The global rank, as returned by                       |
            |                        | :func:`~.dist.get_global_rank`.                       |
            +------------------------+-------------------------------------------------------+
            | ``{local_rank}``       | The local rank of the process, as returned by         |
            |                        | :func:`~.dist.get_local_rank`.                        |
            +------------------------+-------------------------------------------------------+
            | ``{node_rank}``        | The node rank, as returned by                         |
            |                        | :func:`~.dist.get_node_rank`.                         |
            +------------------------+-------------------------------------------------------+

            For example, suppose that checkpoints are stored in the following structure:

            .. code-block::

                my_model/rank_0/ep1.tar
                my_model/rank_1/ep1.tar
                my_model/rank_2/ep1.tar
                ...

            Then, ``path`` should be set to ``my_model/rank_{rank}/ep1.tar``, and all ranks will load the correct
            state.

        state (State): The :class:`~composer.core.state.State` to load the checkpoint into.
        object_store (ObjectStoreProvider, optional): If the ``path`` is in an object store
            (i.e. AWS S3 or Google Cloud Storage), an instance of
            :class:`~.ObjectStoreProvider` which will be used
            to retreive the checkpoint. Otherwise, if the checkpoint is a local filepath, set to ``None``.
            (default: ``None``)
        load_weights_only (bool, optional): Whether or not to only restore the model weights from the checkpoint without
            restoring the associated state. (default: ``False``)
        strict_model_weights (bool, optional): Whether or not to force that the checkpointed weights must exactly
            match the model weights. (default: ``False``)
        chunk_size (int, optional): Chunk size (in bytes) to use when downloading checkpoints.
            Ignored if the checkpoint is a local file path. (default: ``1_048_576`` bytes (1 MB))
        progress_bar (bool, optional): Whether or not to show a progress bar when downloading checkpoints.
            Ignored if the checkpoint is a local file path. (default: ``True``)

    Returns:
        Optional[List[types.StateDict]]: The RNG state dicts, indexed by global rank, if
            :attr:`load_weights_only` is not None. Otherwise, None.
    """
    # download the checkpoint to the node-local folder
    tempdir_ctx = tempfile.TemporaryDirectory() if dist.get_local_rank() == 0 else contextlib.nullcontext(None)
    with tempdir_ctx as tempdir:
        try:
            node_checkpoint_folder = _get_node_checkpoint_download_folder(tempdir)
            composer_states_filepath, extracted_checkpoint_folder, extracted_rank_n = _download_checkpoint(
                path_format=path_format,
                node_checkpoint_folder=node_checkpoint_folder,
                object_store=object_store,
                chunk_size=chunk_size,
                progress_bar=progress_bar,
            )
            rng_state_dicts = _restore_checkpoint(
                state,
                composer_states_filepath,
                extracted_rank_n,
                extracted_checkpoint_folder,
                load_weights_only=load_weights_only,
                strict_model_weights=strict_model_weights,
            )
        finally:
            # Wait for all ranks to finish restoring the checkpoint before releasing the tempdir, since tempdir can
            # be a shared resource between nodes.
            dist.barrier()

    log.info("%s loaded from %s", "Model weights" if load_weights_only else "Trainer checkpoint", path_format)
    return rng_state_dicts


def _get_node_checkpoint_download_folder(path: Optional[str]) -> str:
    """Broadcasts the ``path`` from the LOCAL rank zero to all LOCAL ranks."""
    local_rank_zero = dist.get_local_world_size() * dist.get_node_rank()
    paths = dist.all_gather_object(path)
    local_rank_zero_path = paths[local_rank_zero]
    assert local_rank_zero_path is not None, "local rank zero provides the path"
    return local_rank_zero_path


def _download_checkpoint(
    path_format: str,
    node_checkpoint_folder: str,
    object_store: Optional[ObjectStoreProvider],
    chunk_size: int,
    progress_bar: bool,
) -> Tuple[str, Optional[str], bool]:
    """Download the checkpoint stored at ``path_format``, potentially in ``object_store``, to ``node_checkpoint_folder``.

    Returns a tuple of  (``composer_states_filepath``, ``extracted_checkpoint_folder``, ``extracted_rank_n``).

    *   The ``composer_states_filepath``, is the path to the composer states, which can be passed into
        :meth:`torch.load`.
    *   The ``extracted_checkpoint_folder`` is the path to the checkpoint folder, which can be passed into
        :meth:`deepspeed.DeepSpeedEngine.load_checkpoint`.
    *   The ``extracted_rank_n`` is a boolean flag indicating whether a tarball was extracted on global
        rank greater than 0.
    """
    rank_zero_checkpoint_filepath = os.path.join(node_checkpoint_folder, "rank_0_checkpoint")
    rank_n_checkpoint_filepath = os.path.join(node_checkpoint_folder, f"rank_{dist.get_global_rank()}_checkpoint")
    extracted_checkpoint_folder = None
    extracted_rank_n = False
    if _is_archive(path_format):
        extracted_checkpoint_folder = os.path.join(node_checkpoint_folder, "checkpoint")
        composer_states_filepath = os.path.join(extracted_checkpoint_folder, _COMPOSER_STATES_FILENAME)
    else:
        # it's not an archive; it's just the composer state dict
        # and only rank zero has this file
        extracted_checkpoint_folder = None
        composer_states_filepath = rank_zero_checkpoint_filepath

    try:
        if dist.get_local_rank() == 0:
            # every NODE needs the GLOBAL rank zero checkpoint
            path = _format_path_with_rank_zero(path_format)
            get_file(destination=rank_zero_checkpoint_filepath,
                     path=path,
                     object_store=object_store,
                     chunk_size=chunk_size,
                     progress_bar=progress_bar)
            if extracted_checkpoint_folder is not None:
                try:
                    with tarfile.open(rank_zero_checkpoint_filepath) as tarball:
                        tarball.extractall(extracted_checkpoint_folder)
                except FileNotFoundError:
                    # Not re-raising the file-not-found error as that is irrelevant;
                    # the underlying issue is that the checkpoint file does not exist on the disk
                    # or could not be downloaded
                    raise RuntimeError(f"Checkpoint {path} does not exist")

        if rank_zero_checkpoint_filepath != rank_n_checkpoint_filepath:
            # every RANK needs ITS OWN checkpoint.
            # But, the  global rank zero is a special case -- these files are the same!
            assert dist.get_global_rank() != 0, "invariant violation"

            try:
                get_file(destination=rank_n_checkpoint_filepath,
                         path=_format_path_with_current_rank(path_format),
                         object_store=object_store,
                         chunk_size=chunk_size,
                         progress_bar=progress_bar)
            except GetFileNotFoundException:
                # Allowing not-found errors to be ignored as sometimes there won't be rank-local checkpoints
                # (e.g. when not using deepspeed)
                pass

            if extracted_checkpoint_folder is not None:
                try:
                    # it's an archive and needs to be extracted
                    with tarfile.open(rank_n_checkpoint_filepath) as tarball:
                        tarball.extractall(extracted_checkpoint_folder)
                        extracted_rank_n = True
                except FileNotFoundError:
                    # this will happen most of the time (i.e. whenever deepspeed
                    # is not being used) so not logging anything
                    pass

    finally:
        # Wait for all checkpoints on the node to finish downloading
        # Putting the barrier in a finally so the rank will always block on the barrier,
        # even if it has an exception.
        # Any exception will be re-raised after the barrier passes. The launcher script
        # will detect the process crash and terminate the other ranks
        dist.barrier()

    return composer_states_filepath, extracted_checkpoint_folder, extracted_rank_n


def _restore_checkpoint(
    state: State,
    composer_states_filepath: str,
    extracted_rank_n: bool,
    extracted_checkpoint_folder: Optional[str],
    load_weights_only: bool,
    strict_model_weights: bool,
) -> Optional[List[types.StateDict]]:
    """Restore a checkpoint into ``state`` and returns the rng state dicts (if ``load_weights_only`` is False)."""
    # Now, all ranks load the checkpoint that local rank zero downloaded
    state_dict = torch.load(composer_states_filepath, map_location='cpu')
    log.debug(f"Loaded checkpoint with keys {state_dict.keys()} and state keys {state_dict['state'].keys()}")

    if state.is_model_deepspeed:
        if extracted_checkpoint_folder is None:
            raise RuntimeError("Deepspeed checkpoints require a tarball, not a weights file.")

        global_rank = dist.get_global_rank()
        if global_rank > 0 and not extracted_rank_n:
            raise RuntimeError(f"Deepspeed checkpoint missing for rank {global_rank}")

        load_path_format, _ = state.deepspeed_model.load_checkpoint(
            extracted_checkpoint_folder,
            tag=_DEEPSPEED_TAG,
            load_module_only=load_weights_only,
            load_module_strict=strict_model_weights,
        )
        if load_path_format is None:
            raise RuntimeError(f"Failed to load DeepSpeed checkpoint")
    elif load_weights_only:
        state.load_model_state(state_dict['state'], strict=strict_model_weights)

    if not load_weights_only:
        state.load_state_dict(state_dict['state'])
        return state_dict['rng']


def _format_from_compression(compression: Optional[str]) -> Tuple[str, str]:
    if compression is None:
        file_extension = ".pt"
        write_mode = ""
    elif compression == "gzip":
        file_extension = ".tar.gz"
        write_mode = "w:gz"
    elif compression == "bzip2":
        file_extension = ".tar.bz2"
        write_mode = "w:bz2"
    elif compression == "lzma":
        file_extension = ".tar.lzma"
        write_mode = "w:xz"
    else:
        raise ValueError(f"Unknown compression mode: {compression}")

    return file_extension, write_mode


def _ensure_archive(file_extension: str, write_mode: str) -> Tuple[str, str]:
    if '.tar' not in file_extension:
        file_extension = '.tar'
        write_mode = 'w'
    return file_extension, write_mode


class CheckpointSaver:
    """Manager for saving trainer state to checkpoint files.

    Args:
        save_folder (str): The folder to store checkpoints in. If an absolute path is specified, then
            that path will be used. Otherwise, the ``save_folder`` will be relative
            to the folder returned by :meth:`~composer.utils.run_directory.get_run_directory`.
            If the ``save_folder`` does not exist, it will be created.
        interval (Time or str): The amount of time units to wait between checkpoints.
        compression (str, optional): Compression algorithm to run on checkpoints. Can be ``gzip``, ``bzip2``,
            ``lzma``, or ``None`` for no compression.  (default: ``None``).

    Attributes:
        checkpoint_folder (str): The folder in which checkpoints are stored. If an absolute path was specified for
            ``save_folder`` upon instantiation, then that path will be used. Otherwise, this folder is relative to
            the run directory of the training run (e.g. ``{run_directory}/{save_folder}``).
            If no run directory is provided, then by default, it is of the form
            ``runs/<timestamp>/rank_<GLOBAL_RANK>/<save_folder>`` where ``timestamp``
            is the start time of the run in iso-format, ``GLOBAL_RANK`` is the global rank of the process,
            and ``save_folder`` is the save_folder argument provided upon construction.

            .. seealso:: :mod:`~composer.utils.run_directory` for details on the format of the run directory
                and how to customize it.
        saved_checkpoints (Dict[Timestamp, List[str]]): A dictionary mapping a save timestamp
            to a list of filepaths corresponding to the checkpoints saved at that time.

            .. note:: When using DeepSpeed, the index of a filepath in each list corresponds to the
                global rank of the process that wrote that file. These filepaths are valid only on
                the global rank's node. Otherwise, when not using DeepSpeed, this list will contain
                only one filepath since only rank zero saves checkpoints.
    """

    def __init__(self, save_folder: str, interval: Union[Time, str], compression: Optional[str] = None):
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
        self._save_interval = interval
        self._file_extension, self._write_mode = _format_from_compression(compression=compression)
        self.saved_checkpoints = {}

    def should_checkpoint(self, state: State, event: Event) -> bool:
        """Given the current state and event, determine whether a checkpoint needs to be created.

        Args:
            state (State): The current State of the trainer.
            event (Event): The current Event being executed.

        Returns:
            bool: ``True`` if a checkpoint should be created based on the provided
                state and event and ``False`` otherwise.
        """

        # if we're at the end of training, ensure that we checkpoint regardless of save_event frequency
        if state.get_elapsed_duration() >= 1.0:
            return True

        if event != self.save_event:
            return False
        if self.save_event == Event.EPOCH_END:
            return int(state.timer.epoch) % int(self._save_interval) == 0
        if self.save_event == Event.BATCH_END:
            return int(state.timer.batch) % int(self._save_interval) == 0

        return False

    def save_checkpoint(self, state: State, device: Device) -> None:
        """Save the current state to a new checkpoint file.

        There are 3 cases for the format in which the checkpoint is saved:

        1. The default is to save checkpoints in a ``.pt`` file if DeepSpeed is not being used to
        train the model and there is no compression specified.

        2. If DeepSpeed is being used to train the model and there is no compression, then the checkpoint
        is stored in a ``.tar`` format because DeepSpeed saves model checkpoints
        as multiple files (one for model state, and one for optimizer state).

        3. If compression is being used, then the checkpoint is saved in the file format corresponding to the
        compression type (ex. ``gzip`` compression results in a ``.tar.gz`` file).

        Args:
            state (State): The current State of the trainer.
            device (Device): The Device in use by this process.
        """

        # Even though only rank zero saves the state dict, all states must call `.state_dict`, as individual
        # Algorithms or callbacks may perform distributed operations in their `.state_dict` implementations
        state_dict = {
            'rng': reproducibility.get_rng_state(),
            'state': state.state_dict(),
        }

        if self.save_event == Event.EPOCH_END:
            tag = f"ep{int(state.timer.epoch)}"
        elif self.save_event == Event.BATCH_END:
            tag = f"it{int(state.timer.batch)}"
        else:
            raise ValueError(f"Invalid checkpoint event: {self.save_event}")

        with tempfile.TemporaryDirectory() as tmpdir:
            if state.is_model_deepspeed:
                state.deepspeed_model.save_checkpoint(tmpdir, _DEEPSPEED_TAG)
                # ensure that deepspeed checkpoints are saved in an archive
                self._file_extension, self._write_mode = _ensure_archive(file_extension=self._file_extension,
                                                                         write_mode=self._write_mode)

            composer_states_filepath = os.path.join(tmpdir, _COMPOSER_STATES_FILENAME)
            if dist.get_global_rank() == 0:
                # only rank 0 saves checkpoints
                with open(composer_states_filepath, 'xb') as f:
                    torch.save(state_dict, f)

            checkpoint_filepath = os.path.join(self.checkpoint_folder, f'{tag}{self._file_extension}')
            if not _is_archive(checkpoint_filepath) and dist.get_global_rank() == 0:
                # move the file out of tmpdir to the user-specified location
                shutil.move(composer_states_filepath, checkpoint_filepath)

            if state.is_model_deepspeed or (_is_archive(checkpoint_filepath) and dist.get_global_rank() == 0):
                with tarfile.open(checkpoint_filepath, self._write_mode) as tarball:
                    # add files flat to the tarball with the specified compression
                    tarball.add(tmpdir, arcname="")

            timestamp = state.timer.get_timestamp()
            paths = dist.all_gather_object(checkpoint_filepath) if state.is_model_deepspeed else [checkpoint_filepath]
            self.saved_checkpoints[timestamp] = paths

            log.info(f'Trainer checkpoint saved to {checkpoint_filepath}')

        # Ensure that the non-rank 0 processes don't exit before the checkpoint is saved.
        dist.barrier()
