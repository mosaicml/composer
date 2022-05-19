# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Utilities for working with training checkpoints."""

from __future__ import annotations

import contextlib
import logging
import os
import pathlib
import shutil
import tarfile
import tempfile
import textwrap
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch

from composer.utils import dist, reproducibility
from composer.utils.file_helpers import (FORMAT_NAME_WITH_DIST_AND_TIME_TABLE, GetFileNotFoundException,
                                         format_name_with_dist_and_time, get_file, is_tar)
from composer.utils.object_store import ObjectStore

if TYPE_CHECKING:
    from composer.core.state import State
    from composer.loggers import LoggerDestination
    from composer.loggers.logger import Logger

log = logging.getLogger(__name__)

__all__ = ["load_checkpoint", "save_checkpoint"]

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


def _get_write_mode(name: str) -> str:
    """Get the write mode to use with :func:`tarfile.open`."""
    if name.endswith('.tar'):
        return 'w'
    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        return "w:gz"
    if name.endswith(".tar.bz2"):
        return "w:bz2"
    if name.endswith(".tar.lzma"):
        return "w:xz"
    raise ValueError(f"{name} does not end with a valid tarfile extension.")


def load_checkpoint(
    path: str,
    state: State,
    object_store: Optional[Union[ObjectStore, LoggerDestination]] = None,
    load_weights_only: bool = False,
    strict_model_weights: bool = False,
    chunk_size: int = 1_048_576,
    progress_bar: bool = True,
):
    """Load a checkpoint from a local file, URI, or cloud object store into ``state``.

    Args:
        path (str): The path format string to an existing checkpoint file.

            It can be a path to a file on the local disk, a URL, or if ``object_store`` is set, the object name
            for a checkpoint in a cloud bucket.

            When using `Deepspeed ZeRO <https://www.deepspeed.ai/tutorials/zero/>`_, checkpoints are shareded by rank.
            Instead of hard-coding the rank in the ``path``, use the following format variables:

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

                my_model/ep1-rank0.tar
                my_model/ep1-rank1.tar
                my_model/ep1-rank2.tar
                ...

            Then, ``path`` should be set to ``my_model/ep1-rank{rank}.tar``, and all ranks will load the
            correct state.

        state (State): The :class:`~composer.core.state.State` to load the checkpoint into.
        object_store (Union[ObjectStore, LoggerDestination], optional): If the ``path`` is in an object store
            (i.e. AWS S3 or Google Cloud Storage), an instance of
            :class:`~.ObjectStore` or :class:`~.LoggerDestination` which will be used
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
        Optional[List[Dict[str, Any]]]: The RNG state dicts, indexed by global rank, if
            :attr:`load_weights_only` is not None. Otherwise, None.
    """
    # download the checkpoint to the node-local folder
    tempdir_ctx = tempfile.TemporaryDirectory() if dist.get_local_rank() == 0 else contextlib.nullcontext(None)
    with tempdir_ctx as tempdir:
        try:
            node_checkpoint_folder = _get_node_checkpoint_download_folder(tempdir)
            composer_states_filepath, extracted_checkpoint_folder, extracted_rank_n = _download_checkpoint(
                path=path,
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

    log.info("%s loaded from %s", "Model weights" if load_weights_only else "Trainer checkpoint", path)
    return rng_state_dicts


def _get_node_checkpoint_download_folder(path: Optional[str]) -> str:
    """Broadcasts the ``path`` from the LOCAL rank zero to all LOCAL ranks."""
    local_rank_zero = dist.get_local_world_size() * dist.get_node_rank()
    paths = dist.all_gather_object(path)
    local_rank_zero_path = paths[local_rank_zero]
    assert local_rank_zero_path is not None, "local rank zero provides the path"
    return local_rank_zero_path


def _download_checkpoint(
    path: str,
    node_checkpoint_folder: str,
    object_store: Optional[Union[ObjectStore, LoggerDestination]],
    chunk_size: int,
    progress_bar: bool,
) -> Tuple[str, Optional[str], bool]:
    """Download the checkpoint stored at ``path``, potentially in ``object_store``, to ``node_checkpoint_folder``.

    Returns a tuple of  (``composer_states_filepath``, ``extracted_checkpoint_folder``, ``extracted_rank_n``).

    *   The ``composer_states_filepath``, is the path to the composer states, which can be passed into
        :meth:`torch.load`.
    *   The ``extracted_checkpoint_folder`` is the path to the checkpoint folder, which can be passed into
        :meth:`deepspeed.DeepSpeedEngine.load_checkpoint`.
    *   The ``extracted_rank_n`` is a boolean flag indicating whether a tarball was extracted on global
        rank greater than 0.
    """
    rank_zero_checkpoint_filepath = os.path.join(node_checkpoint_folder, "rank0_checkpoint")
    rank_n_checkpoint_filepath = os.path.join(node_checkpoint_folder, f"rank{dist.get_global_rank()}_checkpoint")
    extracted_checkpoint_folder = None
    extracted_rank_n = False
    if is_tar(path):
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
            path = _format_path_with_rank_zero(path)
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
                         path=_format_path_with_current_rank(path),
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
) -> Optional[List[Dict[str, Any]]]:
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

        load_path, _ = state.deepspeed_model.load_checkpoint(
            extracted_checkpoint_folder,
            tag=_DEEPSPEED_TAG,
            load_module_only=load_weights_only,
            load_module_strict=strict_model_weights,
        )
        if load_path is None:
            raise RuntimeError(f"Failed to load DeepSpeed checkpoint")
    elif load_weights_only:
        state.load_model_state(state_dict['state'], strict=strict_model_weights)

    if not load_weights_only:
        state.load_state_dict(state_dict['state'])
        return state_dict['rng']


def save_checkpoint(state: State,
                    logger: Logger,
                    filename: str = "ep{epoch}-ba{batch}-rank{rank}",
                    *,
                    weights_only: bool = False) -> List[pathlib.Path]:
    state_dict = {
        'state': state.state_dict(),
        'rng': reproducibility.get_rng_state(),
    }
    if weights_only and not state.is_model_deepspeed:
        state_dict['state'] = {"model": state_dict['state']['model']}

    checkpoint_filepath = format_name_with_dist_and_time(filename, logger.run_name, state.timestamp)
    if state.is_model_deepspeed and not is_tar(checkpoint_filepath):
        # Deepspeed requires tarballs; appending `.tar`
        checkpoint_filepath += ".tar"

    with tempfile.TemporaryDirectory() as tmpdir:
        composer_states_filepath = os.path.join(tmpdir, _COMPOSER_STATES_FILENAME)
        if dist.get_global_rank() == 0:
            # Only rank zero saves the composer state dict
            with open(composer_states_filepath, 'xb') as f:
                torch.save(state_dict, f)

        if state.is_model_deepspeed:
            state.deepspeed_model.save_checkpoint(tmpdir, _DEEPSPEED_TAG)

        # Move the checkpoint to the correct location

        checkpoint_dirname = os.path.dirname(checkpoint_filepath)

        if is_tar(checkpoint_filepath) and (state.is_model_deepspeed or dist.get_global_rank() == 0):
            # Either deepspeed (and every rank needs to call this),
            # or not deepspeed (but using an archive), in which case only rank zero should call this.
            if checkpoint_dirname:
                os.makedirs(checkpoint_dirname, exist_ok=True)
            write_mode = _get_write_mode(checkpoint_filepath)
            with tarfile.open(checkpoint_filepath, write_mode) as tarball:
                # add files flat to the tarball with the specified compression
                tarball.add(tmpdir, arcname="")
        elif dist.get_global_rank() == 0:
            # if not an archive, then only saving the states
            # only rank zero saves the state dict
            if checkpoint_dirname:
                os.makedirs(checkpoint_dirname, exist_ok=True)
            shutil.move(composer_states_filepath, checkpoint_filepath)
        else:
            checkpoint_filepath = None

    # Ensure that all processes wait for the checkpoint to be saved.
    dist.barrier()

    if checkpoint_filepath is not None:
        log.info('Saved checkpoint at %s', checkpoint_filepath)

    # Gather the paths across ranks.
    paths = dist.all_gather_object(checkpoint_filepath)
    paths = list(pathlib.Path(path) for path in paths if path is not None)

    return paths


save_checkpoint.__doc__ = f"""Checkpoint the training ``state``.

Args:
    state (State): The training state.
    logger (Logger): The logger.
    filename (str): A format string describing how to name checkpoints.
        (default: ``'ep{{epoch}}-ba{{batch}}-rank{{rank}}'``)

        The following format variables are available:

        {textwrap.indent(FORMAT_NAME_WITH_DIST_AND_TIME_TABLE, prefix='        ')}

        .. note::

            *   By default, only the rank zero process will save a checkpoint file.

            *   When using DeepSpeed, each rank will save a checkpoint file in tarball format. DeepSpeed
                requires tarball format, as it saves model and optimizer states in separate files.
                Ensure that ``'{{rank}}'`` appears within the ``filename``. Otherwise, multiple ranks
                may attempt to write to the same file(s), leading to corrupted checkpoints. If no tarball file
                extension is specified, ``.tar`` will be used.

            *   To use compression (regardless of whether DeepSpeed is enabled), set the file extension
                to ``'.tar.gz'``, ``'.tgz'``, ``'.tar.bzip'``, or ``'.tar.lzma'`` (depending on the desired
                compression algorithm).

        .. warning::

            Using compression will block the training loop while checkpoints are being compressed. As such, we
            recommend saving checkpoints without compression.

        Consider the following scenario, where:

        *   The default ``name='ep{{epoch}}-ba{{batch}}-rank{{rank}}'`` is used.
        *   The current epoch count is ``1``.
        *   The current batch count is ``42``.

        When DeepSpeed is not being used, the rank zero process will save the checkpoint to ``'ep1-ba42-rank0'``.
        When DeepSpeed is being used, each rank (process) will save checkpoints to::

            ep1-ba42-rank0.tar
            ep1-ba42-rank1.tar
            ep1-ba42-rank2.tar
            ...

    weights_only (bool, optional): If ``True``, save only the model weights instead of the entire training state.
        (default: ``False``)

        .. note::

            When using DeepSpeed, this parameter must be ``False``. Weights-only checkpointing is not currently
            compatible with DeepSpeed,

    Returns:
        List[pathlib.Path]: The list of checkpoint files saved, indexed by the rank of the process.

        .. note::

            When using DeepSpeed, each process (rank) saves its own checkpoint file.
            When doing multi-node training, the filepaths are valid only on each process's node;
            Composer does not move checkpoint files between nodes.

            Otherwise, when not using DeepSpeed, each list will contain only one filepath,
            since only the rank zero process saves checkpoints.
"""
