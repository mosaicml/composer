# Copyright 2021 MosaicML. All Rights Reserved.

"""Utilities for working with training checkpoints."""

from __future__ import annotations

import contextlib
import logging
import os
import pathlib
import shutil
import tarfile
import tempfile
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from composer.utils import dist, reproducibility
from composer.utils.file_retriever import GetFileNotFoundException, get_file
from composer.utils.object_store import ObjectStoreProvider

if TYPE_CHECKING:
    from composer.core.state import State
    from composer.core.types import StateDict

log = logging.getLogger(__name__)

__all__ = ["load_checkpoint", "save_checkpoint"]

_COMPOSER_STATES_FILENAME = "composer_states.pt"
_DEEPSPEED_TAG = "deepspeed"  # always tag with the same, deterministic name. We'll rename the tarball to the appropriate name.


def _format_path_with_rank_zero(path_format: str) -> str:
    """Formats ``path_format`` with the rank zero values."""
    return path_format.format(
        rank=0,
        local_rank=0,
        node_rank=0,
    )


def _format_path_with_current_rank(path_format: str) -> str:
    """Formats ``path_format`` formatted with the current rank values."""
    return path_format.format(
        rank=dist.get_global_rank(),
        local_rank=dist.get_local_rank(),
        node_rank=dist.get_node_rank(),
    )


def _is_archive(path: str) -> bool:
    """Returns whether the path is a tar archive."""
    return any(path.endswith(x) for x in (".tar", ".tgz", ".tar.gz", ".tar.bz2", ".tar.lzma"))


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

                my_model/ep1-rank0.tar
                my_model/ep1-rank1.tar
                my_model/ep1-rank2.tar
                ...

            Then, ``path_format`` should be set to ``my_model/ep1-rank{rank}.tar``, and all ranks will load the
            correct state.

        state (State): The :class:`~composer.core.state.State` to load the checkpoint into.
        object_store (ObjectStoreProvider, optional): If the ``path_format`` is in an object store
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
    """Download the checkpoint stored at ``path_format``, potentially in ``object_store``, to
    ``node_checkpoint_folder``.

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
) -> Optional[List[StateDict]]:
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


def format_name(name_format: str, state: State):
    """Format a checkpoint filename according to the ``name_format`` and the training :class:`~.State`.

    The following format variables are available:

    +------------------------+-------------------------------------------------------+
    | Variable               | Description                                           |
    +========================+=======================================================+
    | ``{rank}``             | The global rank, as returned by                       |
    |                        | :func:`~.dist.get_global_rank`.                       |
    +------------------------+-------------------------------------------------------+
    | ``{local_rank}``       | The local rank of the process, as returned by         |
    |                        | :func:`~.dist.get_local_rank`.                        |
    +------------------------+-------------------------------------------------------+
    | ``{world_size}``       | The world size, as returned by                        |
    |                        | :func:`~.dist.get_world_size`.                        |
    +------------------------+-------------------------------------------------------+
    | ``{local_world_size}`` | The local world size, as returned by                  |
    |                        | :func:`~.dist.get_local_world_size`.                  |
    +------------------------+-------------------------------------------------------+
    | ``{node_rank}``        | The node rank, as returned by                         |
    |                        | :func:`~.dist.get_node_rank`.                         |
    +------------------------+-------------------------------------------------------+
    | ``{epoch}``            | The total epoch count, as returned by                 |
    |                        | :meth:`~composer.core.time.Timer.epoch`.              |
    +------------------------+-------------------------------------------------------+
    | ``{batch}``            | The total batch count, as returned by                 |
    |                        | :meth:`~composer.core.time.Timer.batch`.              |
    +------------------------+-------------------------------------------------------+
    | ``{batch_in_epoch}``   | The batch count in the current epoch, as returned by  |
    |                        | :meth:`~composer.core.time.Timer.batch_in_epoch`.     |
    +------------------------+-------------------------------------------------------+
    | ``{sample}``           | The total sample count, as returned by                |
    |                        | :meth:`~composer.core.time.Timer.sample`.             |
    +------------------------+-------------------------------------------------------+
    | ``{sample_in_epoch}``  | The sample count in the current epoch, as returned by |
    |                        | :meth:`~composer.core.time.Timer.sample_in_epoch`.    |
    +------------------------+-------------------------------------------------------+
    | ``{token}``            | The total token count, as returned by                 |
    |                        | :meth:`~composer.core.time.Timer.token`.              |
    +------------------------+-------------------------------------------------------+
    | ``{token_in_epoch}``   | The token count in the current epoch, as returned by  |
    |                        | :meth:`~composer.core.time.Timer.token_in_epoch`.     |
    +------------------------+-------------------------------------------------------+

    .. note::

        If using DeepSpeed, and ``name_format`` does not end with an tarfile archive extension (``'.tar'``, ``'.tgz'``,
        ``'.tar.gz'``, ``'.tar.bz2'``, or ``'.tar.lzma'``), then ``'.tar'`` will be appended. DeepSpeed uses a tarball
        format as it saves model and optimizer states in separate files within the tarball.

    Consider the following scenario, where the current epoch count is ``1`` and the current batch count is ``42``:
    
    *   When not using DeepSpeed, then the rank zero process will call this function:

        .. testsetup:: composer.utils.checkpoint.format_name.no_deepspeed

            from composer.utils.checkpoint import format_name

            state.timer._batch._value = 42
            state.timer._epoch._value = 1

        .. doctest:: composer.utils.checkpoint.format_name.no_deepspeed

            >>> format_name("ep{epoch}-ba{batch}", state)
            'ep1-ba42'

    *   When using DeepSpeed, each rank (process) will call this function. ``'{rank}'`` should appear within
        ``name_format``, so each rank (process) will write to its own file. For example, on the rank zero process:

        .. testsetup:: composer.utils.checkpoint.format_name.deepspeed

            from composer.utils.checkpoint import format_name

            original_is_model_deepspeed = State.is_model_deepspeed

            setattr(State, 'is_model_deepspeed', property(lambda x: True))
            
            state.timer._batch._value = 42
            state.timer._epoch._value = 1

        .. doctest:: composer.utils.checkpoint.format_name.deepspeed

            >>> format_name("ep{epoch}-ba{batch}-rank{rank}", state)
            'ep1-ba42-rank0.tar'
        
        .. testcleanup:: composer.utils.checkpoint.format_name.deepspeed

            setattr(State, 'is_model_deepspeed', original_is_model_deepspeed)
    """
    checkpoint_name = name_format.format(
        rank=dist.get_global_rank(),
        local_rank=dist.get_local_rank(),
        world_size=dist.get_world_size(),
        local_world_size=dist.get_local_world_size(),
        node_rank=dist.get_node_rank(),
        epoch=int(state.timer.epoch),
        batch=int(state.timer.batch),
        batch_in_epoch=int(state.timer.batch_in_epoch),
        sample=int(state.timer.sample),
        sample_in_epoch=int(state.timer.sample_in_epoch),
        token=int(state.timer.token),
        token_in_epoch=int(state.timer.token_in_epoch),
    )
    if state.is_model_deepspeed and not _is_archive(checkpoint_name):
        # Deepspeed requires tarballs; appending `.tar`
        checkpoint_name += ".tar"

    return checkpoint_name


def save_checkpoint(state: State,
                    name_format: str = "ep{epoch}-ba{batch}-rank{rank}",
                    *,
                    weights_only: bool = False) -> List[pathlib.Path]:
    """Checkpoint the training ``state``.

    Args:
        state (State): The current State of the trainer.
        name_format (str): A format string describing how to name checkpoints.
            (default: ``'ep{epoch}-ba{batch}-rank{rank}'``)

            See :func:`.format_name` for the available format variables.

            .. note::

                *   By default, only the rank zero process will save a checkpoint file.

                *   When using DeepSpeed, each rank will save a checkpoint file in tarball format. DeepSpeed
                    requires tarball format, as it saves model and optimizer states in separate files.
                    Ensure that ``'{rank}'`` appears within the ``name_format_string``. Otherwise, multiple ranks
                    may attempt to write to the same file(s), leading to corrupted checkpoints. If no tarball file
                    extension is specified, ``.tar`` will be used.

                *   To use compression (regardless of whether DeepSpeed is enabled), set the file extension
                    to ``'.tar.gz'``, ``'.tgz'``, ``'.tar.bzip'``, or ``'.tar.lzma'`` (depending on the desired
                    compression algorithm).

            .. warning::

                Using compression will block the training loop while checkpoints are being compressed. As such, we
                recommend saving checkpoints without compression.

            Consider the following scenario, where:

            *   The default ``name_format='ep{epoch}-ba{batch}-rank{rank}'`` is used.
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

                Otherwise, when not using DeepSpeed, this list will contain only one filepath,
                since only the rank zero process saves checkpoints.
    """
    state_dict = {
        'state': state.state_dict(),
        'rng': reproducibility.get_rng_state(),
    }
    if weights_only and not state.is_model_deepspeed:
        state_dict['state'] = {"model": state_dict['state']['model']}

    checkpoint_filepath = format_name(name_format, state)

    with tempfile.TemporaryDirectory() as tmpdir:
        composer_states_filepath = os.path.join(tmpdir, _COMPOSER_STATES_FILENAME)
        if dist.get_global_rank() == 0:
            # Only rank zero saves the composer state dict
            with open(composer_states_filepath, 'xb') as f:
                torch.save(state_dict, f)

        if state.is_model_deepspeed:
            state.deepspeed_model.save_checkpoint(tmpdir, _DEEPSPEED_TAG)

        # Move the checkpoint to the correct location

        if _is_archive(checkpoint_filepath) and (state.is_model_deepspeed or dist.get_global_rank() == 0):
            # Either deepspeed (and every rank needs to call this),
            # or not deepspeed (but using an archive), in which case only rank zero should call this.
            os.makedirs(os.path.dirname(checkpoint_filepath), exist_ok=True)
            write_mode = _get_write_mode(checkpoint_filepath)
            with tarfile.open(checkpoint_filepath, write_mode) as tarball:
                # add files flat to the tarball with the specified compression
                tarball.add(tmpdir, arcname="")
        elif dist.get_global_rank() == 0:
            # if not an archive, then only saving the states
            # only rank zero saves the state dict
            os.makedirs(os.path.dirname(checkpoint_filepath), exist_ok=True)
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
