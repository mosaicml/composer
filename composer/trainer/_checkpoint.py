# Copyright 2021 MosaicML. All Rights Reserved.

"""Load and save checkpoints during training."""

from __future__ import annotations

import contextlib
import logging
import os
import tarfile
import tempfile
import textwrap
import urllib.parse
from typing import Iterator, List, Optional, Tuple

import requests
import torch
import tqdm

from composer.core import State, types
from composer.utils import ObjectStoreProvider, dist, iterate_with_pbar

log = logging.getLogger(__name__)

__all__ = ["load_checkpoint"]

_COMPOSER_STATES_FILENAME = "composer_states.pt"
_DEEPSPEED_TAG = "deepspeed"  # always tag with the same, deterministic name. We'll rename the tarball to the appropriate name.


def _format_path_with_rank(path: str, rank: int) -> str:
    """Returns the ``path`` with ``{RANK}`` substituted with the ``rank`` argument."""
    return path.format(RANK=rank)


def _is_pt_file(path: str) -> bool:
    """Returns true if the path is a tar archive and false otherwise."""
    return path.endswith('.pt')


def load_checkpoint(
    path: str,
    state: State,
    object_store: Optional[ObjectStoreProvider] = None,
    load_weights_only: bool = False,
    strict_model_weights: bool = False,
    chunk_size: int = 1_048_576,
    progress_bar: bool = True,
):
    """Load a checkpoint from a local file, URI, or cloud object store into ``state``.

    Args:
        path (str): The template path to an existing checkpoint file.
            It can be a path to a file on local disk, a URL, or if ``object_store`` is set, the object name
            for a checkpoint in a cloud bucket.

            When using `Deepspeed ZeRO <https://www.deepspeed.ai/tutorials/zero/>`_, the :class:`CheckpointSaver`
            shards checkpoints by rank. To load deepspeed checkpoints, specify ``{RANK}`` in this ``path``
            parameter, and the ``RANK`` variable will be substituted with the global rank, thus allowing the correct
            checkpoints to be loaded per-rank.

            For example, suppose that checkpoints are stored in the following structure:

            .. code-block::

                my_model/rank_0/ep1.tar
                my_model/rank_1/ep1.tar
                my_model/rank_2/ep1.tar
                ...

            Then, ``path`` should be set to ``my_model/rank_{RANK}/ep1.tar``, and all ranks will load the correct
            data.
        state (State): The :class:`~composer.core.state.State` to load the checkpoint into.
        object_store (ObjectStoreProvider, optional): If the ``path`` is in an object store
            (i.e. AWS S3 or Google Cloud Storage), an instance of
            :class:`~composer.utils.object_store.ObjectStoreProvider` which will be used
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
    checkpoint_uri_parsed = urllib.parse.urlparse(path)
    if checkpoint_uri_parsed.scheme != "":
        if object_store is not None:
            raise ValueError(
                textwrap.dedent("""\
                    When specifying `object_store`,
                    the `checkpoint` parameter must be the key for the checkpoint in the bucket, NOT a uri."""))

    # download the checkpoint to the node-local folder
    tempdir_ctx = tempfile.TemporaryDirectory() if dist.get_local_rank() == 0 else contextlib.nullcontext(None)
    with tempdir_ctx as tempdir:
        try:
            node_checkpoint_folder = _get_node_checkpoint_download_folder(tempdir)
            composer_checkpoint_filepath, extracted_checkpoint_folder, extracted_rank_n = _download_checkpoint(
                path=path,
                node_checkpoint_folder=node_checkpoint_folder,
                object_store=object_store,
                chunk_size=chunk_size,
                progress_bar=progress_bar,
            )
            rng_state_dicts = _restore_checkpoint(
                state,
                composer_checkpoint_filepath,
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


def _retrieve_checkpoint(
    path: str,
    object_store: Optional[ObjectStoreProvider],
    rank: int,
    chunk_size: int,
    destination_filepath: str,
    ignore_not_found_errors: bool,
    progress_bar: bool,
):
    """Download the checkpoint at ``path`` into ``destination_filepath``, potentially using an ``object_store``."""
    checkpoint_name = _format_path_with_rank(path, rank)
    if object_store is not None:
        try:
            total_size_in_bytes = object_store.get_object_size(checkpoint_name)
        except Exception as e:
            if "ObjectDoesNotExistError" in str(e) and ignore_not_found_errors:
                return
            raise
        _write_to_file_with_pbar(
            destination_filepath=destination_filepath,
            total_size=total_size_in_bytes,
            iterator=object_store.download_object_as_stream(checkpoint_name, chunk_size=chunk_size),
            progress_bar=progress_bar,
            description=f"Downloading {path}",
        )
        return
    checkpoint_uri_parsed = urllib.parse.urlparse(checkpoint_name)

    if checkpoint_uri_parsed.scheme == "":
        if not os.path.exists(checkpoint_name):
            if ignore_not_found_errors:
                return
            else:
                raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_name}")
        # assume it's a local file
        os.symlink(os.path.abspath(checkpoint_name), destination_filepath)
        return
    # it's a url
    with requests.get(checkpoint_name, stream=True) as r:
        r.raise_for_status()
        total_size_in_bytes = r.headers.get('content-length')
        if total_size_in_bytes is not None:
            total_size_in_bytes = int(total_size_in_bytes)
        _write_to_file_with_pbar(
            destination_filepath,
            total_size=total_size_in_bytes,
            iterator=r.iter_content(chunk_size),
            progress_bar=progress_bar,
            description=f"Downloading {path}",
        )


def _write_to_file_with_pbar(
    destination_filepath: str,
    total_size: Optional[int],
    iterator: Iterator[bytes],
    progress_bar: bool,
    description: str,
):
    """Write the contents of ``iterator`` to ``destination_filepath`` while showing a progress bar."""
    if progress_bar:
        if len(description) > 60:
            description = description[:42] + "..." + description[-15:]
        pbar = tqdm.tqdm(desc=description, total=total_size, unit='iB', unit_scale=True)
    else:
        pbar = None
    with open(destination_filepath, "wb") as fp:
        for chunk in iterate_with_pbar(iterator, pbar):
            fp.write(chunk)


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
    object_store: Optional[ObjectStoreProvider],
    chunk_size: int,
    progress_bar: bool,
) -> Tuple[str, Optional[str], bool]:
    """Download the checkpoint stored at ``path``, potentially in ``object_store``, to ``node_checkpoint_folder``.

    Returns a tuple of  (``composer_checkpoint_filepath``, ``extracted_checkpoint_folder``, ``extracted_rank_n``).

    *   The ``composer_checkpoint_filepath``, is the path to the composer states, which can be passed into
        :meth:`torch.load`.
    *   The ``extracted_checkpoint_folder`` is the path to the checkpoint folder, which can be passed into
        :meth:`deepspeed.DeepSpeedEngine.load_checkpoint`.
    *   The ``extracted_rank_n`` is a boolean flag indicating whether a tarball was extracted on global
        rank greater than 0.
    """
    checkpoint_archive_name = path.split(os.path.sep)[-1]
    rank_zero_checkpoint_archive_name = "rank_0." + _format_path_with_rank(checkpoint_archive_name, 0)
    rank_n_checkpoint_archive_name = f"rank_{dist.get_global_rank()}." + _format_path_with_rank(
        checkpoint_archive_name, dist.get_global_rank())
    rank_zero_checkpoint_archive_filepath = os.path.join(node_checkpoint_folder, rank_zero_checkpoint_archive_name)
    rank_n_checkpoint_archive_filepath = os.path.join(node_checkpoint_folder, rank_n_checkpoint_archive_name)
    extracted_checkpoint_folder = None
    extracted_rank_n = False
    if _is_pt_file(rank_zero_checkpoint_archive_filepath):
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
            _retrieve_checkpoint(destination_filepath=rank_zero_checkpoint_archive_filepath,
                                 rank=dist.get_global_rank(),
                                 ignore_not_found_errors=False,
                                 object_store=object_store,
                                 path=path,
                                 chunk_size=chunk_size,
                                 progress_bar=progress_bar)
            if extracted_checkpoint_folder is not None:
                try:
                    with tarfile.open(rank_zero_checkpoint_archive_filepath) as tarball:
                        tarball.extractall(extracted_checkpoint_folder)
                except FileNotFoundError:
                    checkpoint_name = _format_path_with_rank(path, dist.get_global_rank())
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
            _retrieve_checkpoint(destination_filepath=rank_n_checkpoint_archive_filepath,
                                 rank=dist.get_global_rank(),
                                 ignore_not_found_errors=True,
                                 path=path,
                                 object_store=object_store,
                                 chunk_size=chunk_size,
                                 progress_bar=progress_bar)

            if extracted_checkpoint_folder is not None:
                try:
                    # it's an archive and needs to be extracted
                    with tarfile.open(rank_n_checkpoint_archive_filepath) as tarball:
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

    return composer_checkpoint_filepath, extracted_checkpoint_folder, extracted_rank_n


def _restore_checkpoint(
    state: State,
    composer_checkpoint_filepath: str,
    extracted_rank_n: bool,
    extracted_checkpoint_folder: Optional[str],
    load_weights_only: bool,
    strict_model_weights: bool,
) -> Optional[List[types.StateDict]]:
    """Restore a checkpoint into ``state`` and returns the rng state dicts (if ``load_weights_only`` is False)."""
    # Now, all ranks load the checkpoint that local rank zero downloaded
    state_dict = torch.load(composer_checkpoint_filepath, map_location='cpu')
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
