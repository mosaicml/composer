# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Utilities for working with training checkpoints."""

from __future__ import annotations

import contextlib
import fnmatch
import logging
import os
import shutil
import tarfile
import tempfile
import textwrap
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from composer.utils import dist, reproducibility
from composer.utils.file_helpers import (FORMAT_NAME_WITH_DIST_AND_TIME_TABLE, format_name_with_dist,
                                         format_name_with_dist_and_time, get_file, is_tar)
from composer.utils.misc import is_model_deepspeed
from composer.utils.object_store import ObjectStore

if TYPE_CHECKING:
    from composer.core.passes import AlgorithmPass
    from composer.core.state import State
    from composer.loggers import Logger, LoggerDestination

log = logging.getLogger(__name__)

__all__ = ['load_checkpoint', 'save_checkpoint', 'download_checkpoint']

_COMPOSER_STATES_FILENAME = 'composer_states.pt'
_DEEPSPEED_TAG = 'deepspeed'  # always tag with the same, deterministic name. We'll rename the tarball to the appropriate name.


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
    if name.endswith('.tar.gz') or name.endswith('.tgz'):
        return 'w:gz'
    if name.endswith('.tar.bz2'):
        return 'w:bz2'
    if name.endswith('.tar.lzma'):
        return 'w:xz'
    raise ValueError(f'{name} does not end with a valid tarfile extension.')


class PartialFilePath:

    def __init__(self, filename: str, folder: Optional[str] = None):
        self.folder = folder
        self.filename = filename

    def format(self, state: State, is_deepspeed: bool = False) -> str:
        # if filename already has a suffix (e.g. file.pt), this would append to be file.pt.tar
        extra_suffix = '.tar' if is_deepspeed and not is_tar(self.filename) else ''
        if self.folder:
            return os.path.join(
                format_name_with_dist(self.folder, state.run_name),
                format_name_with_dist_and_time(self.filename, state.run_name, state.timestamp),
            ) + extra_suffix
        else:
            return format_name_with_dist_and_time(
                self.filename,
                state.run_name,
                state.timestamp,
            ) + extra_suffix


def load_checkpoint(
    path: str,
    state: State,
    logger: Logger,
    object_store: Optional[Union[ObjectStore, LoggerDestination]] = None,
    load_weights_only: bool = False,
    strict_model_weights: bool = False,
    progress_bar: bool = True,
    ignore_keys: Optional[Union[List[str], Callable[[Dict], None]]] = None,
    exclude_algorithms: Optional[List[str]] = None,
    algorithm_passes: Optional[List[AlgorithmPass]] = None,
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

        state (State): The :class:`~composer.core.State` to load the checkpoint into.
        logger (Logger): The :class:`~composer.logger.Logger` to log any information.
        object_store (Union[ObjectStore, LoggerDestination], optional): If the ``path`` is in an object store
            (i.e. AWS S3 or Google Cloud Storage), an instance of
            :class:`~.ObjectStore` or :class:`~.LoggerDestination` which will be used
            to retreive the checkpoint. Otherwise, if the checkpoint is a local filepath, set to ``None``.
            (default: ``None``)
        load_weights_only (bool, optional): Whether or not to only restore the model weights from the checkpoint without
            restoring the associated state. (default: ``False``)
        strict_model_weights (bool, optional): Whether or not to force that the checkpointed weights must exactly
            match the model weights. (default: ``False``)
        progress_bar (bool, optional): Whether or not to show a progress bar when downloading checkpoints.
            Ignored if the checkpoint is a local file path. (default: ``True``)
        ignore_keys (List[str] | (Dict) -> None, optional): A list of paths for the ``state_dict`` of the checkpoint,
            which, when provided, will be ignored from the state_dict before a checkpoint is loaded. Each path is a list
            of strings specifying the keys to index into ``state_dict`` joined together with `/` as a seperator (as PyTorch
            uses `.` in parameter names). If a prefix is provided, all children are also ignored (see Example 2).
            See :mod:`composer.core.state` for the structure of state_dict.

            Example 1: ``ignore_keys = ["state/model/layer1.weights", "state/model/layer1.bias"]`` would ignore
            layer 1 weights and bias.

            Example 2: ``ignore_keys = ["state/model/*"]`` would ignore the entire model, which would have the same
            effect as the previous example if there was only 1 layer.

            Example 3: ``ignore_keys = ["state/model/layer*.weights"]`` would ignore all weights in the model.

            Example 4: ``ignore_keys = ["state/rank_zero_seed", "rng"]`` would reset all randomness when
            loading the checkpoint.

            If a callable, it should take one argument which is the state_dict. The callable is free to arbitrarily modify
            the state_dict before it is loaded.

            (default: ``None``)
        exclude_algorithms (List[str], optional): A list of algorithm names to exclude from loading.
            By default, algorithms with `required_on_load=True` which were enabled when training the loaded
            checkpoint are automatically applied unless they conflict with a user specified algorithm. These
            algorithms often change the model, and not applying them could result in certain layers not having
            weights loaded.

            Example 1: ``exclude_algorithms = ["BlurPool"]`` would exclude BlurPool from loading.

            Example 2: ``exclude_algorithms = ["FusedLayerNorm", "Alibi"]`` would exclude FusedLayerNorm and Alibi from loading.

            (default: ``None``)
        algorithm_passes (List[AlgorithmPass], optional): A list of algorithm passes to apply to autoloaded algorithms
            to sort them into the correct order. (default: ``None``)

    Returns:
        Optional[List[Dict[str, Any]]]: The RNG state dicts, indexed by global rank, if
            :attr:`load_weights_only` is not None. Otherwise, None.
    """
    # download the checkpoint to the node-local folder
    log.debug('Loading checkpoint at %s', path)
    tempdir_ctx = tempfile.TemporaryDirectory() if dist.get_local_rank() == 0 else contextlib.nullcontext(None)
    with tempdir_ctx as tempdir:
        try:
            node_checkpoint_folder = _get_node_checkpoint_download_folder(tempdir)
            composer_states_filepath, extracted_checkpoint_folder, extracted_rank_n = download_checkpoint(
                path=path,
                node_checkpoint_folder=node_checkpoint_folder,
                object_store=object_store,
                progress_bar=progress_bar,
            )
            rng_state_dicts = _restore_checkpoint(
                state,
                logger,
                composer_states_filepath,
                extracted_rank_n,
                extracted_checkpoint_folder,
                load_weights_only=load_weights_only,
                strict_model_weights=strict_model_weights,
                ignore_keys=ignore_keys,
                exclude_algorithms=exclude_algorithms,
                algorithm_passes=algorithm_passes,
            )
        finally:
            # Wait for all ranks to finish restoring the checkpoint before releasing the tempdir, since tempdir can
            # be a shared resource between nodes.
            dist.barrier()

    log.info('%s loaded from %s', 'Model weights' if load_weights_only else 'Trainer checkpoint', path)
    return rng_state_dicts


def _get_node_checkpoint_download_folder(path: Optional[str]) -> str:
    """Broadcasts the ``path`` from the LOCAL rank zero to all LOCAL ranks."""
    local_rank_zero = dist.get_local_world_size() * dist.get_node_rank()
    paths = dist.all_gather_object(path)
    local_rank_zero_path = paths[local_rank_zero]
    assert local_rank_zero_path is not None, 'local rank zero provides the path'
    return local_rank_zero_path


def download_checkpoint(
    path: str,
    node_checkpoint_folder: str,
    object_store: Optional[Union[ObjectStore, LoggerDestination]],
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
    log.debug('Downloading checkpoint to folder %s', node_checkpoint_folder)
    rank_zero_checkpoint_filepath = os.path.join(node_checkpoint_folder, 'rank0_checkpoint')
    rank_n_checkpoint_filepath = os.path.join(node_checkpoint_folder, f'rank{dist.get_global_rank()}_checkpoint')
    extracted_checkpoint_folder = None
    extracted_rank_n = False
    if is_tar(path):
        extracted_checkpoint_folder = os.path.join(node_checkpoint_folder, 'checkpoint')
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
                     progress_bar=progress_bar)
            if extracted_checkpoint_folder is not None:
                try:
                    with tarfile.open(rank_zero_checkpoint_filepath) as tarball:
                        tarball.extractall(extracted_checkpoint_folder)
                except FileNotFoundError:
                    # Not re-raising the file-not-found error as that is irrelevant;
                    # the underlying issue is that the checkpoint file does not exist on the disk
                    # or could not be downloaded
                    raise RuntimeError(f'Checkpoint {path} does not exist')

        if rank_zero_checkpoint_filepath != rank_n_checkpoint_filepath:
            # every RANK needs ITS OWN checkpoint.
            # But, the  global rank zero is a special case -- these files are the same!
            assert dist.get_global_rank() != 0, 'invariant violation'

            try:
                get_file(destination=rank_n_checkpoint_filepath,
                         path=_format_path_with_current_rank(path),
                         object_store=object_store,
                         progress_bar=progress_bar)
            except FileNotFoundError:
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


def _flatten_keys(obj: Any, paths: List[str], existing_path: str):
    """Recursively flatten the keys of a dictionary or list into a set of paths."""
    # Store path when we reach end, which is either non-Dict or empty Dict
    if isinstance(obj, list) and len(obj) > 0:
        for i, elm in enumerate(obj):
            _flatten_keys(elm, paths, f'{existing_path}/{i}')
    elif isinstance(obj, dict) and len(obj) > 0:
        for k, v in obj.items():
            _flatten_keys(v, paths, f'{existing_path}/{k}')
    # Remove leading /
    paths.append(existing_path.lstrip('/'))


def _remove_paths(obj: Union[list, Dict[str, Any]], exclude_paths: List[List[str]]):
    # First determine the keys which will be recursed on and which will be removed entirely
    # Group the `exclude_paths` by the key
    keys_to_recurse = {}
    keys_to_remove = []
    for exclude_path_parts in exclude_paths:
        key = exclude_path_parts[0]
        if isinstance(obj, list):
            key = int(key)
        if len(exclude_path_parts) == 1:
            keys_to_remove.append(key)
        else:
            if key not in keys_to_recurse:
                keys_to_recurse[key] = []
            keys_to_recurse[key].append(exclude_path_parts[1:])

    # Recurse first, so in the case of a list, the indexing is consistent
    for key, paths_to_recurse in keys_to_recurse.items():
        _remove_paths(obj[key], paths_to_recurse)

    # Sort the keys in reverse order, so in the case of a list, the indexing is consistent
    keys_to_remove.sort(reverse=True)

    # Remove the keys
    for key in keys_to_remove:
        del obj[key]


def glob_filter(exclude_globs: List[str]) -> Callable[[Dict], None]:
    """Provides a function which deletes all subparts of a dictionary based on a list of paths."""

    def filter_func(state_dict: Dict) -> None:
        # Flatten dictionary into paths
        paths = []
        _flatten_keys(state_dict, paths, '/')

        filtered_paths = []
        for exclude_glob in exclude_globs:
            filtered_paths_from_glob = fnmatch.filter(paths, exclude_glob)
            if len(filtered_paths_from_glob) == 0:
                warnings.warn(
                    f'No parts from loaded checkpoint state_dict were ignored by load_ignore_key {exclude_glob}')
            filtered_paths.extend(filtered_paths_from_glob)
        filtered_paths = list(set(filtered_paths))
        filtered_paths_str = ', '.join(filtered_paths)
        if filtered_paths:
            log.info(f'Ignoring the following paths from the loaded checkpoint state_dict: {filtered_paths_str}')

        # Loop through all paths to exclude
        paths_to_remove = [path.split('/') for path in filtered_paths]
        _remove_paths(state_dict, paths_to_remove)

    return filter_func


def _restore_checkpoint(
    state: State,
    logger: Logger,
    composer_states_filepath: str,
    extracted_rank_n: bool,
    extracted_checkpoint_folder: Optional[str],
    load_weights_only: bool,
    strict_model_weights: bool,
    ignore_keys: Optional[Union[List[str], Callable[[Dict], None]]],
    exclude_algorithms: Optional[List[str]],
    algorithm_passes: Optional[List[AlgorithmPass]],
) -> Optional[List[Dict[str, Any]]]:
    """Restore a checkpoint into ``state`` and returns the rng state dicts (if ``load_weights_only`` is False)."""
    # Now, all ranks load the checkpoint that local rank zero downloaded
    state_dict = torch.load(composer_states_filepath, map_location='cpu')
    if ignore_keys:
        # Filter provided list of key paths
        if not callable(ignore_keys):
            ignore_keys = glob_filter(ignore_keys)
        # Call function to modify state_dict
        ignore_keys(state_dict)
    log.debug(f"Loaded checkpoint with keys {state_dict.keys()} and state keys {state_dict['state'].keys()}")

    if is_model_deepspeed(state.model):
        if extracted_checkpoint_folder is None:
            raise RuntimeError('Deepspeed checkpoints require a tarball, not a weights file.')

        global_rank = dist.get_global_rank()
        if global_rank > 0 and not extracted_rank_n:
            raise RuntimeError(f'Deepspeed checkpoint missing for rank {global_rank}')

        load_path, _ = state.deepspeed_model.load_checkpoint(
            extracted_checkpoint_folder,
            tag=_DEEPSPEED_TAG,
            load_module_only=load_weights_only,
            load_module_strict=strict_model_weights,
        )
        if load_path is None:
            raise RuntimeError(f'Failed to load DeepSpeed checkpoint')
    elif load_weights_only:
        state.load_model_state(
            state_dict['state'],
            logger,
            strict=strict_model_weights,
            exclude_algorithms=exclude_algorithms,
            algorithm_passes=algorithm_passes,
        )
    if not load_weights_only:
        state.load_state_dict(
            state_dict['state'],
            logger,
            exclude_algorithms=exclude_algorithms,
            algorithm_passes=algorithm_passes,
        )
        return state_dict['rng']


def save_checkpoint(
    state: State,
    filename: str = 'ep{epoch}-ba{batch}-rank{rank}',
    *,
    weights_only: bool = False,
) -> Union[str, None]:  # noqa: D103

    log.debug('Saving checkpoint to %s', filename)

    is_deepspeed = is_model_deepspeed(state.model)

    state_dict = {
        'state': state.state_dict(),
        'rng': reproducibility.get_rng_state(),
    }
    if weights_only and not is_deepspeed:
        state_dict['state'] = {'model': state_dict['state']['model']}

    save_filename = PartialFilePath(filename).format(state, is_deepspeed)
    dirname = os.path.dirname(save_filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    # only rank 0 saves the state_dict
    if dist.get_global_rank() == 0:
        with open(save_filename, 'wb') as f:
            torch.save(state_dict, f)

        if is_tar(save_filename):
            _compress_file(save_filename, basename=_COMPOSER_STATES_FILENAME)

    # all ranks save for deepspeed
    if is_deepspeed:
        _save_deepspeed_model(state.deepspeed_model, save_filename)

    dist.barrier()  # ensure all ranks saved their files

    if dist.get_global_rank() == 0 or is_deepspeed:
        assert os.path.exists(save_filename), 'Expected file to have been saved.'
        return save_filename
    else:
        # no file saved
        return None


def _compress_file(filename: str, basename: str):
    """Replace a file with its compressed version.

    The contents will be called ``basename`` inside
    the compressed archive.
    """
    write_mode = _get_write_mode(filename)

    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.move(filename, os.path.join(tmpdir, basename))
        with tarfile.open(filename, write_mode) as tarball:
            tarball.add(tmpdir, arcname='')


def _save_deepspeed_model(model, filename: str):
    """Save Deepspeed model and tarball the files."""
    write_mode = _get_write_mode(filename)
    read_mode = 'r' + write_mode[1:]

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_checkpoint(tmpdir, _DEEPSPEED_TAG)

        if os.path.exists(filename):
            # extract to tmpdir to append below
            # not all compression formats support direct append
            with tarfile.open(filename, read_mode) as tar:
                tar.extractall(tmpdir)

        with tarfile.open(filename, write_mode) as tar:
            tar.add(tmpdir, arcname='')


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
