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
from packaging import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from composer.utils import dist, reproducibility
from composer.utils.file_helpers import (FORMAT_NAME_WITH_DIST_AND_TIME_TABLE, format_name_with_dist,
                                         format_name_with_dist_and_time, get_file, is_tar, strip_rank_placeholders)
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
_TORCH_DISTRIBUTED_CHECKPOINTS_FILENAME = f'__{dist.get_global_rank()}_0.distcp'

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

    def _format_with_placeholders(self, extra_suffix: str = ''):
        if self.folder:
            return os.path.join(
                self.folder, self.filename,
            ) + extra_suffix
        else:
            return self.filename + extra_suffix


    def format(self, state: State, is_deepspeed: bool = False, keep_placeholders: bool = False) -> str:
        # if filename already has a suffix (e.g. file.pt), this would append to be file.pt.tar
        extra_suffix = '.tar' if is_deepspeed and not is_tar(self.filename) else ''
        if keep_placeholders:
            return self._format_with_placeholders(extra_suffix=extra_suffix)
        
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


def _is_old_sharded_version_checkpoint_file(
    path: str,
    state: State,
    object_store: Optional[Union[ObjectStore, LoggerDestination]] = None):
    return False
    # with tempfile.TemporaryDirectory() as tmpdir:
    #     chkpt_destination = str(Path(tmpdir) / Path('test_chkpt.pt'))

    #     get_file(path=format_name_with_dist_and_time(path, run_name=state.run_name, timestamp=state.timestamp), 
    #              destination=tmpdir,
    #              object_store=object_store)
    #     if os.path.isdir(chkpt_destination):
    #         return False
    #     else:
    #         state_dict = safe_torch_load(chkpt_destination)
    #         composer_version = state_dict['metadata']['composer_env_info']['composer_version']
    #         if composer_version in ['0.13.1', '0.13.2', '0.13.3']:
    #             return True


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

            When using `Deepspeed ZeRO <https://www.deepspeed.ai/tutorials/zero/>`_, checkpoints are sharded by rank.
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
            to retrieve the checkpoint. Otherwise, if the checkpoint is a local filepath, set to ``None``.
            (default: ``None``)
        load_weights_only (bool, optional): Whether or not to only restore the model weights from the checkpoint without
            restoring the associated state. (default: ``False``)
        strict_model_weights (bool, optional): Whether or not to force that the checkpointed weights must exactly
            match the model weights. (default: ``False``)
        progress_bar (bool, optional): Whether or not to show a progress bar when downloading checkpoints.
            Ignored if the checkpoint is a local file path. (default: ``True``)
        ignore_keys (List[str] | (Dict) -> None, optional): A list of paths for the ``state_dict`` of the checkpoint,
            which, when provided, will be ignored from the state_dict before a checkpoint is loaded. Each path is a list
            of strings specifying the keys to index into ``state_dict`` joined together with `/` as a separator (as PyTorch
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
    if state.fsdp_sharded_state_dict_enabled and not _is_old_sharded_version_checkpoint_file(path, state, object_store):
        rng_state_dicts = load_sharded_checkpoint(source_path=path,
                                                  state=state,
                                                  logger=logger,
                                                  object_store=object_store,
                                                  load_weights_only=load_weights_only,
                                                  strict_model_weights=strict_model_weights,
                                                  progress_bar=progress_bar,
                                                  ignore_keys=ignore_keys,
                                                  exclude_algorithms=exclude_algorithms,
                                                  algorithm_passes=algorithm_passes,
                                                  )
    else:
        # download the checkpoint to the node-local folder
        log.debug('Loading checkpoint at %s', path)
        # Each node gets one unique folder to store checkpoints that is shared amongst all local ranks in that node.
        # If fsdp sharded state_dicts is enabled then EVERY rank gets a unique checkpoint folder.
        tempdir_ctx = (tempfile.TemporaryDirectory() if (state.fsdp_sharded_state_dict_enabled or
                                                        dist.get_local_rank() == 0) else contextlib.nullcontext(None))
        with tempdir_ctx as tempdir:
            try:
                # Get the path to the proper checkpoint folder corresponding to the current rank's node.
                # If fsdp_sharded_state_dict_enabled then just use that rank's unique tempdir.
                node_checkpoint_folder = (tempdir if state.fsdp_sharded_state_dict_enabled else
                                        _get_node_checkpoint_download_folder(tempdir))
                assert node_checkpoint_folder is not None

                composer_states_filepath, extracted_checkpoint_folder, extracted_rank_n = download_checkpoint(
                    path=path,
                    node_checkpoint_folder=node_checkpoint_folder,
                    object_store=object_store,
                    progress_bar=progress_bar,
                    fsdp_sharded_state_dict_enabled=state.fsdp_sharded_state_dict_enabled,
                    deepspeed_sharded_checkpoint=is_model_deepspeed(state.model),
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



def load_sharded_checkpoint(
    source_path: str,
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
    if version.parse(torch.__version__) < version.parse('2.0.0'):
                raise ValueError(f'Sharded checkpoint loading requires torch version >= 2.0.0 Got {torch.__version__}')
    if not os.path.isdir(source_path):
        raise ValueError(f'checkpoint_path must be a directory when using sharded state dict. Got {source_path}')
    from torch.distributed import checkpoint as dist_cp
    from torch.distributed.checkpoint.metadata import Metadata
    from torch.distributed.checkpoint.planner import LoadPlan, LoadPlanner
    from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
    from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

    class DistCPObjectStoreReader(dist_cp.FileSystemReader):
        def __init__(self, source_path, destination_path, object_store): # path to metadata
            self.source_path = source_path
            self.destination_path = destination_path
            self.object_store = object_store
            # Instantiate FileSystemReader with destination path b/c that's where we will download files to.
            # Because we already download the metadata file to the destination, everything will work swimmingly
            super().__init__(destination_path)

        def read_data(self, plan: LoadPlan, planner: LoadPlanner):
            # 1. Download to the destination all files that this rank is responsible for.           
            for plan_item in plan.items:
                relative_file_path = self.storage_data[plan_item.storage_index].relative_path
                file_destination = str(Path(self.destination_path) / Path(relative_file_path))
                if not os.path.exists(file_destination):
                    self.object_store.download_object(object_name=str(Path(self.source_path) / Path(relative_file_path)),
                                                    filename=file_destination)
            
            # 2. Wait for all ranks to finish.
            dist.barrier()

            # 3. Piggyback off of the FileSystemReader to read all the files now that they are downloaded.
            return super().read_data(plan, planner)


    def _get_num_ranks_that_saved_rng(metadata: Metadata):
        rng_inds = []
        for field_name, field_value in metadata.planner_data.items():
            if 'rng' in field_name:
                _, rng_rank_index, _ = field_value
                rng_inds.append(rng_rank_index)
        rng_inds = set(rng_inds)
        return max(rng_inds) + 1


    with tempfile.TemporaryDirectory() as tempdir:
        local_rank0_tempdir = dist.all_gather_object(tempdir)[dist.get_local_world_size() * dist.get_node_rank()]
        if object_store is not None:
            # Download metadata if local rank 0 to destination path
            destination_dir = Path(local_rank0_tempdir) / Path('checkpoints')
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination_dir = str(destination_dir)
            metadata_destination = os.path.join(destination_dir, '.metadata')
            if dist.get_local_rank() == 0:
                object_store.download_object(object_name = str(Path(source_path) / Path('.metadata')), filename=metadata_destination)
            storage_reader  = DistCPObjectStoreReader(source_path=source_path, destination_path=destination_dir, object_store=object_store)

        else:
            storage_reader = dist_cp.FileSystemReader(source_path)

        # We need no_grad becaue we overwrite tensor values with set_() when we do elastic loading and we don't want the set_ op recorded in the computation graph.
        with torch.no_grad(): 
            # 1. Load just model first.
            model_state_dict = {'model' : state.state_dict()['model']}
            dist_cp.load_state_dict(model_state_dict, storage_reader)
            state.load_state_dict(
                model_state_dict,
                logger,
                strict=strict_model_weights,
                exclude_algorithms=exclude_algorithms,
                algorithm_passes=algorithm_passes,
            )

            # 2. Optionally load optimizer
            if not load_weights_only:
                optim_state = load_sharded_optimizer_state_dict(
                                model_state_dict=state.state_dict()['model'],
                                optimizer_key="optimizers",
                                storage_reader=storage_reader)     
                state.load_optim_state(optim_state)

        # 3. Load the rest of state.
        cur_state_dict = state.state_dict()
        if ignore_keys:
            # Filter provided list of key paths
            if not callable(ignore_keys):
                ignore_keys = glob_filter(ignore_keys)
            # Call function to modify state_dict
            ignore_keys(cur_state_dict)

        # Remove model and optimizers because they were already loaded.
        cur_state_dict.pop('model')
        cur_state_dict.pop('optimizers')

        rest_of_the_state_dict = cur_state_dict

        # If we are resuming on more ranks than were used at save time we only want to load in rng's for those ranks
        rng_state_dicts = reproducibility.get_rng_state()
        num_ranks_that_saved_rng = _get_num_ranks_that_saved_rng(storage_reader.read_metadata())
        rest_of_the_state_dict['rng'] = rng_state_dicts[:num_ranks_that_saved_rng] if len(rng_state_dicts) > num_ranks_that_saved_rng else rng_state_dicts
        dist_cp.load_state_dict(rest_of_the_state_dict, storage_reader)
        # We also want to append newly generated rng states for the ranks that don't have an rng state to load in
        # Ii we are resuming on more ranks than were used at save time.
        if len(rng_state_dicts) > num_ranks_that_saved_rng:
            rest_of_the_state_dict['rng'].extend(rng_state_dicts[num_ranks_that_saved_rng:])
        state.load_state_dict(
            rest_of_the_state_dict,
            logger,
        )
        
        return rest_of_the_state_dict['rng']



def _get_node_checkpoint_download_folder(path: Optional[str]) -> str:
    """Broadcasts the ``path`` from the LOCAL rank zero to all LOCAL ranks."""
    local_rank_zero = dist.get_local_world_size() * dist.get_node_rank()
    paths = dist.all_gather_object(path)
    local_rank_zero_path = paths[local_rank_zero]
    assert local_rank_zero_path is not None, 'local rank zero provides the path'
    return local_rank_zero_path


def download_checkpoint(path: str,
                        node_checkpoint_folder: str,
                        object_store: Optional[Union[ObjectStore, LoggerDestination]],
                        progress_bar: bool,
                        fsdp_sharded_state_dict_enabled: bool = False,
                        deepspeed_sharded_checkpoint: bool = False) -> Tuple[str, Optional[str], bool]:
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
        # and only rank zero has this file unless fsdp_sharded_state_dict_enabled then
        # every rank has it's own file.
        extracted_checkpoint_folder = None
        composer_states_filepath = (rank_n_checkpoint_filepath
                                    if fsdp_sharded_state_dict_enabled else rank_zero_checkpoint_filepath)

    checkpoint_is_sharded = fsdp_sharded_state_dict_enabled or deepspeed_sharded_checkpoint
    try:
        if not checkpoint_is_sharded and dist.get_local_rank() == 0:
            # if the checkpoint is not sharded, then local rank 0 on each node needs to download the
            # global rank 0 checkpoint
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
        elif checkpoint_is_sharded:
            # if the checkpoint is sharded, then every rank needs to download its own checkpoint
            try:
                get_file(destination=rank_n_checkpoint_filepath,
                         path=_format_path_with_current_rank(path),
                         object_store=object_store,
                         progress_bar=progress_bar)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    (f'Checkpoint {_format_path_with_current_rank(path)} does not exist, '
                     f'but is required for sharded checkpointing on rank {dist.get_global_rank()}. '
                     'Please ensure that the checkpoint exists and your load_path was specified as a format string'
                     'with the {rank} argument.')) from e

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
        # First we wait for the local rank 0 to finish its download. This prevents timeouts
        # in cases where the local rank 0 is downloading a monolithic checkpoint, and so takes
        # much longer than the other ranks, which have nothing to download
        # Putting the barrier in a finally so the rank will always block on the barrier,
        # even if it has an exception.
        # Any exception will be re-raised after the barrier passes. The launcher script
        # will detect the process crash and terminate the other ranks

        signal_file_path = os.path.join(node_checkpoint_folder, '.local_rank0_completed')
        if dist.get_local_rank() == 0:
            with open(signal_file_path, 'wb') as f:
                f.write(b'local_rank0_completed')
        dist.local_rank_zero_download_and_wait(signal_file_path)
        if dist.get_local_rank() == 0:
            os.remove(signal_file_path)

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


def safe_torch_load(composer_states_filepath: Union[Path, str], map_location: str = 'cpu'):
    """Load a torch checkpoint, catching errors due to backwards compatibility issues.

    Args:
        composer_states_filepath: The path to the checkpoint file.
        map_location: The location to load the checkpoint to.
    """
    try:
        state_dict = torch.load(composer_states_filepath, map_location=map_location)
        return state_dict
    except TypeError as e:
        if 'Accuracy.__new__() missing 1 required positional argument' in str(e):
            raise Exception('As of v0.10.0, torchmetrics introduces a new required argument to Accuracy which '
                            'breaks backwards compatibility. Unfortunately, this means that older checkpoints '
                            'cannot be loaded with the metrics. In order to successfully load this model, please '
                            'pass `load_ignore_keys = ["state/train_metrics/*", "state/eval_metrics/*"]`.') from e
        raise e


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
    
    elif not state.fsdp_sharded_state_dict_enabled:
        state_dict = safe_torch_load(composer_states_filepath)
        if ignore_keys:
            # Filter provided list of key paths
            if not callable(ignore_keys):
                ignore_keys = glob_filter(ignore_keys)
            # Call function to modify state_dict
            ignore_keys(state_dict)
        log.debug(f"Loaded checkpoint with keys {state_dict.keys()} and state keys {state_dict['state'].keys()}")
        if load_weights_only:
            state.load_model_state(
                state_dict['state'],
                logger,
                strict=strict_model_weights,
                exclude_algorithms=exclude_algorithms,
                algorithm_passes=algorithm_passes,
            )
        else:
            state.load_state_dict(
                state_dict['state'],
                logger,
                exclude_algorithms=exclude_algorithms,
                algorithm_passes=algorithm_passes,
            )
            return state_dict['rng']
    else:
        # Returns the rng state dicts (if ``load_weights_only`` is False)
        return load_checkpoint_from_local_shard_files(
                        state=state,
                        checkpoint_path=composer_states_filepath,
                        logger=logger,
                        strict_model_weights=strict_model_weights,
                        ignore_keys=ignore_keys,
                        exclude_algorithms=exclude_algorithms,
                        algorithm_passes=algorithm_passes,
                        load_weights_only=load_weights_only,
        )


def save_checkpoint(
    state: State,
    filename: str = 'ep{epoch}-ba{batch}-rank{rank}',
    *,
    weights_only: bool = False,
    overwrite: bool = False,
) -> Union[str, None]:  # noqa: D103

    if state.fsdp_sharded_state_dict_enabled:
        save_filename = _save_fsdp_sharded_checkpoint(state, filename, weights_only, overwrite)
    elif is_model_deepspeed(state.model):
        save_filename = _save_deepspeed_model(state, filename)
    else:
        save_filename = _save_monolithic_checkpoint(state, filename, weights_only, overwrite)

    dist.barrier()  # ensure all ranks saved their files

    if save_filename is not None:
        assert os.path.exists(save_filename), 'Expected file to have been saved.'
        return save_filename
    else:
        return None
        

def _save_fsdp_sharded_checkpoint(state: State, save_filename:str,
                                  weights_only: bool, overwrite: bool) -> str:
    if version.parse(torch.__version__) < version.parse('2.0.0'):
        raise RuntimeError('To save FSDP sharded checkpoints (fsdp_state_dict_type is either "local" or "sharded") with Composer, you must use torch>=2.0.0.')
    from torch.distributed import checkpoint as dist_checkpoint

    if weights_only:
        state_dict = {'model': state.state_dict()['model']}
    else:
        # Dictionary must be flat to faciliate loading optimizer params.
        state_dict = {
            **state.state_dict(),
            'rng': reproducibility.get_rng_state(),
        }

    # Remove all rank placeholders, so that all ranks get the same folder to save checkpoints.
    # This is important because they all need access to the same metadata file.
    save_filename = strip_rank_placeholders(save_filename)

    # Instead of using a filename, torch.distributed.checkpoint want a directory, so turn filename into a directory
    # because we want the directory to have the same identifying characteristics as the file.
    # Remove .pt suffix to make this a directory.
    save_dir = str(Path(save_filename).parent / Path(save_filename).stem)
    # Remove any trailing hyphens or underscores.
    save_dir = save_dir.rstrip('-').rstrip('_')
    # Fill in remaining placeholders
    save_dir = format_name_with_dist_and_time(
                        save_dir,
                        state.run_name,
                        state.timestamp
                        )
    log.debug('Saving sharded checkpoints to %s...', save_dir)
    if os.path.exists(save_dir):
        if not overwrite and len(os.listdir(save_dir)) > 0:
            raise RuntimeError(textwrap.dedent(f'For saving FSDP sharded checkpoints with Composer, ' 
                                            f'you must use an empty or nonexistent directory, but found {str(save_dir)} '
                                            'to be existing and nonempty and overwrite to be set to False. '
                                            f'Please either delete the contents of {str(save_dir)}, specify a different save_folder '
                                            'or set overwrite to True'))
        elif overwrite:
            os.rmdir(str(save_dir))

    fsw = dist_checkpoint.FileSystemWriter(save_dir)
    dist_checkpoint.save_state_dict(state_dict=state_dict, storage_writer=fsw)

    save_filepath = str(Path(save_dir) / Path(_TORCH_DISTRIBUTED_CHECKPOINTS_FILENAME))
    log.debug('This rank''s checkpoint shards successfully saved to %s', save_filepath)
    return save_filepath



def _save_deepspeed_model(state: State, filename: str) -> str:
    """Save Deepspeed model and tarball the files."""
    save_filename = PartialFilePath(filename).format(state, is_deepspeed=True)
    dirname = os.path.dirname(save_filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    write_mode = _get_write_mode(save_filename)
    read_mode = 'r' + write_mode[1:]

    log.debug('Saving deepspeed checkpoint to %s...', save_filename)     
    with tempfile.TemporaryDirectory() as tmpdir:
        state.deepspeed_model.save_checkpoint(tmpdir, _DEEPSPEED_TAG)

        if os.path.exists(save_filename):
                # extract to tmpdir to append below
                # not all compression formats support direct append
                with tarfile.open(save_filename, read_mode) as tar:
                    tar.extractall(tmpdir)
        with tarfile.open(save_filename, write_mode) as tar:
            tar.add(tmpdir, arcname='')
        log.debug('Successfully saved deepspeed checkpoint to %s', save_filename) 

    return save_filename

def _save_monolithic_checkpoint(state: State, filename: str,
                                weights_only:bool, overwrite: bool) -> Union[str, None]:       
    # only rank 0 saves the state_dict unless state.fsdp_sharded_state_dict_enabled=True.
    if dist.get_global_rank() == 0:
        state_dict = {
            'state': state.state_dict(),
            'rng': reproducibility.get_rng_state(),
        }
        if weights_only:
            state_dict = {'model': state_dict['model']}

        save_filename = PartialFilePath(filename).format(state)
        dirname = os.path.dirname(save_filename)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(save_filename, 'wb') as f:
            log.debug('Saving checkpoint to %s...', save_filename)
            torch.save(state_dict, f)
            log.debug('Successfully saved checkpoint to %s', save_filename)

        if is_tar(save_filename):
            _compress_file(save_filename, basename=_COMPOSER_STATES_FILENAME)
    else:
        save_filename = None 
        log.debug('Global rank 0 is saving checkpoint, but this rank doesn''t save a checkpoint')
    return save_filename

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
