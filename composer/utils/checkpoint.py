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
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import torch
from packaging import version

from composer.utils import dist, reproducibility
from composer.utils.file_helpers import (FORMAT_NAME_WITH_DIST_AND_TIME_TABLE, format_name_with_dist,
                                         format_name_with_dist_and_time, get_file, is_tar)
from composer.utils.misc import is_model_deepspeed, using_torch_2
from composer.utils.object_store import ObjectStore

if TYPE_CHECKING:
    from composer.core import AlgorithmPass, State
    from composer.loggers import Logger, LoggerDestination

log = logging.getLogger(__name__)

__all__ = ['get_save_filename', 'load_checkpoint', 'save_checkpoint', 'download_checkpoint']

_COMPOSER_STATES_FILENAME = 'composer_states.pt'
_DEEPSPEED_TAG = 'deepspeed'  # always tag with the same, deterministic name. We'll rename the tarball to the appropriate name.
_TORCH_DISTRIBUTED_CHECKPOINTS_FILENAME = f'__{dist.get_global_rank()}_0.distcp'


def _get_checkpoint_validation_function() -> Optional[Callable[[Union[Path, str]], bool]]:
    """Get the validation function by name.

    Args:
        name (str): Qualified name of the checkpoint validation function.
                    It should be in the form '{module_name}.{fn_name}'.

    Returns:
        Callable[[Union[Path, str]], bool] The checkpoint validation function that returns
            True given a valid checkpoint and False otherwise.
    """
    name = os.environ.get('CHECKPOINT_VALIDATION_FUNCTION', None)
    if name is None:
        return None
    splits = name.split('.')
    module_name, fn_name = '.'.join(splits[:-1]), splits[-1]
    module = import_module(module_name)
    fn = getattr(module, fn_name)
    log.debug(f'Checkpoint validation function {name} has been found.')
    return fn


def _ensure_valid_checkpoint(checkpoint_filepath: Union[Path, str]) -> Union[Path, str]:
    """Ensures that the checkpoint at checkpoint_filepath is valid.

    using the function specified by the CHECKPOINT_VALIDATION_FUNCTION environment variable.
    If CHECKPOINT_VALIDATION_FUNCTION is not set, we skip validation.

    Args:
        checkpoint_filepath (Union[Path,str]): The path to the checkpoint file.

    Raises:
        ValueError if checkpoint file is invalid.
    """
    # Get the validation function by name.
    validate = _get_checkpoint_validation_function()

    # No function name has been specified.
    if validate is None:
        log.debug('No validation function specified. Skipping checkpoint validation.')
        return checkpoint_filepath

    # Validate the checkpoint.
    if not validate(checkpoint_filepath):
        raise ValueError(f'Checkpoint at {checkpoint_filepath} is invalid.')

    log.debug(f'Checkpoint at {checkpoint_filepath} is valid.')
    return checkpoint_filepath


def _torch_load_with_validation(checkpoint_filepath: Union[Path, str], map_location: str) -> Any:
    """Validates and loads a torch checkpoint.

    Args:
        checkpoint_filepath (Union[Path,str]): The path to the checkpoint file.
        map_location (str): The location to load the checkpoint to.
    """
    return torch.load(_ensure_valid_checkpoint(checkpoint_filepath), map_location=map_location)


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

    def format(self, state: State, is_deepspeed: bool = False, keep_placeholders: bool = False) -> str:
        # if filename already has a suffix (e.g. file.pt), this would append to be file.pt.tar
        extra_suffix = '.tar' if is_deepspeed and not is_tar(self.filename) else ''
        if self.folder:
            if keep_placeholders:
                return os.path.join(
                    self.folder,
                    self.filename,
                ) + extra_suffix
            else:
                return os.path.join(
                    format_name_with_dist(self.folder, state.run_name),
                    format_name_with_dist_and_time(self.filename, state.run_name, state.timestamp),
                ) + extra_suffix
        else:
            if keep_placeholders:
                return self.filename + extra_suffix
            else:
                return format_name_with_dist_and_time(
                    self.filename,
                    state.run_name,
                    state.timestamp,
                ) + extra_suffix


def is_checkpoint_legacy_sharded(object_store: Optional[ObjectStore], source_path: str):
    metadata_path = str(Path(source_path) / Path('.metadata'))
    if object_store is None:
        return not os.path.exists(metadata_path)
    else:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                metadata_destination = os.path.join(str(temp_dir), '.metadata')
                object_store.download_object(object_name=metadata_path, filename=metadata_destination)
            return False
        except FileNotFoundError:
            return True


def load_checkpoint(
    path: str,
    state: State,
    logger: Logger,
    object_store: Optional[Union[ObjectStore, LoggerDestination]] = None,
    load_weights_only: bool = False,
    strict_model_weights: bool = False,
    progress_bar: bool = True,
    ignore_keys: Optional[Union[list[str], Callable[[dict], None]]] = None,
    exclude_algorithms: Optional[list[str]] = None,
    algorithm_passes: Optional[list[AlgorithmPass]] = None,
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
        ignore_keys (list[str] | (dict) -> None, optional): A list of paths for the ``state_dict`` of the checkpoint,
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
        exclude_algorithms (list[str], optional): A list of algorithm names to exclude from loading.
            By default, algorithms with `required_on_load=True` which were enabled when training the loaded
            checkpoint are automatically applied unless they conflict with a user specified algorithm. These
            algorithms often change the model, and not applying them could result in certain layers not having
            weights loaded.

            Example 1: ``exclude_algorithms = ["BlurPool"]`` would exclude BlurPool from loading.

            Example 2: ``exclude_algorithms = ["FusedLayerNorm", "Alibi"]`` would exclude FusedLayerNorm and Alibi from loading.

            (default: ``None``)
        algorithm_passes (list[AlgorithmPass], optional): A list of algorithm passes to apply to autoloaded algorithms
            to sort them into the correct order. (default: ``None``)

    Returns:
        Optional[list[dict[str, Any]]]: The RNG state dicts, indexed by global rank, if
            :attr:`load_weights_only` is not None. Otherwise, None.
    """
    using_legacy_sharded = False
    if state.fsdp_elastic_sharded_enabled:
        assert object_store is None or isinstance(
            object_store,
            ObjectStore), 'For loading sharded checkpoints load_object_store must be set with the class ObjectStore'
        using_legacy_sharded = is_checkpoint_legacy_sharded(object_store, path)

    if state.fsdp_elastic_sharded_enabled and not using_legacy_sharded:
        rng_state_dicts = load_sharded_checkpoint(
            source_path=path,
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
        # Download the checkpoint to the node-local folder
        log.debug('Loading checkpoint at %s', path)
        # Each node gets one unique folder to store checkpoints that is shared amongst all local ranks in that node.
        # If fsdp sharded state_dicts is enabled then EVERY rank gets a unique checkpoint folder.
        needs_unique_checkpoint_folder = state.fsdp_sharded_state_dict_enabled or dist.get_local_rank() == 0
        tempdir_ctx = tempfile.TemporaryDirectory() if needs_unique_checkpoint_folder else contextlib.nullcontext(None)
        with tempdir_ctx as tempdir:
            try:
                # Get the path to the proper checkpoint folder corresponding to the current rank's node.
                # If fsdp_sharded_state_dict_enabled then just use that rank's unique tempdir.
                node_checkpoint_folder = (tempdir if state.fsdp_sharded_state_dict_enabled else
                                          _get_local_rank_zero_path(tempdir))
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
    step_to_resume_from = state.timestamp.batch.value
    max_step_to_resume_from = state.device.tensor_to_device(torch.tensor(state.timestamp.batch.value,
                                                                         dtype=torch.int64))
    min_step_to_resume_from = state.device.tensor_to_device(torch.tensor(state.timestamp.batch.value,
                                                                         dtype=torch.int64))
    dist.all_reduce(max_step_to_resume_from, reduce_operation='MAX')
    dist.all_reduce(min_step_to_resume_from, reduce_operation='MIN')
    if max_step_to_resume_from.data != min_step_to_resume_from.data:
        raise RuntimeError(
            textwrap.dedent(
                f'Timestamp mismatch error: batch to resume from {step_to_resume_from} is not the same on all ranks. '
                'This usually occurs when at least one rank fails to save the last checkpoint '
                'while using sharded checkpointing + autoresume. '
                'Please manually resume by disabling autoresume and explicitly setting load_path '
                'to the most recent checkpoints that all ranks have saved. '
                'E.g. for the 10th batch: trainer = Trainer(autoresume=False, load_path="/path/to/checkpoint/ba10-rank{rank}.pt", ...). '
                'Remember to keep the {rank} placeholder!'))
    return rng_state_dicts


def load_sharded_checkpoint(
    source_path: str,
    state: State,
    logger: Logger,
    object_store: Optional[Union[ObjectStore, LoggerDestination]] = None,
    load_weights_only: bool = False,
    strict_model_weights: bool = False,
    progress_bar: bool = True,
    ignore_keys: Optional[Union[list[str], Callable[[dict], None]]] = None,
    exclude_algorithms: Optional[list[str]] = None,
    algorithm_passes: Optional[list[AlgorithmPass]] = None,
) -> Union[list[dict], None]:
    if not using_torch_2():
        raise ValueError(
            f'Sharded checkpoint loading requires torch version >= 2.0.0. You have torch version {torch.__version__}')

    using_multinode = dist.get_world_size() != dist.get_local_world_size()
    if not version.parse(torch.__version__) >= version.parse('2.0.1') and using_multinode:
        raise ValueError(
            f'Sharded checkpoint loading on >1 node requires torch version >= 2.0.1. You have torch version {torch.__version__}'
        )

    if state.fsdp_config is None:
        raise ValueError('Loading a sharded checkpoint requires passing an FSDP config to Trainer.')
    load_planner = state.fsdp_config['load_planner']
    _validate_load_planner(load_planner)

    from torch.distributed import checkpoint as dist_cp
    from torch.distributed.checkpoint.metadata import Metadata
    from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
    from torch.distributed.checkpoint.planner import LoadPlan, LoadPlanner

    def _get_num_ranks_that_saved_rng(metadata: Metadata):
        rng_inds = []
        for field_name, field_value in metadata.planner_data.items():
            if 'rng' in field_name:
                _, rng_rank_index, _ = field_value
                rng_inds.append(rng_rank_index)
        rng_inds = set(rng_inds)
        return len(rng_inds)

    class FileSystemReaderWithValidation(dist_cp.FileSystemReader):
        """FileSystemReader that validates checkpoint files prior to reading."""

        def __init__(self, path: str):
            if _get_checkpoint_validation_function() is None:
                log.info('No checkpoint validation function found when loading sharded checkpoints.')
            super().__init__(path)

        def read_data(self, plan: LoadPlan, planner: LoadPlanner):
            """Reads data file.

            Raises:
                ValueError if the data file is invalid.
            """
            validated_checkpoint_paths = set()
            for read_item in plan.items:
                data_path = self.path / self.storage_data[read_item.storage_index].relative_path
                if data_path in validated_checkpoint_paths:
                    continue
                _ensure_valid_checkpoint(data_path)
                validated_checkpoint_paths.add(data_path)
            return super().read_data(plan, planner)

        def read_metadata(self) -> Metadata:
            """Reads metadata file.

            Raises:
                ValueError if the metadata file is invalid.
            """
            metadata_file_path = self.path / '.metadata'
            _ensure_valid_checkpoint(metadata_file_path)
            return super().read_metadata()

    # A subclass of FileSystemReaderWithValidation that downloads files from the object store before reading them from the local filesystem.
    class DistCPObjectStoreReader(FileSystemReaderWithValidation):

        def __init__(self, source_path: str, destination_path: str, object_store):
            self.source_path = source_path
            self.destination_path = destination_path
            self.object_store = object_store

            # Download metadata file.
            Path(self.destination_path).mkdir(parents=True, exist_ok=True)
            metadata_destination = os.path.join(self.destination_path, '.metadata')
            if dist.get_local_rank() == 0:
                object_store.download_object(object_name=str(Path(source_path) / Path('.metadata')),
                                             filename=metadata_destination)
            dist.barrier()

            # FileSystemReader takes in a root directory in its constructor, which is the dir where
            # the metadata is expected to be stored. Also, this is parent directory for any shard file relative paths
            # specified in the metadata file.
            super().__init__(destination_path)

        def read_data(self, plan: LoadPlan, planner: LoadPlanner):
            # 1. Download to the destination all files that this rank is responsible for.
            for plan_item in plan.items:
                # Each plan item has a storage index which points to the relative path of the shard file at save time.
                relative_file_path = self.storage_data[plan_item.storage_index].relative_path
                # Download the shard file to the relative path it's associated to and save that relative path
                # to the root directory specified to the FileSystem reader constructor.
                file_destination = str(Path(self.destination_path) / Path(relative_file_path))
                # The file could have already been downloaded as diffeent plan items can point to same file.
                if not os.path.exists(file_destination):
                    self.object_store.download_object(object_name=str(
                        Path(self.source_path) / Path(relative_file_path)),
                                                      filename=file_destination)

            # 2. Wait for all ranks to finish.
            dist.barrier()

            # 3. Piggyback off of the FileSystemReader to read all the files now that they are downloaded.
            return super().read_data(plan, planner)

    # Check to make sure source_path is a directory.
    if object_store is None:
        if os.path.islink(source_path):
            source_path = os.path.join(os.path.dirname(source_path), os.readlink(source_path))
        if os.path.exists(source_path):
            if not os.path.isdir(source_path):
                raise ValueError(f'load_path must be a directory when using sharded state dict. Got {source_path}')
        else:
            raise FileNotFoundError(f'{source_path} not found!')

    download_dir_context = tempfile.TemporaryDirectory if object_store is not None else contextlib.nullcontext

    with download_dir_context() as temp_download_dir:
        if object_store is not None:
            # Get the tempfile made on local rank 0.
            local_rank0_index = dist.get_global_rank() - dist.get_local_rank()
            rank0_download_tempdir = str(dist.all_gather_object(temp_download_dir)[local_rank0_index])
            storage_reader = DistCPObjectStoreReader(source_path=source_path,
                                                     destination_path=str(
                                                         Path(rank0_download_tempdir) / Path('checkpoints')),
                                                     object_store=object_store)
        else:
            storage_reader = FileSystemReaderWithValidation(source_path)

        # We need no_grad because we overwrite tensor values with set_() when we do elastic loading and we don't want the set_ op recorded in the computation graph.
        with torch.no_grad():
            # 1. Load model and metadata first
            if load_weights_only:
                state_dict: Dict[str, Any] = {'state': {'model': state.get_model_state_dict()}}
            else:
                cur_state_dict = state.state_dict()
                # For older versions of torch, we load optimizer separately.
                if version.parse(torch.__version__) < version.parse('2.2.9'):
                    cur_state_dict.pop('optimizers')
                num_rng_ranks = _get_num_ranks_that_saved_rng(storage_reader.read_metadata())
                state_dict: Dict[str, Any] = {
                    'state': cur_state_dict,
                    'rng': reproducibility.get_rng_state()[:num_rng_ranks],
                }

            if ignore_keys:
                # Filter provided list of key paths
                if not callable(ignore_keys):
                    ignore_keys = glob_filter(ignore_keys)
                # Call function to modify state_dict
                ignore_keys(state_dict)
                # Ensure state exists
                state_dict['state'] = state_dict.get('state', {})

            # Only some ranks are meant to load checkpoint
            expect_file = False
            process_group = None
            device_mesh = state.fsdp_device_mesh
            if device_mesh is not None and device_mesh.ndim == 2:
                # If hybrid shard, only rank in first replica saves
                expect_file = (device_mesh.get_local_rank(mesh_dim=0) == 0)
                if expect_file:
                    process_group = device_mesh.get_group(1)  # Shard process_group for first replica
                    log.debug(f'global_rank={dist.get_global_rank()}, {expect_file=}')
            else:
                expect_file = True

            if version.parse(torch.__version__) > version.parse('2.2.9'):
                dist_cp.load(  # type: ignore
                    state_dict=state_dict,
                    storage_reader=storage_reader,
                    planner=load_planner,
                    process_group=process_group,
                )
            else:
                dist_cp.load_state_dict(
                    state_dict=state_dict,
                    storage_reader=storage_reader,
                    planner=load_planner,
                    process_group=process_group,
                )

            state.load_state_dict(
                state_dict['state'],
                logger,
                strict=strict_model_weights,
                exclude_algorithms=exclude_algorithms,
                algorithm_passes=algorithm_passes,
            )

            # 2. Optionally load optimizer
            # if we are using later than 2.2.9 then optimizer will already be loaded
            if version.parse(torch.__version__) < version.parse('2.2.9') and not load_weights_only:
                optim_state = load_sharded_optimizer_state_dict(model_state_dict=state.state_dict()['model'],
                                                                optimizer_key='optimizers',
                                                                storage_reader=storage_reader)
                state._legacy_load_optim_state(optim_state)

    return state_dict.get('rng', None)


def _get_local_rank_zero_path(path: Optional[str]) -> str:
    """Broadcasts the ``path`` from the LOCAL rank zero to all LOCAL ranks."""
    local_rank_zero = dist.get_global_rank() - dist.get_local_rank()
    paths = dist.all_gather_object(path)
    local_rank_zero_path = paths[local_rank_zero]
    assert local_rank_zero_path is not None, 'local rank zero provides the path'
    return local_rank_zero_path


def download_checkpoint(path: str,
                        node_checkpoint_folder: str,
                        object_store: Optional[Union[ObjectStore, LoggerDestination]],
                        progress_bar: bool,
                        fsdp_sharded_state_dict_enabled: bool = False,
                        deepspeed_sharded_checkpoint: bool = False) -> tuple[str, Optional[str], bool]:
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
                     'Please ensure that the checkpoint exists and your load_path was specified as a format string '
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
        # Use busy wait to avoid timeouts on large downloads for non-sharded checkpoints
        if not checkpoint_is_sharded:
            signal_file_path = os.path.join(node_checkpoint_folder,
                                            f'.node_{dist.get_node_rank()}_local_rank0_completed')
            if dist.get_local_rank() == 0:
                with open(signal_file_path, 'wb') as f:
                    f.write(b'local_rank0_completed')

            # Avoid the collective call until the local rank zero has finished trying to download the
            # checkpoint so that we don't timeout for large downloads. This syncs all processes on the
            # node
            with dist.local_rank_zero_download_and_wait(signal_file_path):
                # Then, wait to ensure every node has finished downloading the checkpoint
                dist.barrier()

            if dist.get_local_rank() == 0:
                os.remove(signal_file_path)
        dist.barrier()

    return composer_states_filepath, extracted_checkpoint_folder, extracted_rank_n


def _flatten_keys(obj: Any, paths: list[str], existing_path: str):
    """Recursively flatten the keys of a dictionary or list into a set of paths."""
    # Store path when we reach end, which is either non-dict or empty dict
    if isinstance(obj, list) and len(obj) > 0:
        for i, elm in enumerate(obj):
            _flatten_keys(elm, paths, f'{existing_path}/{i}')
    elif isinstance(obj, dict) and len(obj) > 0:
        for k, v in obj.items():
            _flatten_keys(v, paths, f'{existing_path}/{k}')
    # Remove leading /
    paths.append(existing_path.lstrip('/'))


def _remove_paths(obj: Union[list, dict[str, Any]], exclude_paths: list[list[str]]):
    # Build str(key) to key map to undo cast from glob filtering. Despite typing, some state_dict
    # keys are not strings, so we need to cast them back to their original type.
    str_key_to_key = {}
    if isinstance(obj, dict):
        for key in obj.keys():
            str_key_to_key[str(key)] = key

    # First determine the keys which will be recursed on and which will be removed entirely
    # Group the `exclude_paths` by the key
    keys_to_recurse = {}
    keys_to_remove = []
    for exclude_path_parts in exclude_paths:
        key = exclude_path_parts[0]
        # Cast list indices to int
        if isinstance(obj, list):
            key = int(key)
        # Un-str dict keys if necessary
        if key in str_key_to_key:
            key = str_key_to_key[key]
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


def glob_filter(exclude_globs: list[str]) -> Callable[[dict], None]:
    """Provides a function which deletes all subparts of a dictionary based on a list of paths."""

    def filter_func(state_dict: dict) -> None:
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
        if filtered_paths:
            filtered_paths_str = ', '.join(filtered_paths)
            log.info(f'Ignoring the following paths from the loaded checkpoint state_dict: {filtered_paths_str}')

        # Loop through all paths to exclude
        paths_to_remove = [path.split('/') for path in filtered_paths if len(path) > 0]
        _remove_paths(state_dict, paths_to_remove)

    return filter_func


def _validate_save_planner(save_planner: Optional[Any]) -> None:
    """Checks that ``save_planner`` is an instance of a :class:`~torch.distributed.checkpoint.planner.SavePlanner`.

    TODO(GRT-2456): Remove validation once we deprecate torch 1.13 and can use
    type hints.

    Raises:
        ValueError: If ``save_planner`` is not a
            :class:`~torch.distributed.checkpoint.planner.SavePlanner`.
    """
    from torch.distributed.checkpoint.planner import SavePlanner

    if save_planner is not None and not isinstance(save_planner, SavePlanner):
        raise ValueError((f'save_planner {type(save_planner)} is not a '
                          'torch.distributed.checkpoint.planner.SavePlanner'))


def _validate_load_planner(load_planner: Optional[Any]) -> None:
    """Checks that ``load_planner`` is an instance of a :class:`~torch.distributed.checkpoint.planner.LoadPlanner`.

    TODO(GRT-2456): Remove validation once we deprecate torch 1.13 and can use
    type hints.

    Raises:
        ValueError: If ``load_planner`` is not a
            :class:`~torch.distributed.checkpoint.planner.LoadPlanner`.
    """
    from torch.distributed.checkpoint.planner import LoadPlanner

    if load_planner is not None and not isinstance(load_planner, LoadPlanner):
        raise ValueError((f'load_planner {type(load_planner)} is not a '
                          'torch.distributed.checkpoint.planner.LoadPlanner'))


def safe_torch_load(
    composer_states_filepath: Union[Path, str],
    map_location: str = 'cpu',
    load_fsdp_monolith_rank0_only: bool = False,
) -> dict[str, Any]:
    """Load a torch checkpoint, catching errors due to backwards compatibility issues.

    Args:
        composer_states_filepath: The path to the checkpoint file.
        map_location: The location to load the checkpoint to.
        load_fsdp_monolith_rank0_only: Whether the checkpoint is a monolith FSDP checkpoint.
    """
    try:
        if load_fsdp_monolith_rank0_only:
            log.info(
                'Loading monolith FSDP checkpoint. Only rank 0 will load and broadcast non-weight/optimizer state.')
            state_dict_list = [None]
            model = None
            optimizer = None
            if dist.get_global_rank() == 0:
                state_dict_list[0] = _torch_load_with_validation(composer_states_filepath, map_location=map_location)
                # Don't broadcast model/optimizer state if they exist
                if 'model' in state_dict_list[0]['state']:
                    model = state_dict_list[0]['state']['model']
                    state_dict_list[0]['state']['model'] = None
                if 'optimizers' in state_dict_list[0]['state']:
                    optimizer = state_dict_list[0]['state']['optimizers']
                    state_dict_list[0]['state']['optimizers'] = None

            log.debug('Broadcasting state_dict to all ranks.')
            dist.broadcast_object_list(state_dict_list, src=0)
            state_dict: dict[str, Any] = state_dict_list[0]  # type: ignore

            if dist.get_global_rank() == 0:
                if model is not None:
                    state_dict['state']['model'] = model
                if optimizer is not None:
                    state_dict['state']['optimizers'] = optimizer

            return state_dict
        else:
            return _torch_load_with_validation(composer_states_filepath, map_location=map_location)
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
    ignore_keys: Optional[Union[list[str], Callable[[dict], None]]],
    exclude_algorithms: Optional[list[str]],
    algorithm_passes: Optional[list[AlgorithmPass]],
) -> Optional[list[dict[str, Any]]]:
    """Restore a checkpoint into ``state`` and returns the rng state dicts (if ``load_weights_only`` is False)."""
    # Now, all ranks load the checkpoint that local rank zero downloaded
    state_dict = safe_torch_load(
        composer_states_filepath=composer_states_filepath,
        load_fsdp_monolith_rank0_only=state.load_fsdp_monolith_rank0_only,
    )
    if ignore_keys:
        # Filter provided list of key paths
        if not callable(ignore_keys):
            ignore_keys = glob_filter(ignore_keys)
        # Call function to modify state_dict
        ignore_keys(state_dict)
        # Ensure state exists
        state_dict['state'] = state_dict.get('state', {})
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
        return state_dict.get('rng', None)


def get_save_filename(
    state: State,
    filename: str = 'ep{epoch}-ba{batch}-rank{rank}',
) -> str:
    """Gets full filename of save filename.

    Args:
        state (State): The :class:`~composer.core.State` to load the checkpoint into.
        filename (filename): The name of the save file.

    Returns:
        Full filename of save file.
    """
    if not state.fsdp_sharded_state_dict_enabled:
        is_deepspeed = is_model_deepspeed(state.model)
        return PartialFilePath(filename).format(state, is_deepspeed)

    # Sharded checkpoints get their own little folder.
    assert state.sharded_ckpt_prefix_dir is not None
    save_dirpath = Path(Path(filename).parent) / Path(state.sharded_ckpt_prefix_dir)
    save_dirpath = format_name_with_dist_and_time(str(save_dirpath), state.run_name, state.timestamp)
    # New name is now Trainer.save_folder / sharded_ckpt_prefix_dir / __{dist.get_global_rank()}_0.distcp’ if torch > 2
    # else Trainer.save_folder / sharded_ckpt_prefix_dir / ba{batch}_rank{dist.get_global_rank()}.pt’
    # e.g. path/to/my/checkpoints/ep1-ba2/__1_0.distcp if torch >2 else its path/to/my/checkpoints/ep1-ba2/b2-rank1.pt
    ckpt_filename = _TORCH_DISTRIBUTED_CHECKPOINTS_FILENAME if using_torch_2() else format_name_with_dist_and_time(
        Path(filename).name, state.run_name, state.timestamp)
    return str(Path(save_dirpath) / Path(ckpt_filename))


def _save_checkpoint(
    state: State,
    save_filename: str,
    *,
    weights_only: bool = False,
    ignore_keys: Optional[Union[List[str], Callable[[Dict], None]]] = None,
) -> Union[str, None]:  # noqa: D103

    is_deepspeed = is_model_deepspeed(state.model)

    if weights_only and not is_deepspeed:
        state_dict = {
            'state': {
                'model': state.get_model_state_dict(),
                'integrations': state._get_integrations_state_dict(),
                'metadata': state._get_state_metadata(),
            },
            'rng': reproducibility.get_rng_state(),
        }
    else:
        state_dict = {
            'state': state.state_dict(),
            'rng': reproducibility.get_rng_state(),
        }

    if ignore_keys:
        # Filter provided list of key paths
        if not callable(ignore_keys):
            ignore_keys = glob_filter(ignore_keys)
        # Call function to modify state_dict
        ignore_keys(state_dict)
        # Ensure state exists
        state_dict['state'] = state_dict.get('state', {})

    if state.fsdp_sharded_state_dict_enabled:
        # To load optimizer states with 2.0 <= torch < 2.2.9 , the optimizer state must be at the top
        # level of the state dict because the load_sharded_optimizer_state_dict function
        # requires a top level state dict key for the optimizer.
        # See https://github.com/pytorch/pytorch/blob/v2.0.1/torch/distributed/checkpoint/optimizer.py#L271
        # for more info.
        if using_torch_2() and version.parse(torch.__version__) < version.parse('2.2.9'):
            if not weights_only:
                state_dict['optimizers'] = state_dict['state'].pop('optimizers')
    log.debug('State dict created.')

    dirname = os.path.dirname(save_filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    # Only some ranks are meant to save checkpoint and produce a file
    expect_file = False

    # All ranks save for deepspeed
    if is_deepspeed:
        expect_file = True
        log.debug('Saving deepspeed checkpoints to %s...', save_filename)
        if dist.get_global_rank() == 0:
            with open(save_filename, 'wb') as f:
                torch.save(state_dict, f)
            if is_tar(save_filename):
                _compress_file(save_filename, basename=_COMPOSER_STATES_FILENAME)

        _save_deepspeed_model(state.deepspeed_model, save_filename)

    # Sharded checkpointing for torch >=2.0 uses the torch.distributed.checkpoint module.
    elif state.fsdp_elastic_sharded_enabled:
        if state.fsdp_config is None:
            raise ValueError('Saving a sharded checkpoint requires passing an FSDP config to Trainer.')
        save_planner = state.fsdp_config['save_planner']
        _validate_save_planner(save_planner)

        import torch.distributed.checkpoint as dist_cp

        log.debug(f'Saving sharded checkpoints to {save_filename}...')
        process_group = None
        device_mesh = state.fsdp_device_mesh
        if device_mesh is not None and device_mesh.ndim == 2:
            # If hybrid shard, only rank in first replica saves
            expect_file = (device_mesh.get_local_rank(mesh_dim=0) == 0)
            if expect_file:
                process_group = device_mesh.get_group(1)  # Shard process_group for first replica
                log.debug(f'global_rank={dist.get_global_rank()}, {expect_file=}')
        else:
            expect_file = True

        if expect_file:
            if version.parse(torch.__version__) > version.parse('2.2.9'):
                dist_cp.save(  # type: ignore
                    state_dict=state_dict,
                    storage_writer=dist_cp.FileSystemWriter(dirname),
                    planner=save_planner,
                    process_group=process_group,
                )
            else:
                dist_cp.save_state_dict(
                    state_dict=state_dict,
                    storage_writer=dist_cp.FileSystemWriter(dirname),
                    planner=save_planner,
                    process_group=process_group,
                )
        log.debug('Finished pytorch save state dict')

    # Only rank 0 saves the state_dict unless you are using sharded checkpointing with torch <2.0
    elif dist.get_global_rank() == 0 or state.fsdp_sharded_state_dict_enabled:
        expect_file = True
        log_msg = f'Saving sharded checkpoints to {save_filename}...' if state.fsdp_sharded_state_dict_enabled else f'Saving monolithic checkpoint to {save_filename}'
        with open(save_filename, 'wb') as f:
            log.debug(log_msg)
            torch.save(state_dict, f)

        log.debug(f'Global rank 0 done saving checkpoint to disk at {save_filename}.')

        if is_tar(save_filename):
            _compress_file(save_filename, basename=_COMPOSER_STATES_FILENAME)

    else:
        log.debug(f'Only rank 0 is saving a checkpoint, so rank {dist.get_global_rank()} skips checkpointing.')

    dist.barrier()  # ensure all ranks saved their files

    if expect_file:
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


def save_checkpoint(
    state: State,
    filename: str = 'ep{epoch}-ba{batch}-rank{rank}',
    *,
    weights_only: bool = False,
    ignore_keys: Optional[Union[List[str], Callable[[Dict], None]]] = None,
) -> Union[str, None]:  # noqa: D103
    save_filename = get_save_filename(state, filename)
    return _save_checkpoint(state, save_filename, weights_only=weights_only, ignore_keys=ignore_keys)


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
        list[pathlib.Path]: The list of checkpoint files saved, indexed by the rank of the process.

        .. note::

            When using DeepSpeed, each process (rank) saves its own checkpoint file.
            When doing multi-node training, the filepaths are valid only on each process's node;
            Composer does not move checkpoint files between nodes.

            Otherwise, when not using DeepSpeed, each list will contain only one filepath,
            since only the rank zero process saves checkpoints.
"""
