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
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import torch
from packaging import version
from torch.distributed import checkpoint as dist_cp
from torch.distributed._tensor import DeviceMesh
from torch.distributed.checkpoint.metadata import Metadata
from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
from torch.distributed.checkpoint.planner import LoadPlan, LoadPlanner
from torch.distributed.checkpoint.storage import StorageReader
from torch.distributed.distributed_c10d import ProcessGroup

from composer.utils import dist, reproducibility
from composer.utils.compression import get_compressor, is_compressed_pt
from composer.utils.file_helpers import (
    FORMAT_NAME_WITH_DIST_AND_TIME_TABLE,
    extract_path_from_symlink,
    format_name_with_dist,
    format_name_with_dist_and_time,
    get_file,
    is_tar,
    is_uri,
    maybe_create_object_store_from_uri,
    parse_uri,
)
from composer.utils.misc import ParallelismType, is_model_deepspeed, partial_format
from composer.utils.object_store import ObjectStore
from composer.utils.retrying import retry

if TYPE_CHECKING:
    from composer.core import AlgorithmPass, State
    from composer.loggers import Logger, LoggerDestination

log = logging.getLogger(__name__)

__all__ = ['get_save_filename', 'load_checkpoint', 'save_checkpoint', 'download_checkpoint']

_COMPOSER_STATES_FILENAME = 'composer_states.pt'
_DEEPSPEED_TAG = 'deepspeed'  # always tag with the same, deterministic name. We'll rename the tarball to the appropriate name.
_TORCH_DISTRIBUTED_CHECKPOINTS_FILENAME = f'__{dist.get_global_rank()}_0.distcp'
_TORCH_DISTRIBUTED_CHECKPOINTS_METADATA_FILENAME = '.metadata'


def _get_checkpoint_validation_function(
) -> Optional[Callable[[Union[Path, str], Optional[list[tuple[int, int]]]], bool]]:
    """Get the validation function specified by the environment variable `CHECKPOINT_VALIDATION_FUNCTION`.

    Returns:
        Callable[[Union[Path, str], Optional[int], Optional[int]], bool] The checkpoint validation function that returns
            True given a valid checkpoint and optionally a list of offsets and lengths to check and False otherwise.
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


def _ensure_valid_checkpoint(checkpoint_filepath: Union[Path, str],
                             specs: Optional[list[tuple[int, int]]] = None) -> Union[Path, str]:
    """Ensures that the checkpoint at checkpoint_filepath is valid.

    using the function specified by the CHECKPOINT_VALIDATION_FUNCTION environment variable.
    If CHECKPOINT_VALIDATION_FUNCTION is not set, we skip validation.

    Args:
        checkpoint_filepath (Union[Path,str]): The path to the checkpoint file.
        specs (Optional[list[tuple[int,int]]]): A list of offsets and lengths to check. Defaults to None.

    Raises:
        ValueError if checkpoint file is invalid.
    """
    # Get the validation function by name.
    validate = _get_checkpoint_validation_function()

    # No function name has been specified.
    if validate is None:
        return checkpoint_filepath

    # Validate the checkpoint.
    if not validate(checkpoint_filepath, specs):
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


def _is_rng_key(key: str, value: tuple) -> bool:
    """Check if the key is an RNG key.

    We expect the RNG key to be of the form 'rng.{rank}.cuda|torch|python|numpy'.
    This function ensures that we don't accidentally pick up other keys.
    """
    starts_with_rng = key.startswith('rng')
    ends_with_expected = key.endswith(('cuda', 'torch', 'python', 'numpy'))
    three_parts = isinstance(value, tuple) and len(value) == 3
    if starts_with_rng and ends_with_expected and three_parts:
        return True

    return False


def _get_num_ranks_that_saved_rng(metadata: Metadata):
    rng_inds = []
    for field_name, field_value in metadata.planner_data.items():
        if _is_rng_key(field_name, field_value):
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
        path_to_specs: dict[str, list[tuple[int, int]]] = {}
        for read_item in plan.items:
            item_md = self.storage_data[read_item.storage_index]
            path = os.path.join(self.path, item_md.relative_path)
            path_to_specs.setdefault(path, []).append((item_md.offset, item_md.length))
        for path, spec in path_to_specs.items():
            _ensure_valid_checkpoint(path, spec)
        return super().read_data(plan, planner)

    def read_metadata(self) -> Metadata:
        """Reads metadata file.

        Raises:
            ValueError if the metadata file is invalid.
        """
        metadata_file_path = os.path.join(self.path, '.metadata')
        _ensure_valid_checkpoint(metadata_file_path)
        return super().read_metadata()


@retry(num_attempts=5)
def download_object_or_file(
    object_name: str,
    file_destination: Union[str, Path],
    object_store: Union[ObjectStore, LoggerDestination],
):
    if isinstance(object_store, ObjectStore):
        object_store.download_object(
            object_name=object_name,
            filename=file_destination,
        )
    else:
        object_store.download_file(
            remote_file_name=object_name,
            destination=str(file_destination),
        )


# A subclass of FileSystemReaderWithValidation that downloads files from the object store before reading them from the local filesystem.
class DistCPObjectStoreReader(FileSystemReaderWithValidation):

    def __init__(
        self,
        source_path: str,
        destination_path: str,
        object_store: Optional[Union[ObjectStore, LoggerDestination]] = None,
        device_mesh: Optional[DeviceMesh] = None,
    ):

        if object_store is None:
            if not is_uri(source_path):
                raise ValueError('When object_store is None, source_path must be a URI.')
            object_store = maybe_create_object_store_from_uri(source_path)
            _, _, source_path = parse_uri(source_path)

        self.source_path = source_path
        self.destination_path = destination_path
        self.object_store = object_store
        self.device_mesh = device_mesh

        # Download metadata file.
        Path(self.destination_path).mkdir(parents=True, exist_ok=True)
        metadata_destination = os.path.join(self.destination_path, '.metadata')
        if dist.get_local_rank() == 0:
            metadata_path = str(Path(source_path) / Path('.metadata'))
            assert object_store is not None
            download_object_or_file(metadata_path, metadata_destination, object_store)
        dist.barrier()

        # FileSystemReader takes in a root directory in its constructor, which is the dir where
        # the metadata is expected to be stored. Also, this is parent directory for any shard file relative paths
        # specified in the metadata file.
        super().__init__(destination_path)

    def read_data(self, plan: LoadPlan, planner: LoadPlanner):
        # Download files if not using HSDP or if on first replica with HSDP enabled
        first_replica = True
        if self.device_mesh is not None and self.device_mesh.mesh_dim_names is not None and ParallelismType.DATA_PARALLEL_REPLICATE.value in self.device_mesh.mesh_dim_names:
            hsdp_index = self.device_mesh.mesh_dim_names.index(ParallelismType.DATA_PARALLEL_REPLICATE.value)
            first_replica = self.device_mesh.get_local_rank(mesh_dim=hsdp_index) == 0

        # 1. Collect the relative paths to download for all ranks for deduplication
        relative_file_paths = set()
        for plan_item in plan.items:
            relative_file_paths.add(self.storage_data[plan_item.storage_index].relative_path)
        all_file_paths = dist.all_gather_object(relative_file_paths)

        # 2. Download to the destination all files this rank needs if on first replica
        download_error = False
        if first_replica:
            log.debug(f'Rank {dist.get_global_rank()} starting to download files.')

            # Get the lowest rank in the current node
            local_rank_0 = dist.get_global_rank() - dist.get_local_rank()

            try:
                for plan_item in plan.items:
                    relative_file_path = self.storage_data[plan_item.storage_index].relative_path
                    # Check if the file is scheduled to be downloaded by a lower rank on the same node
                    # i.e. if rank 0 and rank 1 on the same node have the same the same required file,
                    # only rank 0 should download it and not rank 1.
                    is_downloaded = any(
                        relative_file_path in all_file_paths[i] for i in range(local_rank_0, dist.get_global_rank())
                    )

                    # Download the shard file to the relative path it's associated to and save that relative path
                    # to the root directory specified to the FileSystem reader constructor.
                    file_destination = str(Path(self.destination_path) / Path(relative_file_path))

                    # The file could have already been downloaded as different plan items can point to same file.
                    if not is_downloaded and not os.path.exists(file_destination):
                        log.debug(f'Downloading {relative_file_path} to {file_destination}.')
                        object_name = str(Path(self.source_path) / Path(relative_file_path))
                        assert self.object_store is not None
                        download_object_or_file(object_name, file_destination, self.object_store)
                        log.debug(f'Finished downloading {relative_file_path} to {file_destination}.')
            except Exception as e:
                log.error(f'Exception {type(e)} raised during downloading: {str(e)}')
                download_error = True

        # PyTorch will capture any exception of this function,
        # and dist.all_gather_objects(exception) before raising it.
        # If that all_gather_objects fails, the exception is never visible to user.
        # We raise the exception from all ranks to ensure the user sees it.
        download_error_tensor = dist.get_device(None).tensor_to_device(torch.tensor(1 if download_error else 0))
        error_by_rank = dist.all_gather(download_error_tensor)
        failed_ranks = []
        for rank, error in enumerate(list(error_by_rank)):
            if error > 0:
                failed_ranks.append(rank)
                download_error = True

        if download_error:
            raise RuntimeError(
                f'Ranks {failed_ranks} failed to download.',
                'To see the full error please look at the logs for that rank, which are logged via log.error.',
            )

        # 3. Wait for all ranks to finish.
        log.debug(f'Rank {dist.get_global_rank()} finished downloading all files.')
        dist.barrier()
        log.debug('Done waiting for all ranks to finish downloading files.')

        # 4. Broadcast files to all other replicas if HSDP
        if self.device_mesh is not None and self.device_mesh.mesh_dim_names is not None and ParallelismType.DATA_PARALLEL_REPLICATE.value in self.device_mesh.mesh_dim_names:
            # Broadcast file to all replicas
            replicate_index = self.device_mesh.mesh_dim_names.index(ParallelismType.DATA_PARALLEL_REPLICATE.value)
            shard_index = self.device_mesh.mesh_dim_names.index(ParallelismType.DATA_PARALLEL_SHARD.value)
            replicate_process_group = self.device_mesh.get_group(replicate_index)
            shard_process_group = self.device_mesh.get_group(shard_index)
            shard_size = self.device_mesh.size(shard_index)
            rank_in_first_replica = dist.get_global_rank() % shard_size
            sender = dist.get_global_rank() == rank_in_first_replica
            receiver = dist.get_global_rank() != rank_in_first_replica

            # Send list of files to all ranks
            file_list = [[
                file_name for file_name in sorted(os.listdir(self.destination_path)) if file_name.endswith('.distcp')
            ]]
            dist.broadcast_object_list(file_list, src=rank_in_first_replica, group=replicate_process_group)
            file_list = file_list[0]
            log.debug(f'list of files to broadcast: {file_list}')

            # Send each file to the appropriate rank
            for file_name in file_list:
                if dist.get_local_rank() == 0 or (
                    dist.get_global_rank(shard_process_group) == 0  # pyright: ignore[reportGeneralTypeIssues]
                ):  # Only 1 rank per node needs to transfer file
                    full_path = os.path.join(self.destination_path, file_name)
                    log.debug(f'Transferring {full_path=}')
                    file_object = [None]
                    if sender:
                        with open(full_path, 'rb') as f:
                            file_object = [{'content': f.read()}]
                    dist.broadcast_object_list(
                        file_object,
                        src=dist.get_global_rank() % shard_size,
                        group=replicate_process_group,
                    )
                    received_file_object = file_object[0]
                    assert received_file_object is not None
                    if receiver and not os.path.exists(full_path) and dist.get_local_rank() == 0:
                        with open(full_path, 'wb') as f:
                            f.write(received_file_object['content'])

            log.debug(f'Rank {dist.get_global_rank()} finished transferring files to all ranks.')
            dist.barrier()
            log.debug(
                f'Done waiting for all ranks to finish transferring files. Local checkpoint files: {sorted(os.listdir(self.destination_path))}',
            )

        # 5. Piggyback off of the FileSystemReader to read all the files now that they are downloaded.
        return super().read_data(plan, planner)


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


def load_checkpoint(
    path: str,
    state: State,
    logger: Logger,
    object_store: Optional[Union[ObjectStore, LoggerDestination]] = None,
    load_weights_only: bool = False,
    strict_model_weights: bool = True,
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
            match the model weights. (default: ``True``)
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
    path = partial_format(path, run_name=state.run_name)
    log.debug(f'Loading checkpoint from formatted path: {path}')

    if state.fsdp_sharded_state_dict_enabled:
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
        # Each node gets one unique folder to store checkpoints that is shared amongst all local ranks in that node.
        # If fsdp sharded state_dicts is enabled then EVERY rank gets a unique checkpoint folder.
        tempdir_ctx = tempfile.TemporaryDirectory() if dist.get_local_rank() == 0 else contextlib.nullcontext(None)
        with tempdir_ctx as tempdir:
            try:
                # Get the path to the proper checkpoint folder corresponding to the current rank's node.
                # If fsdp_sharded_state_dict_enabled then just use that rank's unique tempdir.
                node_checkpoint_folder = _get_local_rank_zero_path(tempdir)

                composer_states_filepath, extracted_checkpoint_folder, extracted_rank_n = download_checkpoint(
                    path=path,
                    node_checkpoint_folder=node_checkpoint_folder,
                    object_store=object_store,
                    progress_bar=progress_bar,
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

    # Verify all ranks resumed on same step
    step_to_resume_from = state.timestamp.batch.value
    max_step_to_resume_from = state.device.tensor_to_device(
        torch.tensor(state.timestamp.batch.value, dtype=torch.int64),
    )
    min_step_to_resume_from = state.device.tensor_to_device(
        torch.tensor(state.timestamp.batch.value, dtype=torch.int64),
    )
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
                'Remember to keep the {rank} placeholder!',
            ),
        )
    return rng_state_dicts


def dist_cp_load(
    state_dict: dict[str, Any],
    storage_reader: StorageReader,
    load_planner: Optional[LoadPlanner] = None,
):
    if version.parse(torch.__version__) >= version.parse('2.4.0'):
        from torch.distributed.checkpoint.utils import CheckpointException
        try:
            dist_cp.load(
                state_dict=state_dict,
                storage_reader=storage_reader,
                planner=load_planner,
            )
        except CheckpointException as e:
            checkpoint_metadata = storage_reader.read_metadata().state_dict_metadata
            if 'state.metadata' in checkpoint_metadata and 'state.metadata.composer_env_info.composer_version' not in checkpoint_metadata:
                # Torch 2.4 changed the way how state dict is flattened. It broke backward compatibility.
                # Torch issue: https://github.com/pytorch/pytorch/issues/133923.
                # We override the traverse_state_dict so that the load planner could
                # use the old way of flattening the state dict
                log.debug('Trying to load checkpointing saved before torch 2.4')

                import torch.distributed.checkpoint._nested_dict as nested_dict
                import torch.distributed.checkpoint._sharded_tensor_utils as sharded_tensor_util
                from torch.distributed.checkpoint._traverse import traverse_state_dict as traverse_2_4_0

                from composer.trainer._patch_pytorch import traverse_state_dict as backward_compatible_traverse

                nested_dict.traverse_state_dict = backward_compatible_traverse
                sharded_tensor_util.traverse_state_dict = backward_compatible_traverse

                dist_cp.load(
                    state_dict=state_dict,
                    storage_reader=storage_reader,
                    planner=load_planner,
                )
                # Revert the override
                nested_dict.traverse_state_dict = traverse_2_4_0
                sharded_tensor_util.traverse_state_dict = traverse_2_4_0
            else:
                raise e
    else:
        dist_cp.load_state_dict(
            state_dict=state_dict,
            storage_reader=storage_reader,
            planner=load_planner,
            no_dist=(not dist.is_initialized()),
        )


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
    using_multinode = dist.get_world_size() != dist.get_local_world_size()
    if not version.parse(torch.__version__) >= version.parse('2.0.1') and using_multinode:
        raise ValueError(
            f'Sharded checkpoint loading on >1 node requires torch version >= 2.0.1. You have torch version {torch.__version__}',
        )

    if state.fsdp_config is None:
        raise ValueError('Loading a sharded checkpoint requires passing an FSDP config to Trainer.')

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
            if source_path.endswith('.symlink'):
                source_path = extract_path_from_symlink(source_path, object_store=object_store)
            storage_reader = DistCPObjectStoreReader(
                source_path=source_path,
                destination_path=str(Path(rank0_download_tempdir) / Path('checkpoints')),
                object_store=object_store,
                device_mesh=state.device_mesh,
            )
        else:
            storage_reader = FileSystemReaderWithValidation(source_path)

        # We need no_grad because we overwrite tensor values with set_() when we do elastic loading and we don't want the set_ op recorded in the computation graph.
        with torch.no_grad():
            # 1. Load metadata first for backwards compatability check
            # We need to check if the "optimizers" is at the root of the state dict to determine
            # how to load the optimizer state.
            try:
                metadata = storage_reader.read_metadata()
            except AttributeError as e:
                if '_MEM_FORMAT_ENCODING' in str(e):
                    raise ValueError(
                        'Unable to read checkpoint metadata. The checkpoint was likely saved with a '
                        'newer version of torch. Upgrade your torch version to load this checkpoint.',
                    )
                else:
                    raise
            # Retrieve all top-level keys of the metadata.
            top_level_keys = [v[0] for v in metadata.planner_data.values()]
            optimizers_at_root = 'optimizers' in top_level_keys

            # 2. Load model and metadata
            if load_weights_only:
                state_dict: dict[str, Any] = {'state': {'model': state.get_model_state_dict()}}
            else:
                cur_state_dict = state.state_dict()
                # If 'optimizers' is at root-level, we load it separately.
                if optimizers_at_root:
                    cur_state_dict.pop('optimizers')
                num_rng_ranks = _get_num_ranks_that_saved_rng(storage_reader.read_metadata())
                state_dict: dict[str, Any] = {
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

            dist_cp_load(
                state_dict=state_dict,
                storage_reader=storage_reader,
                load_planner=state.fsdp_config.load_planner,
            )
            log.info(f'Loaded state dict')
            state.load_state_dict(
                state_dict['state'],
                logger,
                strict=strict_model_weights,
                exclude_algorithms=exclude_algorithms,
                algorithm_passes=algorithm_passes,
            )

            # 3. Optionally load optimizer
            # If 'optimizers' was not at root-level, then it will already be loaded
            if optimizers_at_root and not load_weights_only:
                optim_state = load_sharded_optimizer_state_dict(
                    model_state_dict=state.state_dict()['model'],
                    optimizer_key='optimizers',
                    storage_reader=storage_reader,
                )
                state._legacy_load_optim_state(optim_state)

    return state_dict.get('rng', None)


def _get_local_rank_zero_path(path: Optional[str]) -> str:
    """Broadcasts the ``path`` from the LOCAL rank zero to all LOCAL ranks."""
    local_rank_zero = dist.get_global_rank() - dist.get_local_rank()
    paths = dist.all_gather_object(path)
    local_rank_zero_path = paths[local_rank_zero]
    assert local_rank_zero_path is not None, 'local rank zero provides the path'
    return local_rank_zero_path


def download_checkpoint(
    path: str,
    node_checkpoint_folder: str,
    object_store: Optional[Union[ObjectStore, LoggerDestination]],
    progress_bar: bool,
    deepspeed_sharded_checkpoint: bool = False,
) -> tuple[str, Optional[str], bool]:
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
        composer_states_filepath = rank_zero_checkpoint_filepath

        if is_compressed_pt(path):
            original_path = path
            path = os.path.splitext(path)[0]
            compressor = get_compressor(original_path)
            with open(path, 'wb') as out_file:
                with compressor.decompress(original_path) as in_file:
                    shutil.copyfileobj(in_file, out_file)

    try:
        if not deepspeed_sharded_checkpoint and dist.get_local_rank() == 0:
            # If the checkpoint is not sharded, then local rank 0 on each node needs to download the
            # global rank 0 checkpoint
            path = _format_path_with_rank_zero(path)
            get_file(
                destination=rank_zero_checkpoint_filepath,
                path=path,
                object_store=object_store,
                progress_bar=progress_bar,
            )
            if extracted_checkpoint_folder is not None:
                try:
                    with tarfile.open(rank_zero_checkpoint_filepath) as tarball:
                        tarball.extractall(extracted_checkpoint_folder)
                except FileNotFoundError:
                    # Not re-raising the file-not-found error as that is irrelevant;
                    # the underlying issue is that the checkpoint file does not exist on the disk
                    # or could not be downloaded
                    raise RuntimeError(f'Checkpoint {path} does not exist')
        elif deepspeed_sharded_checkpoint:
            # If the checkpoint is sharded, then every rank needs to download its own checkpoint
            path = _format_path_with_current_rank(path)
            try:
                get_file(
                    destination=rank_n_checkpoint_filepath,
                    path=path,
                    object_store=object_store,
                    progress_bar=progress_bar,
                )
            except FileNotFoundError as e:
                raise FileNotFoundError((
                    f'Checkpoint {path} does not exist, but is required for sharded checkpointing '
                    f'on rank {dist.get_global_rank()}. Please ensure that the checkpoint exists '
                    'and your load_path was specified as a format string with the {rank} argument.'
                )) from e

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
        if not deepspeed_sharded_checkpoint:
            signal_file_path = os.path.join(
                node_checkpoint_folder,
                dist.get_node_signal_file_name(),
            )
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
                    f'No parts from loaded checkpoint state_dict were ignored by load_ignore_key {exclude_glob}',
                )
            filtered_paths.extend(filtered_paths_from_glob)
        filtered_paths = list(set(filtered_paths))
        if filtered_paths:
            filtered_paths_str = ', '.join(filtered_paths)
            log.info(f'Ignoring the following paths from the loaded checkpoint state_dict: {filtered_paths_str}')

        # Loop through all paths to exclude
        paths_to_remove = [path.split('/') for path in filtered_paths if len(path) > 0]
        _remove_paths(state_dict, paths_to_remove)

    return filter_func


def safe_torch_load(
    composer_states_filepath: Union[Path, str],
    map_location: str = 'cpu',
    load_monolith_rank0_only: bool = False,
) -> dict[str, Any]:
    """Load a torch checkpoint, catching errors due to backwards compatibility issues.

    Args:
        composer_states_filepath: The path to the checkpoint file.
        map_location: The location to load the checkpoint to.
        load_monolith_rank0_only: Whether the checkpoint is a monolith FSDP checkpoint.
    """
    try:
        if load_monolith_rank0_only:
            log.info(
                'Loading monolith FSDP checkpoint. Only rank 0 will load and broadcast non-weight/optimizer state.',
            )
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
            raise Exception(
                'As of v0.10.0, torchmetrics introduces a new required argument to Accuracy which '
                'breaks backwards compatibility. Unfortunately, this means that older checkpoints '
                'cannot be loaded with the metrics. In order to successfully load this model, please '
                'pass `load_ignore_keys = ["state/train_metrics/*", "state/eval_metrics/*"]`.',
            ) from e
        raise e
    except FileNotFoundError as e:
        if 'No such file or directory' in str(e) and dist.get_local_rank() != 0:
            local_rank_zero = dist.get_global_rank() - dist.get_local_rank()
            raise FileNotFoundError(
                f'No such file or directory: {e.filename}. '
                f'This likely implies a download failed on local rank 0, which is global rank {local_rank_zero}'
                f'Please check the logs for global rank {local_rank_zero} to debug the checkpoint download issue.',
            ) from e
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
        load_monolith_rank0_only=state.load_monolith_rank0_only,
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
    assert state.fsdp_config is not None
    remote_prefix = state.fsdp_config.sharded_ckpt_prefix_dir
    assert remote_prefix is not None
    save_dirpath = Path(Path(filename).parent) / Path(remote_prefix)
    save_dirpath = format_name_with_dist_and_time(str(save_dirpath), state.run_name, state.timestamp)
    # New name is now Trainer.save_folder / sharded_ckpt_prefix_dir / __{dist.get_global_rank()}_0.distcp’
    # e.g. path/to/my/checkpoints/ep1-ba2/__1_0.distcp
    ckpt_filename = _TORCH_DISTRIBUTED_CHECKPOINTS_FILENAME
    return str(Path(save_dirpath) / Path(ckpt_filename))


def _save_checkpoint(
    state: State,
    save_filename: str,
    *,
    weights_only: bool = False,
    ignore_keys: Optional[Union[list[str], Callable[[dict], None]]] = None,
) -> Union[str, None]:  # noqa: D103

    is_deepspeed = is_model_deepspeed(state.model)

    if weights_only and not is_deepspeed:
        state_dict = {
            'state': {
                'model': state.get_model_state_dict(),
                'integrations': state._get_integrations_state_dict(),
                'metadata': state._get_state_metadata(),
            },
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

    if state.fsdp_sharded_state_dict_enabled and not weights_only:
        # Only rank 0 saves RNG
        if dist.get_global_rank() > 0:
            state_dict.pop('rng')
        # To load optimizer states with 2.0 <= torch < 2.2.3 , the optimizer state must be at the top
        # level of the state dict because the load_sharded_optimizer_state_dict function
        # requires a top level state dict key for the optimizer.
        # See https://github.com/pytorch/pytorch/blob/v2.0.1/torch/distributed/checkpoint/optimizer.py#L271
        # for more info.
        if version.parse(torch.__version__) < version.parse('2.2.3'):
            state_dict['optimizers'] = state_dict['state'].pop('optimizers')

    log.debug('State dict created.')
    dirname = os.path.dirname(save_filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    # Only some ranks are meant to save checkpoint and produce a file
    expect_file = False

    # Save deepspeed checkpoint
    if is_deepspeed:
        expect_file = True
        log.debug('Saving deepspeed checkpoints to %s...', save_filename)
        if dist.get_global_rank() == 0:
            _write_checkpoint_file(state_dict, save_filename)

        _save_deepspeed_model(state.deepspeed_model, save_filename)
    # Save sharded checkpoint
    elif state.fsdp_sharded_state_dict_enabled:
        if state.fsdp_config is None:
            raise ValueError('Saving a sharded checkpoint requires passing an FSDP config to Trainer.')

        log.debug(f'Saving sharded checkpoints to {save_filename}...')
        process_group = None
        device_mesh = state.device_mesh
        if device_mesh is not None and device_mesh.mesh_dim_names is not None and ParallelismType.DATA_PARALLEL_REPLICATE.value in device_mesh.mesh_dim_names:
            # If hybrid shard, only rank in first replica saves
            hsdp_index = device_mesh.mesh_dim_names.index(ParallelismType.DATA_PARALLEL_REPLICATE.value)
            expect_file = device_mesh.get_local_rank(mesh_dim=hsdp_index) == 0
            if expect_file:
                process_group = device_mesh.get_group(1)  # Shard process_group for first replica
                assert isinstance(process_group, ProcessGroup)  # For type checker
                log.debug(f'Saving on global_rank={dist.get_global_rank()}, {expect_file=}')
        else:
            expect_file = True

        if expect_file:
            if version.parse(torch.__version__) >= version.parse('2.3.0'):
                save_planner = state.fsdp_config.save_planner
                if save_planner is None:
                    if version.parse(torch.__version__) < version.parse('2.4.0'):
                        # Dedup is only broken on <2.4
                        from composer.trainer._patch_pytorch import SavePlannerWithDedupFix

                        save_planner = SavePlannerWithDedupFix()
                    else:
                        from torch.distributed.checkpoint.default_planner import DefaultSavePlanner

                        save_planner = DefaultSavePlanner(dedup_save_to_lowest_rank=True)
                dist_cp.save(
                    state_dict=state_dict,
                    storage_writer=dist_cp.FileSystemWriter(dirname),
                    planner=save_planner,
                    process_group=process_group,
                )
            else:
                dist_cp.save_state_dict(
                    state_dict=state_dict,
                    storage_writer=dist_cp.FileSystemWriter(dirname),
                    planner=state.fsdp_config.save_planner,
                    process_group=process_group,
                )
        log.debug('Finished pytorch save state dict')
    # Save monolith checkpoint
    elif dist.get_global_rank() == 0:
        expect_file = True
        log.debug(f'Saving monolithic checkpoint to {save_filename}')
        _write_checkpoint_file(state_dict, save_filename)
        log.debug(f'Global rank 0 done saving checkpoint to disk at {save_filename}.')
    else:
        log.debug(f'Only rank 0 is saving a checkpoint, so rank {dist.get_global_rank()} skips checkpointing.')

    dist.barrier()  # Ensure all ranks saved their files

    if expect_file:
        assert os.path.exists(save_filename), 'Expected file to have been saved.'
        return save_filename
    else:
        # no file saved
        return None


def _write_checkpoint_file(state_dict: dict[str, Any], filename: str) -> None:
    """Write the given checkpoint state to the given path. Compressing if indicated to do so by the file extension."""
    if is_tar(filename):
        log.debug('Writing checkpoint tar file %s', filename)
        write_mode = _get_write_mode(filename)

        with tempfile.TemporaryDirectory(prefix='checkpoint') as tmpdir:
            with open(os.path.join(tmpdir, _COMPOSER_STATES_FILENAME), 'wb') as f:
                torch.save(state_dict, f)

            with tarfile.open(filename, write_mode) as tarball:
                tarball.add(tmpdir, arcname='')

    elif is_compressed_pt(filename):
        log.debug('Writing compressed checkpoint %s', filename)
        compressor = get_compressor(filename)
        with compressor.compress(filename) as f:
            torch.save(state_dict, f)

    else:
        log.debug('Writing uncompressed checkpoint %s', filename)
        with open(filename, 'wb') as f:
            torch.save(state_dict, f)


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
    ignore_keys: Optional[Union[list[str], Callable[[dict], None]]] = None,
) -> Union[str, None]:  # noqa: D103
    # Clear the cache in case we are near the memory limit to give some space for NCCL.
    torch.cuda.empty_cache()
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

            *   To write to compressed tar files (regardless of whether DeepSpeed is enabled), set the file
                extension to ``'.tar.gz'``, ``'.tgz'``, ``'.tar.bz2'``, or ``'.tar.lzma'`` (depending on the
                desired compression algorithm).

            *   To write to compressed pt files (when DeepSpeed is disabled), set the file extension to
                ``'.pt.bz2'``, ``'.pt.gz'``, ``'.pt.lz4'``, ``'.pt.lzma'``, ``'.pt.lzo'``, ``'.pt.xz'``, ``'.pt.zst'``
                (depending on the desired algorithm). You must have the corresponding CLI tool installed.
                ``lz4`` is a good choice for a modest space saving while being very fast to compress.

        .. warning::

            Using compression will block the training loop while checkpoints are being compressed and the
            compressibility of checkpoints can vary significantly depending on your setup. As such, we
            recommend saving checkpoints without compression by default.

            If you have the ``lz4`` command available on your system, you may want to try saving as ``.pt.lz4``
            as the overhead is minimal (usually less than a second) and the saved space can sometimes
            be significant (1% - 40%).

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
