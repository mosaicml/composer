# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Callback to save checkpoints during training."""

from __future__ import annotations

import logging
import os
import pathlib
import shutil
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any, Callable, Optional, Union

from composer.core import Callback, Event, State, Time, Timestamp
from composer.loggers import Logger, MLFlowLogger, MosaicMLLogger
from composer.utils import (
    FORMAT_NAME_WITH_DIST_AND_TIME_TABLE,
    FORMAT_NAME_WITH_DIST_TABLE,
    PartialFilePath,
    RemoteFilesExistingCheckStatus,
    RemoteUploader,
    checkpoint,
    create_interval_scheduler,
    create_symlink_file,
    dist,
    ensure_folder_has_no_conflicting_files,
    format_name_with_dist,
    format_name_with_dist_and_time,
    is_model_deepspeed,
    parse_uri,
    partial_format,
)
from composer.utils.checkpoint import _TORCH_DISTRIBUTED_CHECKPOINTS_METADATA_FILENAME
from composer.utils.compression import get_compressor, is_compressed_pt
from composer.utils.object_store.mlflow_object_store import MLFLOW_EXPERIMENT_ID_FORMAT_KEY, MLFLOW_RUN_ID_FORMAT_KEY

log = logging.getLogger(__name__)

__all__ = ['CheckpointSaver']


class CheckpointSaver(Callback):  # noqa: D101
    __doc__ = f"""Callback to save checkpoints.

    .. note::

        If the ``folder`` argument is specified when constructing the :class:`.Trainer`, then the :class:`.CheckpointSaver`
        callback need not be constructed manually. However, for advanced checkpointing use cases
        (such as saving a weights-only checkpoint at one interval and the full training state
        at another interval), instance(s) of this :class:`.CheckpointSaver` callback can be specified in the
        ``callbacks`` argument of the :class:`.Trainer`, as shown in the example below.

    Example

    .. testsetup::

        from composer.callbacks.checkpoint_saver import CheckpointSaver

    .. doctest::

        >>> trainer = Trainer(..., callbacks=[
        ...     CheckpointSaver(
        ...         folder='{{run_name}}/checkpoints',
        ...         filename="ep{{epoch}}-ba{{batch}}-rank{{rank}}",
        ...         latest_filename="latest-rank{{rank}}",
        ...         save_interval="1ep",
        ...         weights_only=False,
        ...     )
        ... ])

    Args:
        folder (str, optional): Format string for the save_folder where checkpoints will be saved.
            Default: ``'{{run_name}}/checkpoints'``.

            The following format variables are available:

            {textwrap.indent(FORMAT_NAME_WITH_DIST_TABLE, prefix='            ')}

            .. note::

                When training with multiple devices (i.e. GPUs), ensure that ``'{{rank}}'`` appears in the format.
                Otherwise, multiple processes may attempt to write to the same file.

        filename (str, optional): A format string describing how to name checkpoints.
            Default: ``'ep{{epoch}}-ba{{batch}}-rank{{rank}}.pt'``.

            Checkpoints will be saved approximately to ``{{folder}}/{{filename.format(...)}}``.

            The following format variables are available:

            {textwrap.indent(FORMAT_NAME_WITH_DIST_AND_TIME_TABLE, prefix='            ')}


            .. note::

                *   By default, only the rank zero process will save a checkpoint file.

                *   When using DeepSpeed, each rank will save a checkpoint file in tarball format. DeepSpeed
                    requires tarball format, as it saves model and optimizer states in separate files.
                    Ensure that ``'{{rank}}'`` appears within the ``filename``. Otherwise, multiple ranks
                    may attempt to write to the same file(s), leading to corrupted checkpoints. If no tarball file
                    extension is specified, ``'.tar'`` will be used.

                *   To write to compressed tar files (regardless of whether DeepSpeed is enabled), set the file
                    extension to ``'.tar.gz'``, ``'.tgz'``, ``'.tar.bz2'``, or ``'.tar.lzma'`` (depending on the
                    desired compression algorithm).

                *   To write to compressed pt files (when DeepSpeed is disabled), set the file extension to
                    ``'.pt.bz2'``, ``'.pt.gz'``, ``'.pt.lz4'``, ``'.pt.lzma'``, ``'.pt.lzo'``, ``'.pt.xz'``,
                    ``'.pt.zst'``
                    (depending on the desired algorithm). You must have the corresponding CLI tool installed.
                    ``lz4`` is a good choice for a modest space saving while being very fast to compress.

            .. warning::

                Using compression will block the training loop while checkpoints are being compressed and the
                compressibility of checkpoints can vary significantly depending on your setup. As such, we
                recommend saving checkpoints without compression by default.

                If you have the ``lz4`` command available on your system, you may want to try saving as ``.pt.lz4``
                as the overhead is minimal (usually less than a second) and the saved space can sometimes
                be significant (1% - 40%).

            Consider the following scenario where:

            *   The :attr:`~.State.run_name` is ``'awesome-training-run'``
            *   The default ``folder='{{run_name}}/checkpoints'`` is used.
            *   The default ``name='ep{{epoch}}-ba{{batch}}-rank{{rank}}'`` is used.
            *   The current epoch count is ``1``.
            *   The current batch count is ``42``.

            When DeepSpeed is not being used, the rank zero process will save the checkpoint to
            ``"awesome-training-run/checkpoints/ep1-ba42-rank0"``.

            When DeepSpeed is being used, each rank (process) will save checkpoints to::

                awesome-training-run/checkpoints/ep1-ba42-rank0.tar
                awesome-training-run/checkpoints/ep1-ba42-rank1.tar
                awesome-training-run/checkpoints/ep1-ba42-rank2.tar
                ...

        remote_file_name (str, optional): Format string for the checkpoint's remote file name.
            Default: ``"{{run_name}}/checkpoints/ep{{epoch}}-ba{{batch}}-rank{{rank}}"``.

            After the checkpoint is saved, it will be periodically uploaded.
            The remote file name will be determined by this format string.

            .. seealso:: :doc:`Uploading Files</trainer/file_uploading>` for notes for file uploading.

            The same format variables for ``filename`` are available.

            Leading slashes (``'/'``) will be stripped.

            To disable uploading checkpoints, set this parameter to ``None``.
        latest_filename (str, optional): A format string for a symlink which points to the last saved checkpoint.
            Default: ``'latest-rank{{rank}}.pt'``.

            Symlinks will be created approximately at ``{{folder}}/{{latest_filename.format(...)}}``.

            The same format variables as for ``name`` are available.

            To disable symlinks, set this parameter to ``None``.

            Consider the following scenario, where:

            *   The :attr:`~.State.run_name` is 'awesome-training-run'
            *   The default ``folder='{{run_name}}/checkpoints'`` is used.
            *   The default ``name='ep{{epoch}}-ba{{batch}}-rank{{rank}}'`` is used.
            *   The default ``latest_filename='latest-rank{{rank}}'`` is used.
            *   The current epoch count is ``1``.
            *   The current batch count is ``42``.

            When DeepSpeed is not being used, the rank zero process will save the checkpoint to
            ``'awesome-training-run/checkpoints/ep1-ba42-rank0'``,
            and a symlink will be created at
            ``'awesome-training-run/checkpoints/latest-rank0' -> 'awesome-training-run/checkpoints/ep1-ba42-rank0'``

            When DeepSpeed is being used, each rank (process) will save checkpoints to::

                awesome-training-run/checkpoints/ep1-ba42-rank0.tar
                awesome-training-run/checkpoints/ep1-ba42-rank1.tar
                awesome-training-run/checkpoints/ep1-ba42-rank2.tar
                ...

            Corresponding symlinks will be created at::

                awesome-training-run/checkpoints/latest-rank0.tar -> awesome-training-run/checkpoints/ep1-ba42-rank0.tar
                awesome-training-run/checkpoints/latest-rank1.tar -> awesome-training-run/checkpoints/ep1-ba42-rank1.tar
                awesome-training-run/checkpoints/latest-rank2.tar -> awesome-training-run/checkpoints/ep1-ba42-rank2.tar
                ...
        latest_remote_file_name (str, optional): Format string for the checkpoint's latest symlink remote file name.
            Default: ``'{{run_name}}/checkpoints/latest-rank{{rank}}"``.

            Whenever a new checkpoint is saved, a symlink is created or updated to point to the latest checkpoint's ``remote_file_name``.
            The remote file name will be determined by this format string. This parameter has no effect if ``latest_filename`` or ``remote_file_name`` is ``None``.

            .. seealso:: :doc:`Uploading Files</trainer/file_uploading>` for notes for file uploading.

            The same format variables for ``filename`` are available.

            Leading slashes (``'/'``) will be stripped.

            To disable symlinks in logger, set this parameter to ``None``.

        overwrite (bool, optional): Whether existing checkpoints should be overridden.
            If ``False`` (the default), then the ``folder`` must not exist or must not contain checkpoints which may conflict
            with the current run. Default: ``False``.

        save_interval (Time | str | int | (State, Event) -> bool): A :class:`.Time`, time-string, integer (in epochs),
            or a function that takes (state, event) and returns a boolean whether a checkpoint should be saved.

            If an integer, checkpoints will be saved every n epochs.
            If :class:`.Time` or a time-string, checkpoints will be saved according to this interval.

            .. seealso:: :func:`.checkpoint_periodically`

            If a function, then this function should take two arguments (:class:`.State`, :class:`.Event`).
            The first argument will be the current state of the trainer, and the second argument will be
            be :attr:`.Event.BATCH_CHECKPOINT` or :attr:`.Event.EPOCH_CHECKPOINT` (depending on the current training
            progress). It should return ``True`` if a checkpoint should be saved given the current state and
            event.

        num_checkpoints_to_keep (int, optional): The number of checkpoints to keep locally. The oldest checkpoints
            are removed first. Set to ``-1`` to keep all checkpoints locally. Default: ``-1``.

            Checkpoints will be removed after they have been uploaded. For example, when this callback
            is used in conjunction with the :class:`.RemoteUploaderDownloader`, set this
            parameter to ``0`` to immediately delete checkpoints from the local disk after they have been uploaded to
            the object store.

            This parameter only controls how many checkpoints are kept locally; checkpoints are not deleted from
            remote file systems.

        weights_only (bool): If ``True``, save only the model weights instead of the entire training state.
            This parameter must be ``False`` when using DeepSpeed. Default: ``False``.

        ignore_keys (list[str] | (dict) -> None, optional): A list of paths for the ``state_dict`` of the checkpoint,
            which, when provided, will be ignored from the state_dict before a checkpoint is saved. Each path is a list
            of strings specifying the keys to index into ``state_dict`` joined together with `/` as a separator (as PyTorch
            uses `.` in parameter names). If a prefix is provided, all children are also ignored (see Example 2).
            See :mod:`composer.core.state` for the structure of state_dict.

            Example 1: ``save_ignore_keys = ["state/model/layer1.weights", "state/model/layer1.bias"]`` would ignore
            layer 1 weights and bias.

            Example 2: ``save_ignore_keys = ["state/model/*"]`` would ignore the entire model, which would have the same
            effect as the previous example if there was only 1 layer.

            Example 3: ``save_ignore_keys = ["state/model/layer*.weights"]`` would ignore all weights in the model.

            Example 4: ``save_ignore_keys = ["state/rank_zero_seed", "rng"]`` would reset all randomness when
            saving the checkpoint.

            If a callable, it should take one argument which is the state_dict. The callable is free to arbitrarily modify
            the state_dict before it is loaded.

            (default: ``None``)

    Attributes:
        saved_checkpoints (list[tuple[Timestamp, list[pathlib.Path]]]): The checkpoint timestamps and filepaths.

            This list contains tuples of the save timestamp and the checkpoint filepaths.
            This list will have at most ``num_checkpoints_to_keep`` entries. The latest checkpoint
            will be at the end.

            .. note::

                When using DeepSpeed, the index of a filepath in each list corresponds to the global rank of
                the process that wrote that file. Each filepath is valid only on the process's (rank's) node.

                Otherwise, when not using DeepSpeed, each sub-list will contain only one filepath since only rank zero
                saves checkpoints.
    """

    def __init__(
        self,
        folder: Union[str, pathlib.Path] = '{run_name}/checkpoints',
        filename: Union[str, pathlib.Path] = 'ep{epoch}-ba{batch}-rank{rank}.pt',
        remote_file_name: Optional[Union[str, pathlib.Path]
                                  ] = ('{run_name}/checkpoints/'
                                       'ep{epoch}-ba{batch}-rank{rank}.pt'),
        latest_filename: Optional[Union[str, pathlib.Path]] = 'latest-rank{rank}.pt',
        latest_remote_file_name: Optional[Union[str, pathlib.Path]] = '{run_name}/checkpoints/latest-rank{rank}.pt',
        save_interval: Union[Time, str, int, Callable[[State, Event], bool]] = '1ep',
        *,
        overwrite: bool = False,
        num_checkpoints_to_keep: int = -1,
        weights_only: bool = False,
        ignore_keys: Optional[Union[list[str], Callable[[dict], None]]] = None,
        num_concurrent_uploads: int = 1,
        upload_timeout_in_seconds: int = 3600,
    ):

        backend, _, local_folder = parse_uri(str(folder))
        if local_folder == '':
            local_folder = '.'

        is_remote_folder = backend != ''
        if is_remote_folder:  # If uploading to a remote path, use a temporary directory to save local checkpoints.
            local_folder = os.path.join(tempfile.mkdtemp(), local_folder)

        filename = str(filename)
        remote_file_name = str(remote_file_name) if remote_file_name is not None else None
        latest_filename = str(latest_filename) if latest_filename is not None else None
        latest_remote_file_name = str(latest_remote_file_name) if latest_remote_file_name is not None else None

        # want to fail early if a required CLI tool is missing to ensure no training time is wasted
        for name in [filename, remote_file_name, latest_filename, latest_remote_file_name]:
            if name is not None and is_compressed_pt(name):
                get_compressor(name).check_exists()

        if not callable(save_interval):
            save_interval = create_interval_scheduler(save_interval)
        self.save_interval = save_interval
        self.last_checkpoint_batch: Optional[Time] = None

        self.folder = local_folder

        self.filename = PartialFilePath(filename.lstrip('/'), local_folder)
        self.latest_filename = PartialFilePath(latest_filename.lstrip('/'), local_folder) if latest_filename else None
        self.remote_file_name = PartialFilePath(remote_file_name) if remote_file_name else None
        self.latest_remote_file_name = PartialFilePath(latest_remote_file_name) if latest_remote_file_name else None

        self.overwrite = overwrite
        self.saved_checkpoints: list[str] = []
        self.all_saved_checkpoints_to_timestamp: dict[str, Timestamp] = {}
        self.num_checkpoints_to_keep = num_checkpoints_to_keep
        self.weights_only = weights_only
        self.ignore_keys = ignore_keys

        self.start_batch = None

        self.remote_uploader = None
        self.rank_saves_symlinks: bool = False
        self.tmp_dir_for_symlink = tempfile.TemporaryDirectory()
        self.num_concurrent_uploads = num_concurrent_uploads
        self.upload_timeout_in_seconds = upload_timeout_in_seconds
        # Allow unit test to override this to make it faster
        self._symlink_upload_wait_before_next_try_in_seconds = 30.0
        self.pid = os.getpid()
        self.symlink_count = 0
        self.symlink_upload_tasks = []

        if backend != '':
            self.remote_uploader = RemoteUploader(
                remote_folder=str(folder),
                num_concurrent_uploads=self.num_concurrent_uploads,
            )

    def init(self, state: State, logger: Logger) -> None:
        # If MLFlowLogger is being used, format MLFlow-specific placeholders in the save folder and paths.
        # Assumes that MLFlowLogger comes before CheckpointSaver in the list of loggers.
        for destination in logger.destinations:
            if isinstance(destination, MLFlowLogger):
                mlflow_format_kwargs = {
                    MLFLOW_EXPERIMENT_ID_FORMAT_KEY: destination._experiment_id,
                    MLFLOW_RUN_ID_FORMAT_KEY: destination._run_id,
                }
                self.folder = partial_format(self.folder, **mlflow_format_kwargs)

                self.filename.folder = self.folder
                if self.latest_filename is not None:
                    self.latest_filename.folder = self.folder

                # The remote paths have the placeholders in their filename rather than folder
                if self.remote_file_name is not None:
                    self.remote_file_name.filename = partial_format(
                        self.remote_file_name.filename,
                        **mlflow_format_kwargs,
                    )
                if self.latest_remote_file_name is not None:
                    self.latest_remote_file_name.filename = partial_format(
                        self.latest_remote_file_name.filename,
                        **mlflow_format_kwargs,
                    )
                break

        if self.remote_uploader is not None:
            self.remote_uploader.init()
        folder = format_name_with_dist(self.folder, state.run_name)
        os.makedirs(folder, exist_ok=True)

    def fit_start(self, state: State, logger: Logger) -> None:
        if not self.overwrite:
            # checks that save_folder contains no files with a timestamp after the current timestamp,
            # which has potential for future conflicts.
            folder = format_name_with_dist(self.folder, state.run_name)
            ensure_folder_has_no_conflicting_files(folder, self.filename.filename, state.timestamp)

        dist.barrier()  # holds all ranks until folder check is done

        if is_model_deepspeed(state.model) and self.weights_only:
            raise NotImplementedError('weights_only=True is not supported when using DeepSpeed.')

        self.start_batch = state.timestamp.batch

    def batch_checkpoint(self, state: State, logger: Logger):
        assert callable(self.save_interval)
        if self.save_interval(state, Event.BATCH_CHECKPOINT) and self.last_checkpoint_batch != state.timestamp.batch:
            self._save_checkpoint(
                state,
                logger,
            )

    def epoch_checkpoint(self, state: State, logger: Logger):
        assert callable(self.save_interval)
        if self.save_interval(state, Event.EPOCH_CHECKPOINT) and self.last_checkpoint_batch != state.timestamp.batch:
            self._save_checkpoint(
                state,
                logger,
            )

    def iteration_checkpoint(self, state: State, logger: Logger):
        assert callable(self.save_interval)
        if (
            self.save_interval(state, Event.ITERATION_CHECKPOINT) and
            self.last_checkpoint_batch != state.timestamp.batch
        ):
            self._save_checkpoint(
                state,
                logger,
            )

    def state_dict(self) -> dict[str, Any]:
        state_dict = {}

        all_checkpoints = []
        for save_filename, timestamp in self.all_saved_checkpoints_to_timestamp.items():
            all_checkpoints.append((save_filename, timestamp.state_dict()))

        state_dict['all_saved_checkpoints_to_timestamp'] = all_checkpoints
        return state_dict

    def load_state_dict(self, state: dict[str, Any]):
        if 'all_saved_checkpoints_to_timestamp' in state:
            for (save_filename, timestamp_state) in state['all_saved_checkpoints_to_timestamp']:
                load_timestamp = Timestamp()
                load_timestamp.load_state_dict(timestamp_state)
                self.all_saved_checkpoints_to_timestamp[save_filename] = load_timestamp

    def _upload_checkpoint(
        self,
        remote_file_name: str,
        local_file_name: str,
        local_remote_file_names: list[str],
        logger: Logger,
    ):
        if self.remote_uploader is not None:
            self.remote_uploader.upload_file_async(
                remote_file_name=remote_file_name,
                file_path=pathlib.Path(local_file_name),
                overwrite=self.overwrite,
            )
            local_remote_file_names.append(remote_file_name)
        else:
            logger.upload_file(
                remote_file_name=remote_file_name,
                file_path=local_file_name,
                overwrite=self.overwrite,
            )

    def _save_checkpoint(self, state: State, logger: Logger):
        self.last_checkpoint_batch = state.timestamp.batch

        is_deepspeed = is_model_deepspeed(state.model)

        if is_deepspeed and '{rank}' not in self.filename.filename:
            raise ValueError(f'Save filename {self.filename.filename} must have {{rank}} for deepspeed.')

        # save the checkpoint to the filename
        filename_with_placeholders = self.filename.format(state, is_deepspeed, keep_placeholders=True)
        save_filename = checkpoint.get_save_filename(state, filename_with_placeholders)
        # Store before saving so state_dict in checkpoint has reference to latest checkpoint (itself)
        self.all_saved_checkpoints_to_timestamp[save_filename] = state.timestamp

        saved_path = checkpoint.save_checkpoint(
            state=state,
            filename=filename_with_placeholders,
            weights_only=self.weights_only,
            ignore_keys=self.ignore_keys,
        )
        log.debug(f'Checkpoint locally saved to {saved_path}')

        self.symlink_count += 1
        # Remote checkpoint file names on this rank
        local_remote_file_names = []
        all_remote_filenames = []

        if not saved_path:  # not all ranks save
            if self.remote_file_name is not None and self.remote_uploader is not None:
                all_remote_filenames = dist.all_gather_object(local_remote_file_names)
            return

        metadata_local_file_path = None
        if dist.get_global_rank() == 0 and state.fsdp_sharded_state_dict_enabled:
            metadata_local_file_path = format_name_with_dist_and_time(
                os.path.join(Path(saved_path).parent, _TORCH_DISTRIBUTED_CHECKPOINTS_METADATA_FILENAME),
                state.run_name,
                state.timestamp,
            )

        self.rank_saves_symlinks = dist.get_global_rank() == 0 or not state.fsdp_sharded_state_dict_enabled
        if self.latest_filename is not None and self.num_checkpoints_to_keep != 0:
            symlink = self.latest_filename.format(state, is_deepspeed)
            os.makedirs(os.path.dirname(symlink), exist_ok=True)
            try:
                os.remove(symlink)
            except FileNotFoundError:
                pass
            # Sharded checkpoints for torch >2.0 use directories not files for load_paths
            if state.fsdp_sharded_state_dict_enabled:
                src_path = str(pathlib.Path(saved_path).parent)
            else:
                src_path = saved_path
            if self.rank_saves_symlinks:
                os.symlink(os.path.relpath(src_path, os.path.dirname(symlink)), symlink)

        # if remote file name provided, upload the checkpoint
        if self.remote_file_name is not None:
            if state.fsdp_sharded_state_dict_enabled:
                remote_file_name = self.remote_file_name.format(
                    state,
                    is_deepspeed,
                    keep_placeholders=True,
                ).lstrip('/')
                assert state.fsdp_config is not None
                remote_prefix = state.fsdp_config.sharded_ckpt_prefix_dir
                assert remote_prefix is not None
                ckpt_filename = checkpoint._TORCH_DISTRIBUTED_CHECKPOINTS_FILENAME
                remote_file_name = os.path.join(pathlib.Path(remote_file_name).parent, remote_prefix, ckpt_filename)
                remote_file_name = format_name_with_dist_and_time(remote_file_name, state.run_name, state.timestamp)
                # Upload metadata file.
                # The metadata file contains info related to which shards are saved where.
                if dist.get_global_rank() == 0 and state.fsdp_sharded_state_dict_enabled:
                    metadata_remote_file_name = format_name_with_dist_and_time(
                        os.path.join(Path(remote_file_name).parent, _TORCH_DISTRIBUTED_CHECKPOINTS_METADATA_FILENAME),
                        state.run_name,
                        state.timestamp,
                    )
                    assert metadata_local_file_path is not None
                    self._upload_checkpoint(
                        remote_file_name=metadata_remote_file_name,
                        local_file_name=metadata_local_file_path,
                        local_remote_file_names=local_remote_file_names,
                        logger=logger,
                    )
            else:
                remote_file_name = self.remote_file_name.format(
                    state,
                    is_deepspeed,
                ).lstrip('/')

            log.debug(f'Uploading checkpoint to {remote_file_name}')
            try:
                self._upload_checkpoint(
                    remote_file_name=remote_file_name,
                    local_file_name=saved_path,
                    local_remote_file_names=local_remote_file_names,
                    logger=logger,
                )
            except FileExistsError as e:
                raise FileExistsError(
                    f'Uploading checkpoint failed with error: {e}. overwrite was set to {self.overwrite}. To overwrite checkpoints with Trainer, set save_overwrite to True.',
                ) from e

            if self.remote_uploader is not None:
                all_remote_filenames = dist.all_gather_object(local_remote_file_names)

            # symlinks stay the same with sharded checkpointing
            if self.latest_remote_file_name is not None:
                symlink_name = self.latest_remote_file_name.format(
                    state,
                    is_deepspeed,
                ).lstrip('/') + '.symlink'

                # create and upload a symlink file
                symlink_filename = os.path.join(
                    self.tmp_dir_for_symlink.name,
                    f'latest.{self.symlink_count}.symlink',
                )
                # Sharded checkpoints for torch >2.0 use directories not files for load_paths
                if state.fsdp_sharded_state_dict_enabled:
                    src_path = str(pathlib.Path(remote_file_name).parent)
                else:
                    src_path = remote_file_name
                log.debug(f'Creating symlink file {symlink_filename} -> {src_path}')
                if self.rank_saves_symlinks:
                    create_symlink_file(src_path, symlink_filename)
                    if self.remote_uploader is not None:
                        remote_checkpoint_file_names = []
                        for file_names in all_remote_filenames:
                            remote_checkpoint_file_names += file_names
                        check_remote_files_exist_future = self.remote_uploader.check_remote_files_exist_async(
                            remote_checkpoint_file_names=remote_checkpoint_file_names,
                            max_wait_time_in_seconds=self.upload_timeout_in_seconds,
                            wait_before_next_try_in_seconds=self._symlink_upload_wait_before_next_try_in_seconds,
                        )
                        self.symlink_upload_tasks.append(
                            (check_remote_files_exist_future, symlink_filename, symlink_name),
                        )
                    else:
                        logger.upload_file(
                            remote_file_name=symlink_name,
                            file_path=symlink_filename,
                            overwrite=True,
                        )

        self.saved_checkpoints.append(saved_path)

        if self.num_checkpoints_to_keep >= 0:
            self._rotate_checkpoints(sharding_enabled=state.fsdp_sharded_state_dict_enabled)

    def _rotate_checkpoints(self, sharding_enabled: bool = False):
        while len(self.saved_checkpoints) > self.num_checkpoints_to_keep:
            prefix_dir = None
            checkpoint_to_delete = self.saved_checkpoints.pop(0)
            prefix_dir = str(Path(checkpoint_to_delete).parent)
            if not sharding_enabled:
                os.remove(checkpoint_to_delete)
            else:
                if dist.get_global_rank() == 0:
                    shutil.rmtree(prefix_dir)

    def _log_checkpoint_upload(self, logger: Logger):
        for destination in logger.destinations:
            if isinstance(destination, MosaicMLLogger):
                destination.log_metadata({'checkpoint_uploaded_time': time.time()}, force_flush=True)

    def batch_end(self, state: State, logger: Logger) -> None:
        del state  # unused
        if self.remote_uploader is None:
            return
        self.remote_uploader.check_workers()
        if not self.rank_saves_symlinks:
            return
        undone_symlink_upload_tasks = []
        for (check_remote_files_exist_future, local_symlink_file,
             remote_symlink_file) in reversed(self.symlink_upload_tasks):
            if not check_remote_files_exist_future.done():
                undone_symlink_upload_tasks.insert(
                    0,
                    (check_remote_files_exist_future, local_symlink_file, remote_symlink_file),
                )
                continue
            if check_remote_files_exist_future.done():
                result = check_remote_files_exist_future.result()
                if result == RemoteFilesExistingCheckStatus.EXIST:
                    self.remote_uploader.upload_file_async(
                        remote_file_name=remote_symlink_file,
                        file_path=local_symlink_file,
                        overwrite=True,
                    )
                    self._log_checkpoint_upload(logger)
                    break
                else:
                    raise RuntimeError(f'Failed to check if checkpoint files upload finish: {result}')
        self.symlink_upload_tasks = undone_symlink_upload_tasks

    def fit_end(self, state: State, logger: Logger) -> None:
        del state  # unused
        if self.remote_uploader is None:
            return
        log.info('Waiting for checkpoint uploading to finish')
        self.remote_uploader.wait()
        if self.rank_saves_symlinks and len(self.symlink_upload_tasks) > 0:
            log.debug('Uploading symlink to the latest checkpoint')
            # We only need to upload a symlink pointing to the latest checkpoint files, so we can ignore successful uploads of older checkpoints.
            check_remote_files_exist_future, local_symlink_file, remote_symlink_file = self.symlink_upload_tasks[-1]
            result = check_remote_files_exist_future.result()
            if result == RemoteFilesExistingCheckStatus.EXIST:
                symlink_upload_future = self.remote_uploader.upload_file_async(
                    remote_file_name=remote_symlink_file,
                    file_path=local_symlink_file,
                    overwrite=True,
                )
                symlink_upload_future.result()
                self._log_checkpoint_upload(logger)
            else:
                raise RuntimeError(f'Failed to check if checkpoint files upload finish: {result}')
        log.info('Checkpoint uploading finished!')

    def post_close(self):
        if self.remote_uploader is not None:
            # Wait the symlink file upload to finish and close remote uploader
            try:
                self.remote_uploader.wait_and_close()
            except Exception as e:
                log.error(f'RemoteUploader run into exception {e}')
