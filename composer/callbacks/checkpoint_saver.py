# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Callback to save checkpoints during training."""

from __future__ import annotations

import logging
import os
import pathlib
import tempfile
import textwrap
from typing import Callable, List, Optional, Tuple, Union

from composer.core import Event, State
from composer.core.callback import Callback
from composer.core.time import Time, Timestamp, TimeUnit
from composer.loggers import Logger
from composer.loggers.logger import LogLevel
from composer.utils import checkpoint, dist, is_model_deepspeed
from composer.utils.file_helpers import (FORMAT_NAME_WITH_DIST_AND_TIME_TABLE, FORMAT_NAME_WITH_DIST_TABLE,
                                         create_symlink_file, ensure_folder_has_no_conflicting_files,
                                         format_name_with_dist, format_name_with_dist_and_time, is_tar)

log = logging.getLogger(__name__)

__all__ = ['CheckpointSaver', 'checkpoint_periodically']


def checkpoint_periodically(interval: Union[str, int, Time]) -> Callable[[State, Event], bool]:
    r"""Helper function to create a checkpoint scheduler according to a specified interval.

    Args:
        interval (Union[str, int, :class:`.Time`]): The interval describing how often checkpoints should be
            saved. If an integer, it will be assumed to be in :attr:`.TimeUnit.EPOCH`\s.
            Otherwise, the unit must be either :attr:`.TimeUnit.EPOCH` or :attr:`.TimeUnit.BATCH`.

            Checkpoints will be saved every ``n`` batches or epochs (depending on the unit),
            and at the end of training.

    Returns:
        Callable[[State, Event], bool]: A function that can be passed as the ``save_interval``
            argument into the :class:`.CheckpointSaver`.
    """
    if isinstance(interval, str):
        interval = Time.from_timestring(interval)
    if isinstance(interval, int):
        interval = Time(interval, TimeUnit.EPOCH)

    if interval.unit == TimeUnit.EPOCH:
        save_event = Event.EPOCH_CHECKPOINT
    elif interval.unit == TimeUnit.BATCH:
        save_event = Event.BATCH_CHECKPOINT
    else:
        raise NotImplementedError(
            f'Unknown checkpointing interval: {interval.unit}. Must be TimeUnit.EPOCH or TimeUnit.BATCH.')

    last_checkpoint_batch: Optional[Time] = None

    def save_interval(state: State, event: Event):
        nonlocal last_checkpoint_batch
        elapsed_duration = state.get_elapsed_duration()
        assert elapsed_duration is not None, 'elapsed_duration is set on the BATCH_CHECKPOINT and EPOCH_CHECKPOINT'

        if elapsed_duration >= 1.0:
            # if doing batch-wise checkpointing, and we saved a checkpoint at the batch_checkpoint event
            # right before the epoch_checkpoint event, do not save another checkpoint at the epoch_checkpoint
            # event if the batch count didn't increase.
            if state.timestamp.batch != last_checkpoint_batch:
                last_checkpoint_batch = state.timestamp.batch
                return True

        if save_event == Event.EPOCH_CHECKPOINT:
            count = state.timestamp.epoch
        elif save_event == Event.BATCH_CHECKPOINT:
            count = state.timestamp.batch
        else:
            raise RuntimeError(f'Invalid save_event: {save_event}')

        if event == save_event and int(count) % int(interval) == 0:
            last_checkpoint_batch = state.timestamp.batch
            return True

        return False

    return save_interval


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
        folder (str, optional): Format string for the folder where checkpoints will be saved.
            Default: ``'{{run_name}}/checkpoints'``.

            The following format variables are available:

            {textwrap.indent(FORMAT_NAME_WITH_DIST_TABLE, prefix='            ')}

            .. note::

                When training with multiple devices (i.e. GPUs), ensure that ``'{{rank}}'`` appears in the format.
                Otherwise, multiple processes may attempt to write to the same file.

        filename (str, optional): A format string describing how to name checkpoints.
            Default: ``'ep{{epoch}}-ba{{batch}}-rank{{rank}}'``.

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

                *   To use compression (regardless of whether DeepSpeed is enabled), set the file extension
                    to ``'.tar.gz'``, ``'.tgz'``, ``'.tar.bzip'``, or ``'.tar.lzma'`` (depending on the desired
                    compression algorithm).

            .. warning::

                Using compression will block the training loop while checkpoints are being compressed. As such, we
                recommend saving checkpoints without compression.

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

        artifact_name (str, optional): Format string for the checkpoint's artifact name.
            Default: ``"{{run_name}}/checkpoints/ep{{epoch}}-ba{{batch}}-rank{{rank}}"``.

            After the checkpoint is saved, it will be periodically logged as a file artifact.
            The artifact name will be determined by this format string.

            .. seealso:: :meth:`.Logger.log_file_artifact` for file artifact logging.

            The same format variables for ``filename`` are available.

            Leading slashes (``'/'``) will be stripped.

            To disable logging trace files as file artifacts, set this parameter to ``None``.
        latest_filename (str, optional): A format string for a symlink which points to the last saved checkpoint.
            Default: ``'latest-rank{{rank}}'``.

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
        latest_artifact_name (str, optional): Format string for the checkpoint's latest symlink artifact name.
            Default: ``'{{run_name}}/checkpoints/latest-rank{{rank}}"``.

            Whenever a new checkpoint is saved, a symlink artifact is created or updated to point to the latest checkpoint's ``artifact_name``.
            The artifact name will be determined by this format string. This parameter has no effect if ``latest_filename`` or ``artifact_name`` is ``None``.

            .. seealso:: :func:`.write_symlink` for symlink artifact logging.

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

        weights_only (bool): If ``True``, save only the model weights instead of the entire training state.
            This parmeter must be ``False`` when using DeepSpeed. Default: ``False``.


        num_checkpoints_to_keep (int, optional): The number of checkpoints to keep locally. The oldest checkpoints
            are removed first. Set to ``-1`` to keep all checkpoints locally. Default: ``-1``.

            Checkpoints will be removed after they have been logged as a file artifact. For example, when this callback
            is used in conjunction with the :class:`.ObjectStoreLogger`, set this
            parameter to ``0`` to immediately delete checkpoints from the local disk after they have been uploaded to
            the object store.

            This parameter only controls how many checkpoints are kept locally; checkpoints are not deleted from
            artifact stores.

    Attributes:
        saved_checkpoints (List[Tuple[Timestamp, List[pathlib.Path]]]): The checkpoint timestamps and filepaths.

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
        folder: str = '{run_name}/checkpoints',
        filename: str = 'ep{epoch}-ba{batch}-rank{rank}',
        artifact_name: Optional[str] = '{run_name}/checkpoints/ep{epoch}-ba{batch}-rank{rank}',
        latest_filename: Optional[str] = 'latest-rank{rank}',
        latest_artifact_name: Optional[str] = '{run_name}/checkpoints/latest-rank{rank}',
        save_interval: Union[Time, str, int, Callable[[State, Event], bool]] = '1ep',
        *,
        overwrite: bool = False,
        num_checkpoints_to_keep: int = -1,
        weights_only: bool = False,
    ):
        if not callable(save_interval):
            save_interval = checkpoint_periodically(save_interval)

        self.folder = folder
        self.filename = filename
        self.artifact_name = artifact_name
        self.latest_filename = latest_filename
        self.latest_artifact_name = latest_artifact_name
        self.overwrite = overwrite

        self.save_interval = save_interval
        self.saved_checkpoints: List[Tuple[Timestamp, List[pathlib.Path]]] = []
        self.num_checkpoints_to_keep = num_checkpoints_to_keep
        self.weights_only = weights_only

    def init(self, state: State, logger: Logger) -> None:
        del logger  # unused
        folder = format_name_with_dist(self.folder, state.run_name)
        os.makedirs(folder, exist_ok=True)

    def fit_start(self, state: State, logger: Logger) -> None:
        del logger  # unused
        # Verify safety with self.overwrite. Note that this has to be done at fit_start as opposed to init since it requires state.timestamp
        # from any checkpoints which are loaded, and checkpoint loading happens after Event.INIT.
        if not self.overwrite:
            folder = format_name_with_dist(self.folder, state.run_name)
            ensure_folder_has_no_conflicting_files(folder, self.filename, state.timestamp)
        # Ensure no rank proceeds (and potentially attempts to write to the folder), until all ranks have validated that the folder is safe.
        dist.barrier()
        if is_model_deepspeed(state.model):
            if self.weights_only:
                NotImplementedError(
                    ('Saving checkpoints with `weights_only=True` is not currently supported when using DeepSpeed. '
                     'See https://github.com/mosaicml/composer/issues/685.'))

    def batch_checkpoint(self, state: State, logger: Logger):
        if self.save_interval(state, Event.BATCH_CHECKPOINT):
            # If training is finished, log at the FIT loglevel
            elapsed_duration = state.get_elapsed_duration()
            assert elapsed_duration is not None, 'elapsed_duration is set on Event.BATCH_CHECKPOINT'
            log_level = LogLevel.BATCH if elapsed_duration < 1.0 else LogLevel.FIT
            self._save_checkpoint(state, logger, log_level)

    def epoch_checkpoint(self, state: State, logger: Logger):
        if self.save_interval(state, Event.EPOCH_CHECKPOINT):
            elapsed_duration = state.get_elapsed_duration()
            assert elapsed_duration is not None, 'elapsed_duration is set on Event.BATCH_CHECKPOINT'
            log_level = LogLevel.EPOCH if elapsed_duration < 1.0 else LogLevel.FIT
            self._save_checkpoint(state, logger, log_level)

    def _save_checkpoint(self, state: State, logger: Logger, log_level: LogLevel):
        checkpoint_filepath = os.path.join(format_name_with_dist(self.folder, state.run_name), self.filename)
        checkpoint_filepaths = checkpoint.save_checkpoint(state, checkpoint_filepath, weights_only=self.weights_only)

        if dist.get_global_rank() < len(checkpoint_filepaths):
            # Log the checkpoint as an artifact
            checkpoint_filepath = checkpoint_filepaths[dist.get_global_rank()]
            if self.artifact_name is not None:
                artifact_name = format_name_with_dist_and_time(self.artifact_name, state.run_name,
                                                               state.timestamp).lstrip('/')
                if is_model_deepspeed(state.model) and not is_tar(artifact_name):
                    # Deepspeed requires tarballs; appending `.tar`
                    artifact_name += '.tar'
                logger.file_artifact(log_level=log_level,
                                     artifact_name=artifact_name,
                                     file_path=checkpoint_filepath,
                                     overwrite=self.overwrite)

            if self.latest_filename is not None:
                formatted_folder_path = format_name_with_dist(self.folder, state.run_name)
                symlink_name = os.path.join(
                    formatted_folder_path,
                    format_name_with_dist_and_time(
                        self.latest_filename,
                        state.run_name,
                        state.timestamp,
                    ).lstrip('/'),
                )
                if is_model_deepspeed(state.model) and not is_tar(symlink_name):
                    # Deepspeed requires tarballs; appending `.tar`
                    symlink_name += '.tar'
                symlink_dirname = os.path.dirname(symlink_name)
                if symlink_dirname:
                    os.makedirs(symlink_dirname, exist_ok=True)
                try:
                    os.remove(symlink_name)
                except FileNotFoundError:
                    pass
                relative_checkpoint_path = os.path.relpath(checkpoint_filepath, formatted_folder_path)
                os.symlink(relative_checkpoint_path, symlink_name)
                if self.artifact_name is not None and self.latest_artifact_name is not None:
                    symlink_artifact_name = format_name_with_dist_and_time(self.latest_artifact_name, state.run_name,
                                                                           state.timestamp).lstrip('/') + '.symlink'
                    artifact_name = format_name_with_dist_and_time(self.artifact_name, state.run_name,
                                                                   state.timestamp).lstrip('/')
                    # Always overwrite for symlinks since we use the same filename for latest
                    with tempfile.TemporaryDirectory() as tmpdir:
                        symlink_filename = os.path.join(tmpdir, 'latest.symlink')
                        create_symlink_file(artifact_name, symlink_filename)
                        logger.file_artifact(
                            log_level=log_level,
                            artifact_name=symlink_artifact_name,
                            file_path=symlink_filename,
                            overwrite=True,
                        )

        timestamp = state.timestamp

        self.saved_checkpoints.append((timestamp, checkpoint_filepaths))
        if self.num_checkpoints_to_keep >= 0:
            while len(self.saved_checkpoints) > self.num_checkpoints_to_keep:

                timestamp, checkpoint_filepaths = self.saved_checkpoints[0]
                if dist.get_global_rank() < len(checkpoint_filepaths):
                    # Remove this rank's checkpoint
                    os.remove(checkpoint_filepaths[dist.get_global_rank()])
                del self.saved_checkpoints[0]
