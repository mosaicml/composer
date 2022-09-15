# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Callback to save checkpoints during training."""

from __future__ import annotations

import logging
import os
import tempfile
import textwrap
from typing import Callable, List, Optional, Union

from composer.core import Event, State
from composer.core.callback import Callback
from composer.core.time import Time, TimeUnit
from composer.loggers import Logger
from composer.loggers.logger import LogLevel
from composer.utils import checkpoint, dist, is_model_deepspeed, reproducibility
from composer.utils.checkpoint import PartialFilePath
from composer.utils.file_helpers import (FORMAT_NAME_WITH_DIST_AND_TIME_TABLE, FORMAT_NAME_WITH_DIST_TABLE,
                                         create_symlink_file, ensure_folder_has_no_conflicting_files,
                                         format_name_with_dist)

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
        Callable[[State, Event], bool]: A function that can be passed as the ``checkpoint_save_interval``
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

    def checkpoint_save_interval(state: State, event: Event):
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

    return checkpoint_save_interval


class CheckpointSaver(Callback):  # noqa: D101
    __doc__ = f"""Callback to save checkpoints.

    .. note::

        If the ``checkpoint_save_path`` argument is specified when constructing the :class:`.Trainer`, then the :class:`.CheckpointSaver`
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
        ...         checkpoint_save_path='{{run_name}}/checkpoints',
        ...         checkpoint_filename="ep{{epoch}}-ba{{batch}}-rank{{rank}}",
        ...         latest_checkpoint_filename="latest-rank{{rank}}",
        ...         checkpoint_save_interval="1ep",
        ...         save_model_weights_only=False,
        ...     )
        ... ])

    Args:
        checkpoint_save_path (str, optional): Format string for the checkpoint_save_path where checkpoints will be saved.
            Default: ``'{{run_name}}/checkpoints'``.

            The following format variables are available:

            {textwrap.indent(FORMAT_NAME_WITH_DIST_TABLE, prefix='            ')}

            .. note::

                When training with multiple devices (i.e. GPUs), ensure that ``'{{rank}}'`` appears in the format.
                Otherwise, multiple processes may attempt to write to the same file.

        checkpoint_filename (str, optional): A format string describing how to name checkpoints.
            Default: ``'ep{{epoch}}-ba{{batch}}-rank{{rank}}.pt'``.

            Checkpoints will be saved approximately to ``{{checkpoint_save_path}}/{{checkpoint_filename.format(...)}}``.

            The following format variables are available:

            {textwrap.indent(FORMAT_NAME_WITH_DIST_AND_TIME_TABLE, prefix='            ')}


            .. note::

                *   By default, only the rank zero process will save a checkpoint file.

                *   When using DeepSpeed, each rank will save a checkpoint file in tarball format. DeepSpeed
                    requires tarball format, as it saves model and optimizer states in separate files.
                    Ensure that ``'{{rank}}'`` appears within the ``checkpoint_filename``. Otherwise, multiple ranks
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
            *   The default ``checkpoint_save_path='{{run_name}}/checkpoints'`` is used.
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

            .. seealso:: :doc:`Artifact Logging</trainer/artifact_logging>` for notes for file artifact logging.

            The same format variables for ``checkpoint_filename`` are available.

            Leading slashes (``'/'``) will be stripped.

            To disable logging trace files as file artifacts, set this parameter to ``None``.
        latest_checkpoint_filename (str, optional): A format string for a symlink which points to the last saved checkpoint.
            Default: ``'latest-rank{{rank}}.pt'``.

            Symlinks will be created approximately at ``{{checkpoint_save_path}}/{{latest_checkpoint_filename.format(...)}}``.

            The same format variables as for ``name`` are available.

            To disable symlinks, set this parameter to ``None``.

            Consider the following scenario, where:

            *   The :attr:`~.State.run_name` is 'awesome-training-run'
            *   The default ``checkpoint_save_path='{{run_name}}/checkpoints'`` is used.
            *   The default ``name='ep{{epoch}}-ba{{batch}}-rank{{rank}}'`` is used.
            *   The default ``latest_checkpoint_filename='latest-rank{{rank}}'`` is used.
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
            The artifact name will be determined by this format string. This parameter has no effect if ``latest_checkpoint_filename`` or ``artifact_name`` is ``None``.

            .. seealso:: :doc:`Artifact Logging</trainer/artifact_logging>` for notes for file artifact logging.

            The same format variables for ``checkpoint_filename`` are available.

            Leading slashes (``'/'``) will be stripped.

            To disable symlinks in logger, set this parameter to ``None``.

        overwrite_checkpoints (bool, optional): Whether existing checkpoints should be overwritten.
            If ``False`` (the default), then the ``checkpoint_save_path`` must not exist or must not contain checkpoints which may conflict
            with the current run. Default: ``False``.

        checkpoint_save_interval (Time | str | int | (State, Event) -> bool): A :class:`.Time`, time-string, integer (in epochs),
            or a function that takes (state, event) and returns a boolean whether a checkpoint should be saved.

            If an integer, checkpoints will be saved every n epochs.
            If :class:`.Time` or a time-string, checkpoints will be saved according to this interval.

            .. seealso:: :func:`.checkpoint_periodically`

            If a function, then this function should take two arguments (:class:`.State`, :class:`.Event`).
            The first argument will be the current state of the trainer, and the second argument will be
            be :attr:`.Event.BATCH_CHECKPOINT` or :attr:`.Event.EPOCH_CHECKPOINT` (depending on the current training
            progress). It should return ``True`` if a checkpoint should be saved given the current state and
            event.

        save_model_weights_only (bool): If ``True``, save only the model weights instead of the entire training state.
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
        checkpoint_save_path: str = '{run_name}/checkpoints',
        checkpoint_filename: str = 'ep{epoch}-ba{batch}-rank{rank}.pt',
        artifact_name: Optional[str] = '{run_name}/checkpoints/ep{epoch}-ba{batch}-rank{rank}',
        latest_checkpoint_filename: Optional[str] = 'latest-rank{rank}.pt',
        latest_artifact_name: Optional[str] = '{run_name}/checkpoints/latest-rank{rank}',
        checkpoint_save_interval: Union[Time, str, int, Callable[[State, Event], bool]] = '1ep',
        *,
        overwrite_checkpoints: bool = False,
        num_checkpoints_to_keep: int = -1,
        save_model_weights_only: bool = False,
    ):
        if not callable(checkpoint_save_interval):
            checkpoint_save_interval = checkpoint_periodically(checkpoint_save_interval)
        self.checkpoint_save_interval = checkpoint_save_interval

        self.checkpoint_save_path = checkpoint_save_path

        self.checkpoint_filename = PartialFilePath(checkpoint_filename.lstrip('/'), checkpoint_save_path)
        self.latest_checkpoint_filename = PartialFilePath(latest_checkpoint_filename.lstrip('/'),
                                                          checkpoint_save_path) if latest_checkpoint_filename else None

        self.artifact_name = PartialFilePath(artifact_name) if artifact_name else None
        self.latest_artifact_name = PartialFilePath(latest_artifact_name) if latest_artifact_name else None

        self.overwrite_checkpoints = overwrite_checkpoints
        self.saved_checkpoints: List[str] = []
        self.num_checkpoints_to_keep = num_checkpoints_to_keep
        self.save_model_weights_only = save_model_weights_only

    def init(self, state: State, logger: Logger) -> None:
        checkpoint_save_path = format_name_with_dist(self.checkpoint_save_path, state.run_name)
        os.makedirs(checkpoint_save_path, exist_ok=True)

    def fit_start(self, state: State, logger: Logger) -> None:
        if not self.overwrite_checkpoints:
            # checks that checkpoint_save_path contains no files with a timestamp after the current timestamp,
            # which has potential for future conflicts.
            checkpoint_save_path = format_name_with_dist(self.checkpoint_save_path, state.run_name)
            ensure_folder_has_no_conflicting_files(checkpoint_save_path, self.checkpoint_filename.filename,
                                                   state.timestamp)

        dist.barrier()  # holds all ranks until checkpoint_save_path check is done

        if is_model_deepspeed(state.model) and self.save_model_weights_only:
            raise NotImplementedError('save_model_weights_only=True is not supported when using DeepSpeed.')

    def batch_checkpoint(self, state: State, logger: Logger):
        if self.checkpoint_save_interval(state, Event.BATCH_CHECKPOINT):
            self._save_checkpoint(
                state,
                logger,
                self.get_log_level(state, default=LogLevel.BATCH),
            )

    def epoch_checkpoint(self, state: State, logger: Logger):
        if self.checkpoint_save_interval(state, Event.EPOCH_CHECKPOINT):
            self._save_checkpoint(
                state,
                logger,
                self.get_log_level(state, default=LogLevel.EPOCH),
            )

    def get_log_level(self, state: State, default: LogLevel) -> LogLevel:
        elapsed_duration = state.get_elapsed_duration()
        assert elapsed_duration is not None, 'elapsed_duration is set on Event.BATCH_CHECKPOINT'
        return default if elapsed_duration < 1.0 else LogLevel.FIT

    def get_state_dict(self, state):
        return {
            'state': state.state_dict(),
            'rng': reproducibility.get_rng_state(),
        }

    def _save_checkpoint(self, state: State, logger: Logger, log_level: LogLevel):
        is_deepspeed = is_model_deepspeed(state.model)

        if is_deepspeed and '{rank}' not in self.checkpoint_filename.filename:
            raise ValueError(
                f'Save checkpoint_filename {self.checkpoint_filename.filename} must have {{rank}} for deepspeed.')

        # save the checkpoint to the filename
        checkpoint_filename = self.checkpoint_filename.format(state, is_deepspeed)

        saved_path = checkpoint.save_checkpoint(
            state=state,
            filename=checkpoint_filename,
            weights_only=self.save_model_weights_only,
        )

        if not saved_path:  # not all ranks save
            return

        if self.latest_checkpoint_filename is not None:
            symlink = self.latest_checkpoint_filename.format(state, is_deepspeed)
            os.makedirs(os.path.dirname(symlink), exist_ok=True)
            try:
                os.remove(symlink)
            except FileNotFoundError:
                pass
            os.symlink(os.path.relpath(checkpoint_filename, os.path.dirname(symlink)), symlink)

        # if artifact name provided, upload the checkpoint
        if self.artifact_name is not None:
            artifact_name = self.artifact_name.format(
                state,
                is_deepspeed,
            ).lstrip('/')

            logger.file_artifact(log_level=log_level,
                                 artifact_name=artifact_name,
                                 file_path=checkpoint_filename,
                                 overwrite=self.overwrite_checkpoints)

            if self.latest_artifact_name is not None:
                symlink_name = self.latest_artifact_name.format(
                    state,
                    is_deepspeed,
                ).lstrip('/') + '.symlink'

                # create and upload a symlink file
                with tempfile.TemporaryDirectory() as tmpdir:
                    symlink_filename = os.path.join(tmpdir, 'latest.symlink')
                    create_symlink_file(artifact_name, symlink_filename)
                    logger.file_artifact(
                        log_level=log_level,
                        artifact_name=symlink_name,
                        file_path=symlink_filename,
                        overwrite=True,
                    )

        self.saved_checkpoints.append(checkpoint_filename)

        if self.num_checkpoints_to_keep >= 0:
            self._rotate_checkpoints()

    def _rotate_checkpoints(self):
        while len(self.saved_checkpoints) > self.num_checkpoints_to_keep:
            checkpoint = self.saved_checkpoints.pop(0)
            os.remove(checkpoint)