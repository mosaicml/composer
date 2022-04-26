# Copyright 2021 MosaicML. All Rights Reserved.

"""Early stopping callback."""

from __future__ import annotations

import logging
import os
import pathlib
import textwrap
from ast import Dict
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from async_timeout import Any

from composer.core import Event, State
from composer.core.callback import Callback
from composer.core.time import Time, Timestamp, TimeUnit
from composer.loggers import Logger
from composer.loggers.logger import LogLevel
from composer.utils import checkpoint, dist
from composer.utils.file_helpers import (FORMAT_NAME_WITH_DIST_AND_TIME_TABLE, FORMAT_NAME_WITH_DIST_TABLE,
                                         ensure_folder_is_empty, format_name_with_dist, format_name_with_dist_and_time,
                                         is_tar)

log = logging.getLogger(__name__)

__all__ = []


def checkpoint_periodically(interval: Union[str, int, Time]) -> Callable[[State, Event], bool]:
    """Helper function to create a checkpoint scheduler according to a specified interval.

    Args:
        interval (Union[str, int, Time]): The interval describing how often checkpoints should be
            saved. If an integer, it will be assumed to be in :attr:`~TimeUnit.EPOCH`\\s.
            Otherwise, the unit must be either :attr:`TimeUnit.EPOCH` or :attr:`TimeUnit.BATCH`.

            Checkpoints will be saved every ``n`` batches or epochs (depending on the unit),
            and at the end of training.

    Returns:
        Callable[[State, Event], bool]: A function that can be passed as the ``save_interval``
            argument into the :class:`CheckpointSaver`.
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
            f"Unknown checkpointing interval: {interval.unit}. Must be TimeUnit.EPOCH or TimeUnit.BATCH.")

    last_checkpoint_batch = None

    def save_interval(state: State, event: Event):
        nonlocal last_checkpoint_batch
        elapsed_duration = state.get_elapsed_duration()
        assert elapsed_duration is not None, "elapsed_duration is set on the BATCH_CHECKPOINT and EPOCH_CHECKPOINT"

        if elapsed_duration >= 1.0:
            # if doing batch-wise checkpointing, and we saved a checkpoint at the batch_checkpoint event
            # right before the epoch_checkpoint event, do not save another checkpoint at the epoch_checkpoint
            # event if the batch count didn't increase.
            if state.timer.batch != last_checkpoint_batch:
                last_checkpoint_batch = state.timer.batch
                return True

        if save_event == Event.EPOCH_CHECKPOINT:
            count = state.timer.epoch
        elif save_event == Event.BATCH_CHECKPOINT:
            count = state.timer.batch
        else:
            raise RuntimeError(f"Invalid save_event: {save_event}")

        if event == save_event and int(count) % int(interval) == 0:
            last_checkpoint_batch = state.timer.batch
            return True

        return False

    return save_interval


class CheckpointSaver(Callback):
    __doc__ = f"""Callback to save checkpoints.

    .. note::

        If the ``folder`` argument is specified constructing the :class:`~composer.trainer.trainer.Trainer`,
        then the :class:`.CheckpointSaver` callback need not be constructed manually. However, for advanced
        checkpointing use cases (such as saving a weights-only checkpoint at one interval and the full training state
        at another interval), instance(s) of this :class:`.CheckpointSaver` callback can be specified in the
        ``callbacks`` argument of the :class:`~composer.trainer.trainer.Trainer`, as shown in the example below.

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
    
    .. testcleanup::

        trainer.engine.close()

    Args:
        folder (str, optional): Format string for the folder where checkpoints will be saved.
            (default: ``'{{run_name}}/checkpoints'``)

            The following format variables are available:

            {textwrap.indent(FORMAT_NAME_WITH_DIST_TABLE, prefix='            ')}

            .. note::

                When training with multiple devices (i.e. GPUs), ensure that ``'{{rank}}'`` appears in the format.
                Otherwise, multiple processes may attempt to write to the same file.

        filename (str, optional): A format string describing how to name checkpoints.
            (default: ``'ep{{epoch}}-ba{{batch}}-rank{{rank}}'``)

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

            Consider the following scenario, where:

            *   The :attr:`~.Logger.run_name` is ``'awesome-training-run'``
            *   The default ``folder='{{run_name}}/checkpoints'`` is used.
            *   The default ``name='ep{{epoch}}-ba{{batch}}-rank{{rank}}'`` is used.
            *   The current epoch count is ``1``.
            *   The current batch count is ``42``.

            When DeepSpeed is not being used, the rank zero process will save the checkpoint to ``"awesome-training-run/checkpoints/ep1-ba42-rank0"``.

            When DeepSpeed is being used, each rank (process) will save checkpoints to::

                awesome-training-run/checkpoints/ep1-ba42-rank0.tar
                awesome-training-run/checkpoints/ep1-ba42-rank1.tar
                awesome-training-run/checkpoints/ep1-ba42-rank2.tar
                ...
        
        artifact_name (str, optional): Format string for the checkpoint's artifact name.
            (default: ``'{{run_name}}/checkpoints/ep{{epoch}}-ba{{batch}}-rank{{rank}}"``)
        
            After the checkpoint is saved, it will be periodically logged as a file artifact.
            The artifact name will be determined by this format string.

            .. seealso:: :meth:`~composer.loggers.logger.Logger.log_file_artifact` for file artifact logging.

            The same format variables for ``filename`` are available.

            Leading slashes (``'/'``) will be stripped.

            To disable logging trace files as file artifacts, set this parameter to ``None``.
        latest_filename (str, optional): A format string for a symlink which points to the last saved checkpoint.
            (default: ``'latest-rank{{rank}}'``)
            
            Symlinks will be created approximately at ``{{folder}}/{{latest_filename.format(...)}}``. 

            The same format variables as for ``name`` are available.

            To disable symlinks, set this parameter to ``None``.

            Consider the following scenario, where:

            *   The :attr:`~.Logger.run_name` is 'awesome-training-run'
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
            (default: ``'{{run_name}}/checkpoints/latest-rank{{rank}}"``)
        
            Whenever a new checkpoint is saved, a symlink artifact is created or updated to point to the latest checkpoint's ``artifact_name``.
            The artifact name will be determined by this format string. This parameter has no effect if ``latest_filename`` or ``artifact_name`` is None."

            .. seealso:: :meth:`~composer.loggers.logger.Logger.log_symlink_artifact` for symlink artifact logging.

            The same format variables for ``filename`` are available.

            Leading slashes (``'/'``) will be stripped.

            To disable symlinks in logger, set this parameter to ``None``.

        overwrite (bool, optional): Whether existing checkpoints should be overridden.
            If ``False`` (the default), then the ``folder`` must not exist or be empty.
            (default: ``False``)

        save_interval (Time | str | int | (State, Event) -> bool): A :class:`Time`, time-string, integer (in epochs),
            or a function that takes (state, event) and returns a boolean whether a checkpoint should be saved.

            If an integer, checkpoints will be saved every n epochs.
            If :class:`Time` or a time-string, checkpoints will be saved according to this interval.

            .. seealso:: :func:`.checkpoint_periodically`

            If a function, then this function should take two arguments (:class:`State`, :class:`Event`).
            The first argument will be the current state of the trainer, and the second argument will be
            be :attr:`.Event.BATCH_CHECKPOINT` or :attr:`.EPOCH_CHECKPOINT` (depending on the current training
            progress). It should return ``True`` if a checkpoint should be saved given the current state and
            event.

        weights_only (bool): If ``True``, save only the model weights instead of the entire training state.
            This parmeter must be ``False`` when using DeepSpeed. (default: ``False``)


        num_checkpoints_to_keep (int, optional): The number of checkpoints to keep locally. The oldest checkpoints
            are removed first. Set to ``-1`` to keep all checkpoints locally. (default: ``-1``)

            Checkpoints will be removed after they have been logged as a file artifact. For example, when this callback
            is used in conjunction with the :class:`~composer.loggers.object_store_logger.ObjectStoreLogger`, set this
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
        monitor: str,
        label: str = None,
        comp: Callable = None,
        ceiling: Optional[float] = None,
        min_delta=0.0,
        patience=1,
    ):
        self.monitor = monitor
        self.label = label
        self.comp = comp
        self.min_delta = min_delta
        if self.comp is None:
            self.comp = np.less if 'loss' in monitor.lower() or 'error' in monitor.lower() else np.greater
            if self.comp == np.less:
                self.min_delta *= -1

        self.ceiling = ceiling
        if self.ceiling is None:
            if self.comp == np.less or 'loss' in monitor.lower() or 'error' in monitor.lower():
                self.ceiling = float('inf')
            else:
                self.ceiling = -float('inf')
        self.patience = patience

        self.best = self.ceiling
        self.new_best = False
        self.wait = 0

    def eval_end(self, state: State, logger: Logger) -> None:
        monitored_metric = None
        current_metrics = state.current_metrics
        if self.label in current_metrics:
            if self.monitor in current_metrics[self.label]:
                monitored_metric = current_metrics[self.label][self.monitor]
            else:
                logger.warning(f"Couldn't find the metric {self.monitor} in the current_metrics/{self.label}")
        elif self.label is None:
            if "eval" in current_metrics:
                if self.monitor in current_metrics["eval"]:
                    monitored_metric = current_metrics["eval"][self.monitor]
            elif self.monitor in current_metrics["train"]:
                monitored_metric = current_metrics["eval"][self.monitor]
            else:
                logger.warning(
                    f"Couldn't find the metrics {self.monitor}. Check if it is spelled correctly or check if the label field is correct (train/eval/evaluator_name)."
                )
        else:
            logger.warning(
                f"The label {self.label} isn't in the state's current_metrics. Use the values train, eval, or the name of the Evaluator if using Evaluators."
            )
        if monitored_metric is None:
            logger.warning(
                f"Didn't find the metric {self.monitor} in the current_metrics. Check if the label field ({self.label}) is correct"
            )
            return

        # TODO Anis - remember to convert from tensor to float
        if self.comp(monitored_metric - self.min_delta, self.best):
            self.best, self.new_best = monitored_metric, True
        else:
            self.new_best = False
            self.wait += 1

        if self.wait >= self.patience:
            # stop the training the training
            state.max_duration = state.timer
