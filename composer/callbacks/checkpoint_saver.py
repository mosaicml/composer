# Copyright 2021 MosaicML. All Rights Reserved.

"""Callback to save checkpoints during training."""

from __future__ import annotations

import logging
import os
import textwrap
from collections import OrderedDict
from typing import Callable, Optional, Union

from composer.core import Event, State
from composer.core.callback import Callback
from composer.core.time import Time, TimeUnit
from composer.loggers import Logger
from composer.loggers.logger import LogLevel
from composer.utils import (checkpoint, dist, ensure_folder_is_empty, format_name_with_dist,
                            format_name_with_dist_and_time)

log = logging.getLogger(__name__)

__all__ = ["CheckpointSaver", "checkpoint_periodically"]


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
        if state.get_elapsed_duration() >= 1.0:
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
    """Callback to save checkpoints.

    .. note::

        If the ``save_folder`` argument is specified constructing the :class:`~composer.trainer.trainer.Trainer`,
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
        ...         save_folder_format='{run_name}/checkpoints',
        ...         filename_format="ep{epoch}-ba{batch}-rank{rank}",
        ...         save_latest_format="latest-rank{rank}",
        ...         save_interval="1ep",
        ...         weights_only=False,
        ...     )
        ... ])

    Args:
        save_folder_format (str, optional): Format string for the folder where checkpoints will be saved.

            The following format variables are available:

            +------------------------+-------------------------------------------------------+
            | Variable               | Description                                           |
            +========================+=======================================================+
            | ``{run_name}``         | The name of the training run. See                     |
            |                        | :attr:`~composer.core.logging.Logger.run_name`.       |
            +------------------------+-------------------------------------------------------+
            | ``{rank}``             | The global rank, as returned by                       |
            |                        | :func:`~composer.utils.dist.get_global_rank`.         |
            +------------------------+-------------------------------------------------------+
            | ``{local_rank}``       | The local rank of the process, as returned by         |
            |                        | :func:`~composer.utils.dist.get_local_rank`.          |
            +------------------------+-------------------------------------------------------+
            | ``{world_size}``       | The world size, as returned by                        |
            |                        | :func:`~composer.utils.dist.get_world_size`.          |
            +------------------------+-------------------------------------------------------+
            | ``{local_world_size}`` | The local world size, as returned by                  |
            |                        | :func:`~composer.utils.dist.get_local_world_size`.    |
            +------------------------+-------------------------------------------------------+
            | ``{node_rank}``        | The node rank, as returned by                         |
            |                        | :func:`~composer.utils.dist.get_node_rank`.           |
            +------------------------+-------------------------------------------------------+

            .. note::

                When training with multiple devices (i.e. GPUs), ensure that ``'{rank}'`` appears in the format.
                Otherwise, multiple processes may attempt to write to the same file.

            Consider the following example when using default value of '{run_name}/checkpoints':

            >>> checkpoint_saver = CheckpointSaver(save_folder_format='{run_name}/checkpoints')
            >>> trainer = Trainer(callbacks=[checkpoint_saver], run_name='awesome-training-run')
            >>> checkpoint_saver.save_folder
            'awesome-training-run/checkpoints'

            Default: `'{run_name}/checkpoints'`


        filename_format (str, optional): A format string describing how to name checkpoints.
            (default: ``'ep{epoch}-ba{batch}-rank{rank}'``)

            Checkpoints will be saved approximately to ``{save_folder}/{filename_format.format(...)}``.

            The following format variables are available:

            +------------------------+-------------------------------------------------------+
            | Variable               | Description                                           |
            +========================+=======================================================+
            | ``{run_name}``         | The name of the training run. See                     |
            |                        | :attr:`~composer.core.logging.Logger.run_name`.       |
            +------------------------+-------------------------------------------------------+
            | ``{rank}``             | The global rank, as returned by                       |
            |                        | :func:`~composer.utils.dist.get_global_rank`.         |
            +------------------------+-------------------------------------------------------+
            | ``{local_rank}``       | The local rank of the process, as returned by         |
            |                        | :func:`~composer.utils.dist.get_local_rank`.          |
            +------------------------+-------------------------------------------------------+
            | ``{world_size}``       | The world size, as returned by                        |
            |                        | :func:`~composer.utils.dist.get_world_size`.          |
            +------------------------+-------------------------------------------------------+
            | ``{local_world_size}`` | The local world size, as returned by                  |
            |                        | :func:`~composer.utils.dist.get_local_world_size`.    |
            +------------------------+-------------------------------------------------------+
            | ``{node_rank}``        | The node rank, as returned by                         |
            |                        | :func:`~composer.utils.dist.get_node_rank`.           |
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

                *   By default, only the rank zero process will save a checkpoint file.
                
                *   When using DeepSpeed, each rank will save a checkpoint file in tarball format. DeepSpeed
                    requires tarball format, as it saves model and optimizer states in separate files.
                    Ensure that ``'{rank}'`` appears within the ``filename_format``. Otherwise, multiple ranks
                    may attempt to write to the same file(s), leading to corrupted checkpoints. If no tarball file
                    extension is specified, ``'.tar'`` will be used.

                *   To use compression (regardless of whether DeepSpeed is enabled), set the file extension
                    to ``'.tar.gz'``, ``'.tgz'``, ``'.tar.bzip'``, or ``'.tar.lzma'`` (depending on the desired
                    compression algorithm).
            
            .. warning::

                Using compression will block the training loop while checkpoints are being compressed. As such, we
                recommend saving checkpoints without compression.

            Consider the following scenario, where:

            *   The :attr:`~.Logger.run_name` is 'awesome-training-run'
            *   The default ``save_folder_format='{run_name}/checkpoints'`` is used.
            *   The default ``name_format='ep{epoch}-ba{batch}-rank{rank}'`` is used.
            *   The current epoch count is ``1``.
            *   The current batch count is ``42``.

            When DeepSpeed is not being used, the rank zero process will save the checkpoint to ``"awesome-training-run/checkpoints/ep1-ba42-rank0"``.

            When DeepSpeed is being used, each rank (process) will save checkpoints to::

                awesome-training-run/checkpoints/ep1-ba42-rank0.tar
                awesome-training-run/checkpoints/ep1-ba42-rank1.tar
                awesome-training-run/checkpoints/ep1-ba42-rank2.tar
                ...
        
        artifact_name_format (str, optional):


        save_latest_format (str, optional): A format string for a symlink which points to the last saved checkpoint.
            (default: ``'latest-rank{rank}'``)
            
            Symlinks will be created approximately at ``{save_folder}/{save_latest_format.format(...)}``. 

            The same format variables as for ``name_format`` are available.

            To disable symlinks, set this parameter to ``None``.

            Consider the following scenario, where:

            *   The :attr:`~.Logger.run_name` is 'awesome-training-run'
            *   The default ``save_folder_format='{run_name}/checkpoints'`` is used.
            *   The default ``name_format='ep{epoch}-ba{batch}-rank{rank}'`` is used.
            *   The default ``save_latest_format='latest-rank{rank}'`` is used.
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

        overwrite (bool, optional): Whether existing checkpoints should be overridden.
            If ``False`` (the default), then the ``checkpoint_folder`` must not exist or be empty.
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


        num_checkpoints_to_persist (int, optional): The number of checkpoints to keep locally. The oldest checkpoints
            are removed first. Set to ``-1`` to keep all checkpoints locally. (default: ``-1``)

            Checkpoints will be removed after they have been logged as a file artifact. For example, when this callback
            is used in conjunction with the :class:`~composer.loggers.object_store_logger.ObjectStoreLogger`, set this
            parameter to ``0`` to immediately delete checkpoints from the local disk after they have been uploaded to
            the object store.
            
            This parameter only controls how many checkpoints are kept locally; checkpoints are not deleted from
            artifact stores.

    Attributes:
        saved_checkpoints (Dict[Timestamp, List[str]]): A dictionary mapping a save timestamp
            to a list of filepaths corresponding to the checkpoints saved at that time. This dictionary
            will contain at most ``num_checkpoints_to_persist`` entries.

            .. note:: When using DeepSpeed, the index of a filepath in each list corresponds to the
                global rank of the process that wrote that file. These filepaths are valid only on
                the global rank's node. Otherwise, when not using DeepSpeed, this list will contain
                only one filepath since only rank zero saves checkpoints.
    """

    def __init__(
        self,
        save_folder_format: str = "{run_name}/checkpoints",
        filename_format: str = "ep{epoch}-ba{batch}-rank{rank}",
        artifact_name_format: str = "{run_name}/checkpoints/ep{epoch}-ba{batch}-rank{rank}",
        save_latest_filename_format: Optional[str] = "latest-rank{rank}",
        save_interval: Union[Time, str, int, Callable[[State, Event], bool]] = "1ep",
        *,
        overwrite: bool = False,
        num_checkpoints_to_persist: int = -1,
        weights_only: bool = False,
    ):
        if not callable(save_interval):
            save_interval = checkpoint_periodically(save_interval)

        self.save_folder_format = save_folder_format
        self.filename_format = filename_format
        self.artifact_name_format = artifact_name_format
        self.save_latest_filename_format = save_latest_filename_format
        self.overwrite = overwrite

        self.save_interval = save_interval
        self.saved_checkpoints = OrderedDict()
        self.num_checkpoints_to_persist = num_checkpoints_to_persist
        self.weights_only = weights_only

    def init(self, state: State, logger: Logger) -> None:
        del state  # unused
        save_folder = format_name_with_dist(self.save_folder_format, logger.run_name)
        os.makedirs(save_folder, exist_ok=True)
        if not self.overwrite:
            ensure_folder_is_empty(save_folder)
        # Ensure no rank proceeds (and potentially attempts to write to the folder), until all ranks have validated that the folder is empty.
        dist.barrier()

    def fit_start(self, state: State, logger: Logger) -> None:
        if state.is_model_deepspeed:
            if self.weights_only:
                NotImplementedError(
                    textwrap.dedent(f"""\
                    Saving checkpoints with `weights_only=True` is not currently supported when using DeepSpeed.
                    See https://github.com/mosaicml/composer/issues/685."""))

    def batch_checkpoint(self, state: State, logger: Logger):
        if self.save_interval(state, Event.BATCH_CHECKPOINT):
            # If training is finished, log at the FIT loglevel
            log_level = LogLevel.BATCH if state.get_elapsed_duration() < 1.0 else LogLevel.FIT
            self._save_checkpoint(state, logger, log_level)

    def epoch_checkpoint(self, state: State, logger: Logger):
        if self.save_interval(state, Event.EPOCH_CHECKPOINT):
            log_level = LogLevel.EPOCH if state.get_elapsed_duration() < 1.0 else LogLevel.FIT
            self._save_checkpoint(state, logger, log_level)

    def _save_checkpoint(self, state: State, logger: Logger, log_level: LogLevel):
        checkpoint_filepath_format = os.path.join(format_name_with_dist(self.save_folder_format, logger.run_name),
                                                  self.filename_format)
        checkpoint_filepaths = checkpoint.save_checkpoint(state,
                                                          logger,
                                                          checkpoint_filepath_format,
                                                          weights_only=self.weights_only)

        if dist.get_global_rank() < len(checkpoint_filepaths):
            # Log the checkpoint as an artifact
            checkpoint_filepath = checkpoint_filepaths[dist.get_global_rank()]
            artifact_name = format_name_with_dist_and_time(self.artifact_name_format, logger.run_name,
                                                           state.timer.get_timestamp()).lstrip("/")
            if state.is_model_deepspeed and not _is_archive(artifact_name):
                # Deepspeed requires tarballs; appending `.tar`
                artifact_name += ".tar"
            logger.file_artifact(log_level=log_level,
                                 artifact_name=artifact_name,
                                 file_path=checkpoint_filepath,
                                 overwrite=self.overwrite)

            if self.save_latest_filename_format is not None:
                symlink_name = os.path.join(
                    format_name_with_dist(self.save_folder_format, logger.run_name),
                    format_name_with_dist_and_time(self.save_latest_filename_format, logger.run_name,
                                                   state.timer.get_timestamp()),
                )
                os.makedirs(os.path.dirname(symlink_name), exist_ok=True)
                try:
                    os.remove(symlink_name)
                except FileNotFoundError:
                    pass
                os.symlink(checkpoint_filepath, symlink_name)

        timestamp = state.timer.get_timestamp()

        self.saved_checkpoints[timestamp] = checkpoint_filepaths
        if self.num_checkpoints_to_persist >= 0:
            while len(self.saved_checkpoints) > self.num_checkpoints_to_persist:

                # self.saved_checkpoints is an ordered dict, so the zeroth item will be the oldest checkpoint
                timestamp, checkpoint_filepaths = next(iter(self.saved_checkpoints.items()))
                if dist.get_global_rank() < len(checkpoint_filepaths):
                    # Remove this rank's checkpoint
                    os.remove(checkpoint_filepaths[dist.get_global_rank()])
                del self.saved_checkpoints[timestamp]
