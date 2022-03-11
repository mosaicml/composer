# Copyright 2021 MosaicML. All Rights Reserved.

"""Callback to save checkpoints during training."""

from __future__ import annotations

import logging
import os
import textwrap
from typing import Callable, Optional, Union

from composer.core import Event, State
from composer.core.callback import Callback
from composer.core.logging.logger import Logger
from composer.core.time import Time, TimeUnit
from composer.utils import checkpoint, dist, run_directory

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
        ...         save_folder='checkpoints',
        ...         name_format="ep{epoch}-ba{batch}/rank_{rank}",
        ...         save_latest_format="latest/rank_{rank}",
        ...         save_interval="1ep",
        ...         weights_only=False,
        ...     )
        ... ])

    Args:
        save_folder (str): Folder where checkpoints are saved.

            If an absolute path is specified, then
            that path will be used. Otherwise, the ``save_folder`` will be relative
            to the folder returned by :meth:`~.run_directory.get_run_directory`.
            If the ``save_folder`` does not exist, it will be created.

        name_format (str, optional): A format string describing how to name checkpoints.
            (default: ``'ep{epoch}-ba{batch}/rank_{rank}'``)

            Checkpoints will be saved approximately to ``{save_folder}/{name_format.format(...)}``.

            See :func:`.format_name` for the available format variables.

            .. note::

                *   By default, only the rank zero process will save a checkpoint file.
                
                *   When using DeepSpeed, each rank will save a checkpoint file in tarball format. DeepSpeed
                    requires tarball format, as it saves model and optimizer states in separate files.
                    Ensure that ``'{rank}'`` appears within the ``name_format_string``. Otherwise, multiple ranks
                    may attempt to write to the same file(s), leading to corrupted checkpoints. If no tarball file
                    extension is specified, ``'.tar'`` will be used.

                *   To use compression (regardless of whether DeepSpeed is enabled), set the file extension
                    to ``'.tar.gz'``, ``'.tgz'``, ``'.tar.bzip'``, or ``'.tar.lzma'`` (depending on the desired
                    compression algorithm).
            
            .. warning::

                Using compression will block the training loop while checkpoints are being compressed. As such, we
                recommend saving checkpoints without compression.

            Consider the following scenario, where:

            *   The default ``save_folder='checkpoints'`` is used.
            *   The default ``name_format='ep{epoch}-ba{batch}/rank_{rank}'`` is used.
            *   The current epoch count is ``1``.
            *   The current batch count is ``42``.

            When DeepSpeed is not being used, the rank zero process will save the checkpoint to ``"checkpoints/ep1-ba42/rank_0"``.

            When DeepSpeed is being used, each rank (process) will save checkpoints to::

                checkpoints/ep1-ba42/rank_0.tar
                checkpoints/ep1-ba42/rank_1.tar
                checkpoints/ep1-ba42/rank_2.tar
                ...

        save_latest_format (str, optional): A format string for a symlink which points to the last saved checkpoint.
            (default: ``'latest/rank_{rank}'``)
            
            Symlinks will be created approximately at ``{save_folder}/{save_latest_format.format(...)}``. 

            See :func:`.format_name` for the available format variables.

            To disable symlinks, set this parameter to ``None``.

            Consider the following scenario, where:

            *   The default ``save_folder='checkpoints'`` is used.
            *   The default ``name_format='ep{epoch}-ba{batch}/rank_{rank}'`` is used.
            *   The default ``save_latest_format='latest/rank_{rank}'`` is used.
            *   The current epoch count is ``1``.
            *   The current batch count is ``42``.

            When DeepSpeed is not being used, the rank zero process will save the checkpoint to ``'checkpoints/ep1-ba42/rank_0'``,
            and a symlink will be created at ``'checkpoints/latest/rank_0' -> 'checkpoints/ep1-ba42/rank_0'``

            When DeepSpeed is being used, each rank (process) will save checkpoints to::

                checkpoints/ep1-ba42/rank_0.tar
                checkpoints/ep1-ba42/rank_1.tar
                checkpoints/ep1-ba42/rank_2.tar
                ...

            Corresponding symlinks will be created at::

                checkpoints/latest/rank_0.tar -> checkpoints/ep1-ba42/rank_0.tar
                checkpoints/latest/rank_1.tar -> checkpoints/ep1-ba42/rank_1.tar
                checkpoints/latest/rank_2.tar -> checkpoints/ep1-ba42/rank_2.tar
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

    Attributes:
        checkpoint_folder (str): The folder in which checkpoints are stored. If an absolute path was specified for
            ``save_folder`` upon instantiation, then that path will be used. Otherwise, this folder is relative to
            the run directory of the training run (e.g. ``{run_directory}/{save_folder}``).
            If no run directory is provided, then by default, it is of the form
            ``runs/<timestamp>/rank_<GLOBAL_RANK>/<save_folder>`` where ``timestamp``
            is the start time of the run in iso-format, ``GLOBAL_RANK`` is the global rank of the process,
            and ``save_folder`` is the save_folder argument provided upon construction.

            .. seealso:: :mod:`~.run_directory` for details on the format of the run directory
                and how to customize it.
        saved_checkpoints (Dict[Timestamp, List[str]]): A dictionary mapping a save timestamp
            to a list of filepaths corresponding to the checkpoints saved at that time.

            .. note:: When using DeepSpeed, the index of a filepath in each list corresponds to the
                global rank of the process that wrote that file. These filepaths are valid only on
                the global rank's node. Otherwise, when not using DeepSpeed, this list will contain
                only one filepath since only rank zero saves checkpoints.
    """

    def __init__(
        self,
        save_folder: str = "checkpoints",
        name_format: str = "ep{epoch}-ba{batch}/rank_{rank}",
        save_latest_format: Optional[str] = "latest/rank_{rank}",
        overwrite: bool = False,
        save_interval: Union[Time, str, int, Callable[[State, Event], bool]] = "1ep",
        weights_only: bool = False,
    ):
        if not callable(save_interval):
            save_interval = checkpoint_periodically(save_interval)

        self.checkpoint_folder = os.path.join(run_directory.get_run_directory(), save_folder)
        self.name_format = name_format
        self.save_latest_format = save_latest_format
        self.overwrite = overwrite

        self.save_interval = save_interval
        self.saved_checkpoints = {}
        self.weights_only = weights_only

    def init(self, state: State, logger: Logger) -> None:
        # Each rank will attempt to create the checkpoint folder.
        # If the folder is not parameterized by rank, then exist_ok must be True, as the folder will be the same on all ranks.
        os.makedirs(self.checkpoint_folder, mode=0o775, exist_ok=True)
        if not self.overwrite:
            if any(x.startswith(".") for x in os.listdir(self.checkpoint_folder)):
                raise RuntimeError(
                    textwrap.dedent(f"""\
                    Checkpoint folder {self.checkpoint_folder} is not empty. When using {type(self).__name__}(overwrite=True, ...),
                    the checkpoint folder must not contain any existing checkpoints."""))
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
        del logger  # unused
        if self.save_interval(state, Event.BATCH_CHECKPOINT):
            self._save_checkpoint(state)

    def epoch_checkpoint(self, state: State, logger: Logger):
        del logger  # unused
        if self.save_interval(state, Event.EPOCH_CHECKPOINT):
            self._save_checkpoint(state)

    def _save_checkpoint(self, state: State):
        checkpoint_filepath_format = os.path.join(self.checkpoint_folder, self.name_format)
        checkpoint_filepaths = checkpoint.save_checkpoint(state,
                                                          checkpoint_filepath_format,
                                                          weights_only=self.weights_only)

        if self.save_latest_format is not None and dist.get_global_rank() < len(checkpoint_filepaths):
            checkpoint_filepath = checkpoint_filepaths[dist.get_global_rank()]
            symlink_name = os.path.join(self.checkpoint_folder, checkpoint.format_name(self.save_latest_format, state))
            os.makedirs(os.path.dirname(symlink_name), exist_ok=True)
            try:
                os.remove(symlink_name)
            except FileNotFoundError:
                pass
            os.symlink(checkpoint_filepath, symlink_name)

        timestamp = state.timer.get_timestamp()

        self.saved_checkpoints[timestamp] = checkpoint_filepaths
