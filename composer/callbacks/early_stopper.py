# Copyright 2021 MosaicML. All Rights Reserved.

"""Early stopping callback."""

from __future__ import annotations

import logging
from typing import Callable, Optional, Union

import numpy as np

from composer.core import Event, State
from composer.core.callback import Callback
from composer.core.time import Time, TimeUnit
from composer.loggers import Logger

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


class EarlyStopper(Callback):
    __doc__ = f"""Callback to halt training early.
    """

    def __init__(
        self,
        monitor: str,
        label: str = None,
        comp: Callable = None,
        ceiling: Optional[float] = None,
        min_delta: float =0.0,
        patience: int =1,
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
