# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Base class for logger callback."""

from __future__ import annotations

import pathlib
from abc import ABC
from typing import Any, Dict, Optional

from composer.core.callback import Callback
from composer.core.state import State
from composer.loggers.logger import LogLevel

__all__ = ['LoggerDestination']


class LoggerDestination(Callback, ABC):
    """Base class for logger destination.

    As this class extends :class:`~.callback.Callback`, logger destinations can run on any training loop
    :class:`.Event`. For example, it may be helpful to run on
    :attr:`.Event.EPOCH_END` to perform any flushing at the end of every epoch.

    Example:
        .. doctest::

            >>> from composer.loggers import LoggerDestination
            >>> from composer.trainer import Trainer
            >>> class MyLogger(LoggerDestination):
            ...     def log_hyperparameters(self, data):
            ...         print(f'Batch {int(state.timestamp.batch)}: {data}')
            >>> logger = MyLogger()
            >>> trainer = Trainer(
            ...     ...,
            ...     loggers=[logger]
            ... )
            Batch 0: {'rank_zero_seed': ...}
    """

    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        """Log hyperparameters, configurations, and settings.

        Logs any parameter/configuration/setting that doesn't vary during the run.

        Args:
            hyperparameters (Dict[str, Any]): A dictionary mapping hyperparameter names
                (strings) to their values (Any).
        """
        del hyperparameters  # unused
        pass

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics or parameters that vary during training.

        Args:
            metrics (Dict[str, float]): Dictionary mapping metric name (str) to metric
                scalar value (float)
            step (Optional[int], optional): The current step or batch of training at the
                time of logging. Defaults to None. If not specified the specific
                LoggerDestination implementation will choose a step (usually a running
                counter).
        """
        del metrics, step  # unused
        pass

    def log_traces(self, traces: Dict[str, Any]):
        """Log traces. Logs any debug-related data like algorithm traces.

        Args:
            traces (Dict[str, float]): Dictionary mapping trace names (str) to trace
                (Any).
        """
        del traces
        pass

    def log_file_artifact(
        self,
        state: State,
        log_level: LogLevel,
        artifact_name: str,
        file_path: pathlib.Path,
        *,
        overwrite: bool,
    ):
        """Handle logging of a file artifact stored at ``file_path`` to an artifact named ``artifact_name``.

        Subclasses should implement this method to store logged files (e.g. copy it to another folder or upload it to
        an object store), then it should implement this method. However, not all loggers need to implement this method.
        For example, the :class:`.TQDMLogger` does not implement this method, as it cannot
        handle file artifacts.

        .. note::

            *   This method will block the training loop. For optimal performance, it is recommended that this
                method copy the file to a temporary directory, enqueue the copied file for processing, and return.
                Then, use a background thread(s) or process(s) to read from this queue to perform any I/O.
            *   After this method returns, training can resume, and the contents of ``file_path`` may change (or be may
                deleted). Thus, if processing the file in the background (as is recommended), it is necessary to first
                copy the file to a temporary directory. Otherwise, the original file may no longer exist, or the logged
                artifact can be corrupted (e.g., if the logger destination is reading from file while the training loop
                is writing to it).

        .. seealso:: :doc:`Artifact Logging</trainer/artifact_logging>` for notes for file artifact logging.

        Args:
            state (State): The training state.
            log_level (Union[str, LogLevel]): A :class:`LogLevel`.
            artifact_name (str): The name of the artifact.
            file_path (pathlib.Path): The file path.
            overwrite (bool, optional): Whether to overwrite an existing artifact with the same ``artifact_name``.
                (default: ``False``)
        """
        del state, log_level, artifact_name, file_path, overwrite  # unused
        pass

    def get_file_artifact(
        self,
        artifact_name: str,
        destination: str,
        overwrite: bool = False,
        progress_bar: bool = True,
    ):
        """Handle downloading an artifact named ``artifact_name`` to ``destination``.

        Args:
            artifact_name (str): The name of the artifact.
            destination (str): The destination filepath.
            overwrite (bool): Whether to overwrite an existing file at ``destination``. Defaults to ``False``.
            progress_bar (bool, optional): Whether to show a progress bar. Ignored if ``path`` is a local file.
                (default: ``True``)
        """
        del artifact_name, destination, overwrite, progress_bar  # unused
        raise NotImplementedError

    def can_log_file_artifacts(self) -> bool:
        """Indicates whether LoggerDestination can log file artifacts.

        Defaults to false, should return True for derived logger classes that implement log_file_artifact().

        Returns:
            bool: Whether the class supports logging file artifacts.
        """
        return False
