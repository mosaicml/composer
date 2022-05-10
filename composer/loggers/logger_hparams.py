# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Logger Hyperparameter classes.

Attributes:
    logger_registry (Dict[str, Type[LoggerDestinationHparams]]): The registry of all known
        :class:`.LoggerDestinationHparams`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import yahp as hp

from composer.loggers.file_logger import FileLogger
from composer.loggers.in_memory_logger import InMemoryLogger
from composer.loggers.logger import LogLevel
from composer.loggers.logger_destination import LoggerDestination
from composer.loggers.object_store_logger import ObjectStoreLogger
from composer.loggers.progress_bar_logger import ProgressBarLogger
from composer.loggers.wandb_logger import WandBLogger
from composer.utils import ObjectStoreHparams, import_object

__all__ = [
    "FileLoggerHparams",
    "InMemoryLoggerHparams",
    "LoggerDestinationHparams",
    "ProgressBarLoggerHparams",
    "WandBLoggerHparams",
    "ObjectStoreLoggerHparams",
    "logger_registry",
]


@dataclass
class LoggerDestinationHparams(hp.Hparams, ABC):
    """Base class for logger callback hyperparameters.

    Logger parameters that are added to :class:`~.trainer_hparams.TrainerHparams` (e.g. via YAML or the CLI) are
    initialized in the training loop.
    """

    @abstractmethod
    def initialize_object(self) -> LoggerDestination:
        """Initializes the logger."""
        pass


@dataclass
class FileLoggerHparams(LoggerDestinationHparams):
    """:class:`~composer.loggers.file_logger.FileLogger`
    hyperparameters.

    See :class:`~composer.loggers.file_logger.FileLogger` for documentation.

    Args:
        filename (str, optional): See :class:`~composer.loggers.file_logger.FileLogger`.
        artifact_name (str, optional): See :class:`~composer.loggers.file_logger.FileLogger`.
        capture_stdout (bool, optional): See :class:`~composer.loggers.file_logger.FileLogger`.
        capture_stderr (bool, optional): See :class:`~composer.loggers.file_logger.FileLogger`.
        buffer_size (int, optional): See
            :class:`~composer.loggers.file_logger.FileLogger`.
        log_level (LogLevel, optional): See
            :class:`~composer.loggers.file_logger.FileLogger`.
        log_interval (int, optional): See
            :class:`~composer.loggers.file_logger.FileLogger`.
        flush_interval (int, optional): See
            :class:`~composer.loggers.file_logger.FileLogger`.
    """
    log_level: LogLevel = hp.optional("The maximum verbosity to log. Default: EPOCH", default=LogLevel.EPOCH)
    filename: str = hp.optional("Filename format string for the logfile.", default='{run_name}/logs-rank{rank}.txt')
    artifact_name: Optional[str] = hp.optional("Artifact name format string for the logfile.", default=None)
    capture_stdout: bool = hp.optional("Whether to capture writes to `stdout`", default=True)
    capture_stderr: bool = hp.optional("Whether to capture writes to `stderr`", default=True)
    buffer_size: int = hp.optional("Number of bytes to buffer. Defaults to 1 for line-buffering. "
                                   "See https://docs.python.org/3/library/functions.html#open",
                                   default=1)  # line buffering. Python's default is -1.
    flush_interval: int = hp.optional(
        "Frequency to flush the file, relative to the ``log_level``. "
        "Defaults to 100 of the unit of ``log_level``.",
        default=100)
    log_interval: int = hp.optional(
        "Frequency to record log messages, relative to the ``log_level``."
        "Defaults to 1 (record all messages).",
        default=1)

    def initialize_object(self) -> FileLogger:
        return FileLogger(**asdict(self))


@dataclass
class WandBLoggerHparams(LoggerDestinationHparams):
    """:class:`~composer.loggers.wandb_logger.WandBLogger` hyperparameters.

    Args:
        project (str, optional): WandB project name.
        group (str, optional): WandB group name.
        name (str, optional): WandB run name.
            If not specified, the :attr:`.Logger.run_name` will be used.
        entity (str, optional): WandB entity name.
        tags (str, optional): WandB tags, comma-separated.
        config (Dict[str, Any], optional): WandB run configuration.
        flatten_config (bool, optional): Whether to flatten the run config. (default: ``False``)
        log_artifacts (bool, optional): See :class:`~composer.loggers.wandb_logger.WandBLogger`.
        rank_zero_only (bool, optional): See :class:`~composer.loggers.wandb_logger.WandBLogger`.
        extra_init_params (dict, optional): See
            :class:`~composer.loggers.wandb_logger.WandBLogger`.
    """

    project: Optional[str] = hp.optional(doc="wandb project name", default=None)
    group: Optional[str] = hp.optional(doc="wandb group name", default=None)
    name: Optional[str] = hp.optional(doc="wandb run name", default=None)
    entity: Optional[str] = hp.optional(doc="wandb entity", default=None)
    tags: Optional[str] = hp.optional(doc="wandb tags comma separated", default=None)
    log_artifacts: bool = hp.optional(doc="Whether to log artifacts", default=False)
    rank_zero_only: bool = hp.optional("Whether to log on rank zero only", default=True)
    extra_init_params: Dict[str, Any] = hp.optional(doc="wandb parameters", default_factory=dict)
    config: Dict[str, Any] = hp.optional(doc="Wandb run configuration", default_factory=dict)
    flatten_config: bool = hp.optional(
        doc="Whether to flatten the config, which can make nested fields easier to visualize and query.", default=False)

    def initialize_object(self) -> WandBLogger:
        tags = None
        if self.tags is not None:
            tags = list(set([x.strip() for x in self.tags.split(",") if x.strip() != ""]))

        config_dict = self.config

        if "config" in self.extra_init_params:
            config_dict = self.extra_init_params["config"]

        if self.flatten_config:
            config_dict = self._flatten_dict(config_dict, prefix=[])

        init_params = {
            "project": self.project,
            "name": self.name,
            "group": self.group,
            "entity": self.entity,
            "tags": tags,
            "config": config_dict,
        }
        init_params.update(self.extra_init_params)
        return WandBLogger(
            log_artifacts=self.log_artifacts,
            rank_zero_only=self.rank_zero_only,
            init_params=init_params,
        )

    @classmethod
    def _flatten_dict(cls, data: Dict[str, Any], prefix: List[str]) -> Dict[str, Any]:
        """Flattens a dictionary with list or sub dicts to have dot syntax.

        .. testcode::

            >>> config = {
            ...     "sub_dict":{
            ...         "sub_list":[
            ...             "sub_sub_dict":{
            ...                 "foo": 0,
            ...                 "bar": "baz"
            ...             }
            ...          ]
            ...     },
            ...     "hello": "world"
            ... }
            >>> _flatten_dict(config)
            {
                'sub_dict.sub_list.sub_sub_dict.foo': 0,
                'sub_dict.sub_list.sub_sub_dict.bar': 'baz',
                'hello': 'world',
            }
        """
        all_items = {}
        for key, val in data.items():
            key_items = list(prefix) + [key]
            key_name = ".".join(key_items)
            if isinstance(val, dict):
                all_items.update(cls._flatten_dict(val, key_items))
            elif isinstance(val, list):
                found_sub_dicts = False
                for item in val:
                    if isinstance(item, dict):
                        found_sub_dicts = True
                        for sub_key, sub_val in item.items():
                            if isinstance(sub_val, dict):
                                all_items.update(cls._flatten_dict(sub_val, key_items + [sub_key]))
                            else:
                                all_items.update({sub_key: sub_val})
                if not found_sub_dicts:
                    all_items[key_name] = val
            else:
                all_items[key_name] = val
        return all_items


@dataclass
class ProgressBarLoggerHparams(LoggerDestinationHparams):
    """:class:`~composer.loggers.progress_bar_logger.ProgressBarLogger`
    hyperparameters.

    .. deprecated:: 0.6.0

        This class is deprecated. Instead, please specify the :class:`.ProgressBarLogger` arguments
        directly in the :class:`~composer.trainer.trainer_hparams.TrainerHparams`. This class will be removed
        in v0.7.0.

    Args:
        progress_bar (bool, optional): See :class:`.ProgressBarLogger`.
        log_to_console (bool, optional): See :class:`.ProgressBarLogger`.
        console_log_level (bool, optional): See :class:`.ProgressBarLogger`.
        stream (bool, optional): See :class:`.ProgressBarLogger`.
    """

    progress_bar: bool = hp.optional("Whether to show a progress bar.", default=True)
    log_to_console: Optional[bool] = hp.optional("Whether to print log statements to the console.", default=None)
    console_log_level: LogLevel = hp.optional("The maximum log level.", default=LogLevel.EPOCH)
    stream: str = hp.optional("The stream at which to write the progress bar and log statements.", default="stderr")

    def initialize_object(self) -> ProgressBarLogger:
        return ProgressBarLogger(
            progress_bar=self.progress_bar,
            log_to_console=self.log_to_console,
            console_log_level=self.console_log_level,
            stream=self.stream,
        )


@dataclass
class InMemoryLoggerHparams(LoggerDestinationHparams):
    """:class:`~composer.loggers.in_memory_logger.InMemoryLogger`
    hyperparameters.

    Args:
        log_level (str | LogLevel, optional):
            See :class:`~composer.loggers.in_memory_logger.InMemoryLogger`.
    """
    log_level: LogLevel = hp.optional("The maximum verbosity to log. Default: BATCH", default=LogLevel.BATCH)

    def initialize_object(self) -> LoggerDestination:
        return InMemoryLogger(log_level=self.log_level)


@dataclass
class ObjectStoreLoggerHparams(LoggerDestinationHparams):
    """:class:`~composer.loggers.in_memory_logger.InMemoryLogger`
    hyperparameters.

    Args:
        object_store_hparams (ObjectStoreHparams): The object store provider hparams.
        should_log_artifact (str, optional): The path to a filter function which returns whether an artifact should be
            logged. The path should be of the format ``path.to.module:filter_function_name``.

            The function should take (:class:`~composer.core.state.State`, :class:`.LogLevel`, ``<artifact name>``).
            The artifact name will be a string. The function should return a boolean indicating whether the artifact
            should be logged.

            .. seealso: :func:`composer.utils.import_helpers.import_object`

            Setting this parameter to ``None`` (the default) will log all artifacts.
        object_name (str, optional): See :class:`~composer.loggers.object_store_logger.ObjectStoreLogger`.
        config_artifact_name (str, optional): See :class:`~composer.loggers.object_store_logger.ObjectStoreLogger`.
        num_concurrent_uploads (int, optional): See :class:`~composer.loggers.object_store_logger.ObjectStoreLogger`.
        upload_staging_folder (str, optional): See :class:`~composer.loggers.object_store_logger.ObjectStoreLogger`.
        use_procs (bool, optional): See :class:`~composer.loggers.object_store_logger.ObjectStoreLogger`.
    """
    object_store_hparams: ObjectStoreHparams = hp.required("Object store provider hparams.")
    should_log_artifact: Optional[str] = hp.optional(
        "Path to a filter function which returns whether an artifact should be logged.", default=None)
    object_name: str = hp.optional("A format string for object names", default="{artifact_name}")
    config_artifact_name: Optional[str] = hp.optional(
        "Format string to describe how to store the training configuration.", default="{run_name}/config.yaml")
    num_concurrent_uploads: int = hp.optional("Maximum number of concurrent uploads.", default=4)
    use_procs: bool = hp.optional("Whether to perform file uploads in background processes (as opposed to threads).",
                                  default=True)
    upload_staging_folder: Optional[str] = hp.optional(
        "Staging folder for uploads. If not specified, will use a temporary directory.", default=None)

    def initialize_object(self) -> ObjectStoreLogger:
        return ObjectStoreLogger(
            provider=self.object_store_hparams.provider,
            container=self.object_store_hparams.container,
            provider_kwargs=self.object_store_hparams.get_provider_kwargs(),
            object_name=self.object_name,
            should_log_artifact=import_object(self.should_log_artifact)
            if self.should_log_artifact is not None else None,
            num_concurrent_uploads=self.num_concurrent_uploads,
            upload_staging_folder=self.upload_staging_folder,
            use_procs=self.use_procs,
        )


logger_registry = {
    "file": FileLoggerHparams,
    "wandb": WandBLoggerHparams,
    "progress_bar": ProgressBarLoggerHparams,
    "in_memory": InMemoryLoggerHparams,
    "object_store": ObjectStoreLoggerHparams,
}
