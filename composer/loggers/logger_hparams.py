# Copyright 2021 MosaicML. All Rights Reserved.

"""Logger Hyperparameter classes."""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import yahp as hp

from composer.core.types import JSON
from composer.loggers.file_logger import FileLogger
from composer.loggers.in_memory_logger import InMemoryLogger
from composer.loggers.logger import LogLevel
from composer.loggers.logger_destination import LoggerDestination
from composer.loggers.object_store_logger import ObjectStoreLogger
from composer.loggers.progress_bar_logger import ProgressBarLogger
from composer.loggers.wandb_logger import WandBLogger
from composer.utils import ObjectStoreHparams, dist, import_object

__all__ = [
    "FileLoggerHparams",
    "InMemoryLoggerHparams",
    "LoggerDestinationHparams",
    "ProgressBarLoggerHparams",
    "WandBLoggerHparams",
    "ObjectStoreLoggerHparams",
]


@dataclass
class LoggerDestinationHparams(hp.Hparams, ABC):
    """Base class for logger callback hyperparameters.

    Logger parameters that are added to :class:`~.trainer_hparams.TrainerHparams` (e.g. via YAML or the CLI) are
    initialized in the training loop.
    """

    @abstractmethod
    def initialize_object(self, config: Optional[Dict[str, Any]] = None) -> LoggerDestination:
        """Initializes the logger.

        Args:
            config (dict): The configuration used by the trainer.
                The logger can optionally save this configuration.
        """
        pass


@dataclass
class FileLoggerHparams(LoggerDestinationHparams):
    """:class:`~composer.loggers.file_logger.FileLogger`
    hyperparameters.

    See :class:`~composer.loggers.file_logger.FileLogger` for documentation.

    Args:
        filename (str, optional): See :class:`~composer.loggers.file_logger.FileLogger`
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
    filename: str = hp.optional("The path to the logfile. Can also be `stdout` or `stderr`. Default: stdout",
                                default="stdout")
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

    def initialize_object(self, config: Optional[Dict[str, Any]] = None) -> FileLogger:

        from composer.loggers.file_logger import FileLogger
        return FileLogger(**asdict(self), config=config)


@dataclass
class WandBLoggerHparams(LoggerDestinationHparams):
    """:class:`~composer.loggers.wandb_logger.WandBLogger` hyperparameters.

    Args:
        project (str, optional): WandB project name.
        group (str, optional): WandB group name.
        name (str, optional): WandB run name.
            If not specified, the :attr:`~composer.loggers.logger.Logger.run_name` will be used.
        entity (str, optional): WandB entity name.
        tags (str, optional): WandB tags, comma-separated.
        log_artifacts (bool, optional): See
            :class:`~composer.loggers.wandb_logger.WandBLogger`.
        log_artifacts_every_n_batches (int, optional). See
            :class:`~composer.loggers.wandb_logger.WandBLogger`.
        extra_init_params (dict, optional): See
            :class:`~composer.loggers.wandb_logger.WandBLogger`.
    """

    project: Optional[str] = hp.optional(doc="wandb project name", default=None)
    group: Optional[str] = hp.optional(doc="wandb group name", default=None)
    name: Optional[str] = hp.optional(doc="wandb run name", default=None)
    entity: Optional[str] = hp.optional(doc="wandb entity", default=None)
    tags: str = hp.optional(doc="wandb tags comma separated", default="")
    log_artifacts: bool = hp.optional(doc="Whether to log artifacts", default=False)
    log_artifacts_every_n_batches: int = hp.optional(doc="interval, in batches, to log artifacts", default=100)
    rank_zero_only: bool = hp.optional("Whether to log on rank zero only", default=True)
    extra_init_params: Dict[str, JSON] = hp.optional(doc="wandb parameters", default_factory=dict)
    flatten_hparams: bool = hp.optional(
        doc=
        "Whether the hparams dictionary should be flattened before uploading to WandB. This can make nested fields easier to visualize and query",
        default=False)

    def initialize_object(self, config: Optional[Dict[str, Any]] = None) -> WandBLogger:
        """Initializes the logger.

        The ``config`` is flattened and stored as :attr:`wandb.run.config`.
        The list of algorithms in the ``config`` are appended to :attr:`wandb.run.tags`.

        Args:
            config (Optional[Dict[str, Any]], optional):
                The configuration used by the trainer.

        Returns:
            WandBLogger: An instance of :class:`~composer.loggers.wandb_logger.WandBLogger`.
        """
        tags = list(set([x.strip() for x in self.tags.split(",") if x.strip() != ""]))

        if config is not None:

            def get_flattened_dict(data: Dict[str, Any], _prefix: List[str] = []) -> Dict[str, Any]:
                """Flattens a dictionary with list or sub dicts to have dot syntax.

                i.e. {
                  "sub_dict":{
                    "sub_list":[
                      "sub_sub_dict":{
                        "field1": 12,
                        "field2": "tomatoes"
                      }
                    ]
                  },
                  "field3": "potatoes"
                }

                returns:
                {
                  "sub_dict.sub_list.sub_sub_dict.field1": 12,
                  "sub_dict.sub_list.sub_sub_dict.field2": "tomatoes,
                  "field3": "potatoes",
                }
                """
                all_items = dict()
                for key, val in data.items():
                    key_items = _prefix + [key]
                    key_name = ".".join(key_items)
                    if isinstance(val, dict):
                        all_items.update(get_flattened_dict(val, key_items))
                    elif isinstance(val, list):
                        found_sub_dicts = False
                        for item in val:
                            if isinstance(item, dict):
                                found_sub_dicts = True
                                for sub_key, sub_val in item.items():
                                    if isinstance(sub_val, dict):
                                        all_items.update(get_flattened_dict(sub_val, key_items + [sub_key]))
                                    else:
                                        all_items.update({sub_key: sub_val})
                        if not found_sub_dicts:
                            all_items[key_name] = val
                    else:
                        all_items[key_name] = val
                return all_items

            # extra_init_params may be in ``config`` already. Copy it so we don't get recursive dicts.
            self.extra_init_params = copy.deepcopy(self.extra_init_params)
            if self.flatten_hparams:
                config = get_flattened_dict(data=config)
            if "config" not in self.extra_init_params:
                self.extra_init_params["config"] = {}
            if not isinstance(self.extra_init_params["config"], dict):
                raise TypeError(
                    f"'config' passed to WandB ``extra_init_params`` must be a dictionary. Got {type(self.extra_init_params['config'])}"
                )
            self.extra_init_params["config"].update(config)

        if self.rank_zero_only:
            name = self.name
            group = self.group
        else:
            name = f"{self.name} [RANK_{dist.get_global_rank()}]"
            group = self.group if self.group else self.name
        init_params = {
            "project": self.project,
            "name": name,
            "group": group,
            "entity": self.entity,
            "tags": tags,
        }
        init_params.update(self.extra_init_params)
        return WandBLogger(
            log_artifacts=self.log_artifacts,
            rank_zero_only=self.rank_zero_only,
            log_artifacts_every_n_batches=self.log_artifacts_every_n_batches,
            init_params=init_params,
        )


@dataclass
class ProgressBarLoggerHparams(LoggerDestinationHparams):
    """:class:`~composer.loggers.progress_bar_logger.ProgressBarLogger`
    hyperparameters. This class takes no parameters.
    """

    def initialize_object(self, config: Optional[Dict[str, Any]] = None) -> ProgressBarLogger:
        return ProgressBarLogger(config=config)


@dataclass
class InMemoryLoggerHparams(LoggerDestinationHparams):
    """:class:`~composer.loggers.in_memory_logger.InMemoryLogger`
    hyperparameters.

    Args:
        log_level (str or LogLevel, optional):
            See :class:`~composer.loggers.in_memory_logger.InMemoryLogger`.
    """
    log_level: LogLevel = hp.optional("The maximum verbosity to log. Default: BATCH", default=LogLevel.BATCH)

    def initialize_object(self, config: Optional[Dict[str, Any]] = None) -> LoggerDestination:
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

            .. seealso: :func:`composer.utils.dynamic_import.import_object`

            Setting this parameter to ``None`` (the default) will log all artifacts.
        object_name_format (str, optional): See :class:`~composer.loggers.object_store_logger.ObjectStoreLogger`.
        num_concurrent_uploads (int, optional): See :class:`~composer.loggers.object_store_logger.ObjectStoreLogger`.
        upload_staging_folder (str, optional): See :class:`~composer.loggers.object_store_logger.ObjectStoreLogger`.
        use_procs (bool, optional): See :class:`~composer.loggers.object_store_logger.ObjectStoreLogger`.
    """
    object_store_hparams: ObjectStoreHparams = hp.required("Object store provider hparams.")
    should_log_artifact: Optional[str] = hp.optional(
        "Path to a filter function which returns whether an artifact should be logged.", default=None)
    object_name_format: str = hp.optional("A format string for object names", default="{artifact_name}")
    num_concurrent_uploads: int = hp.optional("Maximum number of concurrent uploads.", default=4)
    use_procs: bool = hp.optional("Whether to perform file uploads in background processes (as opposed to threads).",
                                  default=True)
    upload_staging_folder: Optional[str] = hp.optional(
        "Staging folder for uploads. If not specified, will use a temporary directory.", default=None)

    def initialize_object(self, config: Optional[Dict[str, Any]] = None) -> ObjectStoreLogger:
        return ObjectStoreLogger(
            provider=self.object_store_hparams.provider,
            container=self.object_store_hparams.container,
            provider_kwargs=self.object_store_hparams.get_provider_kwargs(),
            object_name_format=self.object_name_format,
            should_log_artifact=import_object(self.should_log_artifact)
            if self.should_log_artifact is not None else None,
            num_concurrent_uploads=self.num_concurrent_uploads,
            upload_staging_folder=self.upload_staging_folder,
            use_procs=self.use_procs,
        )
