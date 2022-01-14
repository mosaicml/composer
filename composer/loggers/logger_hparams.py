# Copyright 2021 MosaicML. All Rights Reserved.

"""Logger Hyperparameters"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import yahp as hp

from composer.core.logging import BaseLoggerBackend, LogLevel
from composer.core.types import JSON
from composer.loggers.mosaicml_logger import RunType

if TYPE_CHECKING:
    from composer.loggers.file_logger import FileLoggerBackend
    from composer.loggers.mosaicml_logger import MosaicMLLoggerBackend
    from composer.loggers.tqdm_logger import TQDMLoggerBackend
    from composer.loggers.wandb_logger import WandBLoggerBackend


@dataclass
class BaseLoggerBackendHparams(hp.Hparams, ABC):
    """
    Base class for logger backend hyperparameters.

    Logger parameters that are added to
    :class:`~composer.trainer.trainer_hparams.TrainerHparams`
    (e.g. via YAML or the CLI) are initialized in the training loop.
    """

    @abstractmethod
    def initialize_object(self, config: Optional[Dict[str, Any]] = None) -> BaseLoggerBackend:
        """Initializes the logger.

        Args:
            config (dict): The configuration used by the trainer.
                The logger can optionally save this configuration.
        """
        pass


@dataclass
class FileLoggerBackendHparams(BaseLoggerBackendHparams):
    """:class:`~composer.loggers.file_logger.FileLoggerBackend`
    hyperparameters.

    See :class:`~composer.loggers.file_logger.FileLoggerBackend`
    for documentation.
    """
    log_level: LogLevel = hp.optional("The maximum verbosity to log. Default: EPOCH", default=LogLevel.EPOCH)
    filename: str = hp.optional("The path to the logfile. Can also be `stdout` or `stderr`. Default: stdout",
                                default="stdout")
    buffer_size: int = hp.optional("Number of bytes to buffer. Defaults to 1 for line-buffering. "
                                   "See https://docs.python.org/3/library/functions.html#open",
                                   default=1)  # line buffering. Python's default is -1.
    flush_every_n_batches: int = hp.optional(
        "Even if the buffer is not full, write to the file after this many steps. "
        "Defaults to 1 (every step).",
        default=1)
    every_n_epochs: int = hp.optional(
        "Frequency of logging messages for messages of LogLevel.EPOCH and higher."
        "Defaults to 1 (every epoch).",
        default=1)
    every_n_batches: int = hp.optional(
        "Frequency of logging messages for messages of LogLevel.BATCH and higher."
        "Defaults to 1 (every batch).",
        default=1)

    def initialize_object(self, config: Optional[Dict[str, Any]] = None) -> FileLoggerBackend:

        from composer.loggers.file_logger import FileLoggerBackend
        return FileLoggerBackend(**asdict(self), config=config)


@dataclass
class WandBLoggerBackendHparams(BaseLoggerBackendHparams):
    """:class:`~composer.loggers.wandb_logger.WandBLoggerBackend`
    hyperparameters.

    Args:
        project (str, optional): Weights and Biases project name.
        name (str, optional): Weights and Biases run name.
        entity (str, optional): Weights and Biases entity name.
        tags (str, optional): Comma-seperated list of tags to add to the run.
        log_artifacts (bool, optional): Whether to log artifacts. Defaults to False.
        log_artifacts_every_n_batches (int, optional). How frequently to log artifacts. Defaults to 100.
            Only applicable if `log_artifacts` is True.

        extra_init_params (JSON Dictionary, optional): Extra parameters to pass into :func:`wandb.init`.
    """

    project: Optional[str] = hp.optional(doc="wandb project name", default=None)
    name: Optional[str] = hp.optional(doc="wandb run name", default=None)
    entity: Optional[str] = hp.optional(doc="wandb entity", default=None)
    tags: str = hp.optional(doc="wandb tags comma separated", default="")
    log_artifacts: bool = hp.optional(doc="Whether to log artifacts", default=False)
    log_artifacts_every_n_batches: int = hp.optional(doc="interval, in batches, to log artifacts", default=100)
    extra_init_params: Dict[str, JSON] = hp.optional(doc="wandb parameters", default_factory=dict)

    def initialize_object(self, config: Optional[Dict[str, Any]] = None) -> WandBLoggerBackend:
        """Initializes the logger.

        The ``config`` is flattened and stored as :attr:`wandb.run.config`.
        The list of algorithms in the ``config`` are appended to :attr:`wandb.run.tags`.

        Args:
            config (Optional[Dict[str, Any]], optional):
                The configuration used by the trainer.

        Returns:
            WandBLoggerBackend: An instance of :class:`~composer.loggers.wandb_logger.WandBLoggerBackend`.
        """
        tags = list(set([x.strip() for x in self.tags.split(",") if x.strip() != ""]))

        if config is not None:

            def get_flattened_dict(data: Dict[str, Any], _prefix: List[str] = []) -> Dict[str, Any]:
                """
                Flattens a dictionary with list or sub dicts to have dot syntax

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
                                    all_items.update(get_flattened_dict(sub_val, key_items + [sub_key]))
                        if not found_sub_dicts:
                            all_items[key_name] = val
                    else:
                        all_items[key_name] = val
                return all_items

            flattened_config = get_flattened_dict(data=config)
            if "config" not in self.extra_init_params:
                self.extra_init_params["config"] = {}
            self.extra_init_params["config"].update(flattened_config)  # type: ignore

        init_params = {
            "project": self.project,
            "name": self.name,
            "entity": self.entity,
            "tags": tags,
        }

        init_params.update(self.extra_init_params)

        from composer.loggers.wandb_logger import WandBLoggerBackend
        return WandBLoggerBackend(
            log_artifacts=self.log_artifacts,
            log_artifacts_every_n_batches=self.log_artifacts_every_n_batches,
            init_params=init_params,
        )


@dataclass
class TQDMLoggerBackendHparams(BaseLoggerBackendHparams):
    """:class:`~composer.loggers.tqdm_logger.TQDMLoggerBackend`
    hyperparameters.

    See :class:`~composer.loggers.tqdm_logger.TQDMLoggerBackend`
    for documentation.
    """

    def initialize_object(self, config: Optional[Dict[str, Any]] = None) -> TQDMLoggerBackend:
        from composer.loggers.tqdm_logger import TQDMLoggerBackend
        return TQDMLoggerBackend(config=config)


@dataclass
class MosaicMLLoggerBackendHparams(BaseLoggerBackendHparams):
    """:class:`~composer.loggers.mosaicml_logger.MosaicMLLoggerBackend`
    hyperparameters.

    See :class:`~composer.loggers.mosaicml_logger.MosaicMLLoggerBackend`
    for documentation.
    """
    run_name: str = hp.required("The name of the run to write logs for.")
    run_type: RunType = hp.required("The type of the run.")
    run_id: Optional[str] = hp.optional(
        "The name of the run to write logs for. If not provided, a random id "
        "is created.", default=None)
    experiment_name: Optional[str] = hp.optional(
        "The name of the experiment to associate the run with. If "
        "not provided, a random name is created.",
        default=None)
    creds_file: Optional[str] = hp.optional(
        "A file containing the MosaicML api_key. If not provided "
        "will default to the environment variable MOSAIC_API_KEY.",
        default=None)
    flush_every_n_batches: int = hp.optional("Flush the log data buffer every n batches.", default=100)
    max_logs_in_buffer: int = hp.optional(
        "The maximum number of log entries allowed in the buffer "
        "before a forced flush.", default=1000)
    log_level: LogLevel = hp.optional("The maximum verbosity to log. Default: EPOCH", default=LogLevel.EPOCH)

    def initialize_object(self, config: Optional[Dict[str, Any]] = None) -> MosaicMLLoggerBackend:
        from composer.loggers.mosaicml_logger import MosaicMLLoggerBackend
        return MosaicMLLoggerBackend(**asdict(self), config=config)
