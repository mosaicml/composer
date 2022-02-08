# Copyright 2021 MosaicML. All Rights Reserved.

"""Logger Hyperparameters."""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import yahp as hp

from composer.core.logging import LoggerCallback, LogLevel
from composer.core.types import JSON
from composer.loggers.in_memory_logger import InMemoryLogger
from composer.loggers.mosaicml_logger import RunType
from composer.utils import dist

if TYPE_CHECKING:
    from composer.loggers.file_logger import FileLogger
    from composer.loggers.mosaicml_logger import MosaicMLLogger
    from composer.loggers.tqdm_logger import TQDMLogger
    from composer.loggers.wandb_logger import WandBLogger


@dataclass
class LoggerCallbackHparams(hp.Hparams, ABC):
    """Base class for logger backend hyperparameters.

    Logger parameters that are added to
    :class:`~composer.trainer.trainer_hparams.TrainerHparams`
    (e.g. via YAML or the CLI) are initialized in the training loop.
    """

    @abstractmethod
    def initialize_object(self, config: Optional[Dict[str, Any]] = None) -> LoggerCallback:
        """Initializes the logger.

        Args:
            config (dict): The configuration used by the trainer.
                The logger can optionally save this configuration.
        """
        pass


@dataclass
class FileLoggerHparams(LoggerCallbackHparams):
    """:class:`~composer.loggers.file_logger.FileLogger`
    hyperparameters.

    See :class:`~composer.loggers.file_logger.FileLogger`
    for documentation.
    """
    log_level: LogLevel = hp.optional("The maximum verbosity to log. Default: EPOCH", default=LogLevel.EPOCH)
    filename: str = hp.optional("The path to the logfile. Can also be `stdout` or `stderr`. Default: stdout",
                                default="stdout")
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
class WandBLoggerHparams(LoggerCallbackHparams):
    """:class:`~composer.loggers.wandb_logger.WandBLogger`
    hyperparameters.

    Args:
        project (str, optional): Weights and Biases project name.
        group (str, optional): Weights and Biases group name.
        name (str, optional): Weights and Biases run name.
        entity (str, optional): Weights and Biases entity name.
        tags (str, optional): Comma-seperated list of tags to add to the run.
        log_artifacts (bool, optional): Whether to log artifacts. Defaults to False.
        log_artifacts_every_n_batches (int, optional). How frequently to log artifacts. Defaults to 100.
            Only applicable if `log_artifacts` is True.

        extra_init_params (JSON Dictionary, optional): Extra parameters to pass into :func:`wandb.init`.
    """

    project: Optional[str] = hp.optional(doc="wandb project name", default=None)
    group: Optional[str] = hp.optional(doc="wandb group name", default=None)
    name: Optional[str] = hp.optional(doc="wandb run name", default=None)
    entity: Optional[str] = hp.optional(doc="wandb entity", default=None)
    tags: str = hp.optional(doc="wandb tags comma separated", default="")
    log_artifacts: bool = hp.optional(doc="Whether to log artifacts", default=False)
    log_artifacts_every_n_batches: int = hp.optional(doc="interval, in batches, to log artifacts", default=100)
    rank_zero_only: bool = hp.optional("Whether to log on rank zero only", default=False)
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

        name_suffix = f"Rank {dist.get_global_rank()}"
        name = f"{self.name}_{name_suffix}" if self.name else name_suffix
        group = self.name if (not self.group and self.rank_zero_only) else self.group
        init_params = {
            "project": self.project,
            "name": name,
            "group": group,
            "entity": self.entity,
            "tags": tags,
        }
        init_params.update(self.extra_init_params)

        from composer.loggers.wandb_logger import WandBLogger
        return WandBLogger(
            log_artifacts=self.log_artifacts,
            rank_zero_only=self.rank_zero_only,
            log_artifacts_every_n_batches=self.log_artifacts_every_n_batches,
            init_params=init_params,
        )


@dataclass
class TQDMLoggerHparams(LoggerCallbackHparams):
    """:class:`~composer.loggers.tqdm_logger.TQDMLogger`
    hyperparameters.

    See :class:`~composer.loggers.tqdm_logger.TQDMLogger`
    for documentation.
    """

    def initialize_object(self, config: Optional[Dict[str, Any]] = None) -> TQDMLogger:
        from composer.loggers.tqdm_logger import TQDMLogger
        return TQDMLogger(config=config)


@dataclass
class MosaicMLLoggerHparams(LoggerCallbackHparams):
    """:class:`~composer.loggers.mosaicml_logger.MosaicMLLogger`
    hyperparameters.

    See :class:`~composer.loggers.mosaicml_logger.MosaicMLLogger`
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

    def initialize_object(self, config: Optional[Dict[str, Any]] = None) -> MosaicMLLogger:
        from composer.loggers.mosaicml_logger import MosaicMLLogger
        return MosaicMLLogger(**asdict(self), config=config)


@dataclass
class InMemoryLoggerHaparms(LoggerCallbackHparams):
    """:class:`~composer.loggers.in_memory_logger.InMemoryLogger`
    hyperparameters.

    See :class:`~composer.loggers.in_memory_logger.InMemoryLogger`
    for documentation.
    """
    log_level: LogLevel = hp.optional("The maximum verbosity to log. Default: BATCH", default=LogLevel.BATCH)

    def initialize_object(self, config: Optional[Dict[str, Any]] = None) -> LoggerCallback:
        return InMemoryLogger(log_level=self.log_level)
