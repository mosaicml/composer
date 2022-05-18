# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Logger Hyperparameter classes.

Attributes:
    logger_registry (Dict[str, Type[LoggerDestinationHparams]]): The registry of all known
        :class:`.LoggerDestinationHparams`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import yahp as hp

from composer.loggers.file_logger import FileLogger
from composer.loggers.in_memory_logger import InMemoryLogger
from composer.loggers.object_store_logger import ObjectStoreLogger
from composer.loggers.progress_bar_logger import ProgressBarLogger
from composer.loggers.wandb_logger import WandBLogger
from composer.utils import ObjectStoreHparams, import_object

__all__ = [
    "ObjectStoreLoggerHparams",
    "logger_registry",
]


@dataclass
class ObjectStoreLoggerHparams(hp.AutoInitializedHparams):
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
    "file": FileLogger,
    "wandb": WandBLogger,
    "progress_bar": ProgressBarLogger,
    "in_memory": InMemoryLogger,
    "object_store": ObjectStoreLoggerHparams,
}
