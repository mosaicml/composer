# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Logger Hyperparameter classes.

Attributes:
    logger_registry (Dict[str, Type[LoggerDestinationHparams]]): The registry of all known
        :class:`.LoggerDestinationHparams`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Type, Union

import yahp as hp

from composer.loggers.cometml_logger import CometMLLogger
from composer.loggers.file_logger import FileLogger
from composer.loggers.in_memory_logger import InMemoryLogger
from composer.loggers.logger_destination import LoggerDestination
from composer.loggers.progress_bar_logger import ProgressBarLogger
from composer.loggers.remote_uploader_downloader import RemoteUploaderDownloader
from composer.loggers.tensorboard_logger import TensorboardLogger
from composer.loggers.wandb_logger import WandBLogger
from composer.utils.object_store.object_store_hparams import ObjectStoreHparams, object_store_registry

__all__ = [
    'RemoteUploaderDownloaderHparams',
    'logger_registry',
]


@dataclass
class RemoteUploaderDownloaderHparams(hp.Hparams):
    """Hyperparameters for the :class:`~.RemoteUploaderDownloader`.

    Args:
        object_store_hparams (ObjectStoreHparams): The object store provider hparams.
        object_name (str, optional): See :class:`.RemoteUploaderDownloader`.
        num_concurrent_uploads (int, optional): See :class:`.RemoteUploaderDownloader`.
        upload_staging_folder (str, optional): See :class:`.RemoteUploaderDownloader`.
        use_procs (bool, optional): See :class:`.RemoteUploaderDownloader`.
    """

    hparams_registry = {
        'object_store_hparams': object_store_registry,
    }

    object_store_hparams: ObjectStoreHparams = hp.required('Object store provider hparams.')
    object_name: str = hp.auto(RemoteUploaderDownloader, 'object_name')
    num_concurrent_uploads: int = hp.auto(RemoteUploaderDownloader, 'num_concurrent_uploads')
    use_procs: bool = hp.auto(RemoteUploaderDownloader, 'use_procs')
    upload_staging_folder: Optional[str] = hp.auto(RemoteUploaderDownloader, 'upload_staging_folder')
    num_attempts: int = hp.auto(RemoteUploaderDownloader, 'num_attempts')

    def initialize_object(self) -> RemoteUploaderDownloader:
        return RemoteUploaderDownloader(
            object_store_cls=self.object_store_hparams.get_object_store_cls(),
            object_store_kwargs=self.object_store_hparams.get_kwargs(),
            object_name=self.object_name,
            num_concurrent_uploads=self.num_concurrent_uploads,
            upload_staging_folder=self.upload_staging_folder,
            use_procs=self.use_procs,
            num_attempts=self.num_attempts,
        )


logger_registry: Dict[str, Union[Type[LoggerDestination], Type[hp.Hparams]]] = {
    'file': FileLogger,
    'wandb': WandBLogger,
    'progress_bar': ProgressBarLogger,
    'tensorboard': TensorboardLogger,
    'in_memory': InMemoryLogger,
    'object_store': RemoteUploaderDownloaderHparams,
    'comet_ml': CometMLLogger,
}
