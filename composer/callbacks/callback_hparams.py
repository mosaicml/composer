# Copyright 2021 MosaicML. All Rights Reserved.

"""Hyperparameters for callbacks."""
from __future__ import annotations

import abc
import dataclasses
import importlib
import textwrap
from dataclasses import dataclass
from typing import Optional

import yahp as hp

from composer.callbacks.checkpoint_saver import CheckpointSaver
from composer.callbacks.grad_monitor import GradMonitor
from composer.callbacks.lr_monitor import LRMonitor
from composer.callbacks.memory_monitor import MemoryMonitor
from composer.callbacks.run_directory_uploader import RunDirectoryUploader
from composer.callbacks.speed_monitor import SpeedMonitor
from composer.core.callback import Callback
from composer.core.time import Time
from composer.utils.object_store import ObjectStoreProviderHparams

__all__ = [
    "CallbackHparams",
    "GradMonitorHparams",
    "MemoryMonitorHparams",
    "LRMonitorHparams",
    "SpeedMonitorHparams",
    "RunDirectoryUploaderHparams",
    "CheckpointSaverHparams",
]


@dataclass
class CallbackHparams(hp.Hparams, abc.ABC):
    """Base class for callback hyperparameters.

    Callback parameters that are added to the callbacks argument of
    :attr:`~composer.trainer.trainer_hparams.TrainerHparams` (e.g., via YAML or the CLI). See `Trainer with YAHP <https://docs.mosaicml.com/en/latest/tutorials/adding_models_datasets.html#trainer-with-yahp>`_ for more details.
    These are initialized in the training loop.
    """

    @abc.abstractmethod
    def initialize_object(self) -> Callback:
        """Initialize the callback.

        Returns:
            Callback: An instance of the callback.
        """
        pass


@dataclass
class GradMonitorHparams(CallbackHparams):
    """:class:`~.GradMonitor` hyperparamters.

    Args:
        log_layer_grad_norms (bool, optional): 
            See :class:`~.GradMonitor` for documentation.
    """

    log_layer_grad_norms: bool = hp.optional(
        doc="Whether to log gradient norms for individual layers.",
        default=False,
    )

    def initialize_object(self) -> GradMonitor:
        """Initialize the GradMonitor callback.

        Returns:
            GradMonitor: An instance of :class:`~.GradMonitor`.
        """
        return GradMonitor(log_layer_grad_norms=self.log_layer_grad_norms)


@dataclass
class MemoryMonitorHparams(CallbackHparams):
    """:class:`~.MemoryMonitor` hyperparameters.

    There are no parameters as :class:`~.MemoryMonitor` does not take any parameters.
    """

    def initialize_object(self) -> MemoryMonitor:
        """Initialize the MemoryMonitor callback.

        Returns:
            MemoryMonitor: An instance of :class:`~.MemoryMonitor`.
        """
        return MemoryMonitor()


@dataclass
class LRMonitorHparams(CallbackHparams):
    """:class:`~.LRMonitor` hyperparameters.

    There are no parameters as :class:`~.LRMonitor` does not take any parameters.
    """

    def initialize_object(self) -> LRMonitor:
        """Initialize the LRMonitor callback.

        Returns:
            LRMonitor: An instance of :class:`~.LRMonitor`.
        """
        return LRMonitor()


@dataclass
class SpeedMonitorHparams(CallbackHparams):
    """:class:`~.SpeedMonitor` hyperparameters.

    Args:
        window_size (int, optional): See :class:`~.SpeedMonitor` for documentation.
    """
    window_size: int = hp.optional(
        doc="Number of batchs to use for a rolling average of throughput.",
        default=100,
    )

    def initialize_object(self) -> SpeedMonitor:
        """Initialize the SpeedMonitor callback.

        Returns:
            SpeedMonitor: An instance of :class:`~.SpeedMonitor`.
        """
        return SpeedMonitor(window_size=self.window_size)


@dataclass
class CheckpointSaverHparams(CallbackHparams):
    """:class:`~.CheckpointSaver` hyperparameters.
    
    Args:
        save_folder (str, optional): See :class:`~.CheckpointSaver`.
        name_format (str, optional): See :class:`~.CheckpointSaver`.
        save_latest_format (str, optional): See :class:`~.CheckpointSaver`.
        overwrite (str, optional): See :class:`~.CheckpointSaver`.
        weights_only (bool, optional): See :class:`~.CheckpointSaver`.

        save_interval (str, optional): Either a :doc:`time-string </trainer/time>` or a path to a function.

            If a :doc:`time-string </trainer/time>`, checkpoints will be saved according to this interval.

            If a path to a function, it should be of the format ``'path.to.function:function_name'``. The function
            should take (:class:`~.State`, :class:`~.Event`) and return a
            boolean indicating whether a checkpoint should be saved given the current state and event. The event will
            be either :attr:`~composer.core.event.Event.BATCH_CHECKPOINT` or
            :attr:`~composer.core.event.Event.EPOCH_CHECKPOINT`.
    """
    save_folder: str = hp.optional(doc="Folder where checkpoints will be saved.", default="checkpoints")
    name_format: str = hp.optional("Checkpoint name format string.", default="ep{epoch}-ba{batch}/rank_{rank}")
    save_latest_format: Optional[str] = hp.optional("Latest checkpoint symlink format string.",
                                                    default="latest/rank_{rank}")
    overwrite: bool = hp.optional("Whether to override existing checkpoints.", default=False)
    weights_only: bool = hp.optional("Whether to save only checkpoint weights", default=False)
    save_interval: str = hp.optional(textwrap.dedent("""\
        Checkpoint interval or path to a `(State, Event) -> bool` function
        returning whether a checkpoint should be saved."""),
                                     default="1ep")

    def initialize_object(self) -> CheckpointSaver:
        try:
            save_interval = Time.from_timestring(self.save_interval)
        except ValueError:
            # assume it is a module path
            module_path, function_name = self.save_interval.split(":")
            mod = importlib.import_module(module_path)
            save_interval = getattr(mod, function_name)
        return CheckpointSaver(
            save_folder=self.save_folder,
            name_format=self.name_format,
            save_latest_format=self.save_latest_format,
            overwrite=self.overwrite,
            save_interval=save_interval,
            weights_only=self.weights_only,
        )


@dataclass
class RunDirectoryUploaderHparams(CallbackHparams, ObjectStoreProviderHparams):
    """:class:`~.RunDirectoryUploader` hyperparameters.

    Args:
        provider (str):
            See :class:`~.ObjectStoreProviderHparams` for documentation.
        container (str):
            See :class:`~.ObjectStoreProviderHparams` for documentation.
        key_environ (str, optional):
            See :class:`~.ObjectStoreProviderHparams` for documentation.
        secret_environ (str, optional):
            See :class:`~.ObjectStoreProviderHparams` for documentation.
        region (str, optional):
            See :class:`~.ObjectStoreProviderHparams` for documentation.
        host (str, optional):
            See :class:`~.ObjectStoreProviderHparams` for documentation.
        port (int, optional):
            See :class:`~.ObjectStoreProviderHparams` for documentation.
        extra_init_kwargs (Dict[str, Any], optional): Extra keyword arguments to pass into the constructor
            See :class:`~.ObjectStoreProviderHparams` for documentation.
        object_name_prefix (str, optional):
            See :class:`~.RunDirectoryUploader` for documentation.
        num_concurrent_uploads (int, optional):
            See :class:`~.RunDirectoryUploader` for documentation.
        upload_staging_folder (str, optional):
            See :class:`~.RunDirectoryUploader` for documentation.
        use_procs (bool, optional):
            See :class:`~.RunDirectoryUploader` for documentation.
        upload_every_n_batches (int, optional):
            See :class:`~.RunDirectoryUploader` for documentation.
    """

    object_name_prefix: Optional[str] = hp.optional(textwrap.dedent("""\
            A prefix to prepend to all object keys.
            An object's key is this prefix combined with its path relative to the run directory.
            If the container prefix is non-empty, a trailing slash ('/') will
            be added if necessary. If not specified, then the prefix defaults to the run directory. To disable prefixing,
            set to the empty string."""),
                                                    default=None)
    num_concurrent_uploads: int = hp.optional("Maximum number of concurrent uploads. Defaults to 4.", default=4)
    use_procs: bool = hp.optional(
        "Whether to perform file uploads in background processes (as opposed to threads). Defaults to True.",
        default=True)
    upload_staging_folder: Optional[str] = hp.optional(
        "Staging folder for uploads. If not specified, will use a temporary directory.", default=None)
    upload_every_n_batches: int = hp.optional(textwrap.dedent("""\
            Interval at which to scan the run directory for changes and to
            queue uploads of files. Uploads are also queued at the end of the epoch. Defaults to every 100 batches."""),
                                              default=100)

    def initialize_object(self) -> RunDirectoryUploader:
        """Initialize the RunDirectoryUploader callback.

        Returns:
            RunDirectoryUploader: An instance of :class:`~.RunDirectoryUploader`.
        """
        return RunDirectoryUploader(
            object_store_provider_hparams=ObjectStoreProviderHparams(
                **{f.name: getattr(self, f.name) for f in dataclasses.fields(ObjectStoreProviderHparams)}),
            object_name_prefix=self.object_name_prefix,
            num_concurrent_uploads=self.num_concurrent_uploads,
            upload_staging_folder=self.upload_staging_folder,
            use_procs=self.use_procs,
            upload_every_n_batches=self.upload_every_n_batches,
        )
