# Copyright 2021 MosaicML. All Rights Reserved.

"""Callback Hyperparameters"""
from __future__ import annotations

import abc
import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import yahp as hp

from composer.core.callback import Callback

if TYPE_CHECKING:
    from composer.callbacks.benchmarker import Benchmarker
    from composer.callbacks.grad_monitor import GradMonitor
    from composer.callbacks.lr_monitor import LRMonitor
    from composer.callbacks.memory_monitor import MemoryMonitor
    from composer.callbacks.run_directory_uploader import RunDirectoryUploader
    from composer.callbacks.speed_monitor import SpeedMonitor


@dataclass
class CallbackHparams(hp.Hparams, abc.ABC):
    """Base class for callback hyperparameters.
    
    Callback parameters that are added to
    :attr:`composer.trainer.trainer_hparams.TrainerHparams.callbacks`
    (e.g. via YAML or the CLI) are initialized in the training loop.
    """

    @abc.abstractmethod
    def initialize_object(self) -> Callback:
        """Initialize the callback.

        Returns:
            Callback: An instance of the callback.
        """
        pass


@dataclass
class BenchmarkerHparams(CallbackHparams):
    """:class:`~composer.callbacks.benchmarker.Benchmarker` hyperparameters.

    See :class:`~composer.callbacks.benchmarker.Benchmarker` for documentation.
    """
    min_steps: int = hp.optional(
        doc="Minimum number of steps to use for measuring throughput.",
        default=50,
    )
    epoch_list: List[int] = hp.optional(
        doc="List of epochs at which to measure throughput.",
        default_factory=lambda: [0, 1],
    )
    step_list: List[int] = hp.optional(
        doc="List of steps at which to measure throughput.",
        default_factory=lambda: [0, 50],
    )
    all_epochs: bool = hp.optional(
        doc="If true, override epoch_list and profile at all epochs.",
        default=False,
    )

    def initialize_object(self) -> Benchmarker:
        from composer.callbacks.benchmarker import Benchmarker
        return Benchmarker(
            min_steps=self.min_steps,
            epoch_list=self.epoch_list,
            step_list=self.step_list,
            all_epochs=self.all_epochs,
        )


@dataclass
class GradMonitorHparams(CallbackHparams):
    """:class:`~composer.callbacks.grad_monitor.GradMonitor` hyperparamters.

    See :class:`~composer.callbacks.grad_monitor.GradMonitor` for documentation.
    """

    log_layer_grad_norms: bool = hp.optional(
        doc="Whether to log gradient norms for individual layers.",
        default=False,
    )

    def initialize_object(self) -> GradMonitor:
        from composer.callbacks.grad_monitor import GradMonitor
        return GradMonitor(log_layer_grad_norms=self.log_layer_grad_norms)


@dataclass
class MemoryMonitorHparams(CallbackHparams):
    """:class:`~composer.callbacks.memory_monitor.MemoryMonitor` hyperparameters.

    See :class:`~composer.callbacks.memory_monitor.MemoryMonitor` for documentation.
    """

    def initialize_object(self) -> MemoryMonitor:
        from composer.callbacks.memory_monitor import MemoryMonitor
        return MemoryMonitor()


@dataclass
class LRMonitorHparams(CallbackHparams):
    """:class:`~composer.callbacks.lr_monitor.LRMonitor` hyperparameters.

    See :class:`~composer.callbacks.lr_monitor.LRMonitor` for documentation.
    """

    def initialize_object(self) -> LRMonitor:
        from composer.callbacks.lr_monitor import LRMonitor
        return LRMonitor()


@dataclass
class SpeedMonitorHparams(CallbackHparams):
    """:class:`~composer.callbacks.speed_monitor.SpeedMonitor` hyperparameters.

    See :class:`~composer.callbacks.speed_monitor.SpeedMonitor` fpr documentation.
    """
    window_size: int = hp.optional(
        doc="Number of batchs to use for a rolling average of throughput.",
        default=100,
    )

    def initialize_object(self) -> SpeedMonitor:
        from composer.callbacks.speed_monitor import SpeedMonitor
        return SpeedMonitor(window_size=self.window_size)


@dataclass
class RunDirectoryUploaderHparams(CallbackHparams):
    """:class:`~composer.callbacks.torch_profiler.RunDirectoryUploader` hyperparameters.

    See :class:`~composer.callbacks.torch_profiler.RunDirectoryUploader` for documentation.
    """

    provider: str = hp.required("Cloud provider to use.")
    container: str = hp.required("The name of the container (i.e. bucket) to use.")
    object_name_prefix: Optional[str] = hp.optional(textwrap.dedent("""A prefix to prepend to all object keys.
            An object's key is this prefix combined with its path relative to the run directory.
            If the container prefix is non-empty, a trailing slash ('/') will
            be added if necessary. If not specified, then the prefix defaults to the run directory. To disable prefixing,
            set to the empty string."""),
                                                    default=None)
    key: Optional[str] = hp.optional(textwrap.dedent(
        """API key or username to use to connect to the provider. For security. do NOT hardcode the key in the YAML.
        Instead, please specify via CLI arguments, or even better, environment variables."""),
                                     default=None)
    secret: Optional[str] = hp.optional(textwrap.dedent(
        """API secret to use to connect to the provider. For security. do NOT hardcode the key in the YAML.
Instead, please specify via CLI arguments, or even better, environment variables."""),
                                        default=None)
    region: Optional[str] = hp.optional("Cloud region to use", default=None)
    host: Optional[str] = hp.optional("Override hostname for connections", default=None)
    port: Optional[int] = hp.optional("Override port for connections", default=None)
    num_concurrent_uploads: int = hp.optional("Maximum number of concurrent uploads. Defaults to 4.", default=4)
    use_procs: bool = hp.optional(
        "Whether to perform file uploads in background processes (as opposed to threads). Defaults to True.",
        default=True)
    upload_staging_folder: Optional[str] = hp.optional(
        "Staging folder for uploads. If not specified, will use a temporary directory.", default=None)
    extra_init_kwargs: Dict[str, Any] = hp.optional(
        "Extra keyword arguments to pass into the constructor for the specified provider.", default_factory=dict)
    upload_every_n_batches: int = hp.optional(
        textwrap.dedent("""Interval at which to scan the run directory for changes and to
            queue uploads of files. Uploads are also queued at the end of the epoch. Defaults to every 100 batches."""),
        default=100)

    def initialize_object(self) -> RunDirectoryUploader:
        from composer.callbacks.run_directory_uploader import RunDirectoryUploader
        init_kwargs = {}
        for key in ("key", "secret", "host", "port", "region"):
            kwarg = getattr(self, key)
            if getattr(self, key) is not None:
                init_kwargs[key] = kwarg
        init_kwargs.update(self.extra_init_kwargs)
        return RunDirectoryUploader(
            provider=self.provider,
            container=self.container,
            object_name_prefix=self.object_name_prefix,
            num_concurrent_uploads=self.num_concurrent_uploads,
            upload_staging_folder=self.upload_staging_folder,
            use_procs=self.use_procs,
            provider_init_kwargs=init_kwargs,
            upload_every_n_batches=self.upload_every_n_batches,
        )
