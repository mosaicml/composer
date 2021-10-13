# Copyright 2021 MosaicML. All Rights Reserved.

"""Callback Hyperparameters"""
from __future__ import annotations

import abc
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, List

import yahp as hp

from composer.core.callback import Callback

if TYPE_CHECKING:
    from composer.callbacks.benchmarker import Benchmarker
    from composer.callbacks.grad_monitor import GradMonitor
    from composer.callbacks.lr_monitor import LRMonitor
    from composer.callbacks.speed_monitor import SpeedMonitor
    from composer.callbacks.torch_profiler import TorchProfiler


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
class TorchProfilerHparams(CallbackHparams):
    """:class:`~composer.callbacks.torch_profiler.TorchProfiler` hyperparameters.

    See :class:`~composer.callbacks.torch_profiler.TorchProfiler` for documentation.
    """

    tensorboard_trace_handler_dir: str = hp.required("directory to store trace results")
    tensorboard_use_gzip: bool = hp.optional("Whether to use gzip for trace", default=False)
    record_shapes: bool = hp.optional(doc="Whether to record tensor shapes", default=True)
    profile_memory: bool = hp.optional(doc="track tensor memory allocations and frees", default=False)
    with_stack: bool = hp.optional(doc="record stack info", default=True)
    with_flops: bool = hp.optional(doc="estimate flops for operators", default=True)

    skip: int = hp.optional("Number of batches to skip at epoch start", default=0)
    warmup: int = hp.optional("Number of warmup batches in a cycle", default=1)
    active: int = hp.optional("Number of batches to profile in a cycle", default=5)
    wait: int = hp.optional("Number of batches to skip at the end of each cycle", default=0)

    def initialize_object(self) -> TorchProfiler:
        from composer.callbacks.torch_profiler import TorchProfiler
        return TorchProfiler(**asdict(self))
