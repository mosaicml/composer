# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Hyperparameters for callbacks."""
from __future__ import annotations

import abc
import textwrap
from dataclasses import asdict, dataclass
from typing import List, Optional, Union

import yahp as hp

from composer.callbacks.benchmarker import Benchmarker
from composer.callbacks.checkpoint_saver import CheckpointSaver
from composer.callbacks.early_stopper import EarlyStopper
from composer.callbacks.grad_monitor import GradMonitor
from composer.callbacks.lr_monitor import LRMonitor
from composer.callbacks.memory_monitor import MemoryMonitor
from composer.callbacks.mlperf import MLPerfCallback
from composer.callbacks.speed_monitor import SpeedMonitor
from composer.callbacks.threshold_stopper import ThresholdStopper
from composer.core.callback import Callback
from composer.core.time import Time
from composer.utils import import_object

__all__ = [
    "Benchmarker",
    "CallbackHparams",
    "GradMonitorHparams",
    "MemoryMonitorHparams",
    "LRMonitorHparams",
    "SpeedMonitorHparams",
    "CheckpointSaverHparams",
]


@dataclass
class CallbackHparams(hp.Hparams, abc.ABC):
    """Base class for Callback hyperparameters."""

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
    window_length: int = hp.optional(
        doc="Number of steps to use for measuring throughput.",
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
            window_length=self.window_length,
            epoch_list=self.epoch_list,
            step_list=self.step_list,
            all_epochs=self.all_epochs,
        )


@dataclass
class GradMonitorHparams(CallbackHparams):
    """:class:`~.GradMonitor` hyperparamters.

    Args:
        log_layer_grad_norms (bool, optional):
            See :class:`~.GradMonitor` for documentation.
            Default: ``False``.
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
class EarlyStopperHparams(CallbackHparams):
    """:class:`~.EarlyStopper` hyperparameters.

    Args:
        monitor (str): The name of the metric to monitor.
        dataloader_label (str): The label of the dataloader or evaluator associated with the tracked metric. If 
            monitor is in an Evaluator, the dataloader_label field should be set to the Evaluator's label. If 
            monitor is a training metric or an ordinary evaluation metric not in an Evaluator, dataloader_label
            should be set to 'train' or 'eval' respectively.
        comp (str, optional): A string dictating which comparison operator to use to measure
            change in the monitored metric. Set ``comp`` to "less" to use the function :func:`torch.less`,
            and "greater" to use the function :func:`torch.greater`. The comparison operator will be called
            ``comp(current_value, prev_best)``. For example, for metrics where the optimal value is low
            (error, loss, perplexity), use a less than operator.
        min_delta (float, optional): An optional float that requires a new value to exceed the best value by at
            least that amount. Defaults to 0.
        patience (int | str, optional): The interval of time the monitored metric can not improve without stopping
            training. Defaults to 1 epoch. If patience is an integer, it is interpreted as the number of epochs.
    """
    monitor: str = hp.required("The name of the metric to monitor.")
    dataloader_label: str = hp.required("Label of the dataloader/evaluator associated with the metric.")
    comp: Optional[str] = hp.optional("Which comparison operator to use to track change in the metric.", default=None)
    min_delta: float = hp.optional("New metric value must exceed the best value by min_delta to continue training.",
                                   default=0.0)
    patience: Union[int, str] = hp.optional("Interval the trainer can wait without stopping training.", default=1)

    def initialize_object(self) -> EarlyStopper:
        """Initialize the EarlyStopper callback.

        Returns:
            EarlyStopper: An instance of :class:`~.EarlyStopper`.
        """
        return EarlyStopper(monitor=self.monitor,
                            dataloader_label=self.dataloader_label,
                            comp=self.comp,
                            min_delta=self.min_delta,
                            patience=self.patience)


@dataclass
class ThresholdStopperHparams(CallbackHparams):
    """:class:`~.ThresholdStopper` hyperparameters.

    Args:
        monitor (str): The name of the metric to monitor.
        dataloader_label (str): The label of the dataloader or evaluator associated with the tracked metric. If 
            monitor is in an Evaluator, the dataloader_label field should be set to the Evaluator's label. If 
            monitor is a training metric or an ordinary evaluation metric not in an Evaluator, dataloader_label
            should be set to 'train' or 'eval' respectively.
        threshold (float): The threshold that dictates when to halt training. Whether training stops if the metric
            exceeds or falls below the threshold depends on the comparison operator.
        comp (str, optional): A string dictating which comparison operator to use to measure
            change in the monitored metric. Set ``comp`` to "less" to use the function :func:`torch.less`,
            and "greater" to use the function :func:`torch.greater`. The comparison operator will be called
            ``comp(current_value, prev_best)``. For example, for metrics where the optimal value is low
            (error, loss, perplexity), use the less than operator.
        stop_on_batch (bool, optional): A bool that indicates whether to stop training in the middle of an epoch if
            the training metrics satisfy the threshold comparison. Defaults to False.
    """
    monitor: str = hp.required("The name of the metric to monitor.")
    dataloader_label: str = hp.required("Label of the dataloader/evaluator associated with the metric.")
    threshold: float = hp.required("The threshold value to compare the metric to.")
    comp: Optional[str] = hp.optional("Which comparison operator to use to track change in the metric.", default=None)
    stop_on_batch: bool = hp.optional("Whether to stop training in the middle of an epoch if using training metrics.",
                                      default=False)

    def initialize_object(self) -> ThresholdStopper:
        """Initialize the ThresholdStopper callback.

        Returns:
            ThresholdStopper: An instance of :class:`~.ThresholdStopper`.
        """
        return ThresholdStopper(
            monitor=self.monitor,
            dataloader_label=self.dataloader_label,
            threshold=self.threshold,
            comp=self.comp,
            stop_on_batch=self.stop_on_batch,
        )


@dataclass
class MLPerfCallbackHparams(CallbackHparams):
    """:class:`~.MLPerfCallback` hyperparameters.

    Args:
        root_folder (str): The root submission folder
        index (int): The repetition index of this run. The filename created will be
            ``result_[index].txt``.
        benchmark (str, optional): Benchmark name. Currently only ``resnet`` supported. Default: ``resnet``.
        target (float, optional): The target metric before the mllogger marks the stop
            of the timing run. Default: ``0.759`` (resnet benchmark).
        division (str, optional): Division of submission. Currently only ``open`` division supported. Default: ``"open"``.
        metric_name (str, optional): name of the metric to compare against the target. Default: ``"Accuracy"``.
        metric_label (str, optional): label name. The metric will be accessed via ``state.current_metrics[metric_label][metric_name]``.
            Default: ``"eval"``.
        submitter (str, optional): Submitting organization. Default: ``"MosaicML"``.
        system_name (str, optional): Name of the system (e.g. 8xA100_composer). If
            ``None``, system name will default to ``[world_size]x[device_name]_composer``,
            e.g. ``8xNVIDIA_A100_80GB_composer``. Default: ``None``.
        status (str, optional): Submission status. One of (onprem, cloud, or preview).
            Default: ``"onprem"``.
        cache_clear_cmd (str, optional): Command to invoke during the cache clear. This callback
            will call ``subprocess(cache_clear_cmd)``. Default is disabled (``None``)
        host_processors_per_node (int, optional): Total number of host processors per node.  Default: ``None``.

    """

    root_folder: str = hp.required("The root submission folder.")
    index: int = hp.required("The repetition index of this run.")
    benchmark: str = hp.optional("Benchmark name. Default: resnet", default="resnet")
    target: float = hp.optional("The target metric before mllogger marks run_stop. Default: 0.759 (resnet)",
                                default=0.759)
    division: Optional[str] = hp.optional(
        "Division of submission. Currently only open division"
        "is supported. Default: open", default="open")
    metric_name: str = hp.optional('name of the metric to compare against the target. Default: Accuracy',
                                   default='Accuracy')
    metric_label: str = hp.optional(
        'label name. The metric will be accessed via state.current_metrics[metric_label][metric_name]. Default: eval',
        default='eval')
    submitter: str = hp.optional("Submitting organization. Default: MosaicML", default='MosaicML')
    system_name: Optional[str] = hp.optional("Name of the system, defaults to [world_size]x[device_name]", default=None)
    status: str = hp.optional("Submission status. Default: onprem", default="onprem")
    cache_clear_cmd: Optional[str] = hp.optional(
        "Command to invoke during the cache clear. This callback will call subprocess(cache_clear_cmd). Default: Disabled.",
        default=None,
    )
    host_processors_per_node: Optional[int] = hp.optional(
        "Total number of host processors per node.  Default: ``None``.", default=None)

    def initialize_object(self) -> MLPerfCallback:
        """Initialize the MLPerf Callback.

        Returns:
            MLPerfCallback: An instance of :class:`~.MLPerfCallback`
        """
        return MLPerfCallback(**asdict(self))


@dataclass
class CheckpointSaverHparams(CallbackHparams):
    """:class:`~.CheckpointSaver` hyperparameters.

    Args:
        save_folder (str, optional): See :class:`~.CheckpointSaver`.
        filename (str, optional): See :class:`~.CheckpointSaver`.
        artifact_name (str, optional): See :class:`~.CheckpointSaver`.
        latest_filename (str, optional): See :class:`~.CheckpointSaver`.
        overwrite (bool, optional): See :class:`~.CheckpointSaver`.
            Default: ``False``.
        weights_only (bool, optional): See :class:`~.CheckpointSaver`.
            Deafult: ``False``.
        save_interval (str, optional): Either a :doc:`time-string </trainer/time>` or a path to a function.
            If a :doc:`time-string </trainer/time>`, checkpoints will be saved according to this interval.

            If a path to a function, it should be of the format ``'path.to.function:function_name'``. The function
            should take (:class:`~.State`, :class:`~.Event`) and return a
            boolean indicating whether a checkpoint should be saved given the current state and event. The event will
            be either :attr:`.Event.BATCH_CHECKPOINT` or :attr:`.Event.EPOCH_CHECKPOINT`. Default: ``"1ep"``.
        num_checkpoints_to_keep (int, optional): See :class:`~.CheckpointSaver`.
            Default: ``-1``.
    """
    save_folder: str = hp.optional(doc="Folder where checkpoints will be saved.", default="{run_name}/checkpoints")
    filename: str = hp.optional("Checkpoint name format string.", default="ep{epoch}-ba{batch}-rank{rank}")
    artifact_name: str = hp.optional("Checkpoint artifact name format string.",
                                     default="{run_name}/checkpoints/ep{epoch}-ba{batch}-rank{rank}")
    latest_filename: Optional[str] = hp.optional("Latest checkpoint symlink format string.",
                                                 default="latest-rank{rank}")
    overwrite: bool = hp.optional("Whether to override existing checkpoints.", default=False)
    weights_only: bool = hp.optional("Whether to save only checkpoint weights.", default=False)
    save_interval: str = hp.optional(textwrap.dedent("""\
        Checkpoint interval or path to a `(State, Event) -> bool` function
        returning whether a checkpoint should be saved."""),
                                     default="1ep")
    num_checkpoints_to_keep: int = hp.optional(
        "Number of checkpoints to persist locally. Set to -1 to never delete checkpoints.",
        default=-1,
    )

    def initialize_object(self) -> CheckpointSaver:
        try:
            save_interval = Time.from_timestring(self.save_interval)
        except ValueError:
            # assume it is a function path
            save_interval = import_object(self.save_interval)
        return CheckpointSaver(
            folder=self.save_folder,
            filename=self.filename,
            artifact_name=self.artifact_name,
            latest_filename=self.latest_filename,
            overwrite=self.overwrite,
            save_interval=save_interval,
            weights_only=self.weights_only,
            num_checkpoints_to_keep=self.num_checkpoints_to_keep,
        )
