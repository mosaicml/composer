# Copyright 2021 MosaicML. All Rights Reserved.

"""Profilers can be used to gather performance metrics of a given model and training run.

Profilers include:

* :class:`~composer.profiler.profiler.Profiler`:  Produces a trace of the training graph when the Composer Trainer is used.
    Specifically the time of each :class:`Event` (e.g., forward, backward, batch, epoch, etc.) is recorded including
    the latency of each algorithm and callback.
* :class:`~composer.profiler.dataloader_profiler.DataloaderProfiler`: Records the time it takes the data loader to
    return a batch by wrapping the original training and evaluation data loaders.  Implemented as a Composer :class:`~composer.Callback`.
* :class:`~composer.profiler.system_profiler.SystemProfiler`: Records system level metrics such as CPU, system memory,
    disk and network utilization.  Implemented as a Composer :class:`~composer.Callback`.
* :class:`~composer.profiler.torch_profiler.TorchProfiler`: Integrates the Torch Profiler to record GPU stats using the
    Nvidia CUPI API.  Implemented as a Composer :class:`~composer.Callback`.
"""
from composer.profiler.event_handler import ProfilerEventHandler as ProfilerEventHandler
from composer.profiler.profiler import Profiler as Profiler
from composer.profiler.profiler_action import ProfilerAction as ProfilerAction

# All needs to be defined properly for sphinx autosummary
__all__ = [
    "ProfilerEventHandler",
    "Profiler",
    "ProfilerAction",
]
