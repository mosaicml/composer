# Copyright 2021 MosaicML. All Rights Reserved.

"""Profilers can be used to gather performance metrics of a given model and training run.

The starting point is the Composer Trainer :class:`~composer.profiler.Profiler`, which measures the time of each :class:`~comoser.core.event.Event` and
produces a trace.  The Composer Trainer :class:`~composer.profiler.Profiler` is implemented as a standalone object and must be instantiated by
the Trainer.

Additionally the following profilers are defined as Composer :class:`~composer.core.callback.Callback`.  The Trainer instantiates and registers each profiler:

* :class:`~composer.profiler.dataloader_profiler.DataloaderProfiler`: Records the time it takes the data loader to return a batch by wrapping the original training and evaluation data loaders.  Implemented as a Composer :class:`~composer.core.callback.Callback`.
* :class:`~composer.profiler.system_profiler.SystemProfiler`: Records system level metrics such as CPU, system memory, disk and network utilization.  Implemented as a Composer :class:`~composer.core.callback.Callback`.
* :class:`~composer.profiler.torch_profiler.TorchProfiler`: Integrates the Torch Profiler to record GPU stats using the Nvidia CUPI API.  Implemented as a Composer :class:`~composer.core.callback.Callback`.

Composer Trainer profiling can be enabled by specifying an output ``profiler_trace_file`` during :class:`~composer.trainer.Trainer` initialization.
By default, the :class:`~composer.profiler.Profiler`, :class:`~composer.profiler.dataloader_profiler.DataloaderProfiler` and
:class:`~composer.profiler.system_profiler.SystemProfiler` will be active.  Torch profiling is disabled by default.

To activate the :class:`~composer.profiler.torch_profiler.TorchProfiler`, the ``torch_profiler_trace_dir`` must be specified *in addition* to the
``profiler_trace_file`` argument.  If Torch profiling is enabled, the ``profiler_trace_file`` will contain the merged trace data from the other
profilers and the Torch profiler.  The merge allows users to correlate System, Composer Trainer and low-level Torch events durin the training loop.

The following example instantiates a basic dataset, model and enables the Composer Trainer Profilers as well as the Torch Profiler:

.. literalinclude:: ../../../examples/profiler_demo.py
    :language: python
    :linenos:
    :emphasize-lines: 6, 27-49

Traces can be viewed by in a Google Chrome browser navigating to ``chrome://tracing`` and opening the ``profiler_trace_file``.
Here is an example trace file:

.. image:: https://storage.googleapis.com/docs.mosaicml.com/images/profiler/profiler_trace_example.png
    :alt: Example Profiler Trace File
    :align: center

Additonal details an be found in the Profiler Guide.
"""
from composer.profiler._event_handler import ProfilerEventHandler as ProfilerEventHandler
from composer.profiler._profiler import Marker as Marker
from composer.profiler._profiler import Profiler as Profiler
from composer.profiler._profiler_action import ProfilerAction as ProfilerAction

# All needs to be defined properly for sphinx autosummary
__all__ = [
    "Marker",
    "Profiler",
    "ProfilerAction",
    "ProfilerEventHandler",
]

Marker.__module__ = __name__
Profiler.__module__ = __name__
ProfilerAction.__module__ = __name__
ProfilerEventHandler.__module__ = __name__
