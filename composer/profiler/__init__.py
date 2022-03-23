# Copyright 2021 MosaicML. All Rights Reserved.

"""Performance profiling tools.

The profiler gathers performance metrics during a training run that can be used to diagnose bottlenecks and
facilitate model development.

The metrics gathered include:

* Duration of each :class:`.Event` during training
* Time taken by the data loader to return a batch
* Host metrics such as CPU, system memory, disk and network utilization over time
* Execution order, latency and attributes of PyTorch operators and GPU kernels (see :doc:`profiler`)

The following example demonstrates how to setup and perform profiling on a simple training run.

.. literalinclude:: ../../../examples/profiler_demo.py
    :language: python
    :linenos:
    :emphasize-lines: 6, 27-49

It is required to specify an output ``profiler_trace_file`` during :class:`.Trainer` initialization to enable profiling.
The ``profiler_trace_file`` will contain the profiling trace data once the profiling run completes.  By default, the :class:`.Profiler`,
:class:`.DataLoaderProfiler` and :class:`.SystemProfiler` will be active.  The :class:`.TorchProfiler` is **disabled** by default.

To activate the :class:`.TorchProfiler`, the ``torch_profiler_trace_dir`` must be specified *in addition* to the ``profiler_trace_file`` argument.
The ``torch_profiler_trace_dir`` will contain the Torch Profiler traces once the profiling run completes.  The :class:`.Profiler` will
automatically merge the Torch traces in the ``torch_profiler_trace_dir`` into the ``profiler_trace_file``, allowing users to view a unified trace.

The complete traces can be viewed by in a Google Chrome browser navigating to ``chrome://tracing`` and loading the ``profiler_trace_file``.
Here is an example trace file:

.. image:: https://storage.googleapis.com/docs.mosaicml.com/images/profiler/profiler_trace_example.png
    :alt: Example Profiler Trace File
    :align: center

Additonal details an be found in the Profiler Guide.
"""
from composer.profiler._event_handler import ProfilerEventHandler
from composer.profiler._profiler import Marker, Profiler
from composer.profiler._profiler_action import ProfilerAction

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
