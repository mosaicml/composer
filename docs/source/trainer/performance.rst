|:race_car:| Performance
========================

Composer provides a :mod:`~.profiler` to help users collect and analyze detailed timing information of various events in the training loop 
when using the :class:`~.Trainer`.  This information will help with identifying critical bottlenecks in the training pipeline and evaluating 
the effectiveness of potential mitigations.

To get started with profiling, please see the tutorials below.  For details on implementation and advanced features, please see the 
:mod:`~.profiler` API Reference.

Tutorials
---------

.. toctree:: 
    :maxdepth: 1
    :titlesonly:

    performance_tutorials/profiling.md
    performance_tutorials/analyzing_traces.md