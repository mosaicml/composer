composer.profiler
=================


Composer contains a training loop profiler, which can be added via the ``--profiler`` CLI flag:

.. code-block::

    python examples/run_mosaic_trainer.py -f my_model.yaml --profiler


.. autosummary::
    :toctree: generated
    :nosignatures:
    :recursive:

    ~composer.profiler.ProfilerHparams
    ~composer.core.profiler.Profiler
    ~composer.core.profiler.ProfilerAction
    ~composer.core.profiler.Marker
    ~composer.core.profiler.ProfilerEventHandlerHparams
    ~composer.core.profiler.ProfilerEventHandler
    ~composer.profiler.JSONTraceHandlerHparams
    ~composer.profiler.JSONTraceHandler
    ~composer.profiler.DataloaderProfilerHparams
    ~composer.profiler.DataloaderProfiler
    ~composer.profiler.SystemProfilerHparams
    ~composer.profiler.SystemProfiler
    ~composer.profiler.TorchProfilerHparams
    ~composer.profiler.TorchProfiler
