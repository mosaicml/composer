composer.callbacks
==================

.. note::

    To write a custom callback, see :doc:`/core/callback`.

Composer contains built-in callbacks, which can be added via the ``--callbacks`` CLI flag:

.. code-block::

    python examples/run_mosaic_trainer.py -f my_model.yaml --callbacks lr_monitor grad_monitor


Callbacks
---------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :recursive:

    ~composer.callbacks.benchmarker.Benchmarker
    ~composer.callbacks.grad_monitor.GradMonitor
    ~composer.callbacks.lr_monitor.LRMonitor
    ~composer.callbacks.speed_monitor.SpeedMonitor
    ~composer.callbacks.torch_profiler.TorchProfiler

Callback Hyperparameters
------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :recursive:

    ~composer.callbacks.callback_hparams.BenchmarkerHparams
    ~composer.callbacks.callback_hparams.GradMonitorHparams
    ~composer.callbacks.callback_hparams.LRMonitorHparams
    ~composer.callbacks.callback_hparams.SpeedMonitorHparams
    ~composer.callbacks.callback_hparams.TorchProfilerHparams
