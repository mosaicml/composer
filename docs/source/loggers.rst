composer.loggers
================

.. note::

    To write a custom logger, see :doc:`/core/logger`.


Composer contains built-in loggers, which can be added via the ``--loggers`` CLI flag:

.. code-block::

    python examples/run_composer_trainer.py -f my_model.yaml --loggers tqdm file


Backends
--------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :recursive:

    ~composer.loggers.FileLoggerBackend
    ~composer.loggers.TQDMLoggerBackend
    ~composer.loggers.WandBLoggerBackend
    ~composer.loggers.InMemoryLogger


Backend Hyperparameters
-----------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :recursive:

    ~composer.loggers.FileLoggerBackendHparams
    ~composer.loggers.TQDMLoggerBackendHparams
    ~composer.loggers.WandBLoggerBackendHparams
    ~composer.loggers.InMemoryLoggerHparams
