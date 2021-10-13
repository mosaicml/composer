composer.loggers
================

.. note::

    To write a custom logger, see :doc:`/core/logger`.


Composer contains built-in loggers, which can be added via the ``--loggers`` CLI flag:

.. code-block::

    python examples/run_mosaic_trainer.py -f my_model.yaml --loggers tqdm file


.. toctree::
    :maxdepth: 1
    :hidden:

    logger


Backends
--------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :recursive:

    ~composer.loggers.file_logger.FileLoggerBackend
    ~composer.loggers.tqdm_logger.TQDMLoggerBackend
    ~composer.loggers.wandb_logger.WandBLoggerBackend


Backend Hyperparameters
-----------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :recursive:

    ~composer.loggers.logger_hparams.FileLoggerBackendHparams
    ~composer.loggers.logger_hparams.TQDMLoggerBackendHparams
    ~composer.loggers.logger_hparams.WandBLoggerBackendHparams
