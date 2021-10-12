composer.loggers
================

.. csv-table::
    :header: "name", "logger", "description"
    :widths: 20, 40, 30
    :delim: |

    wandb | :class:`WandBLoggerBackend` | logs to `weights and biases <https://wandb.ai>`_
    tqdm | :class:`TQDMLoggerBackend` | creates progress bar
    file | :class:`FileLoggerBackend` | logs to ``stdout`` (default), or file


.. currentmodule:: composer.loggers

.. autosummary::
    :toctree: generated
    :nosignatures:

    wandb_logger.WandBLoggerBackend
    tqdm_logger.TQDMLoggerBackend
    file_logger.FileLoggerBackend

Base Loggers
-------------

.. autosummary::
    :toctree: generated

    Logger
    BaseLoggerBackend
    RankZeroLoggerBackend
