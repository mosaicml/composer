composer.Logger
===============

.. currentmodule:: composer

The trainer includes a :class:`Logger`,
which routes logging calls to logger backends.
Each logger backend inherits from :class:`~composer.core.logging.base_backend.BaseLoggerBackend`,
which inherits from :class:`Callback`.

For example, to define a new logging backend:

.. code-block:: python

    from composer.core.logging import BaseLoggerBackend

    class MyLoggerBackend(BaseLoggerBackend)

        def log_metric(self, epoch, step, log_level, data):
            print(f'Epoch {epoch} Step {step}: {log_level} {data}')

.. note::

    To use Composer's built in loggers, see :doc:`/loggers`.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :recursive:
   
    ~composer.loggers.logger_hparams.BaseLoggerBackendHparams
    ~composer.core.logging.base_backend.BaseLoggerBackend
    ~composer.core.logging.base_backend.RankZeroLoggerBackend

.. autoclass:: Logger
