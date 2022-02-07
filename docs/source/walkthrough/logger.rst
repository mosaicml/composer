composer.logging
================

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


API Reference
*************

For the base logging classes, see :mod:`composer.core.logging`.
For a list of loggers available in composer, see the :mod:`composer.loggers`.
