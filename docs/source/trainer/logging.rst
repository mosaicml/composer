|:wood:| Logging
================

By default, the trainer enables :class:`.TQDMLogger`, which logs
information to a ``tqdm`` progress bar.

To attach other loggers, use the ``loggers`` argument. For example, the
below logs the results to `Weights and
Biases <https://www.wandb.com/>`__ and also saves them to the file
``log.txt``.

.. code:: python

   from composer import Trainer
   from composer.loggers import WandBLogger, FileLogger

   trainer = Trainer(model=model,
                     train_dataloader=train_dataloader,
                     eval_dataloader=eval_dataloader,
                     loggers=[WandBLogger(), FileLogger(filename="log.txt")])

Available Loggers
-----------------

.. currentmodule:: composer.loggers

.. autosummary::
    :nosignatures:

    ~file_logger.FileLogger
    ~wandb_logger.WandBLogger
    ~tqdm_logger.TQDMLogger
    ~in_memory_logger.InMemoryLogger


Default Values
--------------

Several quantities are logged by default during :meth:`.Trainer.fit`:

-  ``trainer/algorithms``: a list of specified algorithms names.
-  ``epoch``: the current epoch.
-  ``trainer/global_step``: the total number of training steps that have
   been performed.
-  ``trainer/batch_idx``: the current training step within the epoch.
-  ``loss/train``: the training loss calculated from the current batch.
-  All the validation metrics specified in the :class:`.ComposerModel`
   object passed to :class:`.Trainer`.

User Logging
------------

The recommended way to log additional information is to define a custom
:class:`.Callback`. Each of its methods has access to :class:`.Logger`.

.. code:: python

   from composer import Callback
   from composer.typing import State, Logger

   class EpochMonitor(Callback):

       def epoch_end(state: State, logger: Logger):
           logger.metric_epoch({"Epoch": state.epoch})

:class:`.Logger` routes all the information to the ``loggers`` provided
to the trainer, and has three primary methods:

-  :meth:`.Logger.metric_fit`
-  :meth:`.Logger.metric_epoch`
-  :meth:`.Logger.metric_batch`

Calls to these methods will log the data into each of the destination
``loggers``, but with different :class:`.LogLevel`.

Similarly, :class:`.Algorithm` classes are also provided the :class:`.Logger`
to log any desired information.

.. seealso::

    :doc:`Algorithms<algorithms>` and :doc:`Callbacks<callbacks>`

Logging Levels
--------------

:class:`.LogLevel` specifies three logging levels that denote where in
the training loop log messages are generated. The logging levels are:

-  :attr:`.LogLevel.FIT`: metrics logged once per training
   run, typically before the first epoch.
-  :attr:`.LogLevel.EPOCH`: metrics logged once per epoch.
-  :attr:`.LogLevel.BATCH`: metrics logged once per batch.

Custom Loggers
--------------

To use a custom destination logger, create a class that inherits from
:class:`.LoggerCallback`. Optionally implement the two following methods:

-  :meth:`.LoggerCallback.will_log`(:class:`.State`, :class:`.LogLevel`:
   returns a boolean to determine if a metric will be logged. This is often
   used to filter messages of a lower log level than desired. The default
   returns ``True`` (i.e. always log).
-  :meth:`.LoggerCallback.log_metric`(``TimeStamp``, ``LogLevel``, ``TLogData``):
   Handles the actual logging of the provided data to an end source. For example,
   write into a log file, or upload to a service.

Here is an example of a :class:`.LoggerCallback` which logs all metrics
into a dictionary:

.. code:: python

   from composer.core.logging import LoggerCallback, LogLevel, TLogData
   from composer.core.time import Timestamp
   from composer.core.types import State

   class DictionaryLogger(LoggerCallback):
       def __init__(self):
           # Dictionary to store logged data
           self.data = {}

       def will_log(state: State, log_level: LogLevel) -> bool:
           return log_level < LogLevel.BATCH

       def log_metric(self, timestamp: Timestamp, log_level: LogLevel, data: TLogData):
           for k, v in data.items():
               if k not in self.data:
                   self.data[k] = []
               self.data[k].append((timestamp, log_level, v))

In addition, :class:`.LoggerCallback` can also implement the typical event-based
hooks of typical callbacks if needed. See :doc:`Callbacks<callbacks>` for
more information.
