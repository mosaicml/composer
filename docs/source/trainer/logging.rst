|:wood:| Logging
================

By default, the trainer enables :class:`.TQDMLogger`, which logs
information to a ``tqdm`` progress bar.

To attach other loggers, use the ``loggers`` argument. For example, the
below logs the results to `Weights and
Biases <https://www.wandb.com/>`__ and also saves them to the file
``log.txt``.

.. testsetup::

    import os
    from composer.utils import run_directory

    try:
        os.remove(os.path.join(run_directory.get_run_directory(), "log.txt"))
    except FileNotFoundError:
        pass

    os.environ["WANDB_MODE"] = "disabled"

.. testcode::

   from composer import Trainer
   from composer.loggers import WandBLogger, FileLogger

   trainer = Trainer(model=model,
                     train_dataloader=train_dataloader,
                     eval_dataloader=eval_dataloader,
                     loggers=[WandBLogger(), FileLogger(filename="log.txt")])

.. testcleanup::

    os.remove(os.path.join(run_directory.get_run_directory(), "log.txt"))

Available Loggers
-----------------

.. currentmodule:: composer.loggers

.. autosummary::
    :nosignatures:

    ~file_logger.FileLogger
    ~wandb_logger.WandBLogger
    ~tqdm_logger.TQDMLogger
    ~in_memory_logger.InMemoryLogger

Automatically Logged Data
-------------------------

The :class:`~composer.trainer.trainer.Trainer` automatically logs the following data:

-  ``trainer/algorithms``: a list of specified algorithms names.
-  ``epoch``: the current epoch.
-  ``trainer/global_step``: the total number of training steps that have
   been performed.
-  ``trainer/batch_idx``: the current training step within the epoch.
-  ``loss/train``: the training loss calculated from the current batch.
-  All the validation metrics specified in the :class:`.ComposerModel`
   object passed to :class:`.Trainer`.

Logging Additional Data
-----------------------

To log additional data, create a custom :class:`.Callback`.
Each of its methods has access to the :class:`.Logger`.

.. testcode::

   from composer import Callback, State
   from composer.loggers import Logger

   class EpochMonitor(Callback):

       def epoch_end(state: State, logger: Logger):
           logger.data_epoch({"Epoch": state.epoch})

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

Custom Logger Destinations
--------------------------

To use a custom logger destination, create a class that inherits from
:class:`.LoggerDestination`. Here is an example which logs all metrics
into a dictionary:

.. testcode::

    from composer.loggers.logger_destination import LoggerDestination
    from composer.loggers.logger import LoggerDataDict, LogLevel
    from composer.core.time import Timestamp
    from composer.core.types import State

    class DictionaryLogger(LoggerDestination):
        def __init__(self, log_level: LogLevel = LogLevel.BATCH):
            self.log_level = log_level
            # Dictionary to store logged data
            self.data = {}

        def log_data(self, state: State, log_level: LogLevel, data: LoggerDataDict):
            if log_level <= self.log_level:
                for k, v in data.items():
                    if k not in self.data:
                        self.data[k] = []
                    self.data[k].append((state.timer.get_timestamp(), log_level, v))

    # Construct a trainer using this logger
    trainer = Trainer(..., loggers=[DictionaryLogger()])

In addition, :class:`.LoggerDestination` can also implement the typical event-based
hooks of typical callbacks if needed. See :doc:`Callbacks<callbacks>` for
more information.
