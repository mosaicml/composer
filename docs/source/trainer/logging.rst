|:wood:| Logging
================

By default, the trainer enables :class:`.ProgressBarLogger`, which logs
information to a ``tqdm`` progress bar.

To attach other loggers, use the ``loggers`` argument. For example, the
below logs the results to `Weights and
Biases <https://www.wandb.com/>`__, and `CometML <https://www.comet.com/?utm_source=mosaicml&utm_medium=partner&utm_campaign=mosaicml_comet_integration>`__,
and also saves them to the file
``log.txt``.

.. testsetup::
    :skipif: not _WANDB_INSTALLED or not _COMETML_INSTALLED

    import os

    os.environ["WANDB_MODE"] = "disabled"
    os.environ["COMET_API_KEY"] = "<comet_api_key>"

.. testcode::
    :skipif: not _WANDB_INSTALLED or not _COMETML_INSTALLED

    from composer import Trainer
    from composer.loggers import WandBLogger, CometMLLogger, FileLogger

    wandb_logger = WandBLogger()
    cometml_logger = CometMLLogger()
    file_logger = FileLogger(filename="log.txt")

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        loggers=[wandb_logger, cometml_logger, file_logger],
    )

.. testcleanup::
    :skipif: not _WANDB_INSTALLED or not _COMETML_INSTALLED

    trainer.engine.close()
    os.remove("log.txt")

Available Loggers
-----------------

.. currentmodule:: composer.loggers

.. autosummary::
    :nosignatures:

    ~file_logger.FileLogger
    ~wandb_logger.WandBLogger
    ~cometml_logger.CometMLLogger
    ~progress_bar_logger.ProgressBarLogger
    ~tensorboard_logger.TensorboardLogger
    ~in_memory_logger.InMemoryLogger
    ~object_store_logger.ObjectStoreLogger

Automatically Logged Data
-------------------------

The :class:`.Trainer` automatically logs the following data:

-  ``trainer/algorithms``: a list of specified algorithm names.
-  ``epoch``: the current epoch.
-  ``trainer/global_step``: the total number of training steps that have
   been performed.
-  ``trainer/batch_idx``: the current training step (batch) within the epoch.
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

        def epoch_end(self, state: State, logger: Logger):
            logger.log_metrics({"Epoch": int(state.timestamp.epoch)})

.. testcleanup::

    # Actually run the callback to ensure it works
    epoch_monitor = EpochMonitor()
    epoch_monitor.run_event(Event.EPOCH_END, state, logger)

Similarly, :class:`.Algorithm` classes are also provided the :class:`.Logger`
to log any desired information.

.. seealso::

    :doc:`Algorithms<algorithms>` and :doc:`Callbacks<callbacks>`


Custom Logger Destinations
--------------------------

To use a custom logger destination, create a class that inherits from
:class:`.LoggerDestination`. Here is an example which logs all metrics
into a dictionary:

.. testcode::

    from typing import Any, Dict, Optional

    from composer.loggers.logger_destination import LoggerDestination
    from composer.core.time import Timestamp
    from composer.core.state import State

    class DictionaryLogger(LoggerDestination):
        def __init__(self):
            # Dictionary to store logged data
            self.data = {}

        def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
            for k, v in self.data.items():
                if k not in self.data:
                    self.data[k] = []
                self.data[k].append((state.timestamp, v))

    # Construct a trainer using this logger
    trainer = Trainer(..., loggers=[DictionaryLogger()])

    # Train!
    trainer.fit()

In addition, :class:`.LoggerDestination` can also implement the typical event-based
hooks of typical callbacks if needed. See :doc:`Callbacks<callbacks>` for
more information.
