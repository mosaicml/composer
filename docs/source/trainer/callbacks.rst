Callbacks
=========

Callbacks enable non-essential code to be executed during any of the
|Event| points. By convention, callbacks should not modify the
training loop by changing the :class:`.State`, but rather be reading and
logging various metrics. Typical callback use cases include logging, timing,
or model introspection.

Using Callbacks
---------------

Built-in callbacks can be accessed in :mod:`composer.callbacks`, and
registered with the `callbacks` argument to the :class:`.Trainer`.

.. code:: python

   from composer import Trainer
   from composer.callbacks import SpeedMonitor, LRMonitor
   from composer.loggers import WandBLogger

   Trainer(model=model,
           train_dataloader=train_dataloader,
           eval_dataloader=None,
           max_duration='1ep',
           callbacks=[SpeedMonitor(window_size=100), LRMonitor()],
           loggers=[WandBLogger()])

This example includes callbacks that measure the model throughput (and
the learning rate) and logs them to weights & biases.
Callbacks control *what* is being logged, whereas loggers specify
*where* the information is being saved. For more information on
loggers, see :doc:`Logging<trainer/logging>`.

Available Callbacks
-------------------

Composer provides several callbacks to monitor and log various
components of training.

.. currentmodule:: composer.callbacks

.. autosummary::
    :nosignatures:

    SpeedMonitor
    LRMonitor
    GradMonitor
    MemoryMonitor
    RunDirectoryUploader


Custom Callbacks
----------------

Custom callbacks should inherit from :class:`.Callback` and override any of the
event-related hooks. For example, below is a simple callback that runs on
|EPOCH_START| and prints the epoch number.

.. code:: python

   from composer import Callback, State, Logger

   class EpochMonitor(Callback):

       def epoch_start(self, state: State, logger: Logger):
           print(f'Epoch: {state.timer.epoch}')

Alternatively, one can override :meth:`.Callback.run_event` to run code
at every event. The below is an equivalent implementation for ``EpochMonitor``:

.. code:: python

   from composer import Callback, Event, Logger, State

   class EpochMonitor(Callback):

       def run_event(self, event: Event, state: State, logger: Logger):
           if event == Event.EPOCH_START:
               print(f'Epoch: {state.timer.epoch}')

.. warning::

    If :meth:`.Callback.run_event` is overriden, the individual methods corresponding
    to each event will be ignored.

Callback Methods
----------------

Here is the list of :class:`.Callback` methods that correspond to each
|Event|.

.. currentmodule:: composer.core

.. autoclass:: Event

.. |Event| replace:: :class:`~composer.core.Event`
.. |EPOCH_START| replace:: :attr:`~composer.core.Event.EPOCH_START`
