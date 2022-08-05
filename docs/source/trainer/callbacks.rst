|:phone:| Callbacks
===================

Callbacks provide hooks that run at each training loop's |Event|.
By convention, callbacks should not modify the
training loop by changing the :class:`.State`, but rather by reading and
logging various metrics. Typical callback use cases include logging, timing,
or model introspection.

Using Callbacks
---------------

Built-in callbacks can be accessed in :mod:`composer.callbacks` and
registered with the ``callbacks`` argument to the :class:`.Trainer`.

.. code:: python

    from composer import Trainer
    from composer.callbacks import SpeedMonitor, LRMonitor
    from composer.loggers import WandBLogger

    Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=None,
        max_duration='1ep',
        callbacks=[SpeedMonitor(window_size=100), LRMonitor()],
        loggers=[WandBLogger()],
    )

This example includes callbacks that measure the model throughput and
learning rate and logs them to Weights & Biases.
Callbacks control *what* is being logged, whereas loggers specify
*where* the information is being saved. For more information on
loggers, see :doc:`Logging<logging>`.

Available Callbacks
-------------------

Composer provides several callbacks to monitor and log various
components of training.

.. currentmodule:: composer.callbacks
.. autosummary::
    :nosignatures:

    ~checkpoint_saver.CheckpointSaver
    ~speed_monitor.SpeedMonitor
    ~lr_monitor.LRMonitor
    ~grad_monitor.GradMonitor
    ~memory_monitor.MemoryMonitor
    ~image_visualizer.ImageVisualizer
    ~mlperf.MLPerfCallback
    ~threshold_stopper.ThresholdStopper
    ~early_stopper.EarlyStopper
    ~export_for_inference.ExportForInferenceCallback

Custom Callbacks
----------------

Custom callbacks should inherit from :class:`.Callback` and override any of the
event-related hooks. For example, below is a simple callback that runs on
|EPOCH_START| and prints the epoch number.

.. code:: python

    from composer import Callback, State, Logger

    class EpochMonitor(Callback):

        def epoch_start(self, state: State, logger: Logger):
            print(f'Epoch: {state.timestamp.epoch}')

Alternatively, one can override :meth:`.Callback.run_event` to run code
at every event. The below is an equivalent implementation for ``EpochMonitor``:

.. code:: python

    from composer import Callback, Event, Logger, State

    class EpochMonitor(Callback):

        def run_event(self, event: Event, state: State, logger: Logger):
            if event == Event.EPOCH_START:
                print(f'Epoch: {state.timestamp.epoch}')

.. warning::

    If :meth:`.Callback.run_event` is overridden, the individual methods corresponding
    to each event will be ignored.

The new callback can then be provided to the trainer.

.. code:: python

    from composer import Trainer

    trainer = Trainer(
        ...,
        callbacks=[EpochMonitor()]
    )

Events
------

Here is the list of supported |Event| for callbacks to hook into.

.. currentmodule:: composer.core

.. autoclass:: Event
    :noindex:

.. |Event| replace:: :class:`~composer.core.Event`
.. |EPOCH_START| replace:: :attr:`~composer.core.Event.EPOCH_START`
