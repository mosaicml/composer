# Copyright 2021 MosaicML. All Rights Reserved.

"""Callbacks provide hooks that can run at each training loop :class:`~composer.core.event.Event`

Callbacks differ from :class:`~composer.core.algorithm.Algorithm` in that they do not modify the training of the model.
They are typically used to for non-essential recording functions such as logging or timing.
By convention, callbacks should not modify the :class:`~composer.core.state.State`.


Each callback inherits from the :class:`~composer.core.callback.Callback` base class.
Callbacks can be implemented in two ways:

#.  Override the individual methods named for each :class:`~composer.core.event.Event`.

    For example,
        >>> class MyCallback(Callback):
        ...     def epoch_start(self, state: State, logger: Logger):
        ...         print(f'Epoch {state.timer.epoch}')
        >>> # construct trainer object with your callback
        >>> trainer = Trainer(
        ...     model=model,
        ...     train_dataloader=train_dataloader,
        ...     eval_dataloader=eval_dataloader,
        ...     optimizers=optimizer,
        ...     max_duration="1ep",
        ...     callbacks=[MyCallback()],
        ... )
        >>> # trainer will run your event as follows
        >>> _ = trainer.engine.run_event(Event.EPOCH_START)
        Epoch Time(0, TimeUnit.EPOCH)

#.  Override :meth:`_run_event` (**not** :meth:`run_event`) to run in response
    to all events. If this method is overridden, then the individual methods
    corresponding to each event name will not be automatically called (however,
    the subclass implementation can invoke these methods as it wishes.)

    For example,
        >>> class MyCallback(Callback):
        ...     def _run_event(self, event: Event, state: State, logger: Logger):
        ...         if event == Event.EPOCH_START:
        ...             print(f'Epoch {state.epoch}/{state.max_epochs}')
        >>> # construct trainer object with your callback
        >>> trainer = Trainer(
        ...     model=model,
        ...     train_dataloader=train_dataloader,
        ...     eval_dataloader=eval_dataloader,
        ...     optimizers=optimizer,
        ...     max_duration="1ep",
        ...     callbacks=[MyCallback()],
        ... )
        >>> # trainer will run your event as follows
        >>> _ = trainer.engine.run_event(Event.EPOCH_START)
        Epoch 0/1
"""
from composer.callbacks.callback_hparams import CallbackHparams as CallbackHparams
from composer.callbacks.callback_hparams import GradMonitorHparams as GradMonitorHparams
from composer.callbacks.callback_hparams import LRMonitorHparams as LRMonitorHparams
from composer.callbacks.callback_hparams import MemoryMonitorHparams as MemoryMonitorHparams
from composer.callbacks.callback_hparams import RunDirectoryUploaderHparams as RunDirectoryUploaderHparams
from composer.callbacks.callback_hparams import SpeedMonitorHparams as SpeedMonitorHparams
from composer.callbacks.grad_monitor import GradMonitor
from composer.callbacks.lr_monitor import LRMonitor
from composer.callbacks.run_directory_uploader import RunDirectoryUploader
from composer.callbacks.speed_monitor import SpeedMonitor

__all__ = [
    "GradMonitor",
    "LRMonitor",
    "RunDirectoryUploader",
    "SpeedMonitor",
]
