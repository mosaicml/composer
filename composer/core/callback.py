# Copyright 2021 MosaicML. All Rights Reserved.

"""Base module for callbacks."""
from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from composer.core.serializable import Serializable

try:
    from typing import final
except ImportError:
    final = lambda x: x  # final is not available in python 3.7

if TYPE_CHECKING:
    from composer import Event, Logger, State

__all__ = ["Callback"]


class Callback(Serializable, abc.ABC):
    """Base class for callbacks.

    Callbacks provide hooks that can run at each training loop :class:`~composer.core.event.Event`.  A callback is
    similar to an :class:`~composer.core.algorithm.Algorithm` in that they are run on specific events. Callbacks differ
    from :class:`~composer.core.algorithm.Algorithm` in that they do not modify the training of the model.  By
    convention, callbacks should not modify the :class:`~composer.core.state.State`. They are typically used to for
    non-essential recording functions such as logging or timing. 

    Callbacks can be implemented in two ways:

    #. Override the individual methods named for each :class:`~composer.core.event.Event`.

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
           >>> # trainer will run MyCallback whenever the EPOCH_START
           >>> # is triggered, like this:
           >>> _ = trainer.engine.run_event(Event.EPOCH_START)
           Epoch Time(0, TimeUnit.EPOCH)

    #. Override :meth:`_run_event` (**not** :meth:`run_event`) if you want a single method to handle all events. 
       If this method is overridden, then the individual methods corresponding to each event name will not be
       automatically called on training loop events. For example, if you override :meth:`_run_event` then
       :meth:`epoch_start` will not be called on the :attr:`~composer.core.event.Event.EPOCH_START` event, :meth:`batch_start`
       will not be called on the :attr:`~composer.core.event.Event.BATCH_START` etc. However, you can invoke
       :meth:`epoch_start`, :meth:`batch_start` etc. in your overriding implementation of :meth:`_run_event`.

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
           >>> # trainer will run MyCallback whenever the EPOCH_START
           >>> # is triggered, like this:
           >>> _ = trainer.engine.run_event(Event.EPOCH_START)
           Epoch 0/1

    :private-members: run_event, _run_event, init, fit_start, epoch_start, batch_start, \
        after_dataloader, before_train_batch, before_forward, after_forward, before_loss, \
        after_loss, before_backward, after_backward, after_train_batch, batch_end, \
        epoch_end, eval_start, eval_batch_start, eval_before_forward, eval_after_forward, \
        eval_batch_end, eval_end, close, post_close
    """

    @final
    def run_event(self, event: Event, state: State, logger: Logger) -> None:
        """This method is called by the engine on each event.

        Args:
            event (Event): The event.
            state (State): The state.
            logger (Logger): The logger.

        :meta private:
        """
        event_cb = getattr(self, event.value)
        return event_cb(state, logger)

    def init(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~Event.INIT` event.

        Args:
            state (State): The global state.
            logger (Logger): The logger.

        :meta private:
        """
        del state, logger  # unused
        pass

    def fit_start(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~Event.FIT_START` event.

        Args:
            state (State): The global state.
            logger (Logger): The logger.

        :meta private:
        """
        del state, logger  # unused
        pass

    def epoch_start(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~Event.EPOCH_START` event.

        Args:
            state (State): The global state.
            logger (Logger): The logger.

        :meta private:
        """
        del state, logger  # unused
        pass

    def batch_start(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~Event.BATCH_START` event.

        Args:
            state (State): The global state.
            logger (Logger): The logger.

        :meta private:
        """
        del state, logger  # unused
        pass

    def after_dataloader(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~Event.AFTER_DATALOADER` event.

        Args:
            state (State): The global state.
            logger (Logger): The logger.

        :meta private:
        """
        del state, logger  # unused
        pass

    def before_train_batch(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~Event.BEFORE_TRAIN_BATCH` event.

        Args:
            state (State): The global state.
            logger (Logger): The logger.

        :meta private:
        """
        del state, logger  # unused
        pass

    def before_forward(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~Event.BEFORE_FORWARD` event.

        Args:
            state (State): The global state.
            logger (Logger): The logger.

        :meta private:
        """
        del state, logger  # unused
        pass

    def after_forward(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~Event.AFTER_FORWARD` event.

        Args:
            state (State): The global state.
            logger (Logger): The logger.

        :meta private:
        """
        del state, logger  # unused
        pass

    def before_loss(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~Event.BEFORE_LOSS` event.

        Args:
            state (State): The global state.
            logger (Logger): The logger.

        :meta private:
        """
        del state, logger  # unused
        pass

    def after_loss(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~Event.AFTER_LOSS` event.

        Args:
            state (State): The global state.
            logger (Logger): The logger.

        :meta private:
        """
        del state, logger  # unused
        pass

    def before_backward(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~Event.BEFORE_BACKWARD` event.

        Args:
            state (State): The global state.
            logger (Logger): The logger.

        :meta private:
        """
        del state, logger  # unused
        pass

    def after_backward(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~Event.AFTER_BACKWARD` event.

        Args:
            state (State): The global state.
            logger (Logger): The logger.

        :meta private:
        """
        del state, logger  # unused
        pass

    def after_train_batch(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~Event.AFTER_TRAIN_BATCH` event.

        Args:
            state (State): The global state.
            logger (Logger): The logger.

        :meta private:
        """
        del state, logger  # unused
        pass

    def batch_end(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~Event.BATCH_END` event.

        .. note::

            :attr:`state.timer.batch`, :attr:`state.timer.batch_in_epoch`,
            :attr:`state.timer.sample`, :attr:`state.timer.sample_in_epoch`,
            :attr:`state.timer.token`, and :attr:`state.timer.token_in_epoch`
            are incremented immediately before :attr:`~Event.BATCH_END`.

        Args:
            state (State): The global state.
            logger (Logger): The logger.

        :meta private:
        """
        del state, logger  # unused
        pass

    def epoch_end(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~Event.EPOCH_END` event.

        .. note::

            :attr:`state.timer.epoch` is incremented immediately before :attr:`~Event.EPOCH_END`.

        Args:
            state (State): The global state.
            logger (Logger): The logger.

        :meta private:
        """
        del state, logger  # unused
        pass

    def eval_start(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~Event.EVAL_START` event.

        Args:
            state (State): The global state.
            logger (Logger): The logger.

        :meta private:
        """
        del state, logger  # unused
        pass

    def eval_batch_start(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~Event.EVAL_BATCH_START` event.

        Args:
            state (State): The global state.
            logger (Logger): The logger.

        :meta private:
        """
        del state, logger  # unused
        pass

    def eval_before_forward(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~Event.EVAL_BATCH_FORWARD` event.

        Args:
            state (State): The global state.
            logger (Logger): The logger.

        :meta private:
        """
        del state, logger  # unused
        pass

    def eval_after_forward(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~Event.EVAL_AFTER_FORWARD` event.

        Args:
            state (State): The global state.
            logger (Logger): The logger.

        :meta private:
        """
        del state, logger  # unused
        pass

    def eval_batch_end(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~Event.EVAL_BATCH_END` event.

        Args:
            state (State): The global state.
            logger (Logger): The logger.

        :meta private:
        """
        del state, logger  # unused
        pass

    def eval_end(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~Event.EVAL_END` event.

        Args:
            state (State): The global state.
            logger (Logger): The logger.

        :meta private:
        """
        del state, logger  # unused
        pass

    def close(self) -> None:
        """Called whenever the trainer finishes training, even when there is an exception.

        It should be used for flushing and closing any files, etc... that may have been opened during the
        :attr:`~Event.INIT` event.

        :meta private:
        """
        pass

    def post_close(self) -> None:
        """This hook is called after :meth:`close` has been invoked for each callback. Very few callbacks should need to
        implement :meth:`post_close`.

        This callback can be used to back up any data that may have been written by other callbacks during
        :meth:`close`.

        :meta private:
        """
        pass
