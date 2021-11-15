# Copyright 2021 MosaicML. All Rights Reserved.

"""Base module for callbacks.
"""
from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from composer.core.serializable import Serializable
from composer.utils.ddp import is_rank_zero

if TYPE_CHECKING:
    from composer import Event, Logger, State


class Callback(Serializable, abc.ABC):
    """Base class for callbacks.
    
    A callback is similar to an
    :class:`Algorithm`, in that
    they are run on specific events. By convention, Callbacks should not
    modify :class:`State`.

    Each method name corresponds to an :class:`Event`.

    Subclasses of callbacks should override these methods to run in response
    to given :class:`Event` invocations.
    """

    def __init__(self) -> None:
        super().__init__()

    def init(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~Event.INIT` event.

        Args:
            state (State): The global state.
            logger (Logger): The logger.
        
        """
        del state, logger  # unused
        pass

    def run_event(self, event: Event, state: State, logger: Logger) -> None:
        """This method is called by the engine on each event. By default, it
        invokes the callback function for the event (for example,
        `self.run_event(Event.TRAINING_START, state, logger)` invokes
        `self.training_start(state, logger)`). If this method is overridden,
        the subclass method should include `super().run_event(event, state, logger)`
        so all callback methods will be invoked.

        Args:
            event (Event): The event.
            state (State): The state.
            logger (Logger): The logger.
        """
        try:
            event_cb = getattr(self, event.value)
        except AttributeError:
            raise ValueError(f'Callback {self} has no method for event {event}')
        return event_cb(state, logger)

    def training_start(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`Event.TRAINING_START` event.

        Args:
            state (State): The global state.
            logger (Logger): The logger.
        
        """
        del state, logger  # unused
        pass

    def epoch_start(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~Event.EPOCH_START` event.

        Args:
            state (State): The global state.
            logger (Logger): The logger.
        
        """
        del state, logger  # unused
        pass

    def batch_start(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~Event.BATCH_START` event.

        Args:
            state (State): The global state.
            logger (Logger): The logger.
        
        """
        del state, logger  # unused
        pass

    def after_dataloader(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~Event.AFTER_DATALOADER` event.

        Args:
            state (State): The global state.
            logger (Logger): The logger.
        
        """
        del state, logger  # unused
        pass

    def before_train_batch(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~Event.BEFORE_TRAIN_BATCH` event.

        Args:
            state (State): The global state.
            logger (Logger): The logger.
        
        """
        del state, logger  # unused
        pass

    def before_forward(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~Event.BEFORE_FORWARD` event.

        Args:
            state (State): The global state.
            logger (Logger): The logger.
        
        """
        del state, logger  # unused
        pass

    def after_forward(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~Event.AFTER_FORWARD` event.

        Args:
            state (State): The global state.
            logger (Logger): The logger.
        
        """
        del state, logger  # unused
        pass

    def before_loss(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~Event.BEFORE_LOSS` event.

        Args:
            state (State): The global state.
            logger (Logger): The logger.
        
        """
        del state, logger  # unused
        pass

    def after_loss(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~Event.AFTER_LOSS` event.

        Args:
            state (State): The global state.
            logger (Logger): The logger.
        
        """
        del state, logger  # unused
        pass

    def before_backward(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~Event.BEFORE_BACKWARD` event.

        Args:
            state (State): The global state.
            logger (Logger): The logger.
        
        """
        del state, logger  # unused
        pass

    def after_backward(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~Event.AFTER_BACKWARD` event.

        Args:
            state (State): The global state.
            logger (Logger): The logger.
        
        """
        del state, logger  # unused
        pass

    def after_train_batch(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~Event.AFTER_TRAIN_BATCH` event.

        Args:
            state (State): The global state.
            logger (Logger): The logger.
        
        """
        del state, logger  # unused
        pass

    def batch_end(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~Event.BATCH_END` event.

        Args:
            state (State): The global state.
            logger (Logger): The logger.
        
        """
        del state, logger  # unused
        pass

    def epoch_end(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~Event.EPOCH_END` event.

        Args:
            state (State): The global state.
            logger (Logger): The logger.
        
        """
        del state, logger  # unused
        pass

    def training_end(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~Event.TRAINING_END` event.

        Args:
            state (State): The global state.
            logger (Logger): The logger.
        
        """
        del state, logger  # unused
        pass

    def eval_start(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~Event.EVAL_START` event.

        Args:
            state (State): The global state.
            logger (Logger): The logger.
        
        """
        del state, logger  # unused
        pass

    def eval_batch_start(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~Event.EVAL_BATCH_START` event.

        Args:
            state (State): The global state.
            logger (Logger): The logger.
        
        """
        del state, logger  # unused
        pass

    def eval_before_forward(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~Event.EVAL_BATCH_FORWARD` event.

        Args:
            state (State): The global state.
            logger (Logger): The logger.
        
        """
        del state, logger  # unused
        pass

    def eval_after_forward(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~Event.EVAL_AFTER_FORWARD` event.

        Args:
            state (State): The global state.
            logger (Logger): The logger.
        
        """
        del state, logger  # unused
        pass

    def eval_batch_end(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~Event.EVAL_BATCH_END` event.

        Args:
            state (State): The global state.
            logger (Logger): The logger.
        
        """
        del state, logger  # unused
        pass

    def eval_end(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`~Event.EVAL_END` event.

        Args:
            state (State): The global state.
            logger (Logger): The logger.
        
        """
        del state, logger  # unused
        pass


class RankZeroCallback(Callback, abc.ABC):
    """Base class for callbacks that only run on the rank zero process.
    """

    def run_event(self, event: Event, state: State, logger: Logger) -> None:
        if not is_rank_zero():
            return
        return super().run_event(event, state, logger)
