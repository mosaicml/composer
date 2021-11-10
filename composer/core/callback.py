# Copyright 2021 MosaicML. All Rights Reserved.

"""Base module for callbacks.
"""
from __future__ import annotations

import abc
from functools import wraps
from types import MethodType
from typing import TYPE_CHECKING, Any, Callable

from composer.core.serializable import Serializable
from composer.utils.ddp import is_rank_zero

if TYPE_CHECKING:
    from composer import Logger, State


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

    .. Note::
    
        :meth:`init` and :meth:`load_state_dict` are executed
        before the DDP fork and will be called on all ranks.
    """

    def __init__(self) -> None:
        from composer.core import Event

        super().__init__()

        # ensure all callbacks are executed only on rank 0
        functions_to_wrap = [*(event.value for event in Event), "state_dict"]

        for fn_name in functions_to_wrap:
            original_fn = getattr(self, fn_name)

            @wraps(original_fn)
            def wrapped_fn(
                backend: RankZeroCallback,
                *args: Any,
                original_fn: Callable[[State, Logger], None] = original_fn,
                **kwargs: Any,
            ) -> None:
                if not is_rank_zero():
                    return
                return original_fn(*args, **kwargs)

            setattr(self, fn_name, MethodType(wrapped_fn, self))
