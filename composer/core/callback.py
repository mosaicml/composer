from __future__ import annotations

import abc
from functools import wraps
from types import MethodType
from typing import TYPE_CHECKING, Any, Callable

from composer.core.serializable import Serializable
from composer.utils.ddp import is_rank_set, is_rank_zero

if TYPE_CHECKING:
    from composer import Logger, State


class Callback(Serializable, abc.ABC):
    """
    Base class for callbacks. Callbacks are similar to Algorithms, in that
    they are run on specific events. By convention, Callbacks should not
    modify ``State``.

    Each method name in ``Callback`` corresponds to an ``Event`` value.
    """

    def init(self, state: State, logger: Logger) -> None:
        pass

    def training_start(self, state: State, logger: Logger) -> None:
        pass

    def epoch_start(self, state: State, logger: Logger) -> None:
        pass

    def batch_start(self, state: State, logger: Logger) -> None:
        pass

    def after_dataloader(self, state: State, logger: Logger) -> None:
        pass

    def before_train_batch(self, state: State, logger: Logger) -> None:
        pass

    def before_forward(self, state: State, logger: Logger) -> None:
        pass

    def after_forward(self, state: State, logger: Logger) -> None:
        pass

    def before_loss(self, state: State, logger: Logger) -> None:
        pass

    def after_loss(self, state: State, logger: Logger) -> None:
        pass

    def before_backward(self, state: State, logger: Logger) -> None:
        pass

    def after_backward(self, state: State, logger: Logger) -> None:
        pass

    def after_train_batch(self, state: State, logger: Logger) -> None:
        pass

    def batch_end(self, state: State, logger: Logger) -> None:
        pass

    def epoch_end(self, state: State, logger: Logger) -> None:
        pass

    def training_end(self, state: State, logger: Logger) -> None:
        pass

    def eval_start(self, state: State, logger: Logger) -> None:
        pass

    def eval_batch_start(self, state: State, logger: Logger) -> None:
        pass

    def eval_before_forward(self, state: State, logger: Logger) -> None:
        pass

    def eval_after_forward(self, state: State, logger: Logger) -> None:
        pass

    def eval_batch_end(self, state: State, logger: Logger) -> None:
        pass

    def eval_end(self, state: State, logger: Logger) -> None:
        pass


class RankZeroCallback(Callback, abc.ABC):
    """
    RankZeroCallback ensures that all callback methods that occur after the DDP fork are only executed on rank 0.
    Note: :meth:`init`, :meth:`load_state_dict` are executed before the DDP fork and will be called on all ranks.
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
                if is_rank_set():
                    if not is_rank_zero():
                        return
                return original_fn(*args, **kwargs)

            setattr(self, fn_name, MethodType(wrapped_fn, self))
