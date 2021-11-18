# Copyright 2021 MosaicML. All Rights Reserved.

"""Base module for callbacks.
"""
from __future__ import annotations

import abc
import warnings
from typing import TYPE_CHECKING

from composer.core.serializable import Serializable
from composer.utils.ddp import is_rank_zero

try:
    from typing import final
except ImportError:
    final = lambda x: x  # final is not available in python 3.7

if TYPE_CHECKING:
    from composer import Event, Logger, State


class Callback(Serializable, abc.ABC):
    """Base class for callbacks.
    
    A callback is similar to an
    :class:`Algorithm`, in that
    they are run on specific events. By convention, Callbacks should not
    modify :class:`State`.

    Subclasses should override :meth:`_run_event`
    (**not** `run_event`) to run in response
    to given :class:`Event` invocations.
    """

    def __init__(self) -> None:
        super().__init__()

    @final
    def run_event(self, event: Event, state: State, logger: Logger) -> None:
        """This method is called by the engine on each event.

        Args:
            event (Event): The event.
            state (State): The state.
            logger (Logger): The logger.
        """
        self._run_event(event, state, logger)

    def _run_event(self, event: Event, state: State, logger: Logger) -> None:
        # default fallback if the callback does not override _run_event
        try:
            event_cb = getattr(self, event.value)
        except AttributeError:
            return
        warnings.warn(
            f"CallbackMethodDeprecationWarning: `self.{event.value}()` will be removed in callbacks."
            "Instead, override `self._run_event()`.",
            category=DeprecationWarning)
        return event_cb(state, logger)


class RankZeroCallback(Callback, abc.ABC):
    """Base class for callbacks that only run on the rank zero process.

    Subclasses should override :meth:`_run_event`
    (**not** `run_event`) to run in response
    to given :class:`Event` invocations.
    """

    @final
    def run_event(self, event: Event, state: State, logger: Logger) -> None:
        if not is_rank_zero():
            return
        return self._run_event(event, state, logger)
