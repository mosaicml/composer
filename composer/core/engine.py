# Copyright 2021 MosaicML. All Rights Reserved.

import contextlib
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import ContextManager, Dict, Optional, Sequence, Union, cast

from composer.core.algorithm import Algorithm
from composer.core.callback import Callback
from composer.core.event import Event
from composer.core.logging import Logger
from composer.core.logging.logger import LogLevel
from composer.core.state import State
from composer.profiler import ProfilerAction

log = logging.getLogger(__name__)
Traces = Dict[str, "Trace"]

_ALWAYS_RECORD_EVENTS = [Event.INIT, Event.FIT_START, Event.EPOCH_START, Event.EPOCH_END]


@dataclass
class Trace():
    """Record of an algorithm's execution.

    Attributes:
        exit_code (int or None): optional return value from algorithm. Default: None.
        order (int or None): order in which the algorithm was executed
                             in the list of algorithms. None means algorithm was not run.
        run (bool): whether the algorithm was run. Default: False
    """
    exit_code: Optional[int] = None
    order: Optional[int] = None
    run: bool = False


def _setup_trace(algorithms: Sequence[Algorithm], event: Event) -> Traces:
    """The default traces of an entire run is an OrderedDict, with the keys of format 'algorithm_name/event' (e.g.
    Blurpool/INIT)."""
    return OrderedDict([(f'{algo}/{event}', Trace()) for algo in algorithms])


class Engine():
    """Coordinator for running and resolving conflicts between algorithms.

    Args:
        state (State): the initial ``State`` of the trainer. Will be modified in-place.
        logger (Optional[Logger]): a ``Logger`` instance to be used for logging algorithm and callback specific metrics.
    """

    def __init__(self, state: State, logger: Optional[Logger] = None):
        if logger is None:
            log.warning("No logger passed to the engine.  Defaulting to an empty logger")
            logger = Logger(state=state, backends=[])

        assert logger is not None
        self.logger = logger
        self.state = state

    def run_event(
        self,
        event: Union[Event, str],
    ) -> Traces:
        """Runs the sequence of algorithms and callbacks.

        Filters algorithms by calling each one's :meth:`~Algorithm.match` function, internally
        checks for conflicting algorithms, then runs each algorithm's :meth:`~Algorithm.apply`
        function to make in-place changes to the :class:`State`.

        The order of algorithm execution is determined by the provided list, plus any changes
        made internally to prevent conflicts.

        Returns traces of the execution, a dictionary with keys formatted as ``<algorithm_name>/<event>``
        (e.g. ``Blurpool/INIT``), and values are the :class:`composer.core.engine.Trace` object,
        which include an optional return code from the algorithm, the order of execution, and whether
        the algorithm was run.

        Callbacks are always ran after algorithms, and do not return a trace.

        Can be called with either the Event enum, or a string of the event value.

        Examples:
            >>> engine = Engine(state, algorithms, logger, callbacks)
            >>> engine.run_event(Event.BEFORE_LOSS) # or
            >>> engine.run_event('before_loss') # also works


        Args:
            event (Event or str): the current :class:`Event`. Can be the enum or a string with the event value.
        Returns:
            Dict[str, Trace]: dictionary of trace for each algorithm.
        """
        duration_marker = None
        event = Event(event)

        if self.state.profiler is not None:
            name = f"event/{event.canonical_name}"
            if (event.is_before_event or event.is_after_event):
                # if not part of an event pair (e.g. init or after dataloader), then don't record an event here
                if event in _ALWAYS_RECORD_EVENTS:
                    actions = [ProfilerAction.ACTIVE, ProfilerAction.WARMUP, ProfilerAction.SKIP]
                else:
                    actions = [ProfilerAction.ACTIVE, ProfilerAction.WARMUP]
                duration_marker = self.state.profiler.marker(name, actions=actions, record_instant_on_start=True)

        if event.is_after_event and duration_marker is not None:
            duration_marker.finish()

        if event == Event.INIT:
            # For the INIT event, run the callbacks first to initialize the loggers
            # For other events, run the algorithms first, so the callbacks have the state
            # after algorithms modify it
            self._run_callbacks(event)
            traces = self._run_algorithms(event)
        else:
            traces = self._run_algorithms(event)
            self._run_callbacks(event)

        if event.is_before_event and duration_marker is not None:
            duration_marker.start()

        return traces

    def _run_algorithms(
        self,
        event: Event,
    ) -> Traces:
        algorithms_to_run = [algo for algo in self.state.algorithms if algo.match(event, self.state)]

        # future collision resolution
        algorithms_to_run = self._compile(algorithms_to_run, event)

        trace = _setup_trace(algorithms_to_run, event)
        for order, algorithm in enumerate(algorithms_to_run):
            marker = None
            if self.state.profiler is not None:
                marker = self.state.profiler.marker(f"algorithm/{algorithm.__class__.__name__}/event/{event.value}", 
                                                    categories=[
                                                        event.value,
                                                        algorithm.__class__.__name__,
                                                    ])
            ctx = cast(ContextManager, contextlib.nullcontext()) if marker is None else marker
            with ctx:
                exit_code = algorithm.apply(event, self.state, self.logger)

            trace_key = f'{algorithm}/{event}'
            trace[trace_key] = Trace(exit_code=exit_code, order=order, run=True)

        if self.logger is not None:
            if event in (Event.INIT, Event.FIT_START):
                log_level = LogLevel.FIT
            if event in (Event.EPOCH_START, Event.EPOCH_END):
                log_level = LogLevel.EPOCH
            else:
                # algs don't run on eval events, so don't have to worry about
                # batch-frequency vs epoch-frequency evaluators
                log_level = LogLevel.BATCH
            self.logger.metric(log_level=log_level, data={key: 1 if tr.run else 0 for key, tr in trace.items()})

        return trace

    def _compile(
        self,
        algorithms_to_run: Sequence[Algorithm],
        event: Event,
    ) -> Sequence[Algorithm]:
        """Runs compilation passes that modify the order and content of a list of algorithms.

        Currently, runs the algorithms in a FILO queue for the before_ and after_ events. For example,
        algorithms will run in order ABCD during before_loss, and in DCBA during after_loss. The motivation
        here is that algorithms can 'undo' their effects upon the exit of an event. Note that events that
        have the pattern _start or _end will still run with ABCD order.

        Intent of this method is to eventually store and handle other algorithms collisions and ordering
        requirements.

        Args:
            algorithms_to_run(Sequence[Algorithm]): sequence of algorithms
            event (Event): the current event

        Returns:
            algorithms_to_run(Sequence[Algorithm]): modified sequence of algorithms
        """
        from composer.algorithms import SelectiveBackprop, StochasticDepth

        # Move selective backprop to the beginning while maintaining order of other algorithms
        algorithms = sorted(algorithms_to_run,
                            key=lambda x: not isinstance(x, SelectiveBackprop) and not isinstance(x, StochasticDepth))

        if event.is_after_event:
            """Establish a FILO queue of algorithms before_ and after_ an event.

            before_loss: A, B, C, D
            after_loss: D, C, B, A
            """
            algorithms = list(reversed(algorithms))

        return algorithms

    def _run_callbacks(
        self,
        event: Union[Event, str],
    ):
        """Runs a sequence of callbacks by calling the function for an event.

        Args:
            event (Event): the current ``Event``
            state (State): the current ``State``
            callbacks (Sequence[Callback]): a sequence of callbacks
        Returns:
            None
        """
        event = Event(event)

        for cb in self.state.callbacks:
            marker = None
            if self.state.profiler is not None:
                marker = self.state.profiler.marker(f"callback/{cb.__class__.__name__}/event/{event.value}",
                                                    categories=[
                                                        event.value,
                                                        cb.__class__.__name__,
                                                    ])
            ctx = cast(ContextManager, contextlib.nullcontext()) if marker is None else marker
            with ctx:
                cb.run_event(event, self.state, self.logger)

    def close(self) -> None:
        """Invoke :meth:`~Callback.close` and :meth:`~Callback.post_close` for each callback.

        :meth:`~Callback.close` is invoked for each callback.
        For all callbacks where :meth:`~Callback.close` did not raise an exception, then
        :meth:`~Callback.post_close` is invoked.

        Does not re-raise any exceptions from :meth:`~Callback.close` and :meth:`~Callback.post_close`.
        Instead, these exceptions are logged.
        """
        callback_to_has_exception: Dict[Callback, bool] = {}
        for callback in self.state.callbacks:
            try:
                callback.close()
            except Exception as e:
                log.error(
                    f"Error running {callback.__class__.__name__}.close(). Skipping {callback.__class__.__name__}.post_close().",
                    exc_info=e,
                    stack_info=True)
                callback_to_has_exception[callback] = True
            else:
                callback_to_has_exception[callback] = False

        if self.state.profiler is not None:
            # Merge traces after close, but before post_close, so the merged file will be uploaded
            self.state.profiler.merge_traces()

        for callback in self.state.callbacks:
            if callback_to_has_exception[callback] is False:
                try:
                    callback.post_close()
                except Exception as e:
                    log.error(f"Error running {callback.__class__.__name__}.post_close().", exc_info=e, stack_info=True)
