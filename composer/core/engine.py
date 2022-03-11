# Copyright 2021 MosaicML. All Rights Reserved.

"""Engine is a coordinator for running algorithms and resolving ordering conflicts among them for composition.

.. currentmodule:: composer

The order in which algorithms are run matters significantly during composition. For example,
:class:`~.SelectiveBackprop` algorithm runs on the :attr:`~.Event.AFTER_DATALOADER` event and must run before any data
augmentations.  :class:`~.engine.Engine` runs re-ordering passes to resolve such ordering issues or conflicts.

.. note::

    * An instance of :class:`~.engine.Engine` is automatically constructed by the :class:`~.trainer.Trainer`
      constructor. A user need not instantiate :class:`~.engine.Engine` class.

    * The design of :class:`~.engine.Engine` is subject to change in future releases to accommodate more complexity as
      we investigate composition of algorithms.


Currently, the following passes are registered:

* **LIFO order for events**

  For the events that follow the ``before_*`` (e.g., :attr:`~.Event.BEFORE_LOSS`) and ``after_*`` (e.g.,
  :attr:`~.Event.AFTER_LOSS`) pattern, the ordering of algorithms is reversed for the ``after_*`` events. For example,
  four given algorithms ``A``, ``B``, ``C`` and ``D`` will run in ``ABCD`` ordering on the ``before_*`` event while
  ``DCBA`` ordering on the ``after_*`` event.

  This allows algorithms to "clean up" their changes. For example, :class:`~.LabelSmoothing` will smooth the labels
  upon on :attr:`~.Event.BEFORE_LOSS` event and then restore the original unsmoothed labels on
  :attr:`~.Event.AFTER_LOSS` event.

* **Run Selective Backprop first**

  :class:`~.SelectiveBackprop` runs after the dataloader returns the batch, and executes an extra forward pass to rank
  and prune the examples in the batch by loss. To ensure a clean estimate of loss, :class:`~.SelectiveBackprop` should
  run before any other data augmentations (e.g., :class:`~.MixUp`) on the :attr:`~.Event.AFTER_DATALOADER` event.

Trace
~~~~~

Traces record whether an algorithm ran at a particular step and event combination and also the order of such executions.
These are logged with the key ``<algorithm_name>/<event>``.

For example, the algorithm :class:`~.LayerFreezing`, which runs at the end of every epoch on :attr:`~.Event.EPOCH_END`,
will emit a series of traces:

.. code-block::

   [STEP=0][layer_freezing/INIT=0]
   [STEP=1][layer_freezing/EPOCH_START=0]
   [STEP=1][layer_freezing/BATCH_START=0]
   ...
   [STEP=2][layer_freezing/BATCH_START=0]
   ...
   [STEP=3][layer_freezing/BATCH_START=0]
   ...
   [STEP=3][layer_freezing/EPOCH_END=1]  # <-- layer freezing ran on step 3 here!
"""
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

__all__ = ["Trace", "Engine", "Traces"]

#: The default traces of an entire run is an OrderedDict.
#: The keys are of format ``<algorithm_name>/<event>`` (e.g.,  ``Blurpool/INIT``) and values are an instance of
#: :class:`Trace`.
Traces = Dict[str, "Trace"]

_ALWAYS_RECORD_EVENTS = [Event.INIT, Event.FIT_START, Event.EPOCH_START, Event.EPOCH_END]


@dataclass
class Trace():
    """Record of an algorithm's execution.

    Attributes:
        exit_code (int or None): Optional return value from an algorithm. Default: None.
        order (int or None): Order in which the algorithm was executed
                             in the list of algorithms. None means algorithm was not run.
        run (bool): Whether the algorithm was run. Default: False
    """
    exit_code: Optional[int] = None
    order: Optional[int] = None
    run: bool = False


def _setup_trace(algorithms: Sequence[Algorithm], event: Event) -> Traces:
    """The default traces of an entire run is an OrderedDict.

    The keys are of format ``<algorithm_name>/<event>`` (e.g.,  ``Blurpool/INIT``) and values are an instance of
    :class:`Trace`.
    """
    return OrderedDict([(f'{algo}/{event}', Trace()) for algo in algorithms])


class Engine():
    """Coordinator for running algorithms and resolving ordering conflicts among them for composition.

    Args:
        state (State): The initial :class:`~.state.State` of the trainer. ``state`` will be modified in-place.
        logger (Optional[Logger]): A :class:`~.logger.Logger` instance to be used for logging algorithm and callback
            specific metrics.
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
        """Runs the sequence of algorithms and callbacks (see :class:`~.callback.Callback`).

        Filters algorithms by calling each one's :meth:`~.Algorithm.match` method, internally checks for conflicting
        algorithms, then runs each algorithm's :meth:`~.Algorithm.apply` method to make in-place changes to the
        ``state``.

        The default order of execution for algorithms is determined by the provided list. However, :class:`Engine` makes
        changes to this order internally to resolve ordering conflicts.

        Returns :data:`Traces` of the execution, a dictionary with keys formatted as ``<algorithm_name>/<event>`` (e.g.,
        ``Blurpool/INIT``), and values are an instance of :class:`~.engine.Trace`.

        Callbacks are always run after algorithms and do not return a trace.

        This method can be called with either the :class:`~.event.Event` enum member values or a string of the event
        name.

        Examples:
            >>> engine = Engine(state, logger)
            >>> engine.run_event(Event.BEFORE_LOSS)
            OrderedDict()
            >>> # calling with a string of the event name also works
            >>> engine.run_event('before_loss')
            OrderedDict()


        Args:
            event (Event or str): The current :class:`~.event.Event`. It can be the enum member values or a
                string with the event value.
        Returns:
            traces (Traces): Ordered dictionary of trace for each algorithm.
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
                duration_marker = self.state.profiler.marker(name, actions=actions)

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
            if len(trace) > 0:
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
            algorithms_to_run(Sequence[Algorithm]): Sequence of algorithms
            event (Event): The current event

        Returns:
            algorithms_to_run(Sequence[Algorithm]): Modified sequence of algorithms
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
            event (Event): The current :class:`~.event.Event`
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
        """Invokes :meth:`~.Callback.close` and :meth:`~.Callback.post_close` for each callback.

        :meth:`~.Callback.close` is invoked for each callback. For all callbacks where :meth:`~.Callback.close` did not
        raise an exception, then :meth:`~.Callback.post_close` is invoked.

        This method does not re-raise any exceptions from :meth:`~.Callback.close` and :meth:`~.Callback.post_close`.
        Instead, these exceptions are logged to the :class:`~.logger.Logger`.
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
            self.state.profiler._merge_traces()

        for callback in self.state.callbacks:
            if callback_to_has_exception[callback] is False:
                try:
                    callback.post_close()
                except Exception as e:
                    log.error(f"Error running {callback.__class__.__name__}.post_close().", exc_info=e, stack_info=True)
