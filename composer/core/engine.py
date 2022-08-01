# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Engine is a coordinator for running algorithms and resolving ordering conflicts among them for composition.

.. currentmodule:: composer

The order in which algorithms are run matters significantly during composition. For example, the
:class:`.SelectiveBackprop` algorithm runs on the :attr:`.Event.AFTER_DATALOADER` event and must run before
any data augmentations. :class:`.Engine` runs re-ordering passes to resolve such ordering issues or conflicts.

.. note::

    * An instance of :class:`.Engine` is automatically constructed by the :class:`.Trainer`
      constructor. A user need not instantiate the :class:`.Engine` class.

    * The design of :class:`.Engine` is subject to change in future releases
      to accommodate more complexity as we investigate composition of algorithms.


Currently, the following passes are registered:

* **LIFO order for events**

  For the events that follow the ``before_*`` (e.g., :attr:`.Event.BEFORE_LOSS`) and ``after_*`` (e.g.,
  :attr:`.Event.AFTER_LOSS`) pattern, the ordering of algorithms is reversed for the ``after_*`` events. For example,
  four given algorithms ``A``, ``B``, ``C``, and ``D`` will run in ``ABCD`` ordering on the ``before_*`` event while
  ``DCBA`` ordering on the ``after_*`` event.

  This allows algorithms to "clean up" their changes. For example, :class:`.LabelSmoothing` will smooth the labels
  upon the :attr:`.Event.BEFORE_LOSS` event and then restore the original unsmoothed labels on the
  :attr:`.Event.AFTER_LOSS` event.

* **Run Selective Backprop first**

  :class:`.SelectiveBackprop` runs after the dataloader returns the batch and executes an extra forward pass to rank
  and prune the examples in the batch by loss. To ensure a clean estimate of loss, :class:`.SelectiveBackprop` should
  run before any other data augmentations (e.g., :class:`.MixUp`) on the :attr:`.Event.AFTER_DATALOADER` event.

Trace
~~~~~

Traces record whether an algorithm ran at a particular step and event combination and also the order of such executions.
These are logged with the key ``<algorithm_name>/<event>``.

For example, the algorithm :class:`.LayerFreezing`, which runs at the end of every epoch on :attr:`.Event.EPOCH_END`,
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

from __future__ import annotations

import atexit
import contextlib
import logging
import warnings
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from typing import ContextManager, Dict, Optional, Sequence, Union, cast

from composer.core.algorithm import Algorithm
from composer.core.callback import Callback
from composer.core.event import Event
from composer.core.state import State
from composer.loggers import Logger, LogLevel
from composer.profiler import ProfilerAction

log = logging.getLogger(__name__)

__all__ = ['Trace', 'Engine', 'Traces']

#: The default traces of an entire run is an OrderedDict.
#: The keys are of format ``<algorithm_name>/<event>`` (e.g.,  ``Blurpool/INIT``) and values are an instance of
#: :class:`Trace`.
Traces = Dict[str, 'Trace']

_ALWAYS_RECORD_EVENTS = [Event.INIT, Event.FIT_START, Event.EPOCH_START, Event.EPOCH_END]
_EVENTS_WHERE_DATALOADER_IS_SET = [e for e in Event if e != Event.INIT]
_EVENTS_WHERE_MAX_DURATION_IS_SET = [
    Event.FIT_START,
    Event.EPOCH_START,
    Event.BATCH_START,
    Event.AFTER_DATALOADER,
    Event.BEFORE_TRAIN_BATCH,
    Event.BEFORE_FORWARD,
    Event.AFTER_FORWARD,
    Event.BEFORE_LOSS,
    Event.AFTER_LOSS,
    Event.BEFORE_BACKWARD,
    Event.AFTER_BACKWARD,
    Event.AFTER_TRAIN_BATCH,
    Event.BATCH_END,
    Event.BATCH_CHECKPOINT,
    Event.EPOCH_END,
    Event.EPOCH_CHECKPOINT,
    Event.FIT_END,
]
_EVAL_EVENTS = [e for e in Event if e.name.startswith('EVAL_')]
_PREDICT_EVENTS = [e for e in Event if e.name.startswith('PREDICT_')]

# Track whether atexit triggered _close(), which indicates whether the python process is shutting down
# If so, do not run close() again via __del__(), as Python machinery (e.g. the ability to do conditional
# imports) are destroyed between close() and __del__().
# Using a global variable instead of an instance variable as _close() is not bound to the instance
_did_atexit_run = False


def _set_atexit_ran():
    global _did_atexit_run
    _did_atexit_run = True


# Since atexit calls hooks in LIFO order, this hook will always be invoked after all atexit-triggered
# _close() calls are invoked
atexit.register(_set_atexit_ran)


@dataclass
class Trace():
    """Record of an algorithm's execution.

    Attributes:
        name (str): The name of the algorithm.
        event (Event): The current event.
        exit_code (int | None): Optional return value from an algorithm. Default: None.
        order (int | None): Order in which the algorithm was executed
                             in the list of algorithms. None means algorithm was not run.
        run (bool): Whether the algorithm was run. Default: False
    """
    name: str = ''
    event: Optional[Event] = None
    exit_code: Optional[int] = None
    order: Optional[int] = None
    run: bool = False


def _setup_trace(algorithms: Sequence[Algorithm], event: Event) -> Traces:
    """The default traces of an entire run is an OrderedDict.

    The keys are of format ``<algorithm_name>/<event>`` (e.g.,  ``Blurpool/INIT``) and values are an instance of
    :class:`Trace`.
    """
    return OrderedDict([(f'{algo}/{event}', Trace(name=algo.__class__.__name__)) for algo in algorithms])


# Track which callbacks are already open, so it is possible to error and instruct the user to call
# previous_trainer.close() if necessary before attempting to reuse a callback
_OPEN_CALLBACKS = weakref.WeakSet()


class Engine():
    """Coordinator for running algorithms and resolving ordering conflicts among them for composition.

    Args:
        state (State): The initial :class:`.State` of the trainer. ``state`` will be modified in-place.
        logger (Logger): A :class:`.Logger` instance to be used for logging algorithm and callback
            specific metrics.
    """

    def __init__(self, state: State, logger: Logger):
        self.logger = logger
        self.state = state
        self._is_closed = False
        atexit.register(self._close, state, logger)

    def run_event(
        self,
        event: Union[Event, str],
    ) -> Traces:
        """Runs the sequence of algorithms and callbacks (see :class:`.Callback`).

        Filters algorithms by calling each one's :meth:`.Algorithm.match` method, internally checks for conflicting
        algorithms, then runs each algorithm's :meth:`.Algorithm.apply` method to make in-place changes to the
        ``state``.

        The default order of execution for algorithms is determined by the provided list. However, :class:`.Engine` makes
        changes to this order internally to resolve ordering conflicts.

        Returns :data:`.Traces` of the execution, a dictionary with keys formatted as ``<algorithm_name>/<event>`` (e.g.,
        ``Blurpool/INIT``), and values are an instance of :class:`.Trace`.

        Callbacks are always run after algorithms and do not return a trace.

        This method can be called with either the :class:`.Event` enum member values or a string of the event name.

        Examples:
            >>> engine = Engine(state, logger)
            >>> engine.run_event(Event.BEFORE_LOSS)
            OrderedDict()
            >>> # calling with a string of the event name also works
            >>> engine.run_event('before_loss')
            OrderedDict()


        Args:
            event (Event | str): The current :class:`.Event`. It can be the enum member values or a
                string with the event value.

        Returns:
            traces (Traces): Ordered dictionary of trace for each algorithm.
        """
        duration_marker = None
        event = Event(event)

        self._debug_log(event, 'Running event')

        if self._is_closed:
            raise RuntimeError(('The engine was already closed and therefore cannot be used again. '
                                'To fix, please create a new Engine (or Trainer)'))

        if self.state.profiler is not None:
            name = f'event/{event.canonical_name}'
            if (event.is_before_event or event.is_after_event):
                # if not part of an event pair (e.g. init or after dataloader), then don't record an event here
                if event in _ALWAYS_RECORD_EVENTS:
                    actions = [ProfilerAction.ACTIVE, ProfilerAction.WARMUP, ProfilerAction.SKIP]
                else:
                    actions = [ProfilerAction.ACTIVE, ProfilerAction.WARMUP]
                duration_marker = self.state.profiler.marker(name, actions=actions)

        if event.is_after_event and duration_marker is not None:
            duration_marker.finish()

        if event in _EVENTS_WHERE_DATALOADER_IS_SET:
            assert self.state.dataloader is not None, f'The trainer should have set state.dataloader for event {event}.'

        if event in _EVENTS_WHERE_MAX_DURATION_IS_SET:
            assert self.state.max_duration is not None, f'The trainer should have set state.max_duration for event {event}.'

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
                marker = self.state.profiler.marker(f'algorithm/{algorithm.__class__.__name__}/event/{event.value}',
                                                    categories=[
                                                        event.value,
                                                        algorithm.__class__.__name__,
                                                    ])
            ctx = cast(ContextManager, contextlib.nullcontext()) if marker is None else marker
            with ctx:
                self._debug_log(event, f'Running algorithm {type(algorithm).__name__}')
                exit_code = algorithm.apply(event, self.state, self.logger)

            trace_key = f'{algorithm}/{event}'
            trace[trace_key] = Trace(name=algorithm.__class__.__name__,
                                     event=event,
                                     exit_code=exit_code,
                                     order=order,
                                     run=True)

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
                self.logger.data(log_level=log_level,
                                 data={f'{tr.name}/{tr.event}': 1 if tr.run else 0 for _, tr in trace.items()})

        return trace

    def _compile(
        self,
        algorithms_to_run: Sequence[Algorithm],
        event: Event,
    ) -> Sequence[Algorithm]:
        """Runs compilation passes that modify the order and content of a list of algorithms.

        Currently, runs the algorithms in a FILO queue for the ``before_`` and ``after_`` events. For example,
        algorithms will run in order ABCD during ``before_loss``, and in DCBA during ``after_loss``. The motivation
        here is that algorithms can 'undo' their effects upon the exit of an event. Note that events that
        have the pattern ``_start`` or ``_end`` will still run with ABCD order.

        The intent of this method is to eventually store and handle other algorithms' collisions and ordering
        requirements.

        Args:
            algorithms_to_run(Sequence[Algorithm]): Sequence of algorithms.
            event (Event): The current event.

        Returns:
            Sequence[Algorithm]: Modified sequence of algorithms.
        """
        from composer.algorithms import CutMix, FusedLayerNorm, MixUp, SelectiveBackprop, StochasticDepth

        # Move selective backprop to the beginning while maintaining order of other algorithms
        algorithms = sorted(algorithms_to_run,
                            key=lambda x: not isinstance(x, SelectiveBackprop) and not isinstance(x, StochasticDepth))

        # Move fused layernorm to the end while maintaining order of other algorithms (FLN only does surgery on leaf modules)
        algorithms = sorted(algorithms, key=lambda x: isinstance(x, FusedLayerNorm))

        # Check for multiple algorithms that try to interpolate the loss at the same time
        interpolation_settings = [a.interpolate_loss for a in algorithms if isinstance(a, (CutMix, MixUp))]
        if sum(interpolation_settings) > 1:
            warnings.warn(
                'Multiple algorithms are trying to interpolate the loss. This can result in strange behavior.')

        if event.is_after_event:
            """Establish a FILO queue of algorithms ``before_`` and ``after_`` an event.

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
            event (Event | str): The current :class:`.Event`.
        """
        event = Event(event)

        if event == Event.INIT:
            # Some callbacks may be open from a previous training run
            # If so, error and instruct the user that they must call `trainer.close()`
            # so callbacks can clean up and reset their state properly
            for cb in self.state.callbacks:
                # If it's not in the set, then the callback is new, so it's closed by definition
                if cb in _OPEN_CALLBACKS:
                    raise RuntimeError(
                        ('Cannot create a new trainer with an open callback or logger from a previous trainer. '
                         'To fix, call trainer.close() before creating this new trainer to ensure that all '
                         'callbacks or loggers shut down properly.'))
                _OPEN_CALLBACKS.add(cb)

        for cb in self.state.callbacks:
            marker = None
            if self.state.profiler is not None:
                marker = self.state.profiler.marker(f'callback/{cb.__class__.__name__}/event/{event.value}',
                                                    categories=[
                                                        event.value,
                                                        cb.__class__.__name__,
                                                    ])
            ctx = cast(ContextManager, contextlib.nullcontext()) if marker is None else marker
            with ctx:
                self._debug_log(event, f'Running callback {type(cb).__name__}')
                cb.run_event(event, self.state, self.logger)

    def __del__(self):
        global _did_atexit_run
        if _did_atexit_run or self._is_closed:
            # Do not attempt to shutdown again, since close() already ran via __atexit__ or was already invoked
            return
        self.close()

    def _debug_log(self, event: Event, msg: str):
        """Helper to include timestamp and event info in log messages."""
        if event in _EVAL_EVENTS:
            log.debug(
                '[ep=%i][ba=%i][eval_ba=%i][event=%s]: %s',
                int(self.state.timestamp.epoch),
                int(self.state.timestamp.batch),
                int(self.state.eval_timestamp.batch),
                event.name,
                msg,
            )
        elif event in _PREDICT_EVENTS:
            log.debug(
                '[ep=%i][ba=%i][predict_ba=%i][event=%s]: %s',
                int(self.state.timestamp.epoch),
                int(self.state.timestamp.batch),
                int(self.state.predict_timestamp.batch),
                event.name,
                msg,
            )
        else:
            log.debug(
                '[ep=%i][ba=%i][event=%s]: %s',
                int(self.state.timestamp.epoch),
                int(self.state.timestamp.batch),
                event.name,
                msg,
            )

    def close(self) -> None:
        """Shutdown the engine.

        As part of the shutdown procedure, :meth:`.Callback.close` and :meth:`.Callback.post_close` are invoked
        for each callback. Note that :meth:`.Callback.post_close` is invoked only for callbacks that did not raise
        an exception during :meth:`.Callback.close`.

        This method does not re-raise any exceptions from :meth:`.Callback.close` and :meth:`.Callback.post_close`.
        Instead, these exceptions are logged as errors.
        """
        self._close(self.state, self.logger)
        # The self._is_closed flag would not be set if `_close` is called via atexit
        # However, in these cases, the engine would never be used again, as Python is shutting
        # down. It is only required to set the flag if the user manually calls `close()` and still holds
        # a reference to the engine.
        self._is_closed = True

    @staticmethod
    def _close(state: State, logger: Logger):
        """The actual shutdown logic, as a static method, so the underlying engine can still be garbage collected."""
        log.debug('Closing the engine')
        callback_to_has_exception: Dict[Callback, bool] = {}
        for callback in state.callbacks:
            try:
                log.debug('Closing callback %s', type(callback).__name__)
                callback.close(state, logger)
            except Exception as e:
                log.error(
                    f'Error running {callback.__class__.__name__}.close(). Skipping {callback.__class__.__name__}.post_close().',
                    exc_info=e,
                    stack_info=True)
                callback_to_has_exception[callback] = True
            else:
                callback_to_has_exception[callback] = False

        for callback in state.callbacks:
            if callback_to_has_exception[callback] is False:
                try:
                    log.debug('Post-closing callback %s', type(callback).__name__)
                    callback.post_close()
                except Exception as e:
                    log.error(f'Error running {callback.__class__.__name__}.post_close().', exc_info=e, stack_info=True)
                else:
                    _OPEN_CALLBACKS.discard(callback)
