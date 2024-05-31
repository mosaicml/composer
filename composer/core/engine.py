# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Engine is a coordinator for running algorithms and resolving ordering conflicts among them for composition.

.. currentmodule:: composer

The order in which algorithms are run matters significantly during composition. For example, the
:class:`.SelectiveBackprop` algorithm runs on the :attr:`.Event.AFTER_DATALOADER` event and must run before
any data augmentations. :class:`.Engine` runs re-ordering passes to resolve such ordering issues or conflicts.

These orderings are enforced by algorithm passes. The default passes registered to the Engine are found in
:mod:`composer.core.passes`. To register a new pass, use :meth:`.Engine.register_pass`, e.g.


.. testsetup::

    # dummy algorithm
    MyAlgorithm = None

.. doctest::

    from composer import Engine, Algorithm, Event
    from typing import Sequence

    def run_last(algorithms: Sequence[Algorithm], event: Event) -> Sequence[Algorithm]:
        algorithms = sorted(algorithms, key=lambda x: isinstance(x, MyAlgorithm))

    engine = Engine(algorithm_passes=run_last)

.. note::

    * An instance of :class:`.Engine` is automatically constructed by the :class:`.Trainer`
      constructor. A user need not instantiate the :class:`.Engine` class. Instead, they should
      specify algorithm_passes to the :class:`.Trainer` constructor, which will be passed to the
      :class:`.Engine` constructor.

.. note::
    * The design of :class:`.Engine` is subject to change in future releases
      to accommodate more complexity as we investigate composition of algorithms.

To generate verbose debug logs for the engine, set the environment variable ``ENGINE_DEBUG=1``.

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
import os
import signal
import sys
import textwrap
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, ContextManager, Optional, Sequence, TypeVar, Union, cast

from composer.core import passes
from composer.core.algorithm import Algorithm
from composer.core.callback import Callback
from composer.core.event import Event
from composer.core.state import State
from composer.loggers import Logger, LoggerDestination
from composer.profiler import ProfilerAction
from composer.utils import ensure_tuple

log = logging.getLogger(__name__)

__all__ = ['Trace', 'Engine', 'Traces']

T = TypeVar('T')

_ALWAYS_RECORD_EVENTS = [Event.INIT, Event.FIT_START, Event.EPOCH_START, Event.EPOCH_END]

#: The default traces of an entire run is an OrderedDict.
#: The keys are of format ``<algorithm_name>/<event>`` (e.g.,  ``Blurpool/INIT``) and values are an instance of
#: :class:`Trace`.
Traces = dict[str, 'Trace']

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


# Catch SIGTERM/SIGINT and instead exit via `sys.exit` using same error code, ensuring atexit
# functions still run. Composer CLI launcher will give a 30 second grace period before sending
# SIGKILL.
def sigterm_handler(signal, frame):
    sys.exit(128 + signal)


try:
    signal.signal(signal.SIGTERM, sigterm_handler)
    signal.signal(signal.SIGINT, sigterm_handler)
except ValueError:
    log.warning('Failed to set signal handler. Checkpoints may not be flushed if the process is killed.')


def _get_default_passes():
    return [
        passes.sort_selective_backprop_first,
        passes.sort_low_precision_layernorm_last,
        passes.set_filo_order,
        passes.warn_if_multiple_loss_interpolation,
    ]


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
        algorithm_passes ([AlgorithmPass | tuple[AlgorithmPass, int] | Sequence[AlgorithmPass | tuple[AlgorithmPass, int]], optional):
            Optional list of passes to change order in which algorithms are applied. These passes are merged with the
            default passes specified in :class:`.Engine`. If ``None``, then no additional passes will be used.
    """

    def __init__(
        self,
        state: State,
        logger: Logger,
        algorithm_passes: Optional[Union[passes.AlgorithmPass,
                                         tuple[passes.AlgorithmPass, int],
                                         Sequence[Union[passes.AlgorithmPass, tuple[passes.AlgorithmPass, int]]],
                                        ]] = None,
    ):
        self.logger = logger
        self.state = state
        self._is_closed = False

        self.algorithm_passes: list[passes.AlgorithmPass] = _get_default_passes()
        if algorithm_passes is not None:
            # Wrap in list if not already a list or if it's a length 2 list specifying a single
            # call to register_pass with type [AlgorithmPass, int]
            if not isinstance(
                algorithm_passes,
                list,
            ) or (len(algorithm_passes) == 2 and isinstance(algorithm_passes[1], int)):
                algorithm_passes = [algorithm_passes]  # type: ignore wrapping list
            algo_passes = algorithm_passes if isinstance(algorithm_passes, list) else [algorithm_passes]
            for algo_pass in algo_passes:
                algo_pass = ensure_tuple(algo_pass)
                if len(algo_pass) == 1 and isinstance(algo_pass[0], Callable):
                    self.register_pass(algo_pass[0])
                elif len(algo_pass) == 2 and isinstance(algo_pass[0], Callable) and isinstance(algo_pass[1], int):
                    self.register_pass(algo_pass[0], algo_pass[1])
                else:
                    raise ValueError(
                        textwrap.dedent(
                            'Received invalid algorithm_pass. Expected either a single AlgorithmPass '
                            f'or a tuple of (AlgorithmPass, int), but received {algo_pass}.',
                        ),
                    )

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
            raise RuntimeError((
                'The engine was already closed and therefore cannot be used again. '
                'To fix, please create a new Engine (or Trainer)'
            ))

        if self.state.profiler is not None:
            name = f'event/{event.canonical_name}'
            if (event.is_before_event or event.is_after_event):
                # if not part of an event pair (e.g. init), then don't record an event here
                if event in _ALWAYS_RECORD_EVENTS:
                    actions = [ProfilerAction.ACTIVE, ProfilerAction.WARMUP, ProfilerAction.SKIP]
                else:
                    actions = [ProfilerAction.ACTIVE, ProfilerAction.WARMUP]
                duration_marker = self.state.profiler.marker(name, actions=actions)

        if event.is_after_event and duration_marker is not None:
            duration_marker.finish()

        self._assert_dataloader_and_duration_set(self.state, event)

        if event == Event.INIT:
            # For the INIT event, run the callbacks first to initialize the loggers
            # For other events, run the algorithms first, so the callbacks have the state
            # after algorithms modify it
            self._check_for_still_open_callbacks()

            # Run loggers first, so they can be initialized before any callbacks that may
            # use them.
            self._run_loggers(event)
            self._run_nonlogger_callbacks(event)
            traces = self._run_algorithms(event)
        else:
            traces = self._run_algorithms(event)
            # Run callbacks first, so any log calls from a callback that are executed lazily
            # get registered before they are flushed by the logger itself.
            self._run_nonlogger_callbacks(event)
            self._run_loggers(event)

        if event.is_before_event and duration_marker is not None:
            duration_marker.start()

        return traces

    def run_marker_only_event(
        self,
        event: Union[Event, str],
    ) -> None:
        """Runs the marker for an event if the profiler is enabled.

        This is primarily used to complete the dataloader marker at the end of the dataloader. In
        this scenario, the dataloader marker has started from Event.BEFORE_DATALOADER, but
        Event.AFTER_DATALOADER cannot be called as no batch was yielded from the dataloader.

        Args:
            event (Event | str): The current :class:`.Event`. It can be the enum member values or a
                string with the event value.
        """
        duration_marker = None
        event = Event(event)

        if self._is_closed:
            raise RuntimeError((
                'The engine was already closed and therefore cannot be used again. '
                'To fix, please create a new Engine (or Trainer)'
            ))

        if self.state.profiler is not None:
            name = f'event/{event.canonical_name}'
            if (event.is_before_event or event.is_after_event):
                # if not part of an event pair (e.g. init), then don't record an event here
                if event in _ALWAYS_RECORD_EVENTS:
                    actions = [ProfilerAction.ACTIVE, ProfilerAction.WARMUP, ProfilerAction.SKIP]
                else:
                    actions = [ProfilerAction.ACTIVE, ProfilerAction.WARMUP]
                duration_marker = self.state.profiler.marker(name, actions=actions)

        if event.is_after_event and duration_marker is not None:
            duration_marker.finish()
        if event.is_before_event and duration_marker is not None:
            duration_marker.start()

    def register_pass(self, algorithm_pass: passes.AlgorithmPass, index: int = -1):
        """Registers an algorithm pass with the Engine.

        Args:
            algorithm_pass (passes.AlgorithmPass): A method that maps a list of
                algorithms to a list of algorithms.
            index (int, optional): The index to insert into the list of passes.
                If -1 (default), the pass will be insert to the end of the list.
        """
        if index == -1:
            index = len(self.algorithm_passes)

        self.algorithm_passes.insert(index, algorithm_pass)

    @staticmethod
    def _assert_dataloader_and_duration_set(state: State, event: Event):
        # correctness checks that dataloader and max duration need to be set for certain events

        # dataloader should be set on all events except INIT/BEFORE_LOAD/AFTER_LOAD/EVAL_STANDALONE_START/EVAL_STANDALONE_END
        if event not in {
            Event.INIT,
            Event.BEFORE_LOAD,
            Event.AFTER_LOAD,
            Event.EVAL_STANDALONE_START,
            Event.EVAL_STANDALONE_END,
        }:
            assert state.dataloader is not None, f'The trainer should have set state.dataloader for event {event}.'

        if event != Event.INIT and event != Event.BEFORE_LOAD and event != Event.AFTER_LOAD and not event.is_predict and not event.is_eval:
            assert state.max_duration is not None, f'The trainer should have set state.max_duration for event {event}.'

    def _run_algorithms(
        self,
        event: Event,
    ) -> Traces:
        algorithms_to_run = [algo for algo in self.state.algorithms if algo.match(event, self.state)]

        # apply algorithm passes
        algorithms_to_run = self._compile(algorithms_to_run, event)

        trace = _setup_trace(algorithms_to_run, event)
        for order, algorithm in enumerate(algorithms_to_run):
            marker = None
            if self.state.profiler is not None:
                marker = self.state.profiler.marker(
                    f'algorithm/{algorithm.__class__.__name__}/event/{event.value}',
                    categories=[
                        event.value,
                        algorithm.__class__.__name__,
                    ],
                )
            ctx = cast(ContextManager, contextlib.nullcontext()) if marker is None else marker
            with ctx:
                self._debug_log(event, f'Running algorithm {type(algorithm).__name__}')
                exit_code = algorithm.apply(event, self.state, self.logger)

            trace_key = f'{algorithm}/{event}'
            trace[trace_key] = Trace(
                name=algorithm.__class__.__name__,
                event=event,
                exit_code=exit_code,
                order=order,
                run=True,
            )

        if len(trace) > 0:
            self.logger.log_traces({
                f'algorithm_traces/{tr.name}/{tr.event}': 1 if tr.run else 0 for _, tr in trace.items()
            })

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
        # run reordering passes on the algorithms
        for passes in self.algorithm_passes:
            algorithms_to_run = passes(algorithms_to_run, event)

        return algorithms_to_run

    def _check_for_still_open_callbacks(self):
        # Some callbacks may be open from a previous training run
        # If so, error and instruct the user that they must call `trainer.close()`
        # so callbacks can clean up and reset their state properly
        for cb in self.state.callbacks:
            # If it's not in the set, then the callback is new, so it's closed by definition
            if cb in _OPEN_CALLBACKS:
                raise RuntimeError((
                    'Cannot create a new trainer with an open callback or logger from a previous trainer. '
                    'To fix, call trainer.close() before creating this new trainer to ensure that all '
                    'callbacks or loggers shut down properly.'
                ))
            _OPEN_CALLBACKS.add(cb)

    def _run_callbacks(
        self,
        event: Union[Event, str],
        callbacks: Optional[Sequence[Callback]] = None,
    ):
        """Runs a sequence of callbacks by calling the function for an event.

        Args:
            event (Event | str): The current :class:`.Event`.
            callbacks (Callback | Sequence[Callback], optional): The callbacks to run.
                If None is specified, will use all the callback in state (self.state).

        """
        event = Event(event)

        callbacks = self.state.callbacks if callbacks is None else callbacks

        for cb in callbacks:
            marker = None
            if self.state.profiler is not None:
                marker = self.state.profiler.marker(
                    f'callback/{cb.__class__.__name__}/event/{event.value}',
                    categories=[
                        event.value,
                        cb.__class__.__name__,
                    ],
                )
            ctx = cast(ContextManager, contextlib.nullcontext()) if marker is None else marker
            with ctx:
                self._debug_log(event, f'Running callback {type(cb).__name__}')
                cb.run_event(event, self.state, self.logger)

    def _run_loggers(self, event: Union[Event, str]):
        loggers = [callback for callback in self.state.callbacks if isinstance(callback, LoggerDestination)]
        self._run_callbacks(event, loggers)

    def _run_nonlogger_callbacks(self, event: Union[Event, str]):
        callbacks = [callback for callback in self.state.callbacks if not isinstance(callback, LoggerDestination)]
        self._run_callbacks(event, callbacks)

    def __del__(self):
        global _did_atexit_run
        if _did_atexit_run or self._is_closed:
            # Do not attempt to shutdown again, since close() already ran via __atexit__ or was already invoked
            return
        self.close()
        atexit.unregister(_set_atexit_ran)
        atexit.unregister(self._close)

    def _debug_log(self, event: Event, msg: str):
        """Helper to include timestamp and event info in log messages."""
        timestamp = f'[ep={int(self.state.timestamp.epoch)}][ba={int(self.state.timestamp.batch)}]'

        # for eval or pr
        if event.is_eval:
            timestamp += f'[eval_ba={int(self.state.eval_timestamp.batch)}]'
        if event.is_predict:
            timestamp += f'[predict_ba={int(self.state.predict_timestamp.batch)}]'

        timestamp += f'[event={event.name}]'

        if os.environ.get('ENGINE_DEBUG', None):
            log.debug(f'{timestamp}: {msg}')

    def close(self) -> None:
        """Shutdown the engine.

        As part of the shutdown procedure, :meth:`.Callback.close` and :meth:`.Callback.post_close` are invoked
        for each callback. Note that :meth:`.Callback.post_close` is invoked only for callbacks that did not raise
        an exception during :meth:`.Callback.close`.

        This method does not re-raise any exceptions from :meth:`.Callback.close` and :meth:`.Callback.post_close`.
        Instead, these exceptions are logged as errors.
        """
        if not self._is_closed:
            self._close(self.state, self.logger)
        # The self._is_closed flag would not be set if `_close` is called via atexit
        # However, in these cases, the engine would never be used again, as Python is shutting
        # down. It is only required to set the flag if the user manually calls `close()` and still holds
        # a reference to the engine.
        self._is_closed = True

    @staticmethod
    def _close(state: State, logger: Logger):
        """The actual shutdown logic, as a static method, so the underlying engine can still be garbage collected."""
        log.debug('Closing the engine.')
        callback_to_has_exception: dict[Callback, bool] = {}
        for callback in state.callbacks:
            try:
                log.debug('Closing callback %s', type(callback).__name__)
                callback.close(state, logger)
            except Exception as e:
                log.error(
                    f'Error running {callback.__class__.__name__}.close(). Skipping {callback.__class__.__name__}.post_close().',
                    exc_info=e,
                    stack_info=True,
                )
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

        # Try to shut down any persistent workers
        try:
            state.train_dataloader._iterator._shutdown_workers()  # type: ignore [reportGeneralTypeIssues]
        except:
            pass

        log.debug('Engine closed.')
