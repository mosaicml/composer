# Copyright 2022 MosaicML. All Rights Reserved.

import os
from typing import Callable, List

import pytest

import composer.callbacks
import composer.loggers
import composer.profiler
from composer.callbacks.mlperf import MLPerfCallback
from composer.core import Event
from composer.core.callback import Callback
from composer.core.engine import Engine
from composer.core.state import State
from composer.loggers import Logger, ObjectStoreLogger
from composer.profiler.profiler import Profiler
from composer.profiler.profiler_action import ProfilerAction


def test_callbacks_map_to_events():
    # callback methods must be 1:1 mapping with events
    # exception for private methods
    cb = Callback()
    excluded_methods = ["state_dict", "load_state_dict", "run_event", "close", "post_close"]
    methods = set(m for m in dir(cb) if (m not in excluded_methods and not m.startswith("_")))
    event_names = set(e.value for e in Event)
    assert methods == event_names


class EventTrackerCallback(Callback):

    def __init__(self) -> None:
        self.event = None

    def run_event(self, event: Event, state: State, logger: Logger) -> None:
        del state, logger  # unused
        self.event = event


@pytest.mark.parametrize('event', list(Event))
def test_run_event_callbacks(event: Event, dummy_state: State):
    callback = EventTrackerCallback()
    logger = Logger(dummy_state)
    dummy_state.callbacks = [callback]
    engine = Engine(state=dummy_state, logger=logger)

    engine.run_event(event)

    assert callback.event == event


def _get_callback_factories() -> List[Callable[..., Callback]]:
    callback_factories: List[Callable[..., Callback]] = [
        x for x in vars(composer.callbacks).values() if isinstance(x, type) and issubclass(x, Callback)
    ]
    callback_factories.extend(
        x for x in vars(composer.loggers).values() if isinstance(x, type) and issubclass(x, Callback))
    callback_factories.extend(
        x for x in vars(composer.profiler).values() if isinstance(x, type) and issubclass(x, Callback))
    callback_factories.remove(ObjectStoreLogger)
    callback_factories.remove(MLPerfCallback)
    callback_factories.append(lambda: ObjectStoreLogger(
        use_procs=False,
        num_concurrent_uploads=1,
        provider='local',
        container='.',
        provider_kwargs={
            'key': os.path.abspath("."),
        },
    ))
    return callback_factories


@pytest.mark.parametrize('callback_factory', _get_callback_factories())
class TestCallbacks:

    @classmethod
    def setup_class(cls):
        pytest.importorskip("wandb", reason="WandB is optional.")

    def test_multiple_fit_start_and_end(self, callback_factory: Callable[[], Callback], dummy_state: State):
        """Test that callbacks do not crash when Event.FIT_START and Event.FIT_END is called multiple times."""
        dummy_state.callbacks.append(callback_factory())
        dummy_state.profiler = Profiler(schedule=lambda _: ProfilerAction.SKIP, trace_handlers=[])
        dummy_state.profiler.bind_to_state(dummy_state)

        logger = Logger(dummy_state)
        engine = Engine(state=dummy_state, logger=logger)

        engine.run_event(Event.INIT)  # always runs just once per engine

        engine.run_event(Event.FIT_START)
        engine.run_event(Event.FIT_END)

        engine.run_event(Event.FIT_START)
        engine.run_event(Event.FIT_END)

    def test_idempotent_close(self, callback_factory: Callable[[], Callback], dummy_state: State):
        """Test that callbacks do not crash when .close() and .post_close() are called multiple times."""
        dummy_state.callbacks.append(callback_factory())
        dummy_state.profiler = Profiler(schedule=lambda _: ProfilerAction.SKIP, trace_handlers=[])
        dummy_state.profiler.bind_to_state(dummy_state)

        logger = Logger(dummy_state)
        engine = Engine(state=dummy_state, logger=logger)

        engine.run_event(Event.INIT)  # always runs just once per engine
        engine.close()
        engine.close()

    def test_multiple_init_and_close(self, callback_factory: Callable[[], Callback], dummy_state: State):
        """Test that callbacks do not crash when INIT/.close()/.post_close() are called multiple times in that order."""
        dummy_state.callbacks.append(callback_factory())
        dummy_state.profiler = Profiler(schedule=lambda _: ProfilerAction.SKIP, trace_handlers=[])
        dummy_state.profiler.bind_to_state(dummy_state)

        logger = Logger(dummy_state)
        engine = Engine(state=dummy_state, logger=logger)

        engine.run_event(Event.INIT)
        engine.close()
        # For good measure, also test idempotent close, in case if there are edge cases with a second call to INIT
        engine.close()

        # Create a new engine, since the engine does allow events to run after it has been closed
        engine = Engine(state=dummy_state, logger=logger)
        engine.close()
        # For good measure, also test idempotent close, in case if there are edge cases with a second call to INIT
        engine.close()
