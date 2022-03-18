# Copyright 2021 MosaicML. All Rights Reserved.

import pytest

from composer.core import Event
from composer.core.callback import Callback
from composer.core.engine import Engine
from composer.core.state import State
from composer.loggers import Logger


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
