# Copyright 2021 MosaicML. All Rights Reserved.

import _pytest.monkeypatch
import pytest

from composer.core import Event
from composer.core.callback import Callback
from composer.core.engine import Engine
from composer.core.logging import Logger
from composer.core.state import State


def test_callbacks_map_to_events():
    # callback methods must be 1:1 mapping with events
    # exception for private methods
    cb = Callback()
    excluded_methods = ["state_dict", "load_state_dict", "setup", "run_event"]
    methods = set(m for m in dir(cb) if (m not in excluded_methods and not m.startswith("_")))
    event_names = set(e.value for e in Event)
    assert methods == event_names


@pytest.mark.parametrize('event', list(Event))
def test_run_event_callbacks(event: Event, dummy_state: State, monkeypatch: _pytest.monkeypatch.MonkeyPatch):
    callback = Callback()
    logger = Logger(dummy_state)
    engine = Engine(state=dummy_state, algorithms=[], logger=logger, callbacks=[callback])

    called = [False]  # storing as an array so it is a pointer, not primitive

    def patched_callback(state: State, logger: Logger):

        called[0] = True

    monkeypatch.setattr(callback, event.value, patched_callback)

    engine.run_event(event)

    assert called[0], "callback method not invoked"
