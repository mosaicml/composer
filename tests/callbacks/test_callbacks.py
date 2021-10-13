# Copyright 2021 MosaicML. All Rights Reserved.

from unittest.mock import Mock

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
    excluded_methods = ["state_dict", "load_state_dict", "setup"]
    methods = set(m for m in dir(cb) if (m not in excluded_methods and not m.startswith("_")))
    event_names = set(e.value for e in Event)
    assert methods == event_names


@pytest.mark.parametrize('event', list(Event))
def test_run_event_callbacks(event: Event, dummy_state: State):
    callbacks = [Mock() for _ in range(5)]
    logger = Logger(dummy_state)
    engine = Engine(state=dummy_state, algorithms=[], logger=logger, callbacks=callbacks)

    engine.run_event(event)

    for cb in callbacks:
        f = getattr(cb, event.value)
        f.assert_called_once_with(dummy_state, logger)
