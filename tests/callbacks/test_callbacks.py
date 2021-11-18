# Copyright 2021 MosaicML. All Rights Reserved.

import _pytest.monkeypatch
import pytest

from composer.core import Event
from composer.core.callback import Callback
from composer.core.engine import Engine
from composer.core.logging import Logger
from composer.core.state import State


class EventTrackerCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.event = None

    def _run_event(self, event: Event, state: State, logger: Logger) -> None:
        self.event = event


@pytest.mark.parametrize('event', list(Event))
def test_run_event_callbacks(event: Event, dummy_state: State, monkeypatch: _pytest.monkeypatch.MonkeyPatch):
    callback = EventTrackerCallback()
    logger = Logger(dummy_state)
    engine = Engine(state=dummy_state, algorithms=[], logger=logger, callbacks=[callback])

    engine.run_event(event)

    assert callback.event == event
