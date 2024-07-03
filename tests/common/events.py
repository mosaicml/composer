# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from composer.core import Callback, Event, State
from composer.loggers import Logger


class EventCounterCallback(Callback):

    def __init__(self) -> None:
        self.event_to_num_calls: dict[Event, int] = {}

        for event in Event:
            self.event_to_num_calls[event] = 0

    def run_event(self, event: Event, state: State, logger: Logger):
        del state, logger  # unused
        self.event_to_num_calls[event] += 1

    def state_dict(self) -> dict[str, Any]:
        return {'events': self.event_to_num_calls}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.event_to_num_calls.update(state['events'])
