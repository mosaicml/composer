# Copyright 2021 MosaicML. All Rights Reserved.

from typing import Optional

from composer.core import Algorithm, Event, Logger, State


class Dummy(Algorithm):

    def __init__(self, dummy_argument: str = 'default'):
        self.dummy_argument = dummy_argument

    def match(self, event: Event, state: State) -> bool:
        return True

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        pass
