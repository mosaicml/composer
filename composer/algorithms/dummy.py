# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import dataclass
from typing import Optional

import yahp as hp

from composer.algorithms import AlgorithmHparams
from composer.core import Algorithm, Event, Logger, State


@dataclass
class DummyHparams(AlgorithmHparams):

    dummy_argument: str = hp.optional(doc="Placeholder dummy argument", default="default")

    def initialize_object(self):
        return Dummy(dummy_argument=self.dummy_argument)


class Dummy(Algorithm):

    def __init__(self, dummy_argument: str = 'default'):
        self.hparams = DummyHparams(dummy_argument)

    def match(self, event: Event, state: State) -> bool:
        return True

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        pass
