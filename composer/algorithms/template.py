# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
from typing import Optional

from composer.core import Algorithm, Event, Logger, State

log = logging.getLogger(__name__)


def my_algorithm(model, alpha: float, beta: float):
    """
    Most of your algorithm logic should be in a functional form here, similar to
    torch.nn.Conv2d vs torch.nn.functional.conv2d
    """
    return model


class MyAlgorithm(Algorithm):
    """
    Add docstrings in Google style. See:

    https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
    """

    def __init__(self, alpha: float, beta: float = 0.5):
        self.alpha = alpha
        self.beta = beta

    def match(self, event: Event, state: State) -> bool:
        return True

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        """
        Implement your algorithm's state change here.
        """
        state.model = my_algorithm(state.model, self.alpha, self.beta)
