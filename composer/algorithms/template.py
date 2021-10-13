# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Optional

import yahp as hp

from composer.algorithms import AlgorithmHparams
from composer.core import Algorithm, Event, Logger, State

log = logging.getLogger(__name__)


@dataclass
class MyAlgorithmHparams(AlgorithmHparams):
    """
    This hparams object is for use with our hparams system for specifying algorithms via YAML
    and argument parser flags.
    """

    alpha: float = hp.optional(doc='optional hparams need a default field. This field would'
                               'not be required for CLI flags, but still required for the YAML config.',
                               default=0.1)
    beta: float = hp.required(doc='required fields need a template_default, which '
                              'is used to generate templates. '
                              'This field is still required for CLI flags.',
                              template_default=0.5)

    def initialize_object(self) -> MyAlgorithm:
        """
        **delete this comment before contributing**

        Factory method that links this hparams object with the algorithm class. For algorithms
        that require special imports, can only run the imports during calls to this function.

        e.g.
        ```
        from composer.algorithms.my_algorithm import MyAlgorithm
        ```
        """
        return MyAlgorithm(**asdict(self))


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
        """
        __init__ is constructed from the same fields as in hparams.
        """
        self.hparams = MyAlgorithmHparams(alpha, beta)

    def match(self, event: Event, state: State) -> bool:
        return True

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        """
        Implement your algorithm's state change here.
        """
        state.model = my_algorithm(state.model, self.hparams.alpha, self.hparams.beta)
