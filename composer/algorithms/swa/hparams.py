# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Optional

import yahp as hp

from composer.algorithms.algorithm_hparams import AlgorithmHparams

log = logging.getLogger(__name__)


@dataclass
class SWAHparams(AlgorithmHparams):

    swa_start: float = hp.optional(
        doc='Percentage of epochs before starting to apply SWA. Default 0.8.',
        default=0.8,
    )
    anneal_epochs: int = hp.optional(
        doc='Number of annealing epochs. Default 10.',
        default=10,
    )
    swa_lr: Optional[float] = hp.optional(
        doc='The final learning rate to anneal towards with this scheduler. '
        'Set to None for no annealing. Default: None',
        default=None,
    )

    def initialize_object(self):
        from composer.algorithms.swa import SWA
        return SWA(**asdict(self))
