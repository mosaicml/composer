# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Optional, cast

import yahp as hp

from composer.algorithms import AlgorithmHparams
from composer.algorithms.stratify_batches.stratify_core import StratifiedBatchSampler
from composer.core import Algorithm, Event, Logger, State


@dataclass
class StratifyBatchesHparams(AlgorithmHparams):
    """See :class:`StratifyBatches`"""
    stratify_how: str = hp.optional(doc="One of {'match', 'balance'}. "
        "'match' attempts to have class distribution in each batch match "
        "the overall class distribution. 'balance' upsamples rare classes such "
        "that the number of samples from each class is as close to equal as "
        "possible within each batch.", default='match')
    targets_attr: Optional[str] = hp.optional(doc='Name of DataLoader attribute '
        'providing class labels.', default=None)

    def initialize_object(self) -> "StratifyBatches":
        return StratifyBatches(**asdict(self))


class StratifyBatches(Algorithm):

    def __init__(self, stratify_how: str = 'match', targets_attr: Optional[str] = None):
        self.stratify_how = stratify_how
        self.targets_attr = targets_attr

    def match(self, event: Event, state: State) -> bool:
        """Apply on Event.AFTER_HPARAMS"""
        return event == Event.AFTER_HPARAMS

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        """Does nothing"""
        pass

    def apply_hparams(self, hparams: Any) -> None:
        # TODO resolve circular import better
        from composer.trainer.trainer_hparams import TrainerHparams
        hparams = cast(TrainerHparams, hparams)
        hparams.dataloader.batch_sampler_factory = partial(
            StratifiedBatchSampler, stratify_how=self.stratify_how, targets_attr=self.targets_attr)
