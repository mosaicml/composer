# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import asdict, dataclass
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
                                    "possible within each batch.",
                                    default='match')
    targets_attr: Optional[str] = hp.optional(doc='Name of DataLoader attribute '
                                              'providing class labels.',
                                              default=None)
    lr_multiplier: float = hp.optional(doc="TODO", default=1.0)
    imbalance: float = hp.optional(doc="TODO", default=0.5)

    def initialize_object(self) -> "StratifyBatches":
        return StratifyBatches(**asdict(self))


class StratifyBatches(Algorithm):

    def __init__(self,
                 stratify_how: str = 'match',
                 targets_attr: Optional[str] = None,
                 lr_multiplier: float = 1.0,
                 imbalance: float = 0.5):
        self.stratify_how = stratify_how
        self.targets_attr = targets_attr
        self.lr_multiplier = lr_multiplier
        self.imbalance = imbalance

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

        def make_sampler(*args, split: str, **kwargs):
            if split.lower() == 'train':
                return StratifiedBatchSampler(*args,
                                              stratify_how=self.stratify_how,
                                              targets_attr=self.targets_attr,
                                              imbalance=self.imbalance,
                                              **kwargs)
            return None  # default sampler for other splits

        hparams.dataloader.batch_sampler_factory = make_sampler
        hparams.optimizer.lr *= self.lr_multiplier
