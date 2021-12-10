# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import yahp as hp
from torch.utils.data import DataLoader

from composer.algorithms import AlgorithmHparams
from composer.algorithms.stratify_batches.stratify_core import StratifiedBatchSampler
from composer.core import Algorithm, Event, Logger, State


def add_stratification(dataloader: DataLoader, stratify_how='match', targets: Optional[Sequence[int]] = None, targets_attr: Optional[str] = None):
    if targets is None:
        dataset = dataloader.dataset
        if targets_attr:
            targets = getattr(dataset, targets_attr)
        # torchvision DatasetFolder subclasses use 'targets'; some torchvision
        # datasets, like caltech101, use 'y' instead
        elif hasattr(dataset, 'targets'):
            targets = dataset.targets
        elif hasattr(dataset, 'y'):
            targets = dataset.y
    else:
        raise AttributeError("Since neither `targets` nor `targets_attr` "
            "were provided, DataLoader.dataset must have an integer vector attribute "
            "named either 'targets' or 'y'.")
    dataloader.batch_sampler = StratifiedBatchSampler(
        targets=targets,
        shuffle=dataloader.shuffle,
        drop_last=dataloader.drop_last)


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
        """Apply on Event.AFTER_DATALOADER"""
        return event == Event.AFTER_DATALOADER

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        add_stratification(state.train_dataloader, stratify_how=self.stratify_how)
