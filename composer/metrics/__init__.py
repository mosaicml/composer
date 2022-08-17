# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A collection of common torchmetrics."""

from typing import Dict, Type, Union

import yahp as hp
from torchmetrics import Metric

from composer.metrics.metrics import CrossEntropy, Dice, LossMetric, MIoU
from composer.metrics.metrics_hparams import MetricHparams
from composer.metrics.nlp import (BinaryF1Score, HFCrossEntropy, LanguageCrossEntropy, MaskedAccuracy, Perplexity,
                                  SequenceCrossEntropy, SequenceCrossEntropyHparams)

__all__ = [
    'MIoU',
    'Dice',
    'CrossEntropy',
    'LossMetric',
    'Perplexity',
    'BinaryF1Score',
    'HFCrossEntropy',
    'LanguageCrossEntropy',
    'MaskedAccuracy',
    'SequenceCrossEntropy',
    'MetricHparams',
    'metric_registry',
    'metric_hparams_registry',
]

metric_hparams_registry = {
    'sequence_cross_entropy': SequenceCrossEntropyHparams,
    # 'MIoU',
    # 'Dice',
    # 'CrossEntropy',
    # 'LossMetric',
    # 'Perplexity',
    # 'BinaryF1Score',
    # 'HFCrossEntropy',
    # 'LanguageCrossEntropy',
    # 'MaskedAccuracy',
    # 'SequenceCrossEntropy'
}

metric_registry: Dict[str, Metric] = {
    # 'MIoU',
    # 'Dice',
    # 'CrossEntropy',
    # 'LossMetric',
    # 'Perplexity',
    # 'BinaryF1Score',
    # 'HFCrossEntropy',
    # 'LanguageCrossEntropy',
    # 'MaskedAccuracy',
    # 'SequenceCrossEntropy'
}
