# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A collection of common torchmetrics."""

from composer.metrics.map import MAP
from composer.metrics.metrics import CrossEntropy, Dice, LossMetric, MIoU
from composer.metrics.nlp import (BinaryF1Score, HFCrossEntropy, InContextLearningLMAccuracy,
                                  InContextLearningLMExpectedCalibrationError,
                                  InContextLearningMCExpectedCalibrationError, InContextLearningMetric,
                                  InContextLearningMultipleChoiceAccuracy, InContextLearningQAAccuracy,
                                  LanguageCrossEntropy, LanguagePerplexity, MaskedAccuracy, Perplexity)

__all__ = [
    'MAP',
    'MIoU',
    'Dice',
    'CrossEntropy',
    'LossMetric',
    'Perplexity',
    'BinaryF1Score',
    'HFCrossEntropy',
    'LanguageCrossEntropy',
    'MaskedAccuracy',
    'LanguagePerplexity',
    'InContextLearningLMAccuracy',
    'InContextLearningMultipleChoiceAccuracy',
    'InContextLearningQAAccuracy',
    'InContextLearningMCExpectedCalibrationError',
    'InContextLearningLMExpectedCalibrationError',
    'InContextLearningMetric',
]

METRIC_DEFAULT_CTORS = {
    'InContextLearningLMAccuracy': InContextLearningLMAccuracy,
    'InContextLearningMultipleChoiceAccuracy': InContextLearningMultipleChoiceAccuracy,
    'InContextLearningQAAccuracy': InContextLearningQAAccuracy
}
