# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A collection of common torchmetrics."""

from composer.metrics.map import MAP
from composer.metrics.metrics import CrossEntropy, Dice, LossMetric, MIoU
from composer.metrics.nlp import (BinaryF1Score, HFCrossEntropy, InContextLearningCodeTracingAveragePassRate,
                                  InContextLearningCodeTracingFullPassRate, InContextLearningLMAccuracy,)
from composer.metrics.nlp import (BinaryF1Score, InContextLearningCodeEvalAccuracy, InContextLearningLMAccuracy,
                                  InContextLearningLMExpectedCalibrationError,
                                  InContextLearningMCExpectedCalibrationError, InContextLearningMetric,
                                  InContextLearningMultipleChoiceAccuracy, InContextLearningQAAccuracy,
                                  LanguageCrossEntropy, LanguagePerplexity, MaskedAccuracy)

__all__ = [
    'MAP',
    'MIoU',
    'Dice',
    'CrossEntropy',
    'LossMetric',
    'BinaryF1Score',
    'LanguageCrossEntropy',
    'MaskedAccuracy',
    'LanguagePerplexity',
    'InContextLearningLMAccuracy',
    'InContextLearningMultipleChoiceAccuracy',
    'InContextLearningQAAccuracy',
    'InContextLearningMCExpectedCalibrationError',
    'InContextLearningLMExpectedCalibrationError',
    'InContextLearningMetric',
    'InContextLearningCodeEvalAccuracy',
    'InContextLearningCodeTracingFullPassRate',
    'InContextLearningCodeTracingAveragePassRate'
]

METRIC_DEFAULT_CTORS = {
    'InContextLearningLMAccuracy': InContextLearningLMAccuracy,
    'InContextLearningMultipleChoiceAccuracy': InContextLearningMultipleChoiceAccuracy,
    'InContextLearningQAAccuracy': InContextLearningQAAccuracy,
    'InContextLearningCodeEvalAccuracy': InContextLearningCodeEvalAccuracy,
}
