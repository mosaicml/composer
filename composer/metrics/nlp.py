# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A collection of common torchmetrics for NLP tasks."""

import logging
from typing import Mapping, Union

import torch
from torch import Tensor
from torchmetrics import Metric

log = logging.getLogger(__name__)

__all__ = [
    'BinaryF1Score',
    'LanguageCrossEntropy',
    'MaskedAccuracy',
    'LanguagePerplexity',
]


class MaskedAccuracy(Metric):
    """Computes accuracy with support for masked indices.

    Adds metric state variables:
        correct (float): The number of instances where the prediction masked the target.
        total (float): The number of total instances that were predicted.

    Args:
        ignore_index (int): The class index to ignore. Default: -100.
        dist_sync_on_step (bool, optional): Synchronize metric state across processes at
            each forward() before returning the value at the step. Default: ``False``.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, ignore_index: int = -100, dist_sync_on_step: bool = False):
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.ignore_index = ignore_index

        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # predictions is a batch x num_classes tensor, take the argmax to get class indices
        preds = torch.argmax(preds, dim=-1)
        assert preds.shape == target.shape

        # mask out the padded indices
        mask = (target != self.ignore_index)
        masked_target = target[mask]
        masked_preds = preds[mask]

        self.correct += torch.sum(masked_preds == masked_target)
        self.total += mask.sum()

    def compute(self):
        assert isinstance(self.correct, Tensor)
        assert isinstance(self.total, Tensor)
        return self.correct.float() / self.total


class LanguageCrossEntropy(Metric):
    """Torchmetric that computes cross entropy on language modeling outputs.

    Adds metric state variables:
        sum_loss (float): The sum of the per-example loss in the batch.
        total_items (float): The number of batches to average across.

    Args:
        dist_sync_on_step (bool, optional): Synchronize metric state across processes at
            each forward() before returning the value at the step. Default: ``False``.
        ignore_index (int, optional): The class index to ignore. Default: ``-100``.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, dist_sync_on_step: bool = False, ignore_index: int = -100):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.ignore_index = ignore_index
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='sum')
        self.add_state('sum_loss', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('total_items', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, output: Union[Mapping, Tensor], target: Tensor) -> None:
        """Updates the internal state with results from a new batch.

        Args:
            output (Mapping): The output from the model, which must contain
                either the Tensor or a Mapping type that contains the loss or model logits.
            target (~torch.Tensor): A Tensor of ground-truth values to compare against.
        """
        if isinstance(output, Mapping):
            logits = output['logits']
        elif isinstance(output, Tensor):
            logits = output
        else:
            raise Exception(f'Type {type(output)} for the output is unsupported.')

        target = target.view(-1)
        logits = logits.view(target.shape[0], -1)
        losses = self.loss_fn(logits, target)

        total_items = (target != self.ignore_index).sum()
        self.total_items += total_items  #type: ignore (third-party)

        # accumulate loss over all batches
        self.sum_loss += losses

    def compute(self) -> Tensor:
        """Aggregate the state over all processes to compute the metric.

        Returns:
            loss: The loss averaged across all batches as a :class:`~torch.Tensor`.
        """
        # Return average loss over entire dataset
        return self.sum_loss / self.total_items  #type: ignore (third-party)


class BinaryF1Score(Metric):
    """Implements F1 Scores for binary classification tasks via sklearn.

    Adds metric state variables:
        true_positive (float): A counter of how many items were correctly classified as positives.
        false_positive (float): A counter of how many items were incorrectly classified as positives.
        false_negative (float): A counter of how many items were incorrectly classified as negatives.

    Args:
        dist_sync_on_step (bool, optional): Synchronize metric state across processes at
            each forward() before returning the value at the step. Default: ``False``.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state('true_positive', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('false_positive', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('false_negative', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, output: Tensor, target: Tensor) -> None:
        """Updates the internal state with results from a new batch.

        Args:
            output (Mapping): The output from the model, which must contain
                either the Tensor or a Mapping type that contains the loss or model logits.
            target (~torch.Tensor): A Tensor of ground-truth values to compare against.
        """
        predictions = torch.argmax(output, dim=1)
        self.true_positive += predictions[(target == 1)].sum()
        self.false_positive += (predictions[(target == 1)] == 0).sum()
        self.false_negative += (predictions[(target == 0)] == 1).sum()

    def compute(self) -> Tensor:
        """Aggregate the state over all processes to compute the metric.

        Returns:
            loss: The loss averaged across all batches as a :class:`~torch.Tensor`.
        """
        assert isinstance(self.true_positive, Tensor)
        assert isinstance(self.false_positive, Tensor)
        assert isinstance(self.false_negative, Tensor)
        f1 = (self.true_positive) / (self.true_positive + (0.5 * (self.false_negative + self.false_positive)))
        return f1


class LanguagePerplexity(LanguageCrossEntropy):
    """Subclasses :class:`~composer.metrics.nlp.LanguageCrossEntropy` to implement perplexity."""

    def compute(self) -> Tensor:
        """Returns torch.exp() of the LanguageCrossEntropy."""
        avg_loss = super().compute()
        return torch.exp(avg_loss)


# For backward compatibility
class InContextLearningMetric:
    """InContextLearningMetric only exists for backwards compatibility of checkpoints that contain pickled metrics."""

    def __init__(self):
        raise RuntimeError(
            f'This class only exists for maintaining backward compatibility for checkpoints that contain pickled metrics. Please instead use https://github.com/mosaicml/llm-foundry/blob/main/scripts/eval/README.md.',
        )

    def __getstate__(self):
        return None

    def __setstate__(self, state):
        pass


InContextLearningCodeEvalAccuracy = InContextLearningMetric
InContextLearningLMAccuracy = InContextLearningMetric
InContextLearningLMExpectedCalibrationError = InContextLearningMetric
InContextLearningMCExpectedCalibrationError = InContextLearningMetric
InContextLearningQAAccuracy = InContextLearningMetric
InContextLearningMultipleChoiceAccuracy = InContextLearningMetric
