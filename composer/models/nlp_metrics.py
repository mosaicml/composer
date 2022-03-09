# Copyright 2021 MosaicML. All Rights Reserved.

"""A collection of common torchmetrics for NLP tasks."""
from typing import Mapping, Union

import torch
from torch import Tensor
from torchmetrics import Metric

from composer.models.loss import soft_cross_entropy

__all__ = ["Perplexity", "BinaryF1Score", "LanguageCrossEntropyLoss", "CrossEntropyLoss", "MaskedAccuracy"]


class MaskedAccuracy(Metric):
    """Computes accuracy with support for masked indicies.

    Adds metric state variables:
        correct (float): The number of instances where the prediction masked the target.
        total (float): The number of total instances that were predicted.

    Args:
        ignore_index (int): The class index to ignore.
        dist_sync_on_step (bool, optional): Synchronize metric state across processes at
            each forward() before returning the value at the step. Default: ``False``.
    """

    def __init__(self, ignore_index: int, dist_sync_on_step=False):
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.ignore_index = ignore_index

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # predictions is a batch x num_classes tensor, take the argmax to get class indicies
        preds = torch.argmax(preds, dim=-1)
        assert preds.shape == target.shape

        # mask out the padded indicies
        mask = (target != self.ignore_index)
        masked_target = target[mask]
        masked_preds = preds[mask]

        self.correct += torch.sum(masked_preds == masked_target)
        self.total += mask.sum()

    def compute(self):
        assert isinstance(self.correct, Tensor)
        assert isinstance(self.total, Tensor)
        return self.correct.float() / self.total


class CrossEntropyLoss(Metric):
    """Computes cross entropy loss.

    Adds metric state variables:
        sum_loss (float): The sum of the per-example loss in the batch.
        total_items (float): The number of batches to average across.

    Args:
        vocab_size (int): The size of the tokenizer vocabulary.
        dist_sync_on_step (bool, optional): Synchronize metric state across processes at
            each forward() before returning the value at the step. Default: ``False``.
        ignore_index (int, optional): The class index to ignore. Default: ``-100``.
    """

    def __init__(self, vocab_size: int, dist_sync_on_step=False, ignore_index: int = -100):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="sum")
        self.add_state("sum_loss", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total_items", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, output: Union[Mapping, Tensor], target: Tensor) -> None:
        """Updates the internal state with results from a new batch.

        Args:
            output (Mapping): The output from the model, which must contain
                either the Tensor or a Mapping type that contains the loss or model logits.
            target (~torch.Tensor): A Tensor of ground-truth values to compare against.
        """

        assert isinstance(output, Tensor)
        output = output.view(-1, self.vocab_size)
        target = target.view(-1)
        losses = self.loss_fn(output, target)

        total_items = (target != self.ignore_index).sum()
        self.total_items += total_items  #type: ignore (third-party)

        # accmulate loss over all batches
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

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("true_positive", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("false_positive", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_negative", default=torch.tensor(0), dist_reduce_fx="sum")

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


class LanguageCrossEntropyLoss(Metric):
    """Hugging Face compatible cross entropy loss.

    Adds metric state variables:
        sum_loss (float): The sum of the per-example loss in the batch.
        total_batches (float): The number of batches to average across.

    Args:
        dist_sync_on_step (bool, optional): Synchronize metric state across processes at
            each forward() before returning the value at the step. Default: ``False``
    """

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("sum_loss", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total_batches", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, output: Union[Mapping, Tensor], target: Tensor) -> None:
        """Updates the internal state with results from a new batch.

        Args:
            output (Mapping): The output from the model, which must contain
                either the Tensor or a Mapping type that contains the loss or model logits.
            target (~torch.Tensor): A Tensor of ground-truth values to compare against.
        """

        # if logit modification algorithms aren't on, we take the loss directly from the model output
        if isinstance(output, Mapping) and 'loss' in output:
            loss = output['loss']
        else:
            if isinstance(output, Mapping):
                logits = output['logits']
            # recompute the loss on our own
            elif isinstance(output, Tensor):
                logits = output
            else:
                raise Exception(f"Type {type(output)} for the output is unsupported.")

            loss = soft_cross_entropy(logits, target)

        # accmulate loss over all batches
        self.sum_loss += loss

        self.total_batches += 1  #type: ignore (third-party)

    def compute(self) -> Tensor:
        """Aggregate the state over all processes to compute the metric.

        Returns:
            loss: The loss averaged across all batches as a :class:`~torch.Tensor`.
        """
        # Return average loss over entire dataset
        return self.sum_loss / self.total_batches  #type: ignore (third-party)


class Perplexity(LanguageCrossEntropyLoss):
    """Subclasses :class:`~composer.models.nlp_metrics.LanguageCrossEntropyLoss` to implement perplexity.

    If an algorithm modifies the loss function and it is no longer directly provided in the output, then this could be
    expensive because it'll compute the loss twice.
    """

    def compute(self) -> Tensor:
        """Returns torch.exp() of the LanguageCrossEntropyLoss."""
        avg_loss = super().compute()
        return torch.exp(avg_loss)
