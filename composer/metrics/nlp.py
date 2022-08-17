# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A collection of common torchmetrics for NLP tasks."""
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import yahp as hp
from torch import ListType, Tensor
from torchmetrics import Metric

from composer.core import State
from composer.loss import soft_cross_entropy
from composer.metrics import MetricHparams

__all__ = [
    'Perplexity', 'BinaryF1Score', 'HFCrossEntropy', 'LanguageCrossEntropy', 'MaskedAccuracy', 'SequenceCrossEntropy'
]


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

    def __init__(self, ignore_index: int, dist_sync_on_step: bool = False):
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.ignore_index = ignore_index

        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: torch.Tensor, target: torch.Tensor, **kwargs):
        # predictions is a batch x num_classes tensor, take the argmax to get class
        # indicies
        del kwargs
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


class LanguageCrossEntropy(Metric):
    """Torchmetric that computes cross entropy on language modeling outputs.

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
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='sum')
        self.add_state('sum_loss', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('total_items', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, output: Union[Mapping, Tensor], target: Tensor, **kwargs) -> None:
        """Updates the internal state with results from a new batch.

        Args:
            output (Mapping): The output from the model, which must contain
                either the Tensor or a Mapping type that contains the loss or model logits.
            target (~torch.Tensor): A Tensor of ground-truth values to compare against.
        """
        del kwargs
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


class SequenceCrossEntropy(Metric):
    """Torchmetric that computes cross entropy on language modeling outputs.

    Adds metric state variables:
        seq_loss (tensor): The loss for each sequence in a batch.
        uid (list): UID for each sequence

    Args:
        vocab_size (int): The size of the tokenizer vocabulary.
        dist_sync_on_step (bool, optional): Synchronize metric state across processes at
            each forward() before returning the value at the step. Default: ``False``.
        ignore_index (int, optional): The class index to ignore. Default: ``-100``.
        reduction (string, optional): Specifies the reduction to apply to token-wise loss
            in each sequence: ``'mean'`` | ``'sum'``. ``'mean'``: the mean across
            non-masked tokens in the sequence is taken. ``'sum'``: loss will be summed
            across tokens. Default: ``'sum'``.
    """

    def __init__(self,
                 vocab_size: int,
                 dist_sync_on_step: bool = False,
                 ignore_index: int = -100,
                 reduction: str = 'sum',
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
        self.reduction = reduction
        self.add_state('seq_loss', default=[], dist_reduce_fx=None)
        self.add_state('uid', default=[], dist_reduce_fx=None)

    def update(self, output: Union[Mapping, Tensor], target: Tensor, state: State, **kwargs) -> None:
        """Updates the internal state with results from a new batch.

        Args:
            output (Mapping): The output from the model, which must contain
                either the Tensor or a Mapping type that contains the loss or model logits.
            target (~torch.Tensor): A Tensor of ground-truth values to compare against.
            state (~.State): A state object.
        """
        del kwargs
        assert isinstance(output, Tensor)
        data_shape = target.shape
        output = output.view(-1, self.vocab_size)
        target = target.view(-1)
        losses = self.loss_fn(output, target)
        losses = losses.reshape(data_shape)
        # Average loss across tokens
        losses = losses.sum(dim=1)
        if self.reduction == 'mean':
            losses /= state.batch['attention_mask'].sum(dim=1)
        # Concatenate losses
        self.seq_loss.extend(losses.tolist())  #type: ignore
        # self.seq_loss = torch.cat([self.seq_loss, losses], dim=0) #type: ignore
        uids = [tuple(item) for item in state.batch['input_ids'].tolist()]
        self.uid.extend(uids)  # type: ignore

    def compute(self) -> Dict[str, Any]:
        """Aggregate the state over all processes to compute the metric.

        Returns:
            seq_loss: The loss for each sequence averaged over tokens as a :class:`~torch.Tensor`.
            uids: The uid for each sequence as List.
        """
        return {'uid': self.uid, 'seq_loss': self.seq_loss}


@dataclass
class SequenceCrossEntropyHparams(MetricHparams):
    """Params for the :class`~.metrics.nlp.SequenceCrossEntropy`.

    Also see the documentation for the :class:`~.metrics.nlp.SequenceCrossEntropy`.

    Args:
        vocab_size (int): The size of the tokenizer vocabulary.
        ignore_index (int, optional): The class index to ignore. Default: ``-100``.
        reduction (str, optional): ``'mean'`` or ``'sum'``. Reduction to apply across
            tokens in a sequence. Default: ``'mean'``.
    """
    vocab_size: Optional[int] = hp.optional(doc='vocab_size', default=None)
    ignore_index: Optional[int] = hp.optional(doc='ignore_index', default=-100)
    reduction: Optional[str] = hp.optional(doc='reduction to apply across tokens', default='mean')

    metric_class = SequenceCrossEntropy

    # def initialize_object(self, **kwargs) -> SequenceCrossEntropy:
    #     return SequenceCrossEntropy(vocab_size=self.vocab_size, ignore_index=self.ignore_index, dist_sync_on_step=self.dist_sync_on_step)


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

    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state('true_positive', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('false_positive', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('false_negative', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, output: Tensor, target: Tensor, **kwargs) -> None:
        """Updates the internal state with results from a new batch.

        Args:
            output (Mapping): The output from the model, which must contain
                either the Tensor or a Mapping type that contains the loss or model logits.
            target (~torch.Tensor): A Tensor of ground-truth values to compare against.
        """
        del kwargs
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


class HFCrossEntropy(Metric):
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

        self.add_state('sum_loss', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('total_batches', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, output: Union[Mapping, Tensor], target: Tensor, *kwargs) -> None:
        """Updates the internal state with results from a new batch.

        Args:
            output (Mapping): The output from the model, which must contain
                either the Tensor or a Mapping type that contains the loss or model logits.
            target (~torch.Tensor): A Tensor of ground-truth values to compare against.
        """
        del kwargs
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
                raise Exception(f'Type {type(output)} for the output is unsupported.')

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


class Perplexity(HFCrossEntropy):
    """Subclasses :class:`~composer.models.nlp_metrics.HFLanguageCrossEntropyLoss` to implement perplexity.

    If an algorithm modifies the loss function and it is no longer directly provided in the output, then this could be
    expensive because it'll compute the loss twice.
    """

    def compute(self) -> Tensor:
        """Returns torch.exp() of the LanguageCrossEntropyLoss."""
        avg_loss = super().compute()
        return torch.exp(avg_loss)
