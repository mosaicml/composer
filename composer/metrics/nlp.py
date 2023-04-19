# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A collection of common torchmetrics for NLP tasks."""
import re
import string
import warnings
from typing import Any, Dict, List, Mapping, Optional, Union

import torch
from torch import Tensor
from torch.nn import functional as F
from torchmetrics import Metric

from composer.loss import soft_cross_entropy

__all__ = [
    'Perplexity',
    'InContextLearningLMAccuracy',
    'InContextLearningMultipleChoiceAccuracy',
    'InContextLearningQAAccuracy',
    'BinaryF1Score',
    'HFCrossEntropy',
    'LanguageCrossEntropy',
    'MaskedAccuracy',
    'LanguagePerplexity',
    'InContextLearningLMExpectedCalibrationError',
    'InContextLearningMCExpectedCalibrationError',
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
        vocab_size (int): The size of the tokenizer vocabulary.
        dist_sync_on_step (bool, optional): Synchronize metric state across processes at
            each forward() before returning the value at the step. Default: ``False``.
        ignore_index (int, optional): The class index to ignore. Default: ``-100``.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, vocab_size: Optional[int] = None, dist_sync_on_step: bool = False, ignore_index: int = -100):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        if vocab_size is not None:
            warnings.warn(
                DeprecationWarning(
                    'The vocab_size argument is deprecated and will be removed in 0.15. It is no longer needed, because the correct shape of output and target is inferred based on the number of target elements.'
                ))

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


class HFCrossEntropy(Metric):
    """Hugging Face compatible cross entropy loss.

    Adds metric state variables:
        sum_loss (float): The sum of the per-example loss in the batch.
        total_batches (float): The number of batches to average across.

    Args:
        dist_sync_on_step (bool, optional): Synchronize metric state across processes at
            each forward() before returning the value at the step. Default: ``False``
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, dist_sync_on_step=False):
        warnings.warn(
            DeprecationWarning(
                "'HFCrossEntropy' is deprecated and will be removed in 0.15. Please use `LanguageCrossEntropy' instead."
            ))

        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state('sum_loss', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('total_batches', default=torch.tensor(0), dist_reduce_fx='sum')

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
                raise Exception(f'Type {type(output)} for the output is unsupported.')

            loss = soft_cross_entropy(logits, target)

        # accumulate loss over all batches
        self.sum_loss += loss

        # Note: This is a slightly different reduction than LanguageCrossEntropy, because LanguageCrossEntropy
        # uses 'sum' reduction in its update call
        self.total_batches += 1  #type: ignore (third-party)

    def compute(self) -> Tensor:
        """Aggregate the state over all processes to compute the metric.

        Returns:
            loss: The loss averaged across all batches as a :class:`~torch.Tensor`.
        """
        # Return average loss over entire dataset
        return self.sum_loss / self.total_batches  #type: ignore (third-party)


class Perplexity(HFCrossEntropy):
    """Subclasses :class:`~composer.metrics.nlp.HFCrossEntropy` to implement perplexity.

    If an algorithm modifies the loss function and it is no longer directly provided in the output, then this could be
    expensive because it'll compute the loss twice.
    """

    def __init__(self, dist_sync_on_step=False):
        warnings.warn(
            DeprecationWarning(
                "'Perplexity' is deprecated and will be removed in 0.15. Please use `LanguagePerplexity' instead."))

        super().__init__(dist_sync_on_step=dist_sync_on_step)

    def compute(self) -> Tensor:
        """Returns torch.exp() of the HFCrossEntropy."""
        avg_loss = super().compute()
        return torch.exp(avg_loss)


class LanguagePerplexity(LanguageCrossEntropy):
    """Subclasses :class:`~composer.metrics.nlp.LanguageCrossEntropy` to implement perplexity."""

    def compute(self) -> Tensor:
        """Returns torch.exp() of the LanguageCrossEntropy."""
        avg_loss = super().compute()
        return torch.exp(avg_loss)


class InContextLearningMetric(Metric):

    def update(self, batch: dict, output_logits: torch.Tensor, labels: torch.Tensor):
        """Abstract interface for computing an in-context learning metrics.

        Args:
            batch (dict): Batch must consist minimally of `input_ids` as well as any other structure needed
                to compute the metric.
            output_logits (torch.Tensor): The model outputs evaluated on the batch `input_ids`
            labels (torch.Tensor): The correct outputs.

        Raises:
            NotImplementedError: Abstract method must be implemented by subclasses
        """
        raise NotImplementedError


class InContextLearningQAAccuracy(InContextLearningMetric):
    r"""Computes accuracy for In-context learning (ICL) question answering (QA) tasks.

    ICL QA tasks consist of some number of example question answering tasks (referred to as the 'context'), followed by a test task where the model must
    match one of the possible answer aliases (referred to as the 'continuation').

    For example, the model may be provided the context below and evaluated on its ability to correctly predict the continuation.

    Context: `Question: Who was president of the United States in 2012?\nAnswer: Barack Obama\nQuestion: Is water wet?\nAnswer: `
    Continuation: [`yes`, `no`]

    Both predictions and answers will be normalized before comparison.

    Adds metric state variables:
        correct (float): The number of instances where the prediction was a prefix for any of the answer aliases.
        total (float): The number of total instances that were predicted.

    Args:
        dist_sync_on_step (bool, optional): Synchronize metric state across processes at
            each forward() before returning the value at the step. Default: ``False``.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, dist_sync_on_step: bool = False):
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state('correct', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0.), dist_reduce_fx='sum')

    def normalize_answer(self, answer: str):
        """Lower text and remove punctuation, articles and extra whitespace.

        Copied from https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py
        """

        def remove_articles(text: str) -> str:
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text: str) -> str:
            return ' '.join(text.split())

        def handle_punc(text: str) -> str:
            exclude = set(string.punctuation + ''.join([u'‘', u'’', u'´', u'`']))
            return ''.join(ch if ch not in exclude else ' ' for ch in text)

        def lower(text: str) -> str:
            return text.lower()

        def replace_underscore(text: str) -> str:
            return text.replace('_', ' ')

        return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(answer))))).strip()

    def update(self, outputs: List[str], labels: List[List[str]]):
        for sample_output, sample_labels in zip(outputs, labels):
            cleaned_sample_output = self.normalize_answer(sample_output)
            cleaned_sample_labels = set(self.normalize_answer(label) for label in sample_labels)
            if any(cleaned_sample_output.startswith(label) for label in cleaned_sample_labels):
                self.correct += torch.tensor(1.0)
            self.total += torch.tensor(1.0)

    def compute(self):
        assert isinstance(self.correct, Tensor)
        assert isinstance(self.total, Tensor)
        return self.correct / self.total


class InContextLearningLMAccuracy(InContextLearningMetric):
    r"""Computes accuracy for In-context learning (ICL) language modeling (LM) tasks.

    ICL LM tasks consist of some number of example language modeling tasks (referred to as the 'context'), followed by a test task where the model must correctly predict all the tokens
    following tokens in some passage (referred to as the 'continuation').

    For example, the model may be provided the context below and evaluated on its ability to correctly predict the continuation. Note: it doesn't matter
    whether the model correctly predicts the context tokens.

    Context: `The dog is->fuzzy\nthe water is->hot\nthe tree is->`
    Continuation: `green`

    Adds metric state variables:
        correct (float): The number of instances where the prediction masked the target.
        total (float): The number of total instances that were predicted.

    Args:
        dist_sync_on_step (bool, optional): Synchronize metric state across processes at
            each forward() before returning the value at the step. Default: ``False``.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, dist_sync_on_step: bool = False):
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state('correct', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0.), dist_reduce_fx='sum')

    def update(self, batch: dict, output_logits: torch.Tensor, labels: torch.Tensor):
        for batch_idx, cont_idx in enumerate(batch['continuation_indices']):
            cont_tok_pred = output_logits[batch_idx].index_select(dim=0, index=cont_idx - 1).argmax(dim=-1)
            cont_tok_targ = labels[batch_idx].index_select(dim=0, index=cont_idx - 1)

            self.correct += (cont_tok_pred == cont_tok_targ).all().int()
            self.total += torch.tensor(1.0)

    def compute(self):
        assert isinstance(self.correct, Tensor)
        assert isinstance(self.total, Tensor)
        return self.correct / self.total


class InContextLearningMultipleChoiceAccuracy(InContextLearningMetric):
    r"""Computes accuracy for In-context learning (ICL) multiple choice (MC) tasks.

    ICL MC tasks consists of a series of questions with some number of possible choices (only one of which can be correct).
    At inference time each possible choice is given to the model as a separate input and the one for which the model assigns
    the lowest perplexity to the choice is considered the model's choice. The model is correct if it "chooses" the right answer.

    Context: `The dog is->fuzzy\nthe water is->hot\nthe tree is->`
    Continuation: `green`

    Adds metric state variables:
        correct (float): The number of instances where the prediction masked the target.
        total (float): The number of total instances that were predicted.

    Args:
        dist_sync_on_step (bool, optional): Synchronize metric state across processes at
            each forward() before returning the value at the step. Default: ``False``.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, dist_sync_on_step: bool = False):
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state('correct', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0.0), dist_reduce_fx='sum')

    def update(self, batch: dict, output_logits: torch.Tensor, labels: torch.Tensor):
        perplexities = []
        for batch_idx, cont_idx in enumerate(batch['continuation_indices']):
            # continuation indices refer to indices in the original input's token space
            cont_tok_logits = output_logits[batch_idx].index_select(dim=0, index=cont_idx - 1)
            # labels have been shifted left by one index, so the cont_idx needs to be shifted as well.
            cont_tok_targ = labels[batch_idx].index_select(dim=0, index=cont_idx - 1)
            cross_entropy = F.cross_entropy(cont_tok_logits, cont_tok_targ)
            perplexity = torch.exp(cross_entropy)
            perplexities.append(perplexity)

        for (start, end), gold_idx in zip(batch['choice_groupings'], batch['gold_indices']):
            subset = perplexities[start:end]
            idx_min = subset.index(min(subset))

            if idx_min == gold_idx:
                self.correct += torch.tensor(1.0)
            self.total += torch.tensor(1.0)

    def compute(self):
        assert isinstance(self.correct, Tensor)
        assert isinstance(self.total, Tensor)
        return self.correct.float() / self.total


class InContextLearningExpectedCalibrationError(InContextLearningMetric):
    """Generic class for Expected Calibration Error (ECE) (cite: https://arxiv.org/pdf/1706.04599.pdf).

    Expected calibration error is calculated by dividing predictions into buckets based on the model's confidence (a probability value between 0 and 1).
    We then calculate the accuracy within each bucket and calculate the average gap between confidence and accuracy
    across buckets, weighted by the number of samples in each bucket.

    Each task must implement its own definition of "confidence" to be computed via the `update` method.

    Adds metric state variables:
    bucket_totals (float): The number of instances where the prediction masked the target per bucket.
    bucket_correct (float): The number of total instances that were predicted per bucket.

    Args:
        dist_sync_on_step (bool, optional): Synchronize metric state across processes at
            each forward() before returning the value at the step. Default: ``False``.
        n_buckets (int): Number of distinct buckets to split the confidence distribution into
    """

    def __init__(self, dist_sync_on_step: bool = False, n_buckets: int = 10):
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.n_buckets = n_buckets
        if n_buckets < 1:
            raise Exception('`n_buckets`')
        self.add_state('bucket_totals', default=torch.zeros(n_buckets), dist_reduce_fx='sum')
        self.add_state('bucket_correct', default=torch.zeros(n_buckets), dist_reduce_fx='sum')

    def update(self, batch: dict, output_logits: torch.Tensor, labels: torch.Tensor):
        pass

    def compute(self):
        assert isinstance(self.bucket_correct, Tensor)
        assert isinstance(self.bucket_totals, Tensor)

        result = torch.tensor(0.0, device=self.bucket_correct.device)
        total_obs = torch.sum(self.bucket_totals)
        for i in range(self.n_buckets):
            if self.bucket_totals[i] == 0:
                continue

            acc_bucket_i = self.bucket_correct[i] / self.bucket_totals[i]
            upper_bound = (i + 1) / self.n_buckets
            lower_bound = i / self.n_buckets
            conf_bucket_i = torch.tensor((upper_bound + lower_bound) / 2, device=self.bucket_correct.device)
            result += (self.bucket_totals[i] / total_obs) * torch.abs(acc_bucket_i - conf_bucket_i)
        return result


class InContextLearningMCExpectedCalibrationError(InContextLearningExpectedCalibrationError):
    r"""Computes Expected Calibration Error (ECE) for In-context learning (ICL) multiple choice (MC) tasks. (source: https://arxiv.org/abs/2012.00955).

    For MC tasks, the model confidence is defined as the softmax of average per-token probability assigned to the top question choice.

    See `InContextLearningExpectedCalibrationError` for more info.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def update(self, batch: Dict[str, Any], output_logits: torch.Tensor, labels: torch.Tensor):
        output_logits = torch.softmax(output_logits, dim=2)
        probabilites = []
        for batch_idx, cont_idx in enumerate(batch['continuation_indices']):
            cont_tok_logits = output_logits[batch_idx].index_select(dim=0, index=cont_idx - 1)
            cont_tok_targ = labels[batch_idx].index_select(dim=0, index=cont_idx - 1)
            probability = cont_tok_logits.index_select(dim=1, index=cont_tok_targ).diagonal().mean()
            probabilites.append(probability)

        for (start, end), gold_idx in zip(batch['choice_groupings'], batch['gold_indices']):
            subset = probabilites[start:end]
            idx_max = subset.index(max(subset))
            confidence = torch.tensor(subset).max() / torch.tensor(subset).sum()

            assert confidence >= 0.0 and confidence <= 1.0
            bucket_idx = int(confidence * self.n_buckets)
            if bucket_idx == self.n_buckets:
                bucket_idx -= 1

            if idx_max == gold_idx:
                self.bucket_correct[bucket_idx] += 1  # pyright: ignore [reportGeneralTypeIssues]

            self.bucket_totals[bucket_idx] += 1  # pyright: ignore [reportGeneralTypeIssues]


class InContextLearningLMExpectedCalibrationError(InContextLearningExpectedCalibrationError):
    r"""Computes Expected Calibration Error (ECE) for In-context learning (ICL) language modeling (LM) tasks. (cite: https://arxiv.org/pdf/1706.04599.pdf).

    For LM tasks, the model confidence is defined as the minimum probability assigned to all tokens in the continuation.

    See `InContextLearningExpectedCalibrationError` for more info.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def update(self, batch: Dict[str, Any], output_logits: torch.Tensor, labels: torch.Tensor):
        output_logits = torch.softmax(output_logits, dim=2)
        for batch_idx, cont_idx in enumerate(batch['continuation_indices']):
            cont_tok_logits = output_logits[batch_idx].index_select(dim=0, index=cont_idx - 1)
            cont_tok_pred = cont_tok_logits.argmax(dim=-1)
            confidence = cont_tok_logits.max(dim=-1).values.min()
            cont_tok_targ = labels[batch_idx].index_select(dim=0, index=cont_idx - 1)
            assert confidence >= 0.0 and confidence <= 1.0
            bucket_idx = int(confidence * self.n_buckets)
            if bucket_idx == self.n_buckets:
                bucket_idx -= 1

            if (cont_tok_pred == cont_tok_targ).all():
                self.bucket_correct[bucket_idx] += 1  # pyright: ignore [reportGeneralTypeIssues]

            self.bucket_totals[bucket_idx] += 1  # pyright: ignore [reportGeneralTypeIssues]
