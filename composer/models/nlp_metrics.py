# Copyright 2021 MosaicML. All Rights Reserved.

from typing import Mapping, Union

import torch
from torch import Tensor
from torchmetrics import Metric

from composer.models.loss import soft_cross_entropy


class LanguageCrossEntropyLoss(Metric):
    """Hugging Face compatible cross entropy loss.

    Args:
        dist_sync_on_step (bool): Synchronize metric state across processes at
            each forward() before returning the value at the step.

    State:
        sum_loss (float): the sum of the per-example loss in the batch.
        total_batches (float): the number of batches to average across.
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
            target (Tensor): A Tensor of ground-truth values to compare against.
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
            loss (Tensor): The loss averaged across all batches.
        """
        # Return average loss over entire dataset
        return self.sum_loss / self.total_batches  #type: ignore (third-party)


class Perplexity(LanguageCrossEntropyLoss):
    """Subclasses :class:`LanguageCrossEntropyLoss` to implement perplexity.

    If an algorithm modifies the loss function and it is no longer directly
    provided in the output, then this could be expensive because it'll compute the loss twice.
    """

    def compute(self) -> Tensor:
        """Returns torch.exp() of the LanguageCrossEntropyLoss.
        """
        avg_loss = super().compute()
        return torch.exp(avg_loss)
