# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import math
import time
from typing import Optional

import pytest
import torch
from torch.nn.functional import cross_entropy

from composer.metrics.nlp import (
    BinaryF1Score,
    LanguageCrossEntropy,
    LanguagePerplexity,
    MaskedAccuracy,
)
from tests.common import device


@pytest.mark.parametrize('ignore_index', [-100])
@pytest.mark.parametrize('num_classes', [2, 3, 4, 5])
def test_masked_accuracy(ignore_index, num_classes):
    """Sanity check to make sure that masked accuracy has reasonable performance.

    Generates random targets and labels, and then ensures that the random targets and labels
    must hit at-chance accuracy.

    Args:
        batch_size (int): how many samples are in each batch
        ignore_index (Optional[int]): if present, the class index to ignore in accuracy calculations.
        num_classes (int): the number of classes in the classification task
    """
    batch_size = int(1e4)
    torchmetrics_masked_acc = MaskedAccuracy(ignore_index=ignore_index)
    # we're only testing binary accuracy -- expected accuracy should be 50%
    generated_preds = torch.rand((batch_size, num_classes))
    true_labels = torch.randint(low=0, high=num_classes - 1, size=(batch_size,))

    if ignore_index is not None:
        labels_mask = torch.rand((batch_size,))
        labels_mask[labels_mask > 0.8] = 1
        labels_mask[labels_mask <= 0.8] = 0
        labels_mask = labels_mask.bool()
        true_labels[labels_mask] = ignore_index

    true_labels = true_labels.float()
    generated_preds = generated_preds.float()

    torchmetrics_masked_acc.update(generated_preds, true_labels)
    final_acc = torchmetrics_masked_acc.compute()
    assert abs(final_acc - (1.0 / num_classes)) < 0.02


@device('cpu', 'gpu')
@pytest.mark.parametrize('ignore_index', [-100])
@pytest.mark.parametrize('batch_size', [1e2, 1e3])
@pytest.mark.parametrize('sequence_length', [128])
@pytest.mark.parametrize('num_classes', [2, 10])
@pytest.mark.parametrize('minibatch_size', [56, 256, 768])
@pytest.mark.parametrize('tensor_device', ['cpu', 'gpu'])
def test_cross_entropy(
    device: str,
    batch_size: float,
    ignore_index: Optional[int],
    sequence_length: int,
    num_classes: int,
    minibatch_size: int,
    tensor_device: str,
):
    """Sanity check to make sure that batched CrossEntropyLoss matches the expected performance.

    Generates a predicted distribution from a normal distribution, and a ground truth from a normal distribution.
    Verifies Cross Entropy Loss against the baseline performance.

    Args:
        device (str): the device to run the test on
        batch_size (int): how many samples are in each batch
        ignore_index (Optional[int]): if present, the class index to ignore in accuracy calculations.
        sequence_length (int): the length of the generated sequence
        num_classes (int): the number of classes in the classification task
        minibatch_size (int): the minibatch size to simulate for model predictions
        tensor_device (str): which device the input tensors to the metric are on
    """

    if device == 'cpu' and tensor_device == 'gpu':
        pytest.skip('Skipping test that would try to use GPU tensors when only CPU is available.')

    batch_size = int(batch_size)
    generated_preds = torch.randn((batch_size, sequence_length, num_classes))
    generated_true = torch.randint(low=0, high=num_classes, size=(batch_size, sequence_length))

    assert ignore_index is not None
    torchmetrics_xent = LanguageCrossEntropy(dist_sync_on_step=False, ignore_index=ignore_index)
    ce_with_keys_metric = LanguageCrossEntropy(dist_sync_on_step=False, ignore_index=ignore_index)

    if tensor_device == 'cpu':
        torchmetrics_xent = torchmetrics_xent.to('cpu')
        ce_with_keys_metric = ce_with_keys_metric.to('cpu')
    elif tensor_device == 'gpu':
        torchmetrics_xent = torchmetrics_xent.to('cuda')
        ce_with_keys_metric = ce_with_keys_metric.to('cuda')

    if device == 'gpu':
        assert torchmetrics_xent.flash_loss_fn is not None

    labels_mask = torch.rand((batch_size, sequence_length))
    labels_mask[labels_mask > 0.8] = 1
    labels_mask[labels_mask <= 0.8] = 0
    labels_mask = labels_mask.bool()
    generated_true[labels_mask] = ignore_index

    num_batches = math.ceil(batch_size / minibatch_size)
    for batch_idx in range(num_batches):
        begin_idx = (batch_idx * minibatch_size)
        end_idx = ((batch_idx + 1) * minibatch_size)
        preds_subset = generated_preds[begin_idx:end_idx]
        true_subset = generated_true[begin_idx:end_idx]

        if tensor_device == 'cpu':
            preds_subset = preds_subset.cpu()
            true_subset = true_subset.cpu()
        elif tensor_device == 'gpu':
            preds_subset = preds_subset.cuda()
            true_subset = true_subset.cuda()

        torchmetrics_xent.update(preds_subset, true_subset)
        ce_with_keys_metric.update(
            {
                'logits': preds_subset.view(-1, num_classes),
                'loss': cross_entropy(preds_subset.view(-1, num_classes), true_subset.view(-1)),
            },
            true_subset.view(-1),
        )

    torchmetrics_loss = torchmetrics_xent.compute()
    ce_with_keys_loss = ce_with_keys_metric.compute()
    correct_loss = cross_entropy(generated_preds.view(-1, num_classes), generated_true.view(-1))
    assert torchmetrics_loss == ce_with_keys_loss
    assert torch.isclose(correct_loss, torchmetrics_loss)


@pytest.mark.parametrize('ignore_index', [-100])
@pytest.mark.parametrize('batch_size', [1e2, 1e3])
@pytest.mark.parametrize('sequence_length', [128])
@pytest.mark.parametrize('num_classes', [2, 10])
@pytest.mark.parametrize('minibatch_size', [56, 256, 768])
def test_torch_cpu_cross_entropy(
    batch_size: float,
    ignore_index: Optional[int],
    sequence_length: int,
    num_classes: int,
    minibatch_size: int,
):
    """Sanity check to make sure that batched CrossEntropyLoss matches the expected performance.

    Generates a predicted distribution from a normal distribution, and a ground truth from a normal distribution.
    Verifies Cross Entropy Loss against the baseline performance.

    Args:
        batch_size (int): how many samples are in each batch
        ignore_index (Optional[int]): if present, the class index to ignore in accuracy calculations.
        sequence_length (int): the length of the generated sequence
        num_classes (int): the number of classes in the classification task
        minibatch_size (int): the minibatch size to simulate for model predictions
    """
    batch_size = int(batch_size)
    generated_preds = torch.randn((batch_size, sequence_length, num_classes))
    generated_true = torch.randint(low=0, high=num_classes, size=(batch_size, sequence_length))

    assert ignore_index is not None
    torchmetrics_xent = LanguageCrossEntropy(dist_sync_on_step=False, ignore_index=ignore_index)
    ce_with_keys_metric = LanguageCrossEntropy(dist_sync_on_step=False, ignore_index=ignore_index)

    labels_mask = torch.rand((batch_size, sequence_length))
    labels_mask[labels_mask > 0.8] = 1
    labels_mask[labels_mask <= 0.8] = 0
    labels_mask = labels_mask.bool()
    generated_true[labels_mask] = ignore_index

    num_batches = math.ceil(batch_size / minibatch_size)
    for batch_idx in range(num_batches):
        begin_idx = (batch_idx * minibatch_size)
        end_idx = ((batch_idx + 1) * minibatch_size)
        preds_subset = generated_preds[begin_idx:end_idx]
        true_subset = generated_true[begin_idx:end_idx]
        torchmetrics_xent.update(preds_subset, true_subset)
        ce_with_keys_metric.update(
            {
                'logits': preds_subset.view(-1, num_classes),
                'loss': cross_entropy(preds_subset.view(-1, num_classes), true_subset.view(-1)),
            },
            true_subset.view(-1),
        )

    torchmetrics_loss = torchmetrics_xent.compute()
    ce_with_keys_loss = ce_with_keys_metric.compute()
    correct_loss = cross_entropy(generated_preds.view(-1, num_classes), generated_true.view(-1))
    assert torchmetrics_loss == ce_with_keys_loss
    assert torch.isclose(correct_loss, torchmetrics_loss)


@pytest.mark.parametrize('batch_size', [1e2, 1e3, 1e4])
@pytest.mark.parametrize('minibatch_size', [256, 768])
def test_binary_f1(batch_size, minibatch_size):
    """Sanity check to make sure that BinaryF1 TorchMetrics implementation matches the sklearn implementation.

    Generates a predicted set of labels, and a random set, and compares the resultant Binary F1 score.

    Args:
        batch_size (int): how many samples are in each batch
        minibatch_size (int): the minibatch size to simulate for model predictions
    """
    pytest.importorskip('sklearn', reason='sklearn is an optional dependency')
    from sklearn.metrics import f1_score

    batch_size = int(batch_size)

    generated_preds = torch.randn(size=(batch_size, 2))
    generated_true = torch.randint(low=0, high=2, size=(batch_size,))

    binary_f1 = BinaryF1Score()

    num_batches = math.ceil(batch_size / minibatch_size)
    for batch_idx in range(num_batches):
        begin_idx = (batch_idx * minibatch_size)
        end_idx = ((batch_idx + 1) * minibatch_size)
        preds_subset = generated_preds[begin_idx:end_idx]
        true_subset = generated_true[begin_idx:end_idx]
        binary_f1.update(preds_subset, true_subset)

    torchmetrics_f1 = binary_f1.compute()
    generated_preds = torch.argmax(generated_preds, dim=1)
    correct_f1 = f1_score(y_true=generated_true, y_pred=generated_preds)
    assert correct_f1 == torchmetrics_f1


def test_language_perplexity():
    batch_size = 1024
    sequence_length = 64
    num_classes = 10
    ignore_index = -100
    minibatch_size = 128

    generated_preds = torch.randn((batch_size, sequence_length, num_classes))
    generated_true = torch.randint(low=0, high=num_classes, size=(batch_size, sequence_length))

    ce_metric = LanguageCrossEntropy(dist_sync_on_step=False)
    perplexity_metric = LanguagePerplexity(dist_sync_on_step=False)

    labels_mask = torch.rand((batch_size, sequence_length))
    labels_mask[labels_mask > 0.8] = 1
    labels_mask[labels_mask <= 0.8] = 0
    labels_mask = labels_mask.bool()
    generated_true[labels_mask] = ignore_index

    num_batches = math.ceil(batch_size / minibatch_size)
    for batch_idx in range(num_batches):
        begin_idx = (batch_idx * minibatch_size)
        end_idx = ((batch_idx + 1) * minibatch_size)
        preds_subset = generated_preds[begin_idx:end_idx]
        true_subset = generated_true[begin_idx:end_idx]

        ce_metric.update(preds_subset, true_subset)
        perplexity_metric.update(preds_subset, true_subset)

    ce = ce_metric.compute()
    perplexity = perplexity_metric.compute()

    assert torch.equal(torch.exp(ce), perplexity)
