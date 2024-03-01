# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Optional

import pytest
import torch
from torch.nn.functional import cross_entropy

from composer.metrics.nlp import (BinaryF1Score, InContextLearningCodeEvalAccuracy,
                                  InContextLearningExpectedCalibrationError, InContextLearningLMAccuracy,
                                  InContextLearningLMExpectedCalibrationError,
                                  InContextLearningMCExpectedCalibrationError, InContextLearningMetric,
                                  InContextLearningMultipleChoiceAccuracy, InContextLearningQAAccuracy,
                                  LanguageCrossEntropy, LanguagePerplexity, MaskedAccuracy)


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


@pytest.mark.parametrize('ignore_index', [-100])
@pytest.mark.parametrize('batch_size', [1e2, 1e3])
@pytest.mark.parametrize('sequence_length', [128])
@pytest.mark.parametrize('num_classes', [2, 10])
@pytest.mark.parametrize('minibatch_size', [56, 256, 768])
def test_cross_entropy(batch_size: float, ignore_index: Optional[int], sequence_length: int, num_classes: int,
                       minibatch_size: int):
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
                'loss': cross_entropy(preds_subset.view(-1, num_classes), true_subset.view(-1))
            }, true_subset.view(-1))

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


def test_in_context_learning_rename_args_no_op():
    batch = {'input': [1, 2, 3]}
    outputs = torch.Tensor([12, 13, 14])
    labels = torch.Tensor([0, 1, 0])
    batch, outputs, labels = InContextLearningMetric.rename_args(batch=batch, outputs=outputs, labels=labels)
    assert batch == {'input': [1, 2, 3]}
    assert torch.all(torch.eq(outputs, torch.tensor([12, 13, 14])))
    assert torch.all(torch.eq(labels, torch.tensor([0, 1, 0])))


def test_in_context_learning_rename_args_output_and_output_logits():
    batch = {'input': [1, 2, 3]}
    outputs = torch.Tensor([12, 13, 14])
    output_logits = torch.Tensor([.1, .2, .3])
    labels = torch.Tensor([0, 1, 0])
    with pytest.raises(ValueError):
        _, _, _ = InContextLearningMetric.rename_args(batch=batch,
                                                      outputs=outputs,
                                                      labels=labels,
                                                      output_logits=output_logits)


def test_in_context_learning_rename_args_rename_output_logits():
    batch = {'input': [1, 2, 3]}
    output_logits = torch.Tensor([.1, .2, .3])
    labels = torch.Tensor([0, 1, 0])
    batch, outputs, labels = InContextLearningMetric.rename_args(batch=batch,
                                                                 labels=labels,
                                                                 output_logits=output_logits)
    assert batch == {'input': [1, 2, 3]}
    assert torch.all(torch.eq(outputs, torch.Tensor([.1, .2, .3])))  # pyright: ignore [reportGeneralTypeIssues]
    assert torch.all(torch.eq(labels, torch.tensor([0, 1, 0])))


def test_in_context_learning_rename_args_fail_on_no_label():
    batch = {'input': [1, 2, 3]}
    output_logits = torch.Tensor([.1, .2, .3])
    with pytest.raises(ValueError):
        _, _, _ = InContextLearningMetric.rename_args(batch=batch, output_logits=output_logits)


def test_in_context_learning_rename_args_fail_on_no_output():
    batch = {'input': [1, 2, 3]}
    with pytest.raises(ValueError):
        _, _, _ = InContextLearningMetric.rename_args(batch=batch)


def test_in_context_learning_lm_accuracy(tiny_gpt2_tokenizer):
    contexts = ['The dog is', 'I love to eat', 'I hate', 'The weather is']
    continuations = [' furry', ' pie', ' long lines', ' snowy']
    pad = tiny_gpt2_tokenizer.pad_token_id
    inputs = [
        tiny_gpt2_tokenizer(context)['input_ids'] + tiny_gpt2_tokenizer(continuation)['input_ids']
        for context, continuation in zip(contexts, continuations)
    ]
    inputs = torch.tensor([input + [pad] * (2048 - len(input)) for input in inputs])

    cont_idxs = []
    for context, continuation in zip(contexts, continuations):
        start = len(tiny_gpt2_tokenizer(context)['input_ids'])
        end = start + len(tiny_gpt2_tokenizer(continuation)['input_ids'])
        cont_idxs.append(torch.tensor(list(range(start, end))))

    batch = {'continuation_indices': cont_idxs, 'labels': inputs.roll(-1), 'input_ids': inputs}
    logits = torch.nn.functional.one_hot(inputs.roll(-1), num_classes=pad + 1).float() * 100
    start, end = cont_idxs[1].tolist()[0] - 1, cont_idxs[1].tolist()[-1]
    logits[1][start:end] = logits[0][start:end].clone()  # make one of the answer's continuations incorrect
    metric = InContextLearningLMAccuracy()
    metric.update(batch, logits, batch['labels'])

    assert metric.compute() == 0.75


def test_in_context_learning_lm_ece(tiny_gpt2_tokenizer):
    contexts = ['The dog is', 'I love to eat', 'I hate', 'The weather is']
    continuations = [' furry', ' pie', ' long lines', ' snowy']
    pad = tiny_gpt2_tokenizer.pad_token_id
    inputs = [
        tiny_gpt2_tokenizer(context)['input_ids'] + tiny_gpt2_tokenizer(continuation)['input_ids']
        for context, continuation in zip(contexts, continuations)
    ]
    inputs = torch.tensor([input + [pad] * (2048 - len(input)) for input in inputs])

    cont_idxs = []
    for context, continuation in zip(contexts, continuations):
        start = len(tiny_gpt2_tokenizer(context)['input_ids'])
        end = start + len(tiny_gpt2_tokenizer(continuation)['input_ids'])
        cont_idxs.append(torch.tensor(list(range(start, end))))

    batch = {'continuation_indices': cont_idxs, 'labels': inputs.roll(-1), 'input_ids': inputs}
    # logits are expected to be unnormalized and will undergo softmax, so we must multiply by 100
    logits = torch.nn.functional.one_hot(inputs.roll(-1), num_classes=pad + 1).float() * 100
    start, end = cont_idxs[1].tolist()[0] - 1, cont_idxs[1].tolist()[-1]
    logits[1][start:end] = logits[0][start:end].clone()  # make one of the answer's continuations incorrect
    metric = InContextLearningLMExpectedCalibrationError()
    metric.update(batch, logits, batch['labels'])
    # all observations fall in the top confidence bucket (95%) but accuracy is only 75%,
    # hence ECE should be 0.2
    assert abs(metric.compute() - 0.2) < 0.0001


def test_in_context_learning_qa_accuracy():
    outputs = ['Correct but then some more text', 'Incorrect', ' the CORREct with weird casing and spacing']
    labels = [['Correct'], ['blah', 'blah2'], ['blah', 'correct']]
    batch = {'cot_delimiter': '', 'labels': labels}
    metric = InContextLearningQAAccuracy()
    metric.update(outputs, labels, batch)

    assert metric.compute() == (2 / 3)


def test_in_context_learning_qa_cot_accuracy():
    outputs = [
        'chain of thought ### Correct but then some more text\n\nanother chain of thought ### Incorrect answer this time',
        'Incorrect', 'chain of thought ### the CORREct with weird casing and spacing',
        'incorrect chain of thought delimiter ## Correct but wrong delimiter'
    ]
    labels = [['Correct'], ['blah', 'blah2'], ['blah', 'correct'], ['correct']]
    batch = {'cot_delimiter': ' ### ', 'labels': labels, 'do_normalization': True, 'stopping_criteria': '\n\n'}
    metric = InContextLearningQAAccuracy()
    metric.update(outputs, labels, batch)

    assert metric.compute() == (2 / 4)


def test_in_context_learning_code_eval_accuracy(monkeypatch):
    outputs = [
        '    return 1 if n <= 1 else fib(n - 1) + fib(n - 1)',  # incorrect
        '   if n <= 1:\n        return 1\n    return fib(n-1) + fib(n-2)',  # incorrect spacing
        '    return n * 2',  # correct
        '    return 2*n',  # correct
        '    return n + 2',  # incorrect
        '    return n + 1'
    ]  # correct
    labels = []
    prompts = ['def fib(n):\n', 'def multiply_by_two(n):\n', 'def add_one(n):\n']
    entry_points = ['fib', 'multiply_by_two', 'add_one']
    test_inputs = [['(1,)', '(2,)', '(4,)'], ['(1,)', '(2,)', '(4,)'], ['(1,)', '(2,)', '(4,)']]
    test_outputs = [['1', '2', '5'], ['2', '4', '8'], ['2', '3', '5']]
    sample_ids = [0, 1, 2]
    languages = ['python', 'python', 'python']
    monkeypatch.setenv('CODE_EVAL_DEVICE', 'LOCAL')
    generations_per_sample = 2

    def repeat(values):
        return [val for val in values for _ in range(generations_per_sample)]

    transformers = pytest.importorskip('transformers')
    tokenizer = transformers.AutoTokenizer.from_pretrained('mosaicml/mpt-7b')  # type: ignore reportUnboundVariable
    tokenizer.pad_token = tokenizer.eos_token
    input_ids = tokenizer.batch_encode_plus(repeat(prompts), return_tensors='pt', padding=True)['input_ids']
    batch = {
        # This tests deterministic beam search rather than sampling
        'input_ids': input_ids,
        'generation_kwargs': {
            'num_beams': 1,
        },
        'prompts': repeat(prompts),
        'pass_at_k': [1],
        'entry_points': repeat(entry_points),
        'test_inputs': repeat(test_inputs),
        'test_outputs': repeat(test_outputs),
        'languages': repeat(languages),
        'dataset_size': len(prompts),
        'generations_per_sample': generations_per_sample,
        'sample_id': repeat(sample_ids),
    }
    metric = InContextLearningCodeEvalAccuracy()
    metric.update(batch, outputs, labels)

    # pass@1 values
    #   program 1: 0
    #   program 2: 1
    #   program 3: .5
    # mean: 0.5
    assert metric.compute() == 0.5


def test_in_context_learning_mc_accuracy(tiny_gpt2_tokenizer):
    contexts = [
        'Q: How do you cook a cake?', 'Q: How do you cook a cake?', 'Q: How old is the earth?',
        'Q: How old is the earth?'
    ]
    continuations = [' A: turn on the oven', ' A: do a backflip', ' A: 2 minutes', ' A: 4.5 billion years']
    gold_indices = [0, 1]
    choice_groupings = [(0, 2), (2, 4)]
    pad = tiny_gpt2_tokenizer.pad_token_id
    inputs = [
        tiny_gpt2_tokenizer(context)['input_ids'] + tiny_gpt2_tokenizer(continuation)['input_ids']
        for context, continuation in zip(contexts, continuations)
    ]
    inputs = torch.tensor([input + [pad] * (2048 - len(input)) for input in inputs])
    attention_mask = ~(inputs == pad)

    cont_idxs = []
    for context, continuation in zip(contexts, continuations):
        start = len(tiny_gpt2_tokenizer(context)['input_ids'])
        end = start + len(tiny_gpt2_tokenizer(continuation)['input_ids'])
        cont_idxs.append(torch.tensor(list(range(start, end))))

    batch = {
        'continuation_indices': cont_idxs,
        'labels': inputs.roll(-1),
        'input_ids': inputs,
        'attention_mask': attention_mask,
        'gold_indices': gold_indices,
        'choice_groupings': choice_groupings
    }
    logits = torch.nn.functional.one_hot(inputs.roll(-1), num_classes=pad + 1).float()

    # for the first two, the correct answer is continuation 0
    # make the answer correct by making continuation 0 more likely for both answers
    start, end = cont_idxs[1].tolist()[0] - 1, cont_idxs[1].tolist()[-1]
    logits[1][start:end] = logits[0][start:end].clone()

    # for the last two, the correct answer is continuation 3
    # make the answer incorrect by making continuation 2 more likely for both answers
    start, end = cont_idxs[3].tolist()[0], cont_idxs[3].tolist()[-1]
    logits[3][start:end] = logits[2][start:end].clone()

    metric = InContextLearningMultipleChoiceAccuracy()

    metric.update(batch, logits, batch['labels'])
    assert metric.compute() == 0.5


def test_in_context_learning_mc_ece(tiny_gpt2_tokenizer):
    contexts = [
        'Q: How do you cook a cake?', 'Q: How do you cook a cake?', 'Q: How old is the earth?',
        'Q: How old is the earth?'
    ]
    continuations = [' turn on the oven', ' do a backflip', ' 2 minutes', ' 4.5 billion years']
    gold_indices = [0, 1]
    choice_groupings = [(0, 2), (2, 4)]
    pad = tiny_gpt2_tokenizer.pad_token_id
    inputs = [
        tiny_gpt2_tokenizer(context)['input_ids'] + tiny_gpt2_tokenizer(continuation)['input_ids']
        for context, continuation in zip(contexts, continuations)
    ]
    inputs = torch.tensor([input + [pad] * (2048 - len(input)) for input in inputs])

    cont_idxs = []
    for context, continuation in zip(contexts, continuations):
        start = len(tiny_gpt2_tokenizer(context)['input_ids'])
        end = start + len(tiny_gpt2_tokenizer(continuation)['input_ids'])
        cont_idxs.append(torch.tensor(list(range(start, end))))

    batch = {
        'continuation_indices': cont_idxs,
        'labels': inputs.roll(-1),
        'input_ids': inputs,
        'gold_indices': gold_indices,
        'choice_groupings': choice_groupings
    }
    logits = torch.nn.functional.one_hot(inputs.roll(-1), num_classes=pad + 1).float() * 100
    # for the first two, the correct answer is continuation 0
    # make the answer correct by making continuation 0 more likely for both answers
    start, end = cont_idxs[1].tolist()[0] - 1, cont_idxs[1].tolist()[-1]
    logits[1][start:end] = logits[0][start:end].clone()

    # for the last two, the correct answer is continuation 3
    # make the answer incorrect by making continuation 2 more likely for both answers
    start, end = cont_idxs[3].tolist()[0] - 1, cont_idxs[3].tolist()[-1]
    logits[3][start:end] = logits[2][start:end].clone()

    metric = InContextLearningMCExpectedCalibrationError()

    metric.update(batch, logits, batch['labels'])

    # accuracy is 50% but confidence is 95%, so ECE is 45%
    assert abs(metric.compute().item() - 0.45) < 0.0001


def test_in_context_learning_ece():
    metric = InContextLearningExpectedCalibrationError(n_buckets=1)
    metric.update(None, None, None)  # pyright: ignore [reportGeneralTypeIssues]
    metric.bucket_totals[0] = 2  # pyright: ignore [reportGeneralTypeIssues]
    metric.bucket_correct[0] = 1  # pyright: ignore [reportGeneralTypeIssues]
    # confidence of bucket = 50%, accuracy = 50% => ECE = 0.0
    assert metric.compute() == 0.0

    metric = InContextLearningExpectedCalibrationError(n_buckets=10)
    # this example corresponds to perfect calibration across all 10 buckets
    metric.update(None, None, None)  # pyright: ignore [reportGeneralTypeIssues]
    for i in range(len(metric.bucket_totals)):  # pyright: ignore [reportGeneralTypeIssues]
        upper_bound = (i + 1) / metric.n_buckets
        lower_bound = i / metric.n_buckets
        conf_bucket_i = (upper_bound + lower_bound) / 2
        metric.bucket_totals[i] = metric.n_buckets * 2  # pyright: ignore [reportGeneralTypeIssues]
        metric.bucket_correct[i] = conf_bucket_i * metric.n_buckets * 2  # pyright: ignore [reportGeneralTypeIssues]
    assert metric.compute() == 0.0

    metric = InContextLearningExpectedCalibrationError(n_buckets=10)
    # this example corresponds to perfect calibration
    metric.update(None, None, None)  # pyright: ignore [reportGeneralTypeIssues]
    metric.bucket_totals[-1] = 2  # pyright: ignore [reportGeneralTypeIssues]
    metric.bucket_correct[-1] = 0  # pyright: ignore [reportGeneralTypeIssues]
    # confidence = 95%, accuracy = 0% => ece = 95%
    assert metric.compute() == 0.95
