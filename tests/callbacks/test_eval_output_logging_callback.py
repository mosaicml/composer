# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import json

import torch
from torch.utils.data import DataLoader

from composer.callbacks import EvalOutputLogging
from composer.core.state import State
from composer.core.time import Timestamp
from composer.datasets.in_context_learning_evaluation import InContextLearningMultipleChoiceTaskDataset
from composer.loggers import InMemoryLogger, Logger
from composer.metrics.nlp import InContextLearningLMAccuracy, InContextLearningMultipleChoiceAccuracy
from tests.common import device


class MockDataset(InContextLearningMultipleChoiceTaskDataset):

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_tok_id = tokenizer.pad_token_id


class MockDataLoader(DataLoader):

    def __init__(self, tokenizer):
        self.dataset = MockDataset(tokenizer)


class MockState(State):

    def __init__(self) -> None:
        self.eval_metrics = {}
        self.metric_outputs = {}
        self.run_name = 'mock_name'
        self.timestamp = Timestamp()

    def add_metric(self, metric_name, metric):
        self.eval_metrics[metric_name] = {}
        self.eval_metrics[metric_name][str(metric)] = metric

    def update_curr_eval(self, dataloader, dataloader_label):
        self._dataloader = dataloader
        self._dataloader_label = dataloader_label


def mock_lm_computation(metric, tokenizer, state):
    contexts = ['The dog is', 'I love to eat', 'I hate', 'The weather is']
    continuations = [' furry', ' pie', ' long lines', ' snowy']
    pad = tokenizer.pad_token_id
    inputs = [
        tokenizer(context)['input_ids'] + tokenizer(continuation)['input_ids']
        for context, continuation in zip(contexts, continuations)
    ]
    inputs = torch.tensor([input + [pad] * (2048 - len(input)) for input in inputs])

    cont_idxs = []
    for context, continuation in zip(contexts, continuations):
        start = len(tokenizer(context)['input_ids'])
        end = start + len(tokenizer(continuation)['input_ids'])
        cont_idxs.append(torch.tensor(list(range(start, end))))

    batch = {'mode': 'icl_task', 'continuation_indices': cont_idxs, 'labels': inputs.roll(-1), 'input_ids': inputs}
    logits = torch.nn.functional.one_hot(inputs.roll(-1), num_classes=pad + 1).float() * 100
    start, end = cont_idxs[1].tolist()[0] - 1, cont_idxs[1].tolist()[-1]
    logits[1][start:end] = logits[0][start:end].clone()  # make one of the answer's continuations incorrect

    state.metric_outputs = metric.update(batch, logits, batch['labels'])
    metric.compute()
    state.batch = batch
    state.outputs = logits
    return state


def mock_mc_computation(metric, tokenizer, state):
    contexts = [
        'Q: How do you cook a cake?', 'Q: How do you cook a cake?', 'Q: How old is the earth?',
        'Q: How old is the earth?'
    ]
    continuations = [' A: turn on the oven', ' A: do a backflip', ' A: 2 minutes', ' A: 4.5 billion years']
    gold_indices = [0, 1]
    choice_groupings = [(0, 2), (2, 4)]
    pad = tokenizer.pad_token_id
    inputs = [
        tokenizer(context)['input_ids'] + tokenizer(continuation)['input_ids']
        for context, continuation in zip(contexts, continuations)
    ]
    inputs = torch.tensor([input + [pad] * (2048 - len(input)) for input in inputs])
    attention_mask = ~(inputs == pad)

    cont_idxs = []
    for context, continuation in zip(contexts, continuations):
        start = len(tokenizer(context)['input_ids'])
        end = start + len(tokenizer(continuation)['input_ids'])
        cont_idxs.append(torch.tensor(list(range(start, end))))

    batch = {
        'mode': 'icl_task',
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

    state.metric_outputs = metric.update(batch=batch, output_logits=logits, labels=batch['labels'])
    state.batch = batch
    state.outputs = logits
    metric.compute()


@device('cpu')
def test_eval_output_logging_lm(device, tiny_gpt2_tokenizer):
    # this test simulates an unrolled version of the eval loop occurring twice
    state = MockState()
    in_memory_logger = InMemoryLogger()
    logger = Logger(state, in_memory_logger)
    lm_metric = InContextLearningLMAccuracy()

    state.add_metric('lm_acc', lm_metric)

    # Construct the callback
    eval_output_logging = EvalOutputLogging(loggers_to_use=['InMemoryLogger'])

    for _ in range(2):
        state.update_curr_eval(
            MockDataLoader(tiny_gpt2_tokenizer),
            'lm_acc',
        )
        mock_lm_computation(state.eval_metrics['lm_acc']['InContextLearningLMAccuracy()'], tiny_gpt2_tokenizer, state)
        state.metric_outputs['metric_name'] = [
            lm_metric.__class__.__name__ for _ in range(0, state.batch['input_ids'].shape[0])
        ]
        eval_output_logging.eval_batch_end(state, logger)
        state.timestamp = Timestamp(batch=state.timestamp.batch.value + 1)
    eval_output_logging.eval_end(state, logger)

    assert f'lm_acc_step_0' in in_memory_logger.tables
    # Only want one table - we log once to a single step value during eval_end()
    assert len(in_memory_logger.tables) == 1
    assert json.loads(in_memory_logger.tables[f'lm_acc_step_0'])['columns'] == [
        'context', 'label', 'output', 'result', 'metric_name', 'input', 'run_name'
    ]
    # We use the same data in each batch
    assert json.loads(in_memory_logger.tables[f'lm_acc_step_0'])['data'] == [
        ['The dog is', ' furry', ' furry', 1, 'InContextLearningLMAccuracy', 'The dog is furry', 'mock_name'],
        ['I love to eat', ' pie', '[PAD]', 0, 'InContextLearningLMAccuracy', 'I love to eat pie', 'mock_name'],
        ['I hate', ' long lines', ' long lines', 1, 'InContextLearningLMAccuracy', 'I hate long lines', 'mock_name'],
        ['The weather is', ' snowy', ' snowy', 1, 'InContextLearningLMAccuracy', 'The weather is snowy', 'mock_name'],
        ['The dog is', ' furry', ' furry', 1, 'InContextLearningLMAccuracy', 'The dog is furry', 'mock_name'],
        ['I love to eat', ' pie', '[PAD]', 0, 'InContextLearningLMAccuracy', 'I love to eat pie', 'mock_name'],
        ['I hate', ' long lines', ' long lines', 1, 'InContextLearningLMAccuracy', 'I hate long lines', 'mock_name'],
        ['The weather is', ' snowy', ' snowy', 1, 'InContextLearningLMAccuracy', 'The weather is snowy', 'mock_name']
    ]


@device('cpu')
def test_eval_output_logging_mc(device, tiny_gpt2_tokenizer):
    # this test simulates an unrolled version of the eval loop occurring twice
    state = MockState()
    in_memory_logger = InMemoryLogger()
    logger = Logger(state, in_memory_logger)
    mc_metric = InContextLearningMultipleChoiceAccuracy()

    state.add_metric('mc_acc', mc_metric)

    # Construct the callback
    eval_output_logging = EvalOutputLogging(loggers_to_use=['InMemoryLogger'])
    for _ in range(2):
        state.update_curr_eval(
            MockDataLoader(tiny_gpt2_tokenizer),
            'mc_acc',
        )
        mock_mc_computation(state.eval_metrics['mc_acc']['InContextLearningMultipleChoiceAccuracy()'],
                            tiny_gpt2_tokenizer, state)
        state.metric_outputs['metric_name'] = [
            mc_metric.__class__.__name__ for _ in range(0, state.batch['input_ids'].shape[0])
        ]
        eval_output_logging.eval_batch_end(state, logger)
        state.timestamp = Timestamp(batch=state.timestamp.batch.value + 1)
    eval_output_logging.eval_end(state, logger)

    assert f'mc_acc_step_0' in in_memory_logger.tables
    # Only want one table - we log once to a single step value during eval_end()
    assert len(in_memory_logger.tables) == 1
    assert json.loads(in_memory_logger.tables[f'mc_acc_step_0'])['columns'] == [
        'context', 'correct_choice', 'correct_choice_idx', 'selected_choice', 'selected_choice_idx', 'all_choices',
        'result', 'metric_name', 'input', 'run_name'
    ]
    # We use the same data for each batch
    assert json.loads(in_memory_logger.tables[f'mc_acc_step_0'])['data'] == [
        [
            'Q: How do you cook a cake?', ' A: turn on the oven', 0, ' A: turn on the oven', 0,
            ['Q: How do you cook a cake? A: turn on the oven', 'Q: How do you cook a cake? A: do a backflip'], 1,
            'InContextLearningMultipleChoiceAccuracy', 'Q: How do you cook a cake? A: turn on the oven', 'mock_name'
        ],
        [
            'Q: How old is the earth?', ' A: 4.5 billion years', 1, ' A: 2 minutes', 0,
            [
                'Q: How old is the earth? A: 2 minutes[PAD][PAD][PAD]',
                'Q: How old is the earth? A: 4.5 billion years[PAD]'
            ], 0, 'InContextLearningMultipleChoiceAccuracy', 'Q: How do you cook a cake? A: do a backflip', 'mock_name'
        ],
        [
            'Q: How do you cook a cake?', ' A: turn on the oven', 0, ' A: turn on the oven', 0,
            ['Q: How do you cook a cake? A: turn on the oven', 'Q: How do you cook a cake? A: do a backflip'], 1,
            'InContextLearningMultipleChoiceAccuracy', 'Q: How do you cook a cake? A: turn on the oven', 'mock_name'
        ],
        [
            'Q: How old is the earth?', ' A: 4.5 billion years', 1, ' A: 2 minutes', 0,
            [
                'Q: How old is the earth? A: 2 minutes[PAD][PAD][PAD]',
                'Q: How old is the earth? A: 4.5 billion years[PAD]'
            ], 0, 'InContextLearningMultipleChoiceAccuracy', 'Q: How do you cook a cake? A: do a backflip', 'mock_name'
        ]
    ]
