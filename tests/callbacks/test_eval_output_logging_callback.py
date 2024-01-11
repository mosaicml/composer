# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import json

import pandas as pd
import pytest
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


class MockDataLoader(DataLoader):

    def __init__(self, tokenizer):
        self.dataset = MockDataset(tokenizer)


class MockState(State):

    def __init__(self) -> None:
        self.eval_metrics = {}
        self.timestamp = Timestamp()

    def add_metric(self, metric_name, metric):
        self.eval_metrics[metric_name] = {}
        self.eval_metrics[metric_name][str(metric)] = metric

    def update_curr_eval(self, dataloader, dataloader_label):
        self._dataloader = dataloader
        self._dataloader_label = dataloader_label


def mock_lm_computation(metric, tokenizer):
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

    batch = {'continuation_indices': cont_idxs, 'labels': inputs.roll(-1), 'input_ids': inputs}
    logits = torch.nn.functional.one_hot(inputs.roll(-1), num_classes=pad + 1).float() * 100
    start, end = cont_idxs[1].tolist()[0] - 1, cont_idxs[1].tolist()[-1]
    logits[1][start:end] = logits[0][start:end].clone()  # make one of the answer's continuations incorrect

    metric.update(batch, logits, batch['labels'])
    metric.compute()


def mock_mc_computation(metric, tokenizer):
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

    cont_idxs = []
    for context, continuation in zip(contexts, continuations):
        start = len(tokenizer(context)['input_ids'])
        end = start + len(tokenizer(continuation)['input_ids'])
        cont_idxs.append(torch.tensor(list(range(start, end))))

    batch = {
        'continuation_indices': cont_idxs,
        'labels': inputs.roll(-1),
        'input_ids': inputs,
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

    metric.update(batch, logits, batch['labels'])
    metric.compute()


@device('cpu')
@pytest.mark.parametrize('subset_samples', [-1, 1])
def test_eval_output_logging(device, subset_samples, tmp_path, tiny_gpt2_tokenizer):
    # this test simulates an unrolled version of the eval loop occurring twice
    state = MockState()
    in_memory_logger = InMemoryLogger()
    logger = Logger(state, in_memory_logger)
    lm_metric = InContextLearningLMAccuracy(cache_responses=True)
    mc_metric = InContextLearningMultipleChoiceAccuracy(cache_responses=True)

    state.add_metric('lm_acc', lm_metric)
    state.add_metric('mc_acc', mc_metric)

    # Construct the callback
    eval_output_logging = EvalOutputLogging(subset_samples, str(tmp_path))

    for i in range(2):

        # simulate a full round of eval
        state.update_curr_eval(
            MockDataLoader(tiny_gpt2_tokenizer),
            'lm_acc',
        )
        eval_output_logging.eval_start(state, logger)
        assert all(
            len(m.response_cache) == 0  # pyright: ignore[reportGeneralTypeIssues]
            for dictionary in state.eval_metrics.values()
            for m in dictionary.values())
        mock_lm_computation(state.eval_metrics['lm_acc']['InContextLearningLMAccuracy()'], tiny_gpt2_tokenizer)
        eval_output_logging.eval_end(state, logger)

        assert 'icl_outputs/lm_acc/InContextLearningLMAccuracy()' in in_memory_logger.tables
        assert json.loads(in_memory_logger.tables['icl_outputs/lm_acc/InContextLearningLMAccuracy()'])['columns'] == [
            'context_tok', 'continuation_tok_target', 'continuation_tok_pred', 'correct'
        ]
        if subset_samples == -1:
            assert json.loads(in_memory_logger.tables['icl_outputs/lm_acc/InContextLearningLMAccuracy()'])['data'] == [[
                'The dog is', ' furry', ' furry', True
            ], ['I love to eat', ' pie', '', False], ['I hate', ' long lines', ' long lines',
                                                      True], ['The weather is', ' snowy', ' snowy', True]]
        else:
            assert len(json.loads(
                in_memory_logger.tables['icl_outputs/lm_acc/InContextLearningLMAccuracy()'])['data']) == subset_samples
        # simulate another eval

        state.update_curr_eval(
            MockDataLoader(tiny_gpt2_tokenizer),
            'mc_acc',
        )
        eval_output_logging.eval_start(state, logger)
        assert all(
            len(m.response_cache) == 0  # pyright: ignore[reportGeneralTypeIssues]
            for dictionary in state.eval_metrics.values()
            for m in dictionary.values())
        mock_mc_computation(state.eval_metrics['mc_acc']['InContextLearningMultipleChoiceAccuracy()'],
                            tiny_gpt2_tokenizer)

        eval_output_logging.eval_end(state, logger)
        assert 'icl_outputs/lm_acc/InContextLearningLMAccuracy()' in in_memory_logger.tables
        assert 'icl_outputs/mc_acc/InContextLearningMultipleChoiceAccuracy()' in in_memory_logger.tables

        # assert lm acc table unchanged
        assert json.loads(in_memory_logger.tables['icl_outputs/lm_acc/InContextLearningLMAccuracy()'])['columns'] == [
            'context_tok', 'continuation_tok_target', 'continuation_tok_pred', 'correct'
        ]
        if subset_samples == -1:
            assert json.loads(in_memory_logger.tables['icl_outputs/lm_acc/InContextLearningLMAccuracy()'])['data'] == [[
                'The dog is', ' furry', ' furry', True
            ], ['I love to eat', ' pie', '', False], ['I hate', ' long lines', ' long lines',
                                                      True], ['The weather is', ' snowy', ' snowy', True]]
        else:
            assert len(json.loads(
                in_memory_logger.tables['icl_outputs/lm_acc/InContextLearningLMAccuracy()'])['data']) == subset_samples

        assert json.loads(
            in_memory_logger.tables['icl_outputs/mc_acc/InContextLearningMultipleChoiceAccuracy()'])['columns'] == [
                'question_tok', 'correct_choice', 'selected_choice', 'correct'
            ]
        if subset_samples == -1:
            assert json.loads(
                in_memory_logger.tables['icl_outputs/mc_acc/InContextLearningMultipleChoiceAccuracy()'])['data'] == [[
                    'Q: How do you cook a cake?', ' A: turn on the oven', ' A: turn on the oven', True
                ], ['Q: How old is the earth?', ' A: 4.5 billion years', ' A: 2 minutes', False]]
        else:
            assert len(
                json.loads(in_memory_logger.tables['icl_outputs/mc_acc/InContextLearningMultipleChoiceAccuracy()'])
                ['data']) == subset_samples

        #simulate eval after all
        eval_output_logging.eval_after_all(state, logger)
        state.timestamp = state.timestamp.to_next_batch()
        assert isinstance(eval_output_logging.destination_file, str)
        assert eval_output_logging.destination_file.endswith(f'eval-outputs-ba{i}.tsv')
        with open(eval_output_logging.destination_file, 'r') as f:
            df = pd.read_csv(f, sep='\t', index_col=None)
        assert set(df['benchmark']) == {
            'mc_acc_InContextLearningMultipleChoiceAccuracy()', 'lm_acc_InContextLearningLMAccuracy()'
        }
        assert df.columns.tolist() == [
            'context_tok', 'continuation_tok_target', 'continuation_tok_pred', 'correct', 'benchmark', 'question_tok',
            'correct_choice', 'selected_choice'
        ]
