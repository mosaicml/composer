# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
import time
from typing import Optional, Set
from unittest.mock import patch

import pytest
from torch.utils.data import DataLoader

from composer.loggers.slack_logger import SlackLogger
from composer.trainer import Trainer
from tests.common import RandomClassificationDataset, SimpleModel


@pytest.mark.parametrize('include_keys', [None, {}, {'loss/train/total'}])
def test_slack_logger_metrics(include_keys: Optional[Set[str]]):
    pytest.importorskip('slack_sdk')
    os.environ['SLACK_LOGGING_API_KEY'] = str('1234')
    os.environ['SLACK_LOGGING_CHANNEL_ID'] = 'C1234'

    slack_logger = SlackLogger(
        formatter_func=(lambda data, **kwargs: [{
            'type': 'section',
            'text': {
                'type': 'mrkdwn',
                'text': f'*{k}:* {v}'
            }
        } for k, v in data.items()]),
        include_keys=include_keys,
        interval_in_seconds=None,
    )

    # Patch for hparams
    with patch('slack_sdk.WebClient.chat_postMessage'):
        trainer = Trainer(
            model=SimpleModel(),
            train_dataloader=DataLoader(RandomClassificationDataset()),
            train_subset_num_batches=2,
            max_duration='1ep',
            loggers=[slack_logger],
        )

    with patch('slack_sdk.WebClient.chat_postMessage') as mock_post_logs:
        trainer.fit()

    if include_keys is None:
        # include all keys by default
        assert mock_post_logs.call_count == 10
    elif len(include_keys) == 0:
        # include no keys if empty set
        assert mock_post_logs.call_count == 0
        return
    elif include_keys == {'loss/train/total'}:
        assert mock_post_logs.call_count == 2
        metrics_key = 'loss/train/total'
        assert (metrics_key in mock_post_logs.call_args_list[0][1]['blocks'][0]['text']['text'])

    mock_post_logs.call_args_list[0][1]['token'] == '1234'
    mock_post_logs.call_args_list[0][1]['channel'] == 'C1234'

    del trainer


def test_slack_logger_hparams():
    pytest.importorskip('slack_sdk')
    os.environ['SLACK_LOGGING_API_KEY'] = str('1234')
    os.environ['SLACK_LOGGING_CHANNEL_ID'] = 'C1234'

    slack_logger = SlackLogger(
        formatter_func=(lambda data, **kwargs: [{
            'type': 'section',
            'text': {
                'type': 'mrkdwn',
                'text': f'*{k}:* {v}'
            }
        } for k, v in data.items()]),
        interval_in_seconds=None,
    )

    # Patch for hparams
    with patch('slack_sdk.WebClient.chat_postMessage') as mock_post_hparams:
        trainer = Trainer(
            model=SimpleModel(),
            train_dataloader=DataLoader(RandomClassificationDataset()),
            train_subset_num_batches=2,
            max_duration='1ep',
            loggers=[slack_logger],
        )
    assert (mock_post_hparams.call_count > 0)


@pytest.mark.parametrize('time_interval', [None, 1])
def test_slack_logger_time_interval(time_interval: Optional[int]):
    pytest.importorskip('slack_sdk')
    os.environ['SLACK_LOGGING_API_KEY'] = str('1234')
    os.environ['SLACK_LOGGING_CHANNEL_ID'] = 'C1234'

    slack_logger = SlackLogger(
        formatter_func=(lambda data, **kwargs: [{
            'type': 'section',
            'text': {
                'type': 'mrkdwn',
                'text': f'*{k}:* {v}'
            }
        } for k, v in data.items()]),
        interval_in_seconds=time_interval,
    )

    # Patch for hparams
    with patch('slack_sdk.WebClient.chat_postMessage') as mock_slack:
        trainer = Trainer(
            model=SimpleModel(),
            train_dataloader=DataLoader(RandomClassificationDataset()),
            train_subset_num_batches=2,
            max_duration='5ep',
            loggers=[slack_logger],
        )
        if time_interval is None:
            assert (mock_slack.call_count > 0)
        else:
            assert (mock_slack.call_count == 0)
        if time_interval is not None:
            time.sleep(time_interval)

        # Log metrics immediately if no time_interval.
        # Otherwise log hparams from previous step.
        trainer.fit()
        if time_interval is None:
            assert (mock_slack.call_count > 2)
        else:
            assert (mock_slack.call_count == 2)


@pytest.mark.parametrize('max_logs', [50, 1])
def test_slack_logger_max_num_logs(max_logs: int):
    pytest.importorskip('slack_sdk')
    os.environ['SLACK_LOGGING_API_KEY'] = str('1234')
    os.environ['SLACK_LOGGING_CHANNEL_ID'] = 'C1234'

    slack_logger = SlackLogger(
        formatter_func=(lambda data, **kwargs: [{
            'type': 'section',
            'text': {
                'type': 'mrkdwn',
                'text': f'*{k}:* {v}'
            }
        } for k, v in data.items()]),
        interval_in_seconds=1,
        max_logs_per_message=max_logs,
    )

    # Patch for hparams
    with patch('slack_sdk.WebClient.chat_postMessage') as mock_slack:
        Trainer(
            model=SimpleModel(),
            train_dataloader=DataLoader(RandomClassificationDataset()),
            train_subset_num_batches=2,
            max_duration='5ep',
            loggers=[slack_logger],
        )
        if max_logs == 1:
            # Since time interval is 1s, will not log
            assert (mock_slack.call_count > 0)
        else:
            # But if max num of logs reached, disregard time interval and log
            assert (mock_slack.call_count == 0)
