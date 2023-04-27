# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Sequence
from unittest.mock import patch

import pytest
from torch.utils.data import DataLoader

from composer.loggers.slack_logger import SlackLogger
from composer.trainer import Trainer
from tests.common import RandomClassificationDataset, SimpleModel


@pytest.mark.parametrize('include_keys', [['loss*']])
@pytest.mark.parametrize('interval', ['1ba', '1ep'])
def test_slack_logger_metrics(include_keys: Sequence[str], interval: str):
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
        log_interval=interval,
    )

    with patch('slack_sdk.WebClient.chat_postMessage') as mock_slack:
        trainer = Trainer(
            model=SimpleModel(),
            train_dataloader=DataLoader(RandomClassificationDataset()),
            train_subset_num_batches=2,
            max_duration=interval,
            loggers=[slack_logger],
        )

    # Log interval is 1 epoch, so no logs should be sent before fit
    assert mock_slack.call_count == 0

    with patch('slack_sdk.WebClient.chat_postMessage') as mock_slack:
        trainer.fit()

    assert mock_slack.call_count == 1
    metrics_key = 'loss/train/total'

    assert (metrics_key in mock_slack.call_args_list[0][1]['blocks'][0]['text']['text'])
    assert mock_slack.call_args_list[0][1]['token'] == '1234'
    assert mock_slack.call_args_list[0][1]['channel'] == 'C1234'

    del trainer
