# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log metrics to slack, using Slack postMessage api."""

from __future__ import annotations

import itertools
import logging
import os
import re
import time
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, Union

from composer.core.time import Time, TimeUnit
from composer.loggers.logger import Logger
from composer.loggers.logger_destination import LoggerDestination
from composer.utils import MissingConditionalImportError

if TYPE_CHECKING:
    from composer.core import State

log = logging.getLogger(__name__)

__all__ = ['SlackLogger']


class SlackLogger(LoggerDestination):
    """Log metrics to slack, using Slack's postMessage api - https://api.slack.com/methods/chat.postMessage.

    First export 2 environment variable to use this logger.
    1. SLACK_LOGGING_API_KEY: To get app credentials, follow tutorial here - https://api.slack.com/tutorials/tracks/posting-messages-with-curl?app_id_from_manifest=A053W1QCEF2.
    2. SLACK_LOGGING_CHANNEL_ID: Channel id to send the message (Open slack channel in web browser to look this up).

    Next write script to output metrics / hparams / traces to slack channel. See example below.

    .. code-block:: python

        trainer = Trainer(
            model=mnist_model(num_classes=10),
            train_dataloader=train_dataloader,
            max_duration='2ep',
            algorithms=[
                LabelSmoothing(smoothing=0.1),
                CutMix(alpha=1.0),
                ChannelsLast(),
            ],
            loggers=[
                SlackLogger(
                    formatter_func=(lambda data: [{
                        'type': 'section',
                        'text': {
                            'type': 'mrkdwn',
                            'text': f'*{k}:* {v}'
                        }
                    } for k, v in data.items()]),
                    include_keys=['loss/train/total'],
                    interval_in_seconds=1
                ),
            ],
        )

        trainer.fit()

    Args:
        formatter_func ((...) -> Any | None): A formatter function that returns list of blocks to be sent to slack.
        include_keys (Sequence[str]): A sequence of metric/logs/traces keys to include in the message.
        log_interval: (int | str | Time): How frequently to log. (default: ``'1ba'``)
        max_logs_per_message (int)(default:50): Maximum number of logs to send in a single message. Note that no more than 50 items are allowed to send in a single message.
        If more than 50 items are stored in buffer, the message flushed without waiting the full time interval.
    """

    def __init__(
        self,
        include_keys: Sequence[str] = (),
        formatter_func: Optional[Callable[..., list[dict[str, Any]]]] = None,
        log_interval: Union[int, str, Time] = '1ba',
        max_logs_per_message: int = 50,
        slack_logging_api_key: Optional[str] = None,
        channel_id: Optional[str] = None,
    ) -> None:
        try:
            import slack_sdk
            self.client = slack_sdk.WebClient()
            del slack_sdk
        except ImportError as e:
            raise MissingConditionalImportError('slack_logger', 'slack_sdk', None) from e

        self.slack_logging_api_key = os.environ.get(
            'SLACK_LOGGING_API_KEY',
            None,
        ) if slack_logging_api_key is None else slack_logging_api_key
        self.channel_id = os.environ.get('SLACK_LOGGING_CHANNEL_ID', None) if channel_id is None else channel_id

        if self.slack_logging_api_key is None:
            print('WARNING: SLACK_LOGGING_API_KEY must be set as environment variable')
        if self.channel_id is None:
            print('WARNING: SLACK_LOGGING_CHANNEL_ID must be set as environment variable')

        self.formatter_func = formatter_func

        if len(include_keys) == 0:
            print('WARNING: The slack logger `include_keys` argument must be a non-empty list of strings.')
        # Create a regex of all keys to include
        self.regex_all_keys = '(' + ')|('.join(include_keys) + ')'

        self.log_interval = Time.from_input(log_interval, TimeUnit.EPOCH)
        if self.log_interval.unit not in (TimeUnit.EPOCH, TimeUnit.BATCH):
            raise ValueError('The `slack logger log_interval` argument must have units of EPOCH or BATCH.')

        self.log_dict, self.last_log_time = {}, time.time()
        self.max_logs_per_message = min(max_logs_per_message, 50)

    def _log_to_buffer(
        self,
        data: dict[str, Any],
        **kwargs,  # can be used to pass additional arguments to the formatter function (eg for headers)
    ):
        """Flush the buffer to slack if the buffer size exceeds max_logs_per_message.

        Buffer will replace existing keys with updated values if keys exist.
        Otherwise, add new key-value pairs.
        If max_logs_per_message is exceeded, flush buffer.
        Otherwise, wait for the next log_interval (batch end or epoch end) to flush the buffer.
        """
        filtered_data = {k: v for k, v in data.items() if re.match(self.regex_all_keys, k) is not None}
        self.log_dict.update(filtered_data)

        if len(self.log_dict.keys()) >= self.max_logs_per_message:
            self._flush_logs_to_slack(**kwargs)

    def _default_log_bold_key_normal_value_pair_with_header(
        self,
        data: dict[str, Any],
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Default formatter function if no formatter func is specified.

        This function will:
        1. Log the key-value pairs in bold (key) and normal (value) text.
        2. When logging metrics, set the step number as the header of the section.

        Args:
            data (dict[str, Any]): Data to be logged.
            **kwargs: Additional arguments to be passed to the formatter function
            (Only supports "header" argument now)

        Returns:
            list[dict[str, str]]: list of blocks to be sent to Slack.
        """
        blocks = [{'type': 'section', 'text': {'type': 'mrkdwn', 'text': f'*{k}:* {v}'}} for k, v in data.items()]
        if len(blocks) > 0 and 'header' in kwargs:
            header = kwargs['header']
            blocks.append({'type': 'header', 'text': {'type': 'plain_text', 'text': f'{header}'}})

        return blocks

    def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        self._log_to_buffer(data=metrics, header=step)

    def log_hyperparameters(self, hyperparameters: dict[str, Any]):
        self._log_to_buffer(data=hyperparameters)

    def log_traces(self, traces: dict[str, Any]):
        self._log_to_buffer(data=traces)

    def epoch_end(self, state: State, logger: Logger) -> None:
        cur_epoch = int(state.timestamp.epoch)  # epoch gets incremented right before EPOCH_END
        unit = self.log_interval.unit

        if unit == TimeUnit.EPOCH and (cur_epoch % int(self.log_interval) == 0 or cur_epoch == 1):
            self._flush_logs_to_slack()

    def batch_end(self, state: State, logger: Logger) -> None:
        cur_batch = int(state.timestamp.batch)
        unit = self.log_interval.unit
        if unit == TimeUnit.BATCH and (cur_batch % int(self.log_interval) == 0 or cur_batch == 1):
            self._flush_logs_to_slack()

    def close(self, state: State, logger: Logger) -> None:
        self._flush_logs_to_slack()

    def _flush_logs_to_slack(self, **kwargs) -> None:
        """Flush buffered metadata to MosaicML.

        Format slack messages through rich message layouts created using Slack Blocks Kit.
        See documentation here: https://api.slack.com/messaging/composing/layouts.
        """
        inx = 0
        while inx < len(self.log_dict.keys()):
            max_log_entries_dict = dict(itertools.islice(self.log_dict.items(), inx, inx + self.max_logs_per_message))
            self._format_and_send_blocks_to_slack(max_log_entries_dict, **kwargs)
            inx += self.max_logs_per_message

        self.log_dict = {}  # reset log_dict

    def _format_and_send_blocks_to_slack(
        self,
        log_entries: dict[str, Any],
        **kwargs,
    ):
        blocks = self.formatter_func(
            log_entries,
            **kwargs,
        ) if self.formatter_func is not None else self._default_log_bold_key_normal_value_pair_with_header(
            log_entries,
            **kwargs,
        )
        try:
            channel_id = self.channel_id
            slack_logging_key = self.slack_logging_api_key
            if channel_id is None:
                raise TypeError('SLACK_LOGGING_CHANNEL_ID cannot be None.')
            if slack_logging_key is None:
                raise TypeError('SLACK_LOGGING_API_KEY cannot be None')
            self.client.chat_postMessage(
                token=f'{self.slack_logging_api_key if self.slack_logging_api_key else ""}',
                channel=channel_id,
                blocks=blocks,
                text=f'Logged {len(log_entries)} items to Slack',
            )
        except Exception as e:
            log.error(f'Error logging to Slack: {e}')
