# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log metrics to slack, using Slack postMessage api."""
import os
import time
from typing import Any, Callable, Dict, List, Optional, Set

from composer.loggers.logger_destination import LoggerDestination
from composer.utils import MissingConditionalImportError


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
                    formatter_func=(lambda data, **kwargs: [{
                        'type': 'section',
                        'text': {
                            'type': 'mrkdwn',
                            'text': f'*{k}:* {v}'
                        }
                    } for k, v in data.items()]),
                    include_keys={'loss/train/total'},
                    interval_in_seconds=1
                ),
            ],
        )

        trainer.fit()

    Args:
        formatter_func ((...) -> Any | None): A formatter function that returns list of blocks to be sent to slack.
        include_keys (Set[str]): A set of metric/logs/traces keys to include in the message. If None, all keys are included.
        interval_in_seconds (int): Interval in seconds to send logs to slack. If None, message is sent after every log call.
        max_logs_per_message (int)(default:50): Maximum number of logs to send in a single message. Note that no more than 50 items are allowed to send in a single message.
        If more than 50 items are stored in buffer, the message flushed without waiting the full time interval.
    """

    def __init__(
        self,
        formatter_func: Optional[Callable[..., List[Dict[str, Any]]]] = None,
        include_keys: Optional[Set[str]] = None,
        interval_in_seconds: Optional[int] = None,
        max_logs_per_message: int = 50,
    ) -> None:
        try:
            import slack_sdk
            self.client = slack_sdk.WebClient()
            del slack_sdk
        except ImportError as e:
            raise MissingConditionalImportError('slack_logger', 'slack_sdk', None) from e

        self.slack_logging_api_key = os.environ.get('SLACK_LOGGING_API_KEY', None)
        self.channel_id = os.environ.get('SLACK_LOGGING_CHANNEL_ID', None)

        if self.slack_logging_api_key is None:
            raise RuntimeError('SLACK_LOGGING_API_KEY must be set as environment variable')
        if self.channel_id is None:
            raise RuntimeError('SLACK_LOGGING_CHANNEL_ID must be set as environment variable')

        self.formatter_func = formatter_func
        self.include_keys = include_keys
        self.interval_in_seconds = interval_in_seconds

        self.logs, self.last_log_time = [], time.time()
        self.max_logs_per_message = min(max_logs_per_message, 50)

    # Rich message layouts can be created using Slack Blocks Kit.
    # See documentation here: https://api.slack.com/messaging/composing/layouts
    def _log_to_slack(
            self,
            data: Dict[str, Any],
            **kwargs,  # can be used to pass additional arguments to the formatter function (eg for headers)
    ):
        filtered_data = {k: v for k, v in data.items() if k in self.include_keys
                        } if self.include_keys is not None else data
        blocks = self.formatter_func(
            filtered_data, **
            kwargs) if self.formatter_func is not None else self._default_log_bold_key_normal_value_pair_with_header(
                filtered_data, **kwargs)

        self.logs += blocks
        now = time.time()

        if len(self.logs) > 0 and self.channel_id is not None and (
                len(self.logs) >= self.max_logs_per_message or self.interval_in_seconds is None or
                now - self.last_log_time >= self.interval_in_seconds):
            self.client.chat_postMessage(token=f'{self.slack_logging_api_key}',
                                         channel=self.channel_id,
                                         blocks=self.logs[:self.max_logs_per_message],
                                         text=str(self.logs[:self.max_logs_per_message]))
            self.last_log_time = now
            self.logs = self.logs[self.max_logs_per_message:]

    def _default_log_bold_key_normal_value_pair_with_header(self, data: Dict[str, Any],
                                                            **kwargs) -> List[Dict[str, Any]]:
        """Default formatter function if no formatter func is specified.

        This function will:
        1. Log the key-value pairs in bold (key) and normal (value) text.
        2. When logging metrics, set the step number as the header of the section.

        Args:
            data (Dict[str, Any]): Data to be logged.
            **kwargs: Additional arguments to be passed to the formatter function (eg header)

        Returns:
            List[Dict[str, str]]: List of blocks to be sent to Slack.
        """
        blocks = [{'type': 'section', 'text': {'type': 'mrkdwn', 'text': f'*{k}:* {v}'}} for k, v in data.items()]
        if len(blocks) > 0 and 'header' in kwargs:
            header = kwargs['header']
            blocks.append({'type': 'header', 'text': {'type': 'plain_text', 'text': f'{header}'}})

        return blocks

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        self._log_to_slack(data=metrics, header=step)

    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        self._log_to_slack(data=hyperparameters)

    def log_traces(self, traces: Dict[str, Any]):
        self._log_to_slack(data=traces)
