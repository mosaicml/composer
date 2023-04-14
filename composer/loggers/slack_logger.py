# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log metrics to slack, using Slack postMessage api - https://api.slack.com/methods/chat.postMessage.

First export 2 environment variable to use this logger.
1. SLACK_LOGGING_API_KEY: To get app credentials, follow tutorial here - https://api.slack.com/tutorials/tracks/posting-messages-with-curl?app_id_from_manifest=A053W1QCEF2
2. SLACK_LOGGING_CHANNEL_ID: Channel id to send the message (Open slack channel in web browser to look this up)

Next write script to output metrics / hparams / traces to slack channel. See example below:

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
            log_metrics_config=SlackLogConfig(
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
            log_hparams_config=SlackLogConfig(),
        )
    ],
)

trainer.fit()
"""

import os
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from composer.loggers.logger_destination import LoggerDestination
from composer.utils import MissingConditionalImportError


class SlackLogConfig:
    """Log config for slack logger, including formatter function, keys to include, and interval in seconds.

    formatter_func ((...) -> Any | None): A formatter function that returns list of blocks to be sent to slack
    include_keys (Set[str]): A set of metric/logs/traces keys to include in the message. If None, no keys are included.
    interval_in_seconds (int): Interval in seconds to send logs to slack. If None, message is sent after every log call.
    Note that no more than 50 items are allowed to send in a single message.
    """

    def __init__(
        self,
        formatter_func: Optional[Callable[..., List[Dict[str, Any]]]] = None,
        include_keys: Optional[Set[str]] = None,
        interval_in_seconds: Optional[int] = None,
    ) -> None:
        self.formatter_func = formatter_func
        self.include_keys = include_keys
        self.interval_in_seconds = interval_in_seconds


class SlackLogger(LoggerDestination):
    """Logger to log metrics to slack.

    This logger uses Slack's postMessage api - https://api.slack.com/methods/chat.postMessage.

    Args:
        log_metrics_config (SlackLogConfig): Configuration for logging metrics to slack.
        log_traces_config (SlackLogConfig): Configuration for logging traces to slack.
        log_hparams_config (SlackLogConfig): Configuration for logging hyperparameters to slack.
    """

    def __init__(
        self,
        log_metrics_config: Optional[SlackLogConfig] = None,
        log_traces_config: Optional[SlackLogConfig] = None,
        log_hparams_config: Optional[SlackLogConfig] = None,
    ) -> None:
        try:
            from slack_sdk import WebClient
            del slack_sdk
        except ImportError as e:
            raise MissingConditionalImportError('health_checker', 'slack_sdk', None) from e

        self.client = WebClient()
        self.slack_logging_api_key = os.environ.get('SLACK_LOGGING_API_KEY', None)
        self.channel_id = os.environ.get('SLACK_LOGGING_CHANNEL_ID', None)

        if self.slack_logging_api_key is None or self.channel_id is None:
            raise RuntimeError(
                'SLACK_LOGGING_API_KEY and SLACK_LOGGING_CHANNEL_ID must be set as environment variables')

        self.log_metrics_config = log_metrics_config
        self.log_traces_config = log_traces_config
        self.log_hparams_config = log_hparams_config

        # Maintain logger state
        now = time.time()
        self.logs_since_last_log, self.last_log_time = [], now
        self.traces_since_last_log, self.last_trace_time = [], now
        self.hparams_since_last_log, self.last_hparams_time = [], now

    # Rich message layouts can be created using Slack Blocks Kit.
    # See documentation here: https://api.slack.com/messaging/composing/layouts
    def _log_to_slack(
        self,
        data: Dict[str, Any],
        last_time_flushed: float,
        blocks_to_log: List[Dict[str, Any]],
        config: SlackLogConfig,
        **kwargs,  # can be used to pass additional arguments to the formatter function (eg for headers)
    ) -> Tuple[float, List[Dict[str, Any]]]:
        filtered_data = {k: v for k, v in data.items() if k in config.include_keys} if config.include_keys else data
        blocks = config.formatter_func(
            filtered_data, **
            kwargs) if config.formatter_func else self._default_log_bold_key_normal_value_pair_with_header(
                filtered_data, **kwargs)

        updated_blocks_to_log = blocks + blocks_to_log

        if len(updated_blocks_to_log) > 50:
            raise RuntimeError('Cannot send more than 50 items in a single slack message')

        if len(updated_blocks_to_log) > 0 and self.channel_id and (
                config.interval_in_seconds is None or time.time() - last_time_flushed >= config.interval_in_seconds):
            self.client.chat_postMessage(token=f'{self.slack_logging_api_key}',
                                         channel=self.channel_id,
                                         blocks=blocks_to_log)
            return time.time(), []
        else:
            return last_time_flushed, updated_blocks_to_log

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
        if self.log_metrics_config is None:
            return
        self.last_log_time, self.logs_since_last_log = self._log_to_slack(data=metrics,
                                                                          last_time_flushed=self.last_log_time,
                                                                          blocks_to_log=self.logs_since_last_log,
                                                                          config=self.log_metrics_config,
                                                                          header=step)

    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        if self.log_hparams_config is None:
            return
        self.last_hparams_time, self.hparams_since_last_log = self._log_to_slack(
            data=hyperparameters,
            last_time_flushed=self.last_hparams_time,
            blocks_to_log=self.hparams_since_last_log,
            config=self.log_hparams_config,
        )

    def log_traces(self, traces: Dict[str, Any]):
        if self.log_traces_config is None:
            return
        self.last_trace_time, self.traces_since_last_log = self._log_to_slack(
            data=traces,
            last_time_flushed=self.last_trace_time,
            blocks_to_log=self.traces_since_last_log,
            config=self.log_traces_config,
        )
