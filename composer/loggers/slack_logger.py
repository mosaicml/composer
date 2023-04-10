from typing import Any, Callable, Dict, Optional

from slack_sdk import WebhookClient

from composer.loggers.logger_destination import LoggerDestination


class SlackLogger(LoggerDestination):
    """Log metrics to slack, using `Slack Webhook URL <https://api.slack.com/messaging/webhooks>`
    
    .. note::
        See example instantiating of SlackLogger in `examples/slack_log_metrics.py`
        Webhook URL will be passed as an environment variable.
        Formatter function can be defined at callsite or default formatter function will be applied if not provided.
         
    Args:
        webhook_url (str): Slack Webhook URL
        log_metrics_formatter_func ((...) -> Any | None): A formatter function that returns list of blocks to be sent to slack
    """

    def __init__(
        self,
        webhook_url: str,
        log_metrics_formatter_func: Optional[Callable] = None,
    ) -> None:
        self.webhook_url = webhook_url
        self.logged_metrics: Dict[str, float] = {}
        self.log_metrics_formatter_func = log_metrics_formatter_func if log_metrics_formatter_func else self._default_log_bold_key_normal_value_pair_with_header

    # Rich message layouts can be created using Slack Blocks Kit.
    # See documentation here: https://api.slack.com/messaging/composing/layouts
    def _log_to_slack(
        self,
        data : Dict[str, Any],
        formatter_fun: Callable,
        **kwargs,
    ):
        client = WebhookClient(url=self.webhook_url)
        blocks = formatter_fun(data, **kwargs)
        client.send(blocks=blocks)
    
    def _default_log_bold_key_normal_value_pair_with_header(self, data: Dict[str, Any], header: Optional[str] = None):
        blocks = []
        if header:
            blocks.append(
                {
                    "type": "header", "text": {"type": "plain_text", "text": f"{header}"}
                }
            )
        blocks += [
            {
                "type": "section", "text": {"type": "mrkdwn", "text": f"*{k}:* {v}"}
            }
            for k, v in data.items()
        ]
        return blocks
        
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        self.logged_metrics.update(metrics)
        self._log_to_slack(metrics, self.log_metrics_formatter_func, header=step)
        
    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        pass

    def log_traces(self, traces: Dict[str, Any]):
        pass

