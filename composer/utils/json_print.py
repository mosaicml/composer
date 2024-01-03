import builtins
import logging
import os
import sys

from composer.utils.json_log_formatter import JsonLogFormatter


def json_print(*args, **kwargs):
    log = logging.getLogger(__name__)
    if not log.handlers:
        stderr_handler = logging.StreamHandler(sys.stderr)
        rank = os.getenv('RANK') if os.getenv('RANK') else 'unknown'
        stderr_handler.setFormatter(JsonLogFormatter(rank=rank))
        log.addHandler(stderr_handler)
    message = ' '.join(str(arg) for arg in args)
    log.info(message, **kwargs)

builtins.print = json_print
