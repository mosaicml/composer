import builtins
import logging
import os
import sys

from json_log_formatter import JsonLogFormatter


def json_print(*args, **kwargs):
    log = logging.getLogger(__name__)
    if not log.handlers:
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setFormatter(JsonLogFormatter(rank=os.getenv('RANK')))
        log.addHandler(stderr_handler)
    message = ' '.join(str(arg) for arg in args)
    log.info(message, **kwargs)

builtins.print = json_print
