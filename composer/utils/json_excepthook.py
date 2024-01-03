import logging
import os
import sys

from composer.utils.json_log_formatter import JsonLogFormatter


def override_excepthook(rank=None):
    log = logging.getLogger(__name__)
    if not log.handlers:
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setFormatter(JsonLogFormatter(rank=os.getenv('RANK')))
        log.addHandler(stderr_handler)

    def log_exception(type, value, tb):
        log.exception('An exception occurred', exc_info=(type, value, tb))

    sys.excepthook = log_exception

if hasattr(sys, "excepthook"):
    override_excepthook()
