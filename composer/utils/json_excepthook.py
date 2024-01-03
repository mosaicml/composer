import logging
import sys

from composer.utils.json_log_formatter import JsonLogFormatter


def override_excepthook():
    log = logging.getLogger(__name__)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(JsonLogFormatter())
    log.addHandler(stderr_handler)

    def log_exception(type, value, tb):
        log.exception()
        sys.__excepthook__(type, value, tb)

    sys.excepthook = log_exception

if hasattr(sys, "excepthook"):
    override_excepthook()
