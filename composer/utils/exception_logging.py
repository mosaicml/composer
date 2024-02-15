# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Overrides the excepthook to log GPU exceptions to stderr and to a file."""

import datetime
import logging
import os
import sys
import traceback
from types import TracebackType

log = logging.getLogger(__name__)

EXCEPTION_FILE_PREFIX  = "gpu_exception_"

def log_exceptions_to_file():
    """Logs GPU exceptions to separate files."""
    exception_handler = sys.excepthook

    def write_to_file(type: type[BaseException], value: BaseException, tb: TracebackType | None ):
        try:
            gpu_rank = os.getenv('LOCAL_RANK')
            if not gpu_rank:
                log.error("LOCAL_RANK env var not set, cannot log GPU exceptions.")
            exception_log_file = f'{EXCEPTION_FILE_PREFIX}{gpu_rank}.txt'
            with open(exception_log_file, "a") as fh:
                fh.write(f'Exception thrown at {datetime.datetime.now()}\n')
                fh.write(f"{value.__class__.__name__}: {value}\n")
                fh.write("".join(traceback.format_exception(type, value, tb)))
        except:
            pass
        exception_handler(type, value, tb)
    sys.excepthook = write_to_file

if hasattr(sys, "excepthook"):
    log_exceptions_to_file()
