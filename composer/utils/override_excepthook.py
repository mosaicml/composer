# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Excepthook override for MosaicML logging"""

import json
import os
import sys
from datetime import datetime

from rich.console import Console
from rich.traceback import Traceback

from composer.loggers.mosaicml_logger import MOSAICML_GPU_EXCEPTION_LOG_FILE_PREFIX_ENV_VAR


def override_excepthook():
    """Override default except hook to log exceptions in a JSONL file and stderr."""
    def log_exception(exc_type, exc_value, tb):
        console = Console(file=sys.stderr, force_terminal=True)
        console.print(Traceback.from_exception(exc_type, exc_value, tb))
        if os.environ.get('LOCAL_RANK') is not None and os.environ.get('NODE_RANK') is not None and os.environ.get('RESUMPTION_ID') is not None:
            exception = {
                'asctime': datetime.now(),
                'gpu_rank': os.environ.get('LOCAL_RANK'),
                'node_rank': os.environ.get('NODE_RANK'),
                'resumption_id': os.environ.get('RESUMPTION_ID'),
                'exception_class': exc_type.__name__, 
                'message': str(exc_value),
                'traceback': Traceback.from_exception(exc_type, exc_value, tb)
            }
            with open(f"{MOSAICML_GPU_EXCEPTION_LOG_FILE_PREFIX_ENV_VAR}{os.environ.get('LOCAL_RANK')}.jsonl", "a") as log_file:
                json.dump(exception, log_file)
                log_file.write("\n")
    sys.excepthook = log_exception

if hasattr(sys, "excepthook"):
    override_excepthook()
