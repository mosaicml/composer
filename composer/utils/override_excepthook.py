# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Excepthook override for MosaicML logging."""

import json
import os
import sys
import time
import warnings
from datetime import datetime
from io import StringIO

from rich.console import Console
from rich.traceback import Traceback

MOSAICML_GPU_EXCEPTION_LOG_FILE_PREFIX_ENV_VAR = 'MOSAICML_GPU_EXCEPTION_LOG_FILE_PREFIX'
MOSAICML_LOG_DIR_ENV_VAR = 'MOSAICML_LOG_DIR'


def override_excepthook():
    """Override default except hook to log exceptions in a JSONL file and stderr."""

    def log_exception(exc_type, exc_value, tb):
        warnings.warn('in override excepthook log exception')
        console = Console(file=sys.stderr, force_terminal=True)
        console.print(Traceback.from_exception(exc_type, exc_value, tb))

        string_io = StringIO()
        string_console = Console(file=string_io, force_terminal=True)
        string_console.print(Traceback.from_exception(exc_type, exc_value, tb))
        traceback_string = string_io.getvalue()
        log_file_prefix = os.environ.get(MOSAICML_GPU_EXCEPTION_LOG_FILE_PREFIX_ENV_VAR)
        local_rank = os.environ.get('LOCAL_RANK')
        if local_rank is not None and os.environ.get('NODE_RANK') is not None and os.environ.get(
                'RESUMPTION_ID') is not None and log_file_prefix is not None:
            exception = {
                'asctime': time.asctime(datetime.now().timetuple()),
                'gpu_rank': local_rank,
                'node_rank': os.environ.get('NODE_RANK'),
                'resumption_id': os.environ.get('RESUMPTION_ID'),
                'exception_class': exc_type.__name__,
                'message': str(exc_value),
                'traceback': traceback_string
            }
            warnings.warn(
                f'Logging exception to {os.environ.get(MOSAICML_LOG_DIR_ENV_VAR)}/{log_file_prefix}{local_rank}.jsonl')
            with open(f'{os.environ.get(MOSAICML_LOG_DIR_ENV_VAR)}/{log_file_prefix}{local_rank}.jsonl',
                      'a') as log_file:
                json.dump(exception, log_file)
                log_file.write('\n')

    sys.excepthook = log_exception
    # training_script = os.environ.get('TRAINING_SCRIPT')
    # if training_script is not None:
    #     with open(training_script) as f:
    #         code = compile(f.read(), training_script, 'exec')
    #         exec(code, globals(), locals())
    # else:
    #     raise ValueError('TRAINING_SCRIPT environment variable not set')
