# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Eval Client base class and implementations."""

from composer.utils.eval_client.eval_client import EvalClient
from composer.utils.eval_client.lambda_eval_client import LambdaEvalClient
from composer.utils.eval_client.local_eval_client import LocalEvalClient

__all__ = [
    'EvalClient',
    'LambdaEvalClient',
    'LocalEvalClient',
]
