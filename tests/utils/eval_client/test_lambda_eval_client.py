# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0
import pytest

from composer.utils import LambdaEvalClient


@pytest.mark.remote
@pytest.mark.parametrize(
    'code, result',
    [
        ['def add_1(x):\n    return x + 1', True],
        ['def add_1(x):\n    return y + 1', False],
        ['def add_1(x):\n    while True:\n        x += 1', False],
        ['def add_1(x): return x + 2', False],
    ],
)
def test_lambda_invoke(code, result):
    """Test invocation function for LambdaEvalClient with code that succeeds, fails compilation, times out, and is incorrect.
    """
    eval_client = LambdaEvalClient()
    assert eval_client.invoke({'code': code, 'input': '(1,)', 'output': '2', 'entry_point': 'add_1'}) == result
