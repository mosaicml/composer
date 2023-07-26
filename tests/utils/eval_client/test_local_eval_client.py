# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0
from composer.utils import LocalEvalClient


def test_local_invoke():
    """Test invocation function for LocalEvalClient with code that succeeds, fails compilation, times out, and is incorrect.
    """
    eval_client = LocalEvalClient()
    assert eval_client.invoke({
        'code': 'def add_1(x):\n    return x + 1',
        'input': '(1,)',
        'output': '2',
        'entry_point': 'add_1'
    })
    assert not eval_client.invoke({
        'code': 'def add_1(x):\n    return y + 1',
        'input': '(1,)',
        'output': '2',
        'entry_point': 'add_1'
    })
    assert not eval_client.invoke({
        'code': 'def add_1(x):\n    while True:\n        x += 1',
        'input': '(1,)',
        'output': '2',
        'entry_point': 'add_1',
    })
    assert not eval_client.invoke({
        'code': 'def add_1(x): return x + 2',
        'input': '(1,)',
        'output': '2',
        'entry_point': 'add_1'
    })
