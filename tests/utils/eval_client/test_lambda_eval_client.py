# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0
import pytest

from composer.utils import LambdaEvalClient


@pytest.mark.remote
@pytest.mark.parametrize(
    'code, result, language',
    [
        ['def add_1(x):\n    return x + 1', True, 'python'],
        ['def add_1(x):\n    return y + 1', False, 'python'],
        ['def add_1(x):\n    while True:\n        x += 1', False, 'python'],
        ['def add_1(x): return x + 2', False, 'python'],
        ['int add_1(int x) {\n\treturn x + 1;\n}', True, 'c++'],
        ['int add_1(int x) {\n\treturn y + 1;\n}', False, 'c++'],
        ['int add_1(int x) {\n\twhile (true) {\n\t\tx += 1;\n\t}\n}', False, 'c++'],
        ['int add_1(int x) {\n\treturn x + 2;\n}', False, 'c++'],
        ['int add_1(int x) {\n\treturn x + 1;\n}', True, 'c'],
        ['int add_1(int x) {\n\treturn y + 1;\n}', False, 'c'],
        ['int add_1(int x) {\n\twhile (true) {\n\t\tx += 1;\n\t}\n}', False, 'c'],
        ['int add_1(int x) {\n\treturn x + 2;\n}', False, 'c'],
        ['function add_1(x) {\n\treturn x+1;\n}', True, 'javascript'],
        ['function add_1(x) {\n\treturn y+1;\n}', False, 'javascript'],
        ['function add_1(x) {\n\twhile (true) {\n\t\tx += 1;\n\t}\n}', False, 'javascript'],
        ['function add_1(x) {\n\treturn x+2;\n}', False, 'javascript'],
    ],
)
def test_lambda_invoke(code, result, language):
    """Test invocation function for LambdaEvalClient with code that succeeds, fails compilation, times out, and is incorrect in C, C++, Python, JS.
    """
    eval_client = LambdaEvalClient()
    input = '(1,)' if language == 'python' else '1'
    assert eval_client.invoke([[[{
        'code': code,
        'input': input,
        'output': '2',
        'entry_point': 'add_1',
        'language': language
    }]]]) == [[[result]]]
