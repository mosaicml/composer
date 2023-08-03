# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Eval client for local evaluation."""
import logging
import multiprocessing
import os
import subprocess
import types
from typing import Dict, List

from composer.utils.eval_client.eval_client import EvalClient

__all__ = ['LocalEvalClient']
log = logging.getLogger(__name__)

TIMEOUT = 5  # in seconds


class LocalEvalClient(EvalClient):
    """Utility for creating a client for and invoking local evaluations."""

    def invoke(self, payload: List[List[List[Dict[str, str]]]]) -> List[List[List[bool]]]:
        """Invoke a batch of provided payloads for code evaluations."""
        return [[[self.invoke_helper(test_case)
                  for test_case in generation_group]
                 for generation_group in prompt_group]
                for prompt_group in payload]

    def invoke_helper(self, payload: Dict[str, str]) -> bool:
        """Invoke a provided dictionary payload to a multiprocessing subprocess that performs code eval."""
        ret = multiprocessing.Value('b', 0)  # Store result of test case in shared memory
        p = multiprocessing.Process(target=self.update_offline_helper,
                                    args=(payload['code'], payload['input'], payload['output'], payload['entry_point'],
                                          payload['language'], ret))  # Evaluate test case in an independent subprocess
        p.start()
        p.join(TIMEOUT)  # wait for timeout to terminate
        p.terminate()
        return bool(ret.value)

    def update_offline_helper(self, code_gen: str, test_input: str, test_output: str, entry_point: str, language: str,
                              val: multiprocessing.Value):  # type: ignore
        """Helper function to evaluate test case in a subprocess.

        This function compiles the code generation,
        and runs the function from the entry point, before running the test input through the function and
        checking it against the test output.

        Args:
            code_gen (str): The code generation to be evaluated.
            test_input (str): The input of the test case
            test_output (str): The output of the test case
            entry_point (str): The name of the function to call
            language (str): The language of the code generation
            val (multiprocessing.Value): The value in which to save the final value of the test case
        """
        if language == 'python':
            mod = types.ModuleType('test_module')
            val.value = 0
            result = None
            expected_result = None
            try:
                exec(code_gen, mod.__dict__)

                result = mod.__dict__[entry_point](*eval(test_input))
                syntax_compiled = True
                try:
                    expected_result = eval(test_output)
                except:
                    expected_result = test_output

            except Exception as _:
                syntax_compiled = False  # don't print out warning b/c this can happen frequently

            if syntax_compiled:
                val.value = int(result == expected_result)
            else:
                val.value = 0
        elif language == 'c++':
            prefix = '#include <iostream>\nusing namespace std;\n'
            code_gen = prefix + code_gen

            ending = '\nint main() {\n    cout << (' + entry_point + '(' + test_input + ') == ' + test_output + ');\n    return 0;\n}'
            code_gen = code_gen + ending
            with open('test_code.cpp', 'w') as f:
                f.write(code_gen)

            if subprocess.run(['g++', 'test_code.cpp', '-o', 'test_code'],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE).returncode == 0:
                run_process = subprocess.run('./test_code', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                output = run_process.stdout.decode()
            else:
                output = '0'

            if 'test_code.cpp' in os.listdir():
                os.remove('test_code.cpp')
            if 'test_code' in os.listdir():
                os.remove('test_code')
            val.value = output == '1'
        elif language == 'javascript':
            ending = '\nconsole.log(JSON.stringify(' + entry_point + '(' + test_input + ')) === JSON.stringify(' + test_output + '));'
            code_gen = code_gen + ending

            with open('test_code.js', 'w') as f:
                f.write(code_gen)

            run_process = subprocess.run(['node', 'test_code.js'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output = run_process.stdout.decode()
            if 'test_code.js' in os.listdir():
                os.remove('test_code.js')
            val.value = output == 'true\n'

        elif language == 'c':
            prefix = '#include <stdio.h>\n'
            code_gen = prefix + code_gen

            ending = '\nint main() {\n    printf("%d", (' + entry_point + '(' + test_input + ') == ' + test_output + '));\n    return 0;\n}'
            code_gen = code_gen + ending
            with open('test_code.c', 'w') as f:
                f.write(code_gen)

            if subprocess.run(['gcc', 'test_code.c', '-o', 'test_code'], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE).returncode == 0:
                run_process = subprocess.run('./test_code', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                output = run_process.stdout.decode()
            else:
                output = '0'

            if 'test_code.c' in os.listdir():
                os.remove('test_code.c')
            if 'test_code' in os.listdir():
                os.remove('test_code')
            val.value = output == '1'
        else:
            raise ValueError(f'Language {language} not supported.')
