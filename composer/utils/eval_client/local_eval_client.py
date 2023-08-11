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
            prefix = '#include <iostream>\n#include <vector>\n#include <string>\n#include <map>\n#include <math.h>\nusing namespace std;\nbool issame(vector<string> a,vector<string>b){\n    if (a.size()!=b.size()) return false;\n    for (int i=0;i<a.size();i++)\n    {\n    if (a[i]!=b[i]) return false;\n    }\n    return true;\n}\nbool issame(float a, float b){\n    return abs(a - b) < 1e-4;\n}\nbool issame(bool a, bool b){\n    return a == b;\n}\nbool issame(int a, int b){\n    return a == b;\n}\nbool issame(vector<int> a,vector<int>b){\n    if (a.size()!=b.size()) return false;\n    for (int i=0;i<a.size();i++)\n    {\n        if (a[i]!=b[i]) return false;\n    }\n    return true;\n}\nbool issame(string a, string b){\n    return a == b;\n}\nbool issame(vector<float> a,vector<float>b){\n    if (a.size()!=b.size()) return false;\n    for (int i=0;i<a.size();i++)\n    {\n        if (abs(a[i]-b[i])>1e-4) return false;\n    }\n    return true;\n}\nbool issame(double a, double b){\n    return abs(a - b) < 1e-3;\n}\nbool issame(vector<vector<int>> a,vector<vector<int>> b){\n    if (a.size()!=b.size()) return false;\n\n    for (int i=0;i<a.size();i++)\n    {\n        if (a[i].size()!=b[i].size()) return false;\n        for (int j=0;j<a[i].size();j++)\n            if (a[i][j]!=b[i][j]) return false;\n    }\n    return true;\n}\nbool issame(map<char,int> a,map<char,int> b){\n    if (a.size()!=b.size()) return false;\n    map <char,int>::iterator it;\n    for (it=a.begin();it!=a.end();it++)\n    {\n        char w1=it->first;\n        int w2=it->second;\n        if (b.find(w1)==b.end()) return false;\n        if (b[w1]!=w2) return false;\n    }\n\n    return true;\n}\nbool issame(long long a, long long b){\n    return a == b;\n}\n'
            code_gen = prefix + code_gen

            ending = '\nint main() {\n    cout << issame(' + entry_point + '(' + test_input + '), ' + test_output + ');\n    return 0;\n}'
            code_gen = code_gen + ending
            with open('test_code.cpp', 'w') as f:
                f.write(code_gen)
            compilation_process = subprocess.run(['g++', '-std=c++11', 'test_code.cpp', '-o', 'test_code'],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
            if compilation_process.returncode == 0:
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
            prefix = '#include <stdio.h>\n#include <stdbool.h>\n#include <math.h>\n#include <stdlib.h>\nbool issame_int(int a, int b){\n    return a == b;\n}\nbool issame_bool(bool a, bool b){\n    return a == b;\n}\nbool issame_float(float a, float b){\n    return fabs(a-b) < 1e-4;\n}\n#define issame(a, b) _Generic((a), int: issame_int, bool: issame_bool, float: issame_float)(a, b)\n'
            code_gen = prefix + code_gen

            ending = '\nint main() {\n    bool val = issame(' + entry_point + '(' + test_input + ') , ' + test_output + ');\n    printf("%d", val);\n    return 0;\n}'
            code_gen = code_gen + ending
            with open('test_code.c', 'w') as f:
                f.write(code_gen)
            compilation_process = subprocess.run(['gcc', 'test_code.c', '-o', 'test_code'], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
            if compilation_process.returncode == 0:
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
