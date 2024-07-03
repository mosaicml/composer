# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Eval client for local evaluation."""
import logging
import multiprocessing
import os
import subprocess
import textwrap
import types

from composer.utils import dist
from composer.utils.eval_client.eval_client import EvalClient

__all__ = ['LocalEvalClient']
log = logging.getLogger(__name__)

TIMEOUT = 5  # in seconds


class LocalEvalClient(EvalClient):
    """Utility for creating a client for and invoking local evaluations."""

    def invoke(self, payload: list[list[list[dict[str, str]]]]) -> list[list[list[bool]]]:
        """Invoke a batch of provided payloads for code evaluations."""
        ret = []
        for prompt_group in payload:
            ret_prompt_group = []
            for generation_group in prompt_group:
                ret_generation_group = []
                for test_case in generation_group:
                    ret_generation_group.append(self.invoke_helper(test_case))
                ret_prompt_group.append(ret_generation_group)
            ret.append(ret_prompt_group)
        return ret

    def invoke_helper(self, payload: dict[str, str]) -> bool:
        """Invoke a provided dictionary payload to a multiprocessing subprocess that performs code eval."""
        ret = multiprocessing.Value('b', 0)  # Store result of test case in shared memory
        p = multiprocessing.Process(
            target=self.update_offline_helper,
            args=(
                payload['code'],
                payload['input'],
                payload['output'],
                payload['entry_point'],
                payload['language'],
                ret,
            ),
        )  # Evaluate test case in an independent subprocess
        p.start()
        p.join(TIMEOUT)  # wait for timeout to terminate
        p.terminate()
        return bool(ret.value)  # pyright: ignore[reportGeneralTypeIssues]

    def update_offline_helper(
        self,
        code_gen: str,
        test_input: str,
        test_output: str,
        entry_point: str,
        language: str,
        val: multiprocessing.Value,  # type: ignore
    ):
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
        rank = dist.get_global_rank()
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
            # since we can't run tostring based comparisons in C++, initialize a series of overloaded equality functions
            prefix = '''\
                #include <iostream>
                #include <vector>
                #include <string>
                #include <map>
                #include <math.h>
                using namespace std;

                bool issame(vector<string> a,vector<string>b){
                    if (a.size()!=b.size()) return false;
                    for (int i=0;i<a.size();i++)
                    {
                        if (a[i]!=b[i]) return false;
                    }
                    return true;
                }

                bool issame(float a, float b){
                    return abs(a - b) < 1e-4;
                }

                bool issame(bool a, bool b){
                    return a == b;
                }

                bool issame(int a, int b){
                    return a == b;
                }

                bool issame(vector<int> a,vector<int>b){
                    if (a.size()!=b.size()) return false;
                    for (int i=0;i<a.size();i++)
                    {
                        if (a[i]!=b[i]) return false;
                    }
                    return true;
                }

                bool issame(string a, string b){
                    return a == b;
                }

                bool issame(vector<float> a,vector<float>b){
                    if (a.size()!=b.size()) return false;
                    for (int i=0;i<a.size();i++)
                    {
                        if (abs(a[i]-b[i])>1e-4) return false;
                    }
                    return true;
                }

                bool issame(double a, double b){
                    return abs(a - b) < 1e-3;
                }

                bool issame(vector<vector<int>> a,vector<vector<int>> b){
                    if (a.size()!=b.size()) return false;
                    for (int i=0;i<a.size();i++)
                    {
                        if (a[i].size()!=b[i].size()) return false;
                        for (int j=0;j<a[i].size();j++)
                        {
                            if (a[i][j]!=b[i][j]) return false;
                        }
                    }
                    return true;
                }

                bool issame(map<char,int> a,map<char,int> b){
                    if (a.size()!=b.size()) return false;
                    map <char,int>::iterator it;
                    for (it=a.begin();it!=a.end();it++)
                    {
                        char w1=it->first;
                        int w2=it->second;
                        if (b.find(w1)==b.end()) return false;
                        if (b[w1]!=w2) return false;
                    }
                    return true;
                }

                bool issame(long long a, long long b){
                    return a == b;
                }
            '''
            code_gen = textwrap.dedent(prefix) + code_gen

            # print out the result of the equality check to console, use double brackets to override f string defaults
            ending = f'''\
                int main() {{
                    cout << issame({entry_point}({test_input}), {test_output});
                    return 0;
                }}
            '''
            code_gen = code_gen + '\n' + textwrap.dedent(ending)
            with open(f'test_code_{rank}.cpp', 'w') as f:
                f.write(code_gen)
            compilation_process = subprocess.run(
                [
                    'g++',
                    '-std=c++11',
                    f'test_code_{rank}.cpp',
                    '-o',
                    f'test_code_{rank}',
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if compilation_process.returncode == 0:
                run_process = subprocess.run(f'./test_code_{rank}', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                output = run_process.stdout.decode()
            else:
                output = '0'

            if f'test_code_{rank}.cpp' in os.listdir():
                os.remove(f'test_code_{rank}.cpp')
            if f'test_code_{rank}' in os.listdir():
                os.remove(f'test_code_{rank}')
            val.value = output == '1'
        elif language == 'javascript':
            ending = '\nconsole.log(JSON.stringify(' + entry_point + '(' + test_input + ')) === JSON.stringify(' + test_output + '));'
            code_gen = code_gen + ending

            with open(f'test_code_{rank}.js', 'w') as f:
                f.write(code_gen)

            run_process = subprocess.run(
                ['node', f'test_code_{rank}.js'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            output = run_process.stdout.decode()
            if f'test_code_{rank}.js' in os.listdir():
                os.remove(f'test_code_{rank}.js')
            val.value = output == 'true\n'

        elif language == 'c':
            # since we can't run tostring based comparisons in C, initialize a series of overloaded equality functions
            prefix = '''\
                #include <stdio.h>
                #include <stdbool.h>
                #include <math.h>
                #include <stdlib.h>

                bool issame_int(int a, int b){
                    return a == b;
                }

                bool issame_bool(bool a, bool b){
                    return a == b;
                }

                bool issame_float(float a, float b){
                    return fabs(a-b) < 1e-4;
                }

                #define issame(a, b) _Generic((a), int: issame_int, bool: issame_bool, float: issame_float)(a, b)
            '''
            code_gen = textwrap.dedent(prefix) + code_gen

            # print out the result of the equality check to console, use double brackets to override f string defaults
            ending = f'''\
                int main() {{
                    bool val = issame({entry_point}({test_input}), {test_output});
                    printf("%d", val);
                    return 0;
                }}
            '''
            code_gen = code_gen + '\n' + textwrap.dedent(ending)
            with open(f'test_code_{rank}.c', 'w') as f:
                f.write(code_gen)
            compilation_process = subprocess.run(
                ['gcc', f'test_code_{rank}.c', '-o', f'test_code_{rank}'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if compilation_process.returncode == 0:
                run_process = subprocess.run(f'./test_code_{rank}', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                output = run_process.stdout.decode()
            else:
                output = '0'

            if f'test_code_{rank}.c' in os.listdir():
                os.remove(f'test_code_{rank}.c')
            if f'test_code_{rank}' in os.listdir():
                os.remove(f'test_code_{rank}')
            val.value = output == '1'
        else:
            raise ValueError(f'Language {language} not supported.')
