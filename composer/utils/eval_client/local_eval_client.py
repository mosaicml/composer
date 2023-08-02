# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Eval client for local evaluation."""
import logging
import multiprocessing
import types
from typing import Dict

from composer.utils.eval_client.eval_client import EvalClient

__all__ = ['LocalEvalClient']
log = logging.getLogger(__name__)

TIMEOUT = 5  # in seconds


class LocalEvalClient(EvalClient):
    """Utility for creating a client for and invoking local evaluations."""

    def invoke(self, payload: Dict[str, str]) -> bool:
        """Invoke a provided dictionary payload to a multiprocessing subprocess that performs code eval."""
        ret = multiprocessing.Value('b', 0)  # Store result of test case in shared memory
        p = multiprocessing.Process(target=self.update_offline_helper,
                                    args=(payload['code'], payload['input'], payload['output'], payload['entry_point'], payload['language'],
                                          ret))  # Evaluate test case in an independent subprocess
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
            return val.value
        elif language == 'c++':
            with open('test_code.cpp', 'w') as f:
                f.write(code_gen)
            # Command to compile the C++ program 
            compile_command = ["g++", "readme.cpp", "-o", "readme"] 
            
            # Command to execute the compiled program 
            run_command = ["./readme"] 
            
            # Compile the C++ program 
            compile_process = subprocess.run(compile_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) 
            
            # Check if the compilation was successful 
            if compile_process.returncode == 0: 
                print("Compilation successful.") 
                # Run the compiled program 
                run_process = subprocess.run(run_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) 
                
                # Get the output and error messages from the program 
                output = run_process.stdout.decode() 
                error = run_process.stderr.decode() 
                
                # Print the output and error messages 
                print("Output:") 
                print(output) 
                
                print("Error:") 
                print(error) 
            else: 
                # Print the compilation error messages 
                print("Compilation failed.") 
                print(compile_process.stderr.decode()) 

            print(os.listdir())
            os.remove('readme.cpp')
            print(os.listdir())
            os.remove('readme')
            print(os.listdir())
        elif language == 'javascript':
        
        elif language == 'c':


