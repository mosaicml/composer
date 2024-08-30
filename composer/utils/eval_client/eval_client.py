# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2023 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Abstract class for utilities that access and run code on serverless eval clients."""

import abc

__all__ = ['EvalClient']


class EvalClient(abc.ABC):
    """Abstract class for implementing eval clients, such as LambdaEvalClient."""

    def invoke(self, payload: list[list[list[dict[str, str]]]]) -> list[list[list[bool]]]:
        """Invoke a provided batch of dictionary payload to the client.

        For code generation, the payload is a list of list of lists of JSONs. The lists are organized in a nested structure, with the outer list
        being grouped by the prompt. For each prompt, the model generates a series of possible continuations, which we term generation beams.
        As a result, in the nested list for each prompt, the lists are grouped by the generation beam, since each prompt produces some set number. In the
        final tier of nesting for each generation, the JSONs are grouped by test case. We note that the evaluation client is agnostic to the list
        structure and instead iterates over each JSON payload for a test cases, converting each JSON to a boolean independently, only maintaining
        the list shape. The JSON for each test case containing the following attributes:

        {
            'code': <code to be evaluated>,
            'input': <test input>,
            'output': <test output>,
            'entry_point': <entry point>,
            'language': <language>,

        }

        The JSON is formatted as [[[request]]] so that the client can batch requests. The outermost list is for the generations of a
        given prompt, the middle list is for the beam generations of a given prompt, and the innermost list is for each test cases.
        Args:
            payload: the materials of the batched HTTPS request to the client organized by prompt, beam generation, and test case.

        Returns:
            Whether the test case passed or failed.
        """
        del payload  # unused
        raise NotImplementedError(f'{type(self).__name__}.invoke is not implemented')

    def close(self):
        """Close the object store."""
        pass
