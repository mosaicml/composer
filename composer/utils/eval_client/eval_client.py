# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2023 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Abstract class for utilities that access and run code on serverless eval clients."""

import abc
from typing import Dict

__all__ = ['EvalClient']


class EvalClient(abc.ABC):
    """Abstract class for implementing eval clients, such as LambdaEvalClient."""

    def invoke(self, payload: Dict[str, str]) -> bool:
        """Invoke a provided dictionary payload to the client.

        For code generation, the payload is a JSON with the following attributes:
            {
                'code': <code to be evaluated>,
                'input': <test input>,
                'output': <test output>,
                'entry_point': <entry point>,

            }

        Args:
            payload: the materials of the HTTPS request to the client.

        Returns:
            Whether the test case passed or failed.
        """
        del payload  # unused
        raise NotImplementedError(f'{type(self).__name__}.invoke is not implemented')

    def close(self):
        """Close the object store."""
        pass
