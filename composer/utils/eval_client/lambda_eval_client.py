# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""AWS Lambda compatible eval client."""
import json
import logging
import os

import requests

from composer.utils.eval_client.eval_client import EvalClient

__all__ = ['LambdaEvalClient']
log = logging.getLogger(__name__)


class LambdaEvalClient(EvalClient):
    """Utility for creating a client for and invoking an AWS Lambda."""

    def __init__(self) -> None:
        """Checks that the requisite environment variables are in the EvalClient.

        There must be CODE_EVAL_URL for the URL of the lambda API and CODE_EVAL_APIKEY
        for the API key of the lambda API.
        """
        if 'CODE_EVAL_URL' not in os.environ or 'CODE_EVAL_APIKEY' not in os.environ:
            raise Exception(
                'Please set environment variable CODE_EVAL_URL to the URL of the lambda API '
                'and CODE_EVAL_APIKEY to the API key of the API gateway.',
            )
        log.debug('Running code eval on lambda.')

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
        """Invoke a provided dictionary payload to the client."""
        response = requests.post(
            os.environ['CODE_EVAL_URL'],
            data=json.dumps(payload),
            headers={'x-api-key': os.environ['CODE_EVAL_APIKEY']},
        )
        response = response.json()
        return 'statusCode' in response and response['statusCode'] == 200
