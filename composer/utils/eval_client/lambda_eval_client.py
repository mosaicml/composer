# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""AWS Lambda compatible eval client."""
import json
import logging
import os
from typing import Dict

from composer.utils.eval_client.eval_client import EvalClient
from composer.utils.import_helpers import MissingConditionalImportError

__all__ = ['LambdaEvalClient']
log = logging.getLogger(__name__)


class LambdaEvalClient(EvalClient):
    """Utility for creating a client for and invoking an AWS Lambda."""

    def __init__(self) -> None:
        """Initialize the LambdaEvalClient by attempting to create a boto3 client.

        Uses CODE_EVAL_ARN and CODE_EVAL_REGION environment variables for boto3 env vars.
        """
        if 'CODE_EVAL_ARN' not in os.environ or 'CODE_EVAL_REGION' not in os.environ:
            raise Exception('Please set environment variable CODE_EVAL_ARN to the ARN of the lambda function '
                            'and CODE_EVAL_REGION to the region of the lambda function.')
        try:
            import boto3
        except ImportError as e:
            raise MissingConditionalImportError('streaming', 'boto3') from e
        log.debug('Running code eval on lambda.')
        self.client = boto3.Session().client('lambda', region_name=os.environ['CODE_EVAL_REGION'])

    def invoke(self, payload: Dict[str, str]) -> bool:
        """Invoke a provided dictionary payload to the client."""
        response = self.client.invoke(
            FunctionName=os.environ['CODE_EVAL_ARN'],
            InvocationType='RequestResponse',
            LogType='None',
            Payload=bytes(json.dumps(payload), 'utf-8'),
        )
        response = json.load(response['Payload'])
        return 'statusCode' in response and response['statusCode'] == 200

    def close(self):
        self.client.close()
