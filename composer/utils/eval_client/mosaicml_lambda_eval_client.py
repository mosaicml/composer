# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""MCLI compatible eval client."""
import logging
import os
import time
from http import HTTPStatus

import mcli
import numpy as np

from composer.utils.eval_client.eval_client import EvalClient

__all__ = ['MosaicMLLambdaEvalClient']
log = logging.getLogger(__name__)


class MosaicMLLambdaEvalClient(EvalClient):
    """Utility for creating a client for and invoking an AWS Lambda through MCLI."""

    def __init__(self, backoff: int = 3, num_retries: int = 5) -> None:
        """Checks that the requisite environment variables are in the EvalClient.

        `MOSAICML_ACCESS_TOKEN_ENV_VAR` environment variable must be set to access the platform.
        """
        from composer.loggers.mosaicml_logger import \
            MOSAICML_ACCESS_TOKEN_ENV_VAR  # in-line import to avoid circular import

        if MOSAICML_ACCESS_TOKEN_ENV_VAR not in os.environ:
            raise RuntimeError('Cannot use MosaicML Lambda Client Eval without setting MOSAICML_ACCESS_TOKEN_ENV_VAR.')
        log.debug('Running code eval through MosaicMLLambdaEvalClient.')
        self.backoff = backoff
        self.num_retries = num_retries

    def invoke(self, payload: list[list[list[dict[str, str]]]]) -> list[list[list[bool]]]:
        """Invoke a batch of provided payloads for code evaluations."""
        num_beams = len(payload[0])
        num_tests = [len(generation_payload[0]) for generation_payload in payload]
        cum_tests = (np.cumsum([0] + num_tests[:-1]) * num_beams).tolist()
        test_cases = [
            test_case for generation_payload in payload for beam_payload in generation_payload
            for test_case in beam_payload
        ]
        ret_helper = [False] * len(test_cases)
        for i in range(self.num_retries):
            try:
                ret_helper = mcli.get_code_eval_output(test_cases).data  # pyright: ignore[reportGeneralTypeIssues]
                break
            except mcli.MAPIException as e:
                if e.status >= 500:
                    if i == self.num_retries - 1:
                        log.error(f'Failed to get code eval output after {self.num_retries} retries. Error: {e}')
                    log.warning(f'Failed to get code eval output, retrying in {self.backoff**i} seconds.')
                    time.sleep(self.backoff**i)
                elif e.status == HTTPStatus.UNAUTHORIZED:
                    raise RuntimeError(
                        'Failed to get code eval output due to UNAUTHORIZED error. '
                        'Please ensure you have access to MosaicMLLambdaEvalClient.',
                    ) from e
                else:
                    log.error(f'Failed to get code eval output with unexpected MAPIException. Error: {e}')
                    break
            except TimeoutError as e:
                if i == self.num_retries - 1:
                    log.error(f'Failed to get code eval output after {self.num_retries} retries. Error: {e}')
                log.warning(f'Failed to get code eval output, retrying in {self.backoff**i} seconds.')
                time.sleep(self.backoff**i)
            except Exception as e:
                log.error(f'Failed to get code eval output with unexpected error. Error: {e}')
                break

        ret = []
        for i in range(len(payload)):
            ret_payload = []
            for j in range(num_beams):
                ret_num_beams = []
                for k in range(num_tests[i]):
                    ret_num_beams.append(ret_helper[cum_tests[i] + j * num_tests[i] + k])
                ret_payload.append(ret_num_beams)
            ret.append(ret_payload)
        return ret
