# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from composer.utils import retry


@pytest.mark.parametrize('with_args', [True, False])
def test_retry(with_args: bool):
    num_tries = 0
    return_after = 2

    if with_args:
        decorator = retry(RuntimeError, num_attempts=3, initial_backoff=0.01, max_jitter=0.01)
        return_after = 2
    else:
        decorator = retry
        # Need to return immediately to avoid timeouts
        return_after = 0

    @decorator
    def flaky_function():
        nonlocal num_tries
        if num_tries < return_after:
            num_tries += 1
            raise RuntimeError('Called too soon!')
        return "Third time's a charm"

    assert flaky_function() == "Third time's a charm"
