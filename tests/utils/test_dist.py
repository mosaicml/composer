# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest

from composer.utils import dist


@pytest.mark.world_size(2)
def test_run_local_rank_first_context_raises_error():
    # This mocking is necessary because there is a fixture called configure_dist that
    # initializes dist for ALL tests, so we need to pretend that dist is not initialized
    with patch('composer.utils.dist.dist.is_initialized') as mock_dist_is_initialized:
        mock_dist_is_initialized.return_value = False
        with pytest.raises(RuntimeError) as e:
            with dist.run_local_rank_zero_first():
                pass
        # Verify error raised is intended
        assert 'the distributed package is not available or has not been initialized' in str(e)


@pytest.mark.world_size(2)
def test_run_local_rank_first_context_runs_properly():
    # There is a fixture called configure_dist that initializes dist for ALL tests,
    # so dist is initialized here and this code should run without error
    with dist.run_local_rank_zero_first():
        pass
