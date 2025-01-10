# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
import time
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


@pytest.mark.world_size(2)
def test_get_node_signal_file_name():
    file_name = dist.get_node_signal_file_name()
    gathered_file_names = dist.all_gather_object(file_name)

    assert len(gathered_file_names) == 2
    assert gathered_file_names[0] == gathered_file_names[1]
    assert gathered_file_names[0] == file_name
    assert file_name.startswith('._signal_file_node0_')
    assert len(file_name) == len('._signal_file_node0_') + 6


@pytest.mark.world_size(2)
def test_write_signal_file(tmp_path):
    file_name = dist.get_node_signal_file_name()
    file_path = os.path.join(tmp_path, file_name)
    dist.write_signal_file(file_name, tmp_path)

    # tmp_path will be different on each rank, and only rank zero
    # should have written a file
    if dist.get_local_rank() == 0:
        assert os.path.exists(file_path)
    else:
        assert not os.path.exists(file_path)


@pytest.mark.world_size(2)
def test_busy_wait_for_local_rank_zero(tmp_path):
    gathered_tmp_path = dist.all_gather_object(tmp_path)[0]

    dist.barrier()
    start_time = time.time()
    assert os.listdir(gathered_tmp_path) == ['mlruns']
    with dist.busy_wait_for_local_rank_zero(gathered_tmp_path):
        if dist.get_local_rank() == 0:
            time.sleep(0.5)

    end_time = time.time()
    total_time = end_time - start_time
    gathered_times = dist.all_gather_object(total_time)
    assert os.listdir(gathered_tmp_path) == ['mlruns']
    assert len(gathered_times) == 2
    assert abs(gathered_times[0] - gathered_times[1]) < 0.1
