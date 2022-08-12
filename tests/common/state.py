# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict

from composer.core import State
from composer.utils import is_model_deepspeed
from tests.common.compare import deep_compare


def _del_wct_timestamp_fields(timestamp_state_dict: Dict[str, Any]):
    del timestamp_state_dict['Timestamp']['total_wct']
    del timestamp_state_dict['Timestamp']['epoch_wct']
    del timestamp_state_dict['Timestamp']['batch_wct']


def assert_state_equivalent(state1: State, state2: State):
    """Assert that ``state1`` is equivalent to ``state2``, ignoring wall clock timestamp fields."""
    assert state1.serialized_attributes == state2.serialized_attributes
    assert is_model_deepspeed(state1.model) == is_model_deepspeed(state2.model)

    # Using a loose tolerance for GPU states as GPU determinism does not work properly
    is_gpu = next(state1.model.parameters()).device.type == 'cuda'
    atol = 0.1 if is_gpu else 0.0
    rtol = 0.1 if is_gpu else 0.0

    state_dict_1 = state1.state_dict()
    state_dict_2 = state2.state_dict()

    # Remove any wall clock timestamp fields
    _del_wct_timestamp_fields(state_dict_1['timestamp'])
    _del_wct_timestamp_fields(state_dict_2['timestamp'])

    # Remove run_name since we use timestamp as part of name
    del state_dict_1['run_name']
    del state_dict_2['run_name']

    # Compare the state dicts
    deep_compare(state_dict_1, state_dict_2, atol=atol, rtol=rtol)
