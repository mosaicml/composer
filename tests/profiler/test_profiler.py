# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from composer.core import State
from composer.profiler import Profiler, ProfilerAction, SystemProfiler, TorchProfiler, cyclic_schedule


@pytest.mark.parametrize('repeat', [1, 0])
def test_cyclic_schedule(dummy_state: State, repeat: int):
    # tests that get_action works correctly given the state
    skip_first = 1
    wait = 2
    warmup = 3
    active = 4
    schedule = cyclic_schedule(skip_first=1, wait=2, warmup=3, active=4, repeat=repeat)

    assert schedule(dummy_state) == ProfilerAction.SKIP  # skip first epoch

    for _ in range(skip_first):
        dummy_state.timestamp = dummy_state.timestamp.to_next_batch()
    assert schedule(dummy_state) == ProfilerAction.SKIP

    for _ in range(wait):
        dummy_state.timestamp = dummy_state.timestamp.to_next_batch()

    assert schedule(dummy_state) == ProfilerAction.WARMUP

    for _ in range(warmup):
        dummy_state.timestamp = dummy_state.timestamp.to_next_batch()

    assert schedule(dummy_state) == ProfilerAction.ACTIVE

    for _ in range(active + wait + warmup):
        dummy_state.timestamp = dummy_state.timestamp.to_next_batch()

    if repeat == 0:
        assert schedule(dummy_state) == ProfilerAction.ACTIVE
    else:
        assert schedule(dummy_state) == ProfilerAction.SKIP


def test_profiler_init(minimal_state: State):
    # Construct a profiler and assert that it created the correct callbacks from the arguments
    mock_trace_handler = MagicMock()
    profiler = Profiler(
        trace_handlers=[mock_trace_handler],
        schedule=cyclic_schedule(),
        torch_prof_profile_memory=True,
        sys_prof_cpu=True,
    )
    profiler.bind_to_state(minimal_state)
    assert any(isinstance(cb, TorchProfiler) for cb in minimal_state.callbacks)
    assert any(isinstance(cb, SystemProfiler) for cb in minimal_state.callbacks)


def test_marker(dummy_state: State):
    mock_trace_handler = MagicMock()
    profiler = Profiler(
        trace_handlers=[mock_trace_handler],
        schedule=cyclic_schedule(),
    )
    profiler.bind_to_state(dummy_state)
    dummy_state.profiler = profiler
    marker = profiler.marker('name',
                             actions=[ProfilerAction.SKIP, ProfilerAction.WARMUP, ProfilerAction.ACTIVE],
                             categories=['cat1'])
    marker.start()  # call #1
    with pytest.raises(RuntimeError):
        marker.start()  # cannot call start twice without finishing
    marker.finish()  # call #2
    with pytest.raises(RuntimeError):
        marker.finish()  # cannot call finish twice without a start before

    with marker:
        pass  # call #3 and #4

    @marker
    def func_to_profile(foo: str):
        assert foo == 'hi'

    func_to_profile(foo='hi')  # call 5 and 6

    @marker()
    def func_to_profile2(bar: int):
        assert bar == 6

    func_to_profile2(bar=6)  # call 7 and 8

    marker.instant()

    assert mock_trace_handler.process_duration_event.call_count == 8
    assert mock_trace_handler.process_instant_event.call_count == 1
