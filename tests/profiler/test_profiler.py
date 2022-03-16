# Copyright 2021 MosaicML. All Rights Reserved.

from unittest.mock import MagicMock

import pytest

from composer.core import State
from composer.profiler import Profiler, ProfilerAction


@pytest.mark.parametrize("repeat", [1, 0])
def test_profiler_get_action(dummy_state: State, repeat: int):
    # tests that get_action works correctly given the state
    skip_first = 1
    wait = 2
    warmup = 3
    active = 4
    profiler = Profiler(
        state=dummy_state,
        event_handlers=[],
        skip_first=skip_first,
        warmup=warmup,
        wait=wait,
        active=active,
        repeat=repeat,
    )

    assert profiler.get_action(int(dummy_state.timer.batch_in_epoch)) == ProfilerAction.SKIP  # skip first epoch

    for _ in range(skip_first):
        dummy_state.timer.on_batch_complete()
    assert profiler.get_action(int(dummy_state.timer.batch_in_epoch)) == ProfilerAction.SKIP

    for _ in range(wait):
        dummy_state.timer.on_batch_complete()

    assert profiler.get_action(int(dummy_state.timer.batch_in_epoch)) == ProfilerAction.WARMUP

    for _ in range(warmup):
        dummy_state.timer.on_batch_complete()

    assert profiler.get_action(int(dummy_state.timer.batch_in_epoch)) == ProfilerAction.ACTIVE

    for _ in range(active + wait + warmup):
        dummy_state.timer.on_batch_complete()

    if repeat == 0:
        assert profiler.get_action(int(dummy_state.timer.batch_in_epoch)) == ProfilerAction.ACTIVE
    else:
        assert profiler.get_action(int(dummy_state.timer.batch_in_epoch)) == ProfilerAction.SKIP


def test_marker(dummy_state: State):
    mock_event_handler = MagicMock()
    profiler = Profiler(
        state=dummy_state,
        event_handlers=[mock_event_handler],
    )
    marker = profiler.marker("name",
                             actions=[ProfilerAction.SKIP, ProfilerAction.WARMUP, ProfilerAction.ACTIVE],
                             categories=["cat1"])
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
        assert foo == "hi"

    func_to_profile(foo="hi")  # call 5 and 6

    @marker()
    def func_to_profile2(bar: int):
        assert bar == 6

    func_to_profile2(bar=6)  # call 7 and 8

    marker.instant()

    assert mock_event_handler.process_duration_event.call_count == 8
    assert mock_event_handler.process_instant_event.call_count == 1
