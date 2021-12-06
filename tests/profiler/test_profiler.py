# Copyright 2021 MosaicML. All Rights Reserved.

from typing import Optional
from unittest.mock import MagicMock

import pytest

from composer.core.profiler import MosaicProfiler, MosaicProfilerAction
from composer.core.types import State


@pytest.mark.parametrize("skip_first_epoch", [True, False])
@pytest.mark.parametrize("repeat", [1, None])
def test_profiler_get_action(dummy_state: State, skip_first_epoch: bool, repeat: Optional[int]):
    # tests that get_action works correctly given the state
    wait = 2
    active = 3
    profiler = MosaicProfiler(
        state=dummy_state,
        event_handlers=[],
        skip_first_epoch=skip_first_epoch,
        wait=wait,
        active=active,
        repeat=repeat,
    )

    dummy_state.epoch = 0
    dummy_state.step = wait + 1

    if skip_first_epoch:
        assert profiler.get_action() == MosaicProfilerAction.SKIP
    else:
        assert profiler.get_action() == MosaicProfilerAction.ACTIVE

        dummy_state.step = wait + active + wait + 1

        if repeat is None:
            assert profiler.get_action() == MosaicProfilerAction.ACTIVE
        else:
            assert profiler.get_action() == MosaicProfilerAction.SKIP

    dummy_state.epoch = 1
    dummy_state.step = dummy_state.steps_per_epoch + wait + 1

    assert profiler.get_action() == MosaicProfilerAction.ACTIVE

    dummy_state.step = dummy_state.steps_per_epoch + wait + active + wait + 1

    if repeat is None:
        assert profiler.get_action() == MosaicProfilerAction.ACTIVE
    else:
        assert profiler.get_action() == MosaicProfilerAction.SKIP


def test_marker(dummy_state: State):
    mock_event_handler = MagicMock()
    profiler = MosaicProfiler(
        state=dummy_state,
        event_handlers=[mock_event_handler],
    )
    marker = profiler.marker("name", always_record=True, categories=["cat1"])
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
