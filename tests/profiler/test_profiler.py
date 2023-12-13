# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
from typing import Union
from unittest.mock import MagicMock

import pytest
import torch
from packaging import version

from composer.core import State
from composer.profiler import Profiler, ProfilerAction, SystemProfiler, TorchProfiler, cyclic_schedule
from composer.profiler.utils import export_memory_timeline_html


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
        torch_prof_memory_filename=None,
        sys_prof_cpu=True,
    )
    profiler.bind_to_state(minimal_state)
    assert any(isinstance(cb, TorchProfiler) for cb in minimal_state.callbacks)
    assert any(isinstance(cb, SystemProfiler) for cb in minimal_state.callbacks)


def test_marker(dummy_state: State):
    mock_trace_handler = MagicMock()
    profiler = Profiler(trace_handlers=[mock_trace_handler],
                        schedule=cyclic_schedule(),
                        torch_prof_memory_filename=None)
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


@pytest.mark.parametrize('torch_prof_with_stack', [True, False])
@pytest.mark.parametrize('torch_prof_record_shapes', [True, False])
@pytest.mark.parametrize('torch_prof_profile_memory', [True, False])
@pytest.mark.parametrize('torch_prof_memory_filename', [None, 'test.html'])
def test_profiler_error_message(torch_prof_with_stack: bool, torch_prof_record_shapes: bool,
                                torch_prof_profile_memory: bool, torch_prof_memory_filename: Union[None, str]) -> None:
    # Construct a profiler and assert that it triggers the ValueError if the arguments are invalid
    if (torch_prof_memory_filename is not None and
            not (torch_prof_with_stack and torch_prof_record_shapes and torch_prof_profile_memory)):
        with pytest.raises(ValueError):
            _ = Profiler(
                trace_handlers=[MagicMock()],
                schedule=cyclic_schedule(),
                torch_prof_with_stack=torch_prof_with_stack,
                torch_prof_record_shapes=torch_prof_record_shapes,
                torch_prof_profile_memory=torch_prof_profile_memory,
                torch_prof_memory_filename=torch_prof_memory_filename,
            )
    else:
        _ = Profiler(
            trace_handlers=[MagicMock()],
            schedule=cyclic_schedule(),
            torch_prof_with_stack=torch_prof_with_stack,
            torch_prof_record_shapes=torch_prof_record_shapes,
            torch_prof_profile_memory=torch_prof_profile_memory,
            torch_prof_memory_filename=torch_prof_memory_filename,
        )


@pytest.mark.gpu
def test_memory_timeline(tmp_path: pathlib.Path) -> None:
    if version.parse(torch.__version__) <= version.parse('2.1.0.dev'):
        # memory timeline is supported after PyTorch 2.1.0.
        return
    import torch.profiler._memory_profiler as _memory_profiler

    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 1024, bias=True),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 1024, bias=False),
        torch.nn.Softmax(dim=1),
    ).to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    x = torch.ones((1024, 1024), device='cuda')
    targets = torch.ones((1024, 1024), device='cuda')
    with torch.profiler.profile(record_shapes=True, with_stack=True, profile_memory=True) as prof:
        y = model(x)
        loss = torch.nn.functional.mse_loss(y, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    memory_profile = prof._memory_profile()
    timeline = memory_profile.timeline

    # this checks the default memory timeline event value (t == -1) for preexisting tensors
    assert all((t == -1) if action == _memory_profiler.Action.PREEXISTING else (t > 0) for t, action, _, _ in timeline)

    fig = export_memory_timeline_html(
        prof,
        os.path.join(tmp_path, 'test_memory_timeline.html'),
        yxis_step_size=0.01,
        return_fig=True,
    )
    assert fig is not None, 'export_memory_timeline_html should return a figure when return_fig=True'
    _, end = fig.gca().get_ylim()
    assert round(end, 2) == 0.06
