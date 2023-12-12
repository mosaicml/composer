# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib

import pytest
import torch
from packaging import version

from composer.profiler.utils import export_memory_timeline_html


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
