# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import pathlib

import pytest
from torch.utils.data import DataLoader

from composer.profiler import Profiler
from composer.profiler.json_trace_handler import JSONTraceHandler
from composer.profiler.profiler_schedule import cyclic_schedule
from composer.trainer import Trainer
from tests.common import RandomClassificationDataset, SimpleModel


# This test shouldn't run with the Torch profiler enabled, not providing a model or data can cause a seg fault
@pytest.mark.filterwarnings(
    r'ignore:The profiler is enabled\. Using the profiler adds additional overhead when training\.:UserWarning')
def test_json_trace_profiler_handler(tmp_path: pathlib.Path):
    # Construct the trainer
    profiler = Profiler(
        schedule=cyclic_schedule(wait=0, warmup=0, active=1000, repeat=0),
        trace_handlers=[JSONTraceHandler(
            folder=str(tmp_path),
            merged_trace_filename='trace.json',
        )],
        sys_prof_cpu=False,
        sys_prof_net=False,
        sys_prof_disk=False,
        sys_prof_memory=False,
        torch_prof_record_shapes=False,
        torch_prof_profile_memory=False,
        torch_prof_with_stack=False,
        torch_prof_with_flops=False,
    )
    trainer = Trainer(
        model=SimpleModel(),
        train_dataloader=DataLoader(RandomClassificationDataset()),
        max_duration='2ep',
        profiler=profiler,
    )

    # Train
    trainer.fit()

    # Validate that the trace file contains expected events
    profiler_file = os.path.join(tmp_path, 'trace.json')
    with open(profiler_file, 'r') as f:
        trace_json = json.load(f)
        has_epoch_start_event = False
        has_epoch_end_event = False
        for event in trace_json:
            if event['name'] == 'event/epoch' and event['ph'] == 'B':
                has_epoch_start_event = True
            if event['name'] == 'event/epoch' and event['ph'] == 'E':
                has_epoch_end_event = True
        assert has_epoch_start_event
        assert has_epoch_end_event
