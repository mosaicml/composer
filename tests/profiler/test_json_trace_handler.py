# Copyright 2022 MosaicML. All Rights Reserved.

import json
import os
import pathlib

import pytest

from composer.profiler.profiler_hparams import CyclicProfilerScheduleHparams, JSONTraceHparams
from composer.trainer import TrainerHparams


# This test shouldn't run with the Torch profiler enabled, not providing a model or data can cause a seg fault
@pytest.mark.timeout(30)
def test_json_trace_profiler_handler(composer_trainer_hparams: TrainerHparams, tmpdir: pathlib.Path):
    profiler_file = os.path.join(tmpdir, 'trace.json')
    json_trace_handler_params = JSONTraceHparams(folder=str(tmpdir), merged_trace_filename='trace.json')

    composer_trainer_hparams.prof_trace_handlers = [json_trace_handler_params]
    composer_trainer_hparams.prof_schedule = CyclicProfilerScheduleHparams(
        wait=0,
        warmup=0,
        active=1000,
        repeat=0,
    )
    composer_trainer_hparams.max_duration = "2ep"
    composer_trainer_hparams.sys_prof_cpu = False
    composer_trainer_hparams.sys_prof_net = False
    composer_trainer_hparams.sys_prof_disk = False
    composer_trainer_hparams.sys_prof_memory = False
    composer_trainer_hparams.torch_prof_record_shapes = False
    composer_trainer_hparams.torch_prof_profile_memory = False
    composer_trainer_hparams.torch_prof_with_stack = False
    composer_trainer_hparams.torch_prof_with_flops = False

    trainer = composer_trainer_hparams.initialize_object()
    trainer.fit()

    with open(profiler_file, "r") as f:
        trace_json = json.load(f)
        has_epoch_start_event = False
        has_epoch_end_event = False
        for event in trace_json:
            if event["name"] == "event/epoch" and event["ph"] == "B":
                has_epoch_start_event = True
            if event["name"] == "event/epoch" and event["ph"] == "E":
                has_epoch_end_event = True
        assert has_epoch_start_event
        assert has_epoch_end_event
