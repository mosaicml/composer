# Copyright 2021 MosaicML. All Rights Reserved.

import json
import os

import pytest

from composer.profiler import JSONTraceHandlerHparams, ProfilerHparams
from composer.trainer import TrainerHparams
from composer.utils import run_directory


@pytest.mark.timeout(10)
def test_json_trace_profiler_hanlder(composer_trainer_hparams: TrainerHparams):
    json_trace_handler_params = JSONTraceHandlerHparams(flush_every_n_batches=1,)

    profiler_hparams = ProfilerHparams(
        trace_event_handlers=[json_trace_handler_params],
        skip_first=0,
        warmup=0,
        wait=0,
        active=1000,
        repeat=0,
    )

    composer_trainer_hparams.profiler = profiler_hparams
    composer_trainer_hparams.max_duration = "2ep"

    trainer = composer_trainer_hparams.initialize_object()
    trainer.fit()

    profiler_file = os.path.join(run_directory.get_run_directory(), "composer_profiler", "rank_0.trace.json")

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
