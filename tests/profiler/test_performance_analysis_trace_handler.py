# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from torch.utils.data import DataLoader

from composer.profiler import Profiler
from composer.profiler.performance_analysis_trace_handler import PerformanceAnalyzerTraceHandler
from composer.profiler.profiler_schedule import cyclic_schedule
from composer.trainer import Trainer
from tests.common import RandomClassificationDataset, RandomSlowClassificationDataset, SimpleModel


@pytest.mark.filterwarnings(
    r'ignore:The profiler is enabled\. Using the profiler adds additional overhead when training\.:UserWarning')
def test_performance_analysis_trace_handler_with_small_model():
    # Construct the trainer
    profiler = Profiler(
        schedule=cyclic_schedule(skip_first=1, wait=0, warmup=0, active=1000, repeat=0),
        trace_handlers=[PerformanceAnalyzerTraceHandler()],
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


@pytest.mark.filterwarnings(
    r'ignore:The profiler is enabled\. Using the profiler adds additional overhead when training\.:UserWarning')
def test_performance_analysis_trace_handler_with_dataloader_bottleneck():
    # Construct the trainer
    profiler = Profiler(
        schedule=cyclic_schedule(skip_first=1, wait=0, warmup=0, active=1000, repeat=0),
        trace_handlers=[PerformanceAnalyzerTraceHandler()],
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
        train_dataloader=DataLoader(RandomSlowClassificationDataset()),
        max_duration='2ep',
        profiler=profiler,
    )

    # Train
    trainer.fit()
