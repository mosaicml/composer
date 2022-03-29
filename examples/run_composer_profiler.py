# Copyright 2021 MosaicML. All Rights Reserved.

"""Entrypoint that performs a profiling run on the provided yahp hparams file This example is interchangable with
run_composer_trainer.py."""
import argparse
import logging
import sys
import warnings
from typing import Type

import composer
from composer.profiler import CyclicProfilerScheduleHparams, JSONTraceHparams
from composer.trainer import TrainerHparams

logger = logging.getLogger(__name__)


def warning_on_one_line(message: str, category: Type[Warning], filename: str, lineno: int, file=None, line=None):
    # From https://stackoverflow.com/questions/26430861/make-pythons-warnings-warn-not-mention-itself
    return f'{category.__name__}: {message} (source: {filename}:{lineno})\n'


def main() -> None:
    warnings.formatwarning = warning_on_one_line

    if len(sys.argv) == 1:
        sys.argv = [sys.argv[0], "--help"]

    parser = argparse.ArgumentParser(parents=[TrainerHparams.get_argparse(cli_args=True)])
    parser.add_argument("--detailed",
                        default=False,
                        action="store_true",
                        help="Whether to record all system level statistics and torch tensor shapes and stack traces.")

    args, _ = parser.parse_known_args()
    hparams = TrainerHparams.create(cli_args=True)  # reads cli args from sys.argv
    logging.getLogger(composer.__name__).setLevel(hparams.log_level)

    # Configure the Composer profiler
    hparams.prof_trace_handlers = [JSONTraceHparams(folder="composer_traces")]
    if args.detailed:
        hparams.sys_prof_disk = True
        hparams.sys_prof_memory = True
        hparams.sys_prof_net = True
        hparams.torch_prof_record_shapes = True
        hparams.torch_prof_with_stack = True

    hparams.max_duration = "2ep"
    if hparams.prof_schedule is None:
        schedule_hparams = CyclicProfilerScheduleHparams()
        hparams.prof_schedule = schedule_hparams
        if schedule_hparams.repeat != 0 and hparams.train_subset_num_batches is None:
            cycle_len = schedule_hparams.wait + schedule_hparams.warmup + schedule_hparams.active
            num_profiling_batches = schedule_hparams.skip_first + cycle_len * schedule_hparams.repeat
            hparams.train_subset_num_batches = num_profiling_batches

        # Disable dataset shuffle, since shuffle is not supported when using subset_num_batches
        hparams.train_dataset.shuffle = False

    # Disable validation
    # First, set the val dataset to the train dataset, to avoid any issues with initialization
    # We never run evaluation so it doesn't matter
    hparams.val_dataset = hparams.train_dataset
    hparams.validate_every_n_batches = -1
    hparams.validate_every_n_epochs = -1

    # Create the trainer and train
    trainer = hparams.initialize_object()
    trainer.fit()


if __name__ == "__main__":
    main()
