# Copyright 2021 MosaicML. All Rights Reserved.

"""Entrypoint that performs a profiling run on the provided yahp hparams file
This example is interchangable with run_mosaic_trainer.py
"""
import argparse
import logging

import composer
from composer.profiler import ProfilerHparams
from composer.profiler.profiler_hparams import DataloaderProfilerHparams, SystemProfilerHparams, TorchProfilerHparams
from composer.trainer import Trainer, TrainerHparams

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig()
    logging.captureWarnings(True)

    parser = argparse.ArgumentParser(parents=[TrainerHparams.get_argparse(cli_args=True)])
    parser.add_argument(
        '--datadir',
        default=None,
        help='set the datadir for the train dataset',
    )
    parser.add_argument("--detailed",
                        default=False,
                        action="store_true",
                        help="Whether to record all system level statistics and torch tensor shapes and stack traces.")

    args, _ = parser.parse_known_args()
    hparams = TrainerHparams.create(cli_args=True)  # reads cli args from sys.argv
    logging.getLogger(composer.__name__).setLevel(hparams.log_level)
    if args.datadir is not None:
        hparams.set_datadir(args.datadir)
        logger.info(f'Set dataset dirs in hparams to: {args.datadir}')

    # Configure the mosaic profiler
    if hparams.profiler is None:
        if args.detailed:
            hparams.profiler = ProfilerHparams(profilers=[  # type: ignore
                DataloaderProfilerHparams(),
                SystemProfilerHparams(profile_disk=True, profile_memory=True, profile_net=True),
                TorchProfilerHparams(record_shapes=True, with_stack=True),
            ])
        else:
            hparams.profiler = ProfilerHparams()
    hparams.max_epochs = 2
    if hparams.profiler.repeat != 0 and hparams.train_subset_num_batches is None:
        cycle_len = hparams.profiler.wait + hparams.profiler.warmup + hparams.profiler.active
        num_profiling_batches = hparams.profiler.skip_first + cycle_len * hparams.profiler.repeat
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
    trainer = Trainer.create_from_hparams(hparams=hparams)
    trainer.fit()


if __name__ == "__main__":
    main()
