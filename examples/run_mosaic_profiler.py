# Copyright 2021 MosaicML. All Rights Reserved.

"""Entrypoint that performs a profiling run on the provided yahp hparams file
This example is interchangable with run_mosaic_trainer.py
"""
import logging

import composer
from composer.profiler import ProfilerHparams
from composer.trainer import Trainer, TrainerHparams

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig()
    logging.captureWarnings(True)

    hparams = TrainerHparams.create(cli_args=True)  # reads cli args from sys.argv
    logging.getLogger(composer.__name__).setLevel(hparams.log_level)

    # Configure the mosaic profiler
    if hparams.profiler is None:
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
