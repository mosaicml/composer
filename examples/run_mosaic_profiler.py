# Copyright 2021 MosaicML. All Rights Reserved.

"""Entrypoint that performs a profiling run on the provided yahp hparams file
This example is interchangable with run_mosaic_trainer.py
"""
import logging

import composer
from composer.callbacks.callback_hparams import TorchProfilerHparams
from composer.datasets.hparams import SyntheticHparamsMixin
from composer.profiler.profiler_hparams import MosaicProfilerHparams
from composer.trainer.trainer import Trainer
from composer.trainer.trainer_hparams import TrainerHparams

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig()
    logging.captureWarnings(True)

    hparams = TrainerHparams.create(cli_args=True)  # reads cli args from sys.argv
    logging.getLogger(composer.__name__).setLevel(hparams.log_level)

    # Configure the mosaic profiler
    if hparams.mosaic_profiler is None:
        hparams.mosaic_profiler = MosaicProfilerHparams()
    hparams.max_epochs = 2
    if hparams.mosaic_profiler.repeat is not None:
        hparams.train_subset_num_batches = (hparams.mosaic_profiler.wait +
                                            hparams.mosaic_profiler.active) * hparams.mosaic_profiler.repeat

    # Configure the torch profiler
    hparams.callbacks.append(
        TorchProfilerHparams(
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            skip=0,
            warmup=hparams.mosaic_profiler.wait,
            active=hparams.mosaic_profiler.active,
            wait=0,
            repeat=hparams.mosaic_profiler.repeat,
        ))

    # Use synthetic data
    hparams.train_dataset.shuffle = False
    if not isinstance(hparams.train_dataset, SyntheticHparamsMixin):
        raise RuntimeError("Train dataset does not support synthetic data")
    hparams.train_dataset.use_synthetic = True

    # Disable validation
    # fix the val dataset to the train dataset, to avoid any issues with initialization
    # We never run evaluation so it doesn't matter
    # TODO(ravi) -- after #120 is merged, set the evaluators to the empty list.
    hparams.val_dataset = hparams.train_dataset
    # disable validation
    hparams.validate_every_n_batches = -1
    hparams.validate_every_n_epochs = -1
    trainer = Trainer.create_from_hparams(hparams=hparams)
    trainer.fit()


if __name__ == "__main__":
    main()
