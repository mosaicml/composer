# Copyright 2021 MosaicML. All Rights Reserved.

"""Entrypoint that performs a profiling run on the provided yahp hparams file
This example is interchangable with run_mosaic_trainer.py
"""
import argparse
import logging

import composer
from composer.profiler.profiler_hparams import MosaicProfilerHparams
from composer.trainer.trainer import Trainer
from composer.trainer.trainer_hparams import TrainerHparams

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig()
    logging.captureWarnings(True)

    parser = argparse.ArgumentParser(parents=[TrainerHparams.get_argparse(cli_args=True)])
    parser.add_argument(
        '--datadir',
        default=None,
        help='set the datadir for both train and eval datasets',
    )

    args, _ = parser.parse_known_args()
    hparams = TrainerHparams.create(cli_args=True)  # reads cli args from sys.argv
    logging.getLogger(composer.__name__).setLevel(hparams.log_level)
    if args.datadir is not None:
        hparams.set_datadir(args.datadir)
        logger.info(f'Set dataset dirs in hparams to: {args.datadir}')
    if hparams.mosaic_profiler is None:
        hparams.mosaic_profiler = MosaicProfilerHparams()
    hparams.max_epochs = 2
    if hparams.mosaic_profiler.repeat is not None:
        hparams.train_subset_num_batches = (hparams.mosaic_profiler.wait + hparams.mosaic_profiler.active) *  hparams.mosaic_profiler.repeat
    hparams.train_dataset.shuffle = False
    # disable validation
    hparams.validate_every_n_batches = -1
    hparams.validate_every_n_epochs = -1
    trainer = Trainer.create_from_hparams(hparams=hparams)
    trainer.fit()


if __name__ == "__main__":
    main()
