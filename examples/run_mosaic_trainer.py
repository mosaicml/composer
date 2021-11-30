# Copyright 2021 MosaicML. All Rights Reserved.

"""Entrypoint that runs the mosaic trainer on a provided YAML hparams file.

Adds a --datadir flag to conveniently set a common
data directory for both train and validation datasets.

Example that trains MNIST with label smoothing::

    >>> python examples/run_mosaic_trainer.py
    -f composer/yamls/models/classify_mnist_cpu.yaml
    --algorithms label_smoothing --alpha 0.1
    --datadir ~/datasets
"""
import argparse
import logging

import composer
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
    trainer = Trainer.create_from_hparams(hparams=hparams)
    trainer.fit()


if __name__ == "__main__":
    main()
