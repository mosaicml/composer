# Copyright 2021 MosaicML. All Rights Reserved.

"""Entrypoint that runs the mosaic trainer on a provided YAML hparams file.

Adds a --datadir flag to conveniently set a common
data directory for both train and validation datasets.

Example that trains MNIST with label smoothing::

    >>> python examples/run_mosaic_trainer.py
    -f composer/yamls/models/classify_mnist_cpu.yaml
    --algorithms label_smoothing
    --datadir ~/datasets
"""
import argparse
import logging

import deepspeed

import composer
from composer.trainer.deepspeed_trainer import DeepSpeedTrainer
from composer.trainer.deepspeed_trainer_hparams import DeepSpeedTrainerHparams

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig()
    logging.captureWarnings(True)

    parser = argparse.ArgumentParser(parents=[DeepSpeedTrainerHparams.get_argparse(cli_args=True)])
    parser.add_argument(
        '--datadir',
        default=None,
        help='set the datadir for both train and eval datasets',
    )

    deepspeed.add_config_arguments(parser)

    args, _ = parser.parse_known_args()
    hparams = DeepSpeedTrainerHparams.create(cli_args=True)  # reads cli args from sys.argv
    logging.getLogger(composer.__name__).setLevel(hparams.log_level)
    if args.datadir is not None:
        hparams.set_datadir(args.datadir)
        logger.info(f'Set dataset dirs in hparams to: {args.datadir}')
    trainer = DeepSpeedTrainer.create_from_hparams(hparams=hparams, deepspeed_config=args.deepspeed_config)
    trainer.fit()


if __name__ == "__main__":
    main()
