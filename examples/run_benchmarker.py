# Copyright 2021 MosaicML. All Rights Reserved.

"""Entrypoint that runs the mosaic trainer on a provided YAML hparams file.

Example that trains MNIST with label smoothing::

    >>> python examples/run_mosaic_trainer.py
    -f composer/yamls/models/classify_mnist_cpu.yaml
    --algorithms label_smoothing
    --datadir ~/datasets
"""
import argparse
import logging

import composer
from composer.benchmarker.benchmarker import Benchmarker
from composer.benchmarker.benchmarker_hparams import BenchmarkerHparams
from composer.trainer.trainer import Trainer
from composer.trainer.trainer_hparams import TrainerHparams

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig()
    logging.captureWarnings(True)

    parser = argparse.ArgumentParser(parents=[BenchmarkerHparams.get_argparse(cli_args=True)])

    parser.parse_known_args()
    # hparams = TrainerHparams.create(cli_args=True)  # reads cli args from sys.argv
    hparams = BenchmarkerHparams.create(cli_args=True) # reads cli args from sys.argv
    logging.getLogger(composer.__name__).setLevel(hparams.log_level)

    benchmarker = Benchmarker.create_from_hparams(hparams=hparams)
    benchmarker.run_timing_benchmark()
    # trainer = Trainer.create_from_hparams(hparams=hparams)
    # trainer.fit()


if __name__ == "__main__":
    main()
