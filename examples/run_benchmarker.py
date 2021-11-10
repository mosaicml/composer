# Copyright 2021 MosaicML. All Rights Reserved.

"""Entrypoint that runs the `Benchmarker` on a provided YAML hparams file."""
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
    hparams = BenchmarkerHparams.create(cli_args=True)  # reads cli args from sys.argv
    logging.getLogger(composer.__name__).setLevel(hparams.log_level)

    benchmarker = Benchmarker.create_from_hparams(hparams=hparams)
    benchmarker.run_timing_benchmark()


if __name__ == "__main__":
    main()
