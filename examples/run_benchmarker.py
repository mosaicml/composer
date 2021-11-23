# Copyright 2021 MosaicML. All Rights Reserved.

"""Entrypoint that runs the `Benchmarker` on a provided YAML hparams file."""
import logging

import composer
from composer.benchmarker.benchmarker import Benchmarker
from composer.benchmarker.benchmarker_hparams import BenchmarkerHparams

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig()
    logging.captureWarnings(True)

    hparams = BenchmarkerHparams.create(cli_args=True)  # reads cli args from sys.argv
    logging.getLogger(composer.__name__).setLevel(hparams.log_level)

    benchmarker = Benchmarker.create_from_hparams(hparams=hparams)
    benchmarker.run_timing_benchmark()


if __name__ == "__main__":
    main()
