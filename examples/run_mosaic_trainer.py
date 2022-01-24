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
import sys
import warnings
from typing import Type

import yahp as hp

from composer.trainer.trainer import Trainer


def warning_on_one_line(message: str, category: Type[Warning], filename: str, lineno: int, file=None, line=None):
    # From https://stackoverflow.com/questions/26430861/make-pythons-warnings-warn-not-mention-itself
    return f'{category.__name__}: {message} (source: {filename}:{lineno})\n'


def main() -> None:
    warnings.formatwarning = warning_on_one_line

    if len(sys.argv) == 1:
        sys.argv = [sys.argv[0], "--help"]

    import inspect
    print(inspect.signature(Trainer))

    trainer = hp.create(Trainer, cli_args=True)
    trainer.fit()


if __name__ == "__main__":
    main()
