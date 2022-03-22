# Copyright 2021 MosaicML. All Rights Reserved.

"""Entrypoint that runs the Composer trainer on a provided YAML hparams file.

Adds a --datadir flag to conveniently set a common
data directory for both train and validation datasets.

Example that trains MNIST with label smoothing::

    >>> python examples/run_composer_trainer.py
    -f composer/yamls/models/classify_mnist_cpu.yaml
    --algorithms label_smoothing --alpha 0.1
    --datadir ~/datasets
"""
import os
import sys
import textwrap
import warnings
from typing import Type

import yaml

from composer.loggers.logger import LogLevel
from composer.loggers.logger_hparams import WandBLoggerHparams
from composer.trainer import TrainerHparams


def warning_on_one_line(message: str, category: Type[Warning], filename: str, lineno: int, file=None, line=None):
    # From https://stackoverflow.com/questions/26430861/make-pythons-warnings-warn-not-mention-itself
    return f'{category.__name__}: {message} (source: {filename}:{lineno})\n'


def main() -> None:
    warnings.formatwarning = warning_on_one_line

    if len(sys.argv) == 1:
        sys.argv = [sys.argv[0], "--help"]

    hparams = TrainerHparams.create(cli_args=True)  # reads cli args from sys.argv

    # if using wandb, store the config inside the wandb run
    for logger_hparams in hparams.loggers:
        if isinstance(logger_hparams, WandBLoggerHparams):
            logger_hparams.config = hparams.to_dict()

    trainer = hparams.initialize_object()

    # Log the config to an artifact store
    hparams_path = os.path.join(trainer.logger.run_name, "hparams.yaml")
    os.makedirs(os.path.dirname(hparams_path), exist_ok=True)
    try:
        with open(hparams_path, "x") as f:
            # Storing the config (ex. hparams) in a separate file so they can be modified before resuming
            f.write(hparams.to_yaml())
    except FileExistsError as e:
        with open(hparams_path, "r") as f:
            # comparing the parsed hparams to ignore whitespace and formatting differences
            if hparams.to_dict() != yaml.safe_load(f):
                raise RuntimeError(
                    textwrap.dedent(f"""\
                        The hparams in the existing checkpoint folder {hparams_path}
                        differ from those being used in the current training run.
                        Please specify a new checkpoint folder.""")) from e
    trainer.logger.file_artifact(LogLevel.FIT, artifact_name="hparams.yaml", file_path=hparams_path, overwrite=True)

    # Print the config to the terminal
    print("*" * 30)
    print("Config:")
    print(hparams.to_yaml())
    print("*" * 30)

    trainer.fit()


if __name__ == "__main__":
    main()
