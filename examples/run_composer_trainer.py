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
import contextlib
import os
import sys
import tempfile
import textwrap
import warnings
from typing import Type

import yaml

from composer.loggers import LogLevel
from composer.trainer import TrainerHparams
from composer.utils import dist


def warning_on_one_line(message: str, category: Type[Warning], filename: str, lineno: int, file=None, line=None):
    # From https://stackoverflow.com/questions/26430861/make-pythons-warnings-warn-not-mention-itself
    return f'{category.__name__}: {message} (source: {filename}:{lineno})\n'


def main() -> None:
    warnings.formatwarning = warning_on_one_line

    if len(sys.argv) == 1:
        sys.argv = [sys.argv[0], "--help"]

    hparams = TrainerHparams.create(cli_args=True)  # reads cli args from sys.argv
    trainer = hparams.initialize_object()
    if dist.get_global_rank() == 0:
        if hparams.run_name is None:
            ctx = tempfile.TemporaryDirectory()
        else:
            ctx = contextlib.nullcontext(hparams.run_name)
        with ctx as hparams_dir:
            hparams_path = os.path.join(hparams_dir, "hparams.yaml")
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
            trainer.logger.file_artifact(LogLevel.FIT, artifact_name="hparams.yaml", file_path=hparams_path)

    trainer.fit()


if __name__ == "__main__":
    main()
