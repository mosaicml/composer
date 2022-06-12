# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import sys
import tempfile
import warnings

from composer.loggers import LogLevel, WandBLoggerHparams
from composer.trainer import TrainerHparams
from composer.utils import dist


def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return f'{category.__name__}: {message} (source: {filename}:{lineno})\n'


def main() -> None:
    warnings.formatwarning = warning_on_one_line

    if len(sys.argv) == 1:
        sys.argv = [sys.argv[0], "--help"]

    hparams = TrainerHparams.create(
        cli_args=True)  # reads cli args from sys.argv

    # if using wandb, store the config inside the wandb run
    for logger_hparams in hparams.loggers:
        if isinstance(logger_hparams, WandBLoggerHparams):
            logger_hparams.config = hparams.to_dict()

    trainer = hparams.initialize_object()

    # Only log the config once, since it should be the same on all ranks.
    if dist.get_global_rank() == 0:
        with tempfile.NamedTemporaryFile(mode="x+") as f:
            f.write(hparams.to_yaml())
            trainer.logger.file_artifact(
                LogLevel.FIT,
                artifact_name=f"{trainer.logger.run_name}/hparams.yaml",
                file_path=f.name,
                overwrite=True)

    # Print the config to the terminal
    if dist.get_local_rank() == 0:
        print("*" * 30)
        print("Config:")
        print(hparams.to_yaml())
        print("*" * 30)

    trainer.fit()


if __name__ == "__main__":
    main()
