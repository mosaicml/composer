#!/usr/bin/env python
# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Entrypoint that runs the Composer trainer on a provided YAML hparams file.

Example that trains MNIST with label smoothing:

.. code-block:: console

    python examples/run_composer_trainer.py -f composer/yamls/models/classify_mnist_cpu.yaml --algorithms label_smoothing --alpha 0.1
"""

import logging
import os
import sys
import tempfile
import warnings
from typing import Type

from composer.loggers import LogLevel
from composer.trainer.trainer_hparams import TrainerHparams
from composer.utils import dist


def _warning_on_one_line(message: str, category: Type[Warning], filename: str, lineno: int, file=None, line=None):
    # From https://stackoverflow.com/questions/26430861/make-pythons-warnings-warn-not-mention-itself
    return f'{category.__name__}: {message} (source: {filename}:{lineno})\n'


def _main():
    warnings.formatwarning = _warning_on_one_line

    global_rank = dist.get_global_rank()

    logging.basicConfig(
        # Example of format string
        # 2022-06-29 11:22:26,152: rank0[822018][MainThread]: INFO: composer.trainer.trainer: Using precision Precision.FP32
        # Including the PID and thread name to help with debugging dataloader workers and callbacks that spawn background
        # threads / processes
        format=f'%(asctime)s: rank{global_rank}[%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s')

    if len(sys.argv) == 1:
        sys.argv.append('--help')

    hparams = TrainerHparams.create(cli_args=True)  # reads cli args from sys.argv

    trainer = hparams.initialize_object()

    # if using wandb, store the config inside the wandb run
    try:
        import wandb
    except ImportError:
        pass
    else:
        if wandb.run is not None:
            wandb.config.update(hparams.to_dict())

    # Only log the config once, since it should be the same on all ranks.
    if dist.get_global_rank() == 0:
        with tempfile.TemporaryDirectory() as tmpdir:
            hparams_name = os.path.join(tmpdir, 'hparams.yaml')
            with open(hparams_name, 'w+') as f:
                f.write(hparams.to_yaml())
            trainer.logger.file_artifact(
                LogLevel.FIT,
                artifact_name=f'{trainer.state.run_name}/hparams.yaml',
                file_path=f.name,
                overwrite=True,
            )

    # Print the config to the terminal and log to artifact store if on each local rank 0
    if dist.get_local_rank() == 0:
        print('*' * 30)
        print('Config:')
        print(hparams.to_yaml())
        print('*' * 30)

    trainer.fit()


if __name__ == '__main__':
    _main()
