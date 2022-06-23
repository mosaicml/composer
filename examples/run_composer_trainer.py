#!/usr/bin/env python
# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Entrypoint to train via hparams.

.. deprecated:: 0.8.0

    This entrypoint was deprecated in composer v0.8.0 and will be removed in v0.9.0.
    Instead, use the `composer-train` command -- e.g.

    .. code-block:: console

        composer-train -f path/to/hparams.yaml

    Or, with the launcher script:

    .. code-block:: console

        composer -c composer-train -f path/to/hparams.yaml
"""

import warnings

from composer.trainer.trainer_hparams import train_via_hparams

if __name__ == '__main__':
    warnings.warn(('examples/run_composer_trainer.py is deprecated. Instead, please use the `composer-train` command. '
                   'This entrypoint will be removed in Composer v0.9.0.'),
                  category=DeprecationWarning)
    train_via_hparams()
