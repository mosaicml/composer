# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Replaces all instances of `torch.nn.Dropout` with a `GyroDropout`.

By masking Dropout layer, this usually improves accuracy.
"""

from composer.algorithms.gyro_dropout.gyro_dropout import GyroDropout, GyroDropoutLayer, apply_gyro_dropout

__all__ = ['GyroDropoutLayer', 'GyroDropout', 'apply_gyro_dropout']
