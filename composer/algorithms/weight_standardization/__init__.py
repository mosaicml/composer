# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from composer.algorithms.weight_standardization.weight_standardization import \
    WeightStandardization as WeightStandardization
from composer.algorithms.weight_standardization.weight_standardization import \
    apply_weight_standardization as apply_weight_standardization

__all__ = ['WeightStandardization', 'apply_weight_standardization']
