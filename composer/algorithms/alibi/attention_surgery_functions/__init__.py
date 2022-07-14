# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from composer.algorithms.alibi.attention_surgery_functions.utils import replacement_policy_mapping_builder

# Import files that add to the `replacement_policy_mapping_builder`
from composer.algorithms.alibi.attention_surgery_functions import _bert, _gpt2
del _bert
del _gpt2

__all__ = ['replacement_policy_mapping_builder']