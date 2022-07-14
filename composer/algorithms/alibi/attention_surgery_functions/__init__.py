# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from composer.algorithms.alibi.attention_surgery_functions.utils import replacement_policy_mapping_builder

# Import files that add functions to the `replacement_policy_mapping_builder` registry in order to actually
# register those functions.
from composer.algorithms.alibi.attention_surgery_functions import ( # pyright: reportUnusedImport=none
    _bert,
    _gpt2,
)

__all__ = ['replacement_policy_mapping_builder']