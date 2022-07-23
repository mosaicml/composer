# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# Import files that add functions to the `replacement_policy_mapping_builder` registry in order to actually
# register those functions.
try:
    from composer.algorithms.alibi.attention_surgery_functions import _bert, _gpt2  # pyright: reportUnusedImport=none
    from composer.algorithms.alibi.attention_surgery_functions.utils import replacement_policy_mapping_builder

except ImportError as e:
    # This triggers `alibi.py` to check for the `transformers` package
    replacement_policy_mapping_builder = {}

__all__ = ['replacement_policy_mapping_builder']
