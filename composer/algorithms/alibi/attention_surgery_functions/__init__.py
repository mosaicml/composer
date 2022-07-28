# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# Import files that add functions to the `policy_registry` registry in order to actually
# register those functions.
from composer.utils import MissingConditionalImportError

try:
    from composer.algorithms.alibi.attention_surgery_functions import _bert, _gpt2  # pyright: reportUnusedImport=none
    from composer.algorithms.alibi.attention_surgery_functions.utils import policy_registry
except ImportError as e:
    raise MissingConditionalImportError(extra_deps_group='nlp', conda_package='transformers') from e

__all__ = ['policy_registry']
