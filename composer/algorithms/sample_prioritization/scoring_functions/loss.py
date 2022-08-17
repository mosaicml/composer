# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from composer.algorithms.sample_prioritization import register_scoring_fxn


@register_scoring_fxn('loss')  #  type: ignore
def dummy_fxn():
    """Dummy function to hold 'loss' in scoring function registry until actual loss
    function can be obtained."""
    pass
