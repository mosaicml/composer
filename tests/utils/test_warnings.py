# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import warnings

import pytest

from composer.utils.warnings import VersionedDeprecationWarning


def test_versioned_deprecation_warning():

    def deprecated_function():
        warnings.warn(VersionedDeprecationWarning('This function is deprecated.', remove_version='0.20.0'))

    with pytest.warns(
        VersionedDeprecationWarning,
        match='This function is deprecated. It will be removed in version 0.20.0.',
    ):
        deprecated_function()
