# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Utilities for warnings."""

__all__ = ['VersionedDeprecationWarning']


class VersionedDeprecationWarning(DeprecationWarning):
    """A custom deprecation warning class that includes version information.

    Attributes:
        message (str): The deprecation message describing why the feature is deprecated.
        remove_version (str): The version in which the feature will be removed.
    """

    def __init__(self, message: str, remove_version: str) -> None:
        super().__init__(message + f' It will be removed in version {remove_version}.')
