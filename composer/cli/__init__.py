# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Composer CLI."""

import sys

from composer.cli.launcher import main

__all__ = ["main"]

if __name__ == "__main__":
    sys.exit(main())
