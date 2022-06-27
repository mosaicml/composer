# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0
from functools import partial

from composer.utils import import_object


def test_dynamic_import_object():
    assert import_object('functools:partial') is partial
