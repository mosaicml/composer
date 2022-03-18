# Copyright 2021 MosaicML. All Rights Reserved.
from functools import partial

from composer.utils import import_object


def test_dynamic_import_object():
    assert import_object("functools:partial") is partial
