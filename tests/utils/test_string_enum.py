# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# disabling unused class checks in this test, as string enum checks happen during class construction
# pyright: reportUnusedClass=none

import pytest

from composer.utils.string_enum import StringEnum


def test_string_enum_invalid_name():
    with pytest.raises(ValueError):
        # names must be uppercase
        class TestStringEnum(StringEnum):
            names_must_be_uppercase = 'names_must_be_uppercase'


def test_string_enum_invalid_value():
    with pytest.raises(ValueError):

        class TestStringEnum(StringEnum):
            VALUES_MUST_BE_LOWERCASE = 'VALUES_MUST_BE_LOWERCASE'


def test_string_enum_comparision():

    class TestStringEnum(StringEnum):
        HELLO_WORLD = 'hello_world'

    with pytest.warns(UserWarning):
        assert TestStringEnum.HELLO_WORLD == 'hello_world'

    with pytest.warns(UserWarning):
        assert TestStringEnum.HELLO_WORLD == 'HeLlO_WoRlD'


def test_missing():

    class TestStringEnum(StringEnum):
        HELLO_WORLD = 'hello_world'

    real_val = TestStringEnum.HELLO_WORLD
    assert real_val == TestStringEnum(real_val)
    assert real_val == TestStringEnum('HeLlO_WoRlD')

    with pytest.raises(ValueError):
        TestStringEnum('unknown_name')

    with pytest.raises(TypeError):
        TestStringEnum(object())
