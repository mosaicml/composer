# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from composer.utils import convert_flat_dict_to_nested_dict, convert_nested_dict_to_flat_dict, extract_hparams


def test_convert_nested_dict_to_flat_dict():
    test_nested_dict = {'a': 1, 'b': {'c': 2, 'd': 3}, 'e': {'f': {'g': 4}}}

    expected_flat_dict = {'a': 1, 'b/c': 2, 'b/d': 3, 'e/f/g': 4}
    actual_flat_dict = convert_nested_dict_to_flat_dict(test_nested_dict)
    assert actual_flat_dict == expected_flat_dict


def test_convert_flat_dict_to_nested_dict():
    expected_nested_dict = {'a': 1, 'b': {'c': 2, 'd': 3}, 'e': {'f': {'g': 4}}}

    test_flat_dict = {'a': 1, 'b/c': 2, 'b/d': 3, 'e/f/g': 4}
    actual_nested_dict = convert_flat_dict_to_nested_dict(test_flat_dict)
    assert actual_nested_dict == expected_nested_dict


def test_extract_hparams():

    class Foo:

        def __init__(self):
            self.g = 7
            self.h = {'i': 8, 'j': 9}
            self.k = np.arange(10)
            self.q = range(20)

    class Bar:

        def __init__(self):
            self.local_hparams = {'m': 11, 'n': 12}

    locals_dict = {'a': 1, 'b': {'c': 2, 'd': 3}, 'e': [4, 5, 6], 'f': Foo(), 'p': Bar()}

    expected_parsed_dict = {
        'a': 1,
        'b': {
            'c': 2,
            'd': 3
        },
        'e': [4, 5, 6],
        'f': {
            'Foo': {
                'g': 7,
                'h': {
                    'i': 8,
                    'j': 9
                },
                'q': range(0, 20)
            }
        },
        'p': {
            'Bar': {
                'm': 11,
                'n': 12
            }
        },
    }

    parsed_dict = extract_hparams(locals_dict)
    assert parsed_dict == expected_parsed_dict
