# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from composer.utils.misc import partial_format


def test_partial_format():
    # No args provided
    assert partial_format('{foo} {bar} {}') == '{foo} {bar} {}'

    # Keyword args
    assert partial_format('{foo} {bar}', foo='Hello') == 'Hello {bar}'
    assert partial_format('{foo} {bar}', foo='Hello', bar='World') == 'Hello World'

    # Positional args
    assert partial_format('{} {}', 'Hello') == 'Hello {}'
    assert partial_format('{} {}', 'Hello', 'World') == 'Hello World'

    # Positional and keyword args
    assert partial_format('{foo} {}', 'World') == '{foo} World'
    assert partial_format('{foo} {}', foo='Hello') == 'Hello {}'
    assert partial_format('{foo} {}', 'World', foo='Hello') == 'Hello World'
