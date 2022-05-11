# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from composer.core import Event


@pytest.mark.parametrize('event', list(Event))
def test_event_values(event: Event):
    assert event.name.lower() == event.value
