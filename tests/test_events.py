# Copyright 2022 MosaicML. All Rights Reserved.

import pytest

from composer.core import Event


@pytest.mark.parametrize('event', list(Event))
def test_event_values(event: Event):
    assert event.name.lower() == event.value
