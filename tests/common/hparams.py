# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict, Optional, Type, Union

import yahp as hp
import yaml


def assert_in_registry(constructor: Callable, registry: Dict[str, Callable]):
    """Assert that the ``registry`` contains ``constructor``."""
    registry_entries = set(registry.values())
    assert constructor in registry_entries, f"Constructor {constructor.__name__} is missing from the registry."


def assert_yaml_loads(
    constructor: Callable,
    yaml_dict: Optional[Dict[str, Any]] = None,
    expected: Optional[Union[Type[object], object]] = None,
):
    """Assert that ``yaml_dict``, when passed to ``constructor``, returns ``expected``.

    Args:
        constructor (Callable): The constructor to test (such as an Hparams class)
        yaml_dict (Dict[str, Any], optional): The YAML. Defaults to ``None``, which is equivalent
            to an empty dictionary.
        expected (Any, optional): The expected type or instance. If ``expected`` is a type, then
            it is expected that the constructor, when build from the ``yaml_dict``, returns an instance
            of this type. Otherwise, it is expected that the constructed instance is equal to ``expected``
    """
    yaml_dict = {} if yaml_dict is None else yaml_dict
    # ensure that yaml_dict is actually a dictionary of only json-serializable objects
    yaml_dict = yaml.safe_load(yaml.safe_dump(yaml_dict))
    # TODO(ravi) remove the following line once auto-yahp is merged
    assert isinstance(constructor, type) and issubclass(constructor, hp.Hparams)
    # TODO(ravi) change the following line to using the `hp.create` syntax
    instance = constructor.create(data=yaml_dict, cli_args=False)
    if expected is not None:
        if isinstance(expected, (type, tuple)):
            assert isinstance(instance, expected)
        else:
            assert instance == expected
