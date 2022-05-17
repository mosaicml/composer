from typing import Any, Callable, Dict, Optional, Sequence, Type

import yahp as hp


def assert_registry_contains_entry(subclass: Type, registry: Dict[str, Callable], ignore_list: Sequence[Callable] = ()):
    """Assert that the ``registry`` contains """
    registry_entries = set(registry.values())
    registry_entries.union(ignore_list)
    assert subclass in registry_entries, f"Class {type(subclass).__name__} is missing from the registry."


def assert_is_constructable_from_yaml(
    constructor: Callable,
    yaml_dict: Optional[Dict[str, Any]] = None,
    expected: Any = None,
):
    yaml_dict = {} if yaml_dict is None else yaml_dict
    instance = hp.create(constructor, yaml_dict)
    if expected is not None:
        if isinstance(expected, type):
            assert isinstance(instance, expected)
        else:
            assert instance == expected
