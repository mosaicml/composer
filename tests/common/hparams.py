from typing import Any, Callable, Dict, Optional, Sequence, Type

import yahp as hp
import yaml


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
