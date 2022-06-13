# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict, Optional, TypeVar

import yahp as hp
import yaml

T = TypeVar('T')


def assert_in_registry(constructor: Callable, registry: Dict[str, Callable]):
    """Assert that the ``registry`` contains ``constructor``."""
    registry_entries = set(registry.values())
    assert constructor in registry_entries, f'Constructor {constructor.__name__} is missing from the registry.'


def construct_from_yaml(
    constructor: Callable[..., T],
    yaml_dict: Optional[Dict[str, Any]] = None,
) -> T:
    """Build ``constructor`` from ``yaml_dict``

    Args:
        constructor (Callable): The constructor to test (such as an Hparams class)
        yaml_dict (Dict[str, Any], optional): The YAML. Defaults to ``None``, which is equivalent
            to an empty dictionary.
    """
    yaml_dict = {} if yaml_dict is None else yaml_dict
    # ensure that yaml_dict is actually a dictionary of only json-serializable objects
    yaml_dict = yaml.safe_load(yaml.safe_dump(yaml_dict))
    instance = hp.create(constructor, yaml_dict, cli_args=False)
    return instance
