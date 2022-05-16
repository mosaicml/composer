from typing import Any, Callable, Dict

import yahp as hp


def test_registry(registry: Dict[str, Callable], defaults: Dict[str, Any]):
    for constructor in registry.values():
        hp.create(constructor, data=defaults, cli_args=False)
