# Copyright 2021 MosaicML. All Rights Reserved.

from typing import Protocol

from composer.core.types import StateDict

# Putting the typing annotation for Serializable here to avoid
# https://bugs.python.org/issue45121
class Serializable(Protocol):
    def state_dict(self) -> StateDict:
        pass

    def load_state_dict(self, state: StateDict) -> None:
        pass
