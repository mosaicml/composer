# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Protocol

# Putting the typing annotation for Serializable here to avoid
# https://bugs.python.org/issue45121
class Serializable(Protocol):

    def state_dict(self) -> Dict[str, Any]:
        pass

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        pass
