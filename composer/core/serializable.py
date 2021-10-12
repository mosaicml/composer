from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from composer.core.types import StateDict


class Serializable(Protocol):

    def state_dict(self) -> StateDict:
        """state_dict() returns a dictionary representing the internal state.
        The returned dictionary must be pickale-able via ::meth::`torch.save()`.
    
        Returns:
            StateDict: The state of the object
        """
        return {}

    def load_state_dict(self, state: StateDict) -> None:
        """load_state_dict() restores the state of the object

        Args:
            state (StateDict): The state of the object, as previously returned by ::meth::`state_dict`
        """
        pass
