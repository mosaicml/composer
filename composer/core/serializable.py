# Copyright 2021 MosaicML. All Rights Reserved.

"""Serialization interface used by checkpointing."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from composer.core.types import StateDict

__all__ = ["Serializable"]


class Serializable:
    """Interface for serialization; used by checkpointing."""

    def state_dict(self) -> StateDict:
        """Returns a dictionary representing the internal state.

        The returned dictionary must be pickale-able via :func:`torch.save`.

        Returns:
            StateDict: The state of the object
        """
        return {}

    def load_state_dict(self, state: StateDict) -> None:
        """Restores the state of the object.

        Args:
            state (StateDict): The state of the object, as previously returned by :meth:`.state_dict`
        """
        pass
