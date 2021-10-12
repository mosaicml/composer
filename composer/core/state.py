from __future__ import annotations

import logging
import warnings
from dataclasses import Field, dataclass, field, fields
from itertools import count
from typing import TYPE_CHECKING, Callable, ContextManager, Iterable, Optional, Sequence, Union

import torch.nn.modules.utils
from torch.nn.parallel import DistributedDataParallel

import composer.core.types as types
from composer.core.callback import Callback
from composer.core.precision import Precision
from composer.core.serializable import Serializable
from composer.utils import ensure_tuple, make_empty_tensor
from composer.utils.ddp import get_global_rank, is_rank_set
from composer.utils.precision import default_precision_factory

if TYPE_CHECKING:
    from composer.core.algorithm import Algorithm

logger = logging.getLogger(__name__)

# These fields will be serialized directly using torch.save / torch.load
DIRECT_SERIALIZATION_FIELDS = [
    "train_batch_size",
    "eval_batch_size",
    "last_batch_size",
    "grad_accum",
    "_precision",
    "max_epochs",
    "epoch",
    "step",
    "seed",
]

# These fields will be serialized using .state_dict(), and loaded with .load_state_dict()
STATE_DICT_SERIALIZATION_FIELDS = [
    "model",
    "optimizers",
    "schedulers",
    "algorithms",
    "callbacks",
    "scaler",
]

# These fields will not be serialized
SKIP_SERIALIZATION_FIELDS = [
    "loss",
    "batch",
    "outputs",
    "precision",
    "train_dataloader",
    "eval_dataloader",
    "world_size",
    "nproc_per_node",
    "precision",
    "precision_context",
]


@dataclass
class State(Serializable):
    """
    State attributes for the trainer. Algorithms can modify
    these states in-place as needed.
    """

    # model
    model: types.Model

    # data configurations
    train_batch_size: int
    eval_batch_size: int
    grad_accum: int

    # stopping conditions
    max_epochs: int

    # precision
    # storing precision internally so strings can be passed into the constructor and setter
    # but the getter will always return a Precision enum
    precision: Union[str, types.Precision]  # type: ignore
    _precision: types.Precision = field(init=False)  # but store an enum internally
    precision_context: Callable[[Union[str, Precision]], ContextManager] = \
        field(default_factory=default_precision_factory)

    # timing information
    epoch: int = 0  # epoch counter
    step: int = 0  # global step counter

    # transient tensors within training loop
    loss: types.Tensors = field(default_factory=make_empty_tensor)
    last_batch_size: int = 0

    batch: types.Batch = field(default_factory=dict)
    outputs: types.Tensors = field(default_factory=make_empty_tensor)

    # optimizers
    optimizers: Optional[types.Optimizers] = None
    schedulers: Optional[types.Schedulers] = None

    # scaler
    scaler: Optional[types.Scaler] = None

    # dataloaders
    train_dataloader: Optional[types.DataLoader] = None
    eval_dataloader: Optional[types.DataLoader] = None

    # algorithms
    algorithms: Sequence[Algorithm] = tuple()
    callbacks: Sequence[Callback] = tuple()

    # machine info
    world_size: int = 1
    nproc_per_node: int = 1

    # random seed
    seed: Optional[int] = None

    @property
    def global_rank(self) -> int:
        return get_global_rank()

    @property
    def local_rank(self) -> int:
        return self.global_rank % self.nproc_per_node

    @property
    def is_rank_zero(self) -> bool:
        return self.global_rank == 0

    @property
    def is_rank_set(self) -> bool:
        return is_rank_set()

    def get_epochs(self) -> Iterable[int]:
        return range(self.epoch, self.max_epochs) if self.max_epochs else count(self.epoch)

    def update_last(self, batch: types.Batch):
        """Convenience function to update the state after the
        dataloader with the batch.

        Args:
            batch (Batch): the batch returned by the dataloader
        """
        self.batch = batch

    def state_dict(self) -> types.StateDict:
        """Returns the state as a :class:`dict`.
        """
        state_dict: types.StateDict = {}

        for state_field in fields(self):
            if state_field.name in SKIP_SERIALIZATION_FIELDS:
                continue
            elif state_field.name in DIRECT_SERIALIZATION_FIELDS:
                state_dict[state_field.name] = getattr(self, state_field.name)
                continue
            elif state_field.name in STATE_DICT_SERIALIZATION_FIELDS:
                state_value = getattr(self, state_field.name)
                if state_field.name == "model":
                    # Save model directly instead of by class name, since model may be wrapped by DistributedDataParallel
                    serialized_value = state_value.state_dict()
                else:
                    serialized_value = {
                        obj.__class__.__qualname__: obj.state_dict()
                        for obj in ensure_tuple(state_value)
                        if obj is not None
                    }
                state_dict[state_field.name] = serialized_value
            else:
                raise RuntimeError(f"Unable to serialize field {state_field.name}")
        state_dict["_is_model_ddp_wrapped"] = isinstance(self.model, DistributedDataParallel)
        return state_dict

    def load_state_dict(self, state: types.StateDict):
        """Loads the state.

        Args:
            state_dict (types.StateDict): object returned from call to :meth:`state_dict`.

        """
        for state_field in fields(self):
            if state_field.name in SKIP_SERIALIZATION_FIELDS:
                continue
            elif state_field.name in DIRECT_SERIALIZATION_FIELDS:
                setattr(self, state_field.name, state[state_field.name])
            elif state_field.name in STATE_DICT_SERIALIZATION_FIELDS:
                state_value = getattr(self, state_field.name)
                serialized_value = state[state_field.name]

                if state_field.name == "model":
                    if state["_is_model_ddp_wrapped"] and not isinstance(self.model, DistributedDataParallel):
                        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(serialized_value, "module.")
                    state_value.load_state_dict(serialized_value)
                else:
                    for target in ensure_tuple(state_value):
                        if target is None:
                            continue
                        if target.__class__.__qualname__ not in serialized_value:
                            warnings.warn(
                                f"{target.__class__.__qualname__} was not found in the state_dict. Its state will NOT be restored",
                                category=UserWarning)
                            continue
                        source = serialized_value[target.__class__.__qualname__]
                        target.load_state_dict(source)
            else:
                raise RuntimeError(f"Unable to load field {state_field.name}")

    @property
    def batch_idx(self) -> int:
        """batch_idx is the index of the batch in the current epoch.
        """
        return self.step - self.epoch * self.steps_per_epoch

    @property
    def steps_per_epoch(self) -> int:
        if self.train_dataloader is None:
            raise RuntimeError("To determine the number of steps per epoch, state.train_dataloader must be set.")
        return len(self.train_dataloader)

    @property
    def precision(self) -> types.Precision:
        return self._precision

    @precision.setter
    def precision(self, precision: Union[str, types.Precision]):  # type: ignore
        self._precision = Precision(precision)

    @property
    def batch_pair(self) -> types.BatchPair:
        return types.as_batch_pair(self.batch)

    @property
    def batch_dict(self) -> types.BatchDict:
        return types.as_batch_dict(self.batch)


def is_field_serialized(f: Field) -> bool:
    if f.name in STATE_DICT_SERIALIZATION_FIELDS or f.name in DIRECT_SERIALIZATION_FIELDS:
        return True
    elif f.name in SKIP_SERIALIZATION_FIELDS:
        return False
    else:
        raise RuntimeError(f"Serialization method for field {f.name} not specified")
