# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Callable, ContextManager, Optional, Sequence, Union

import torch
import torch.nn.modules.utils
from torch.nn.parallel import DistributedDataParallel

import composer.core.types as types
from composer.core.callback import Callback
from composer.core.precision import Precision
from composer.core.serializable import Serializable
from composer.utils import ensure_tuple
from composer.utils.ddp import get_global_rank, get_local_rank, get_local_world_size, get_world_size
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
    "precision",
    "precision_context",
]


@dataclass
class State(Serializable):
    """The class used to store the state of the trainer.

    Contains variables that the trainer tracks throughout the training loop.
    Note that the entire state is serialized when the trainer is checkpointed
    so that it can be used restore the trainer and continue training from a
    checkpoint. Algorithms are able to modify this object in-place.

    Attributes:
        model (types.Model, often BaseMosaicModel): The model, typically as a subclass of :class:`BaseMosaicModel`.
        train_batch_size (int): The global batch size used for training.
        eval_batch_size (int): The batch size used for evaluation.
        grad_accum (int): The number of gradient accumulation steps to use. The size of each microbatch is ``train_batch_size / num_gpus / grad_accum``.
        max_epochs (int): The maximum number of epochs to train for.
        precision (str | Precision): The numerical precision to use for training. Should be one of ``[fp32, amp]``.
        precision_context ((precision: Precision) -> ContextManager): Function to produce a context manager to mandate precision.

        epoch (int): The index of the current epoch.
        step (int): The index of the current step/batch (measured globally).

        batch (types.Batch): The most recently retrieved batch.
        loss (types.Tensors): The most recently computed loss.
        last_batch_size (int): The size of the batch last returned from the dataloader. This can be different from the current size of ``batch`` if algorithms have modified the ``batch``.
        outputs (types.Tensors): The most recently computed output from the model's forward pass.

        optimizers (types.Optimizers): The optimizers being used to train the model. Multiple optimizers are not currently supported.
        schedulers (types.Schedulers): The learning rate schedulers, typically wrapped in :class:`ComposableScheduler`.
        scaler (torch.cuda.amp.GradScaler, optional): The gradient scaler in use for mixed precision training.

        train_dataloader (types.DataLoader): The dataloader used for training.
        eval_dataloader (types.DataLoader): The dataloader used for evaluation.

        algorithms (Sequence[Algorithm]): The algorithms used for training.
        callbacks (Sequence[Callback]): The callbacks used for training.
    """

    # model
    model: types.Model

    # data configurations
    train_batch_size: int
    eval_batch_size: int
    grad_accum: int

    # stopping conditions
    max_epochs: int

    # dataloaders
    train_dataloader: types.DataLoader
    eval_dataloader: types.DataLoader

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
    loss: types.Tensors = field(default_factory=lambda: torch.zeros(size=(1,)))
    last_batch_size: int = 0

    batch: types.Batch = field(default_factory=dict)
    outputs: types.Tensors = field(default_factory=lambda: torch.zeros(size=(1,)))

    # optimizers
    optimizers: Optional[types.Optimizers] = None
    schedulers: Optional[types.Schedulers] = None

    # scaler
    scaler: Optional[types.Scaler] = None

    # algorithms
    algorithms: Sequence[Algorithm] = tuple()
    callbacks: Sequence[Callback] = tuple()

    @property
    def world_size(self) -> int:
        return get_world_size()

    @property
    def global_rank(self) -> int:
        return get_global_rank()

    @property
    def local_world_size(self) -> int:
        return get_local_world_size()

    @property
    def local_rank(self) -> int:
        return get_local_rank()

    @property
    def is_rank_zero(self) -> bool:
        return self.global_rank == 0

    def state_dict(self) -> types.StateDict:
        """Returns the state as a :class:`dict`."""
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
        """int: batch_idx is the index of the batch in the current epoch."""
        return self.step - self.epoch * self.steps_per_epoch

    @property
    def steps_per_epoch(self) -> int:
        """int: The number of steps (batches) per epoch."""
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
        """:class:`~types.BatchPair`: The current batch, represented as a :class:`~types.BatchPair`.

        Raises:
            TypeError: If the current batch is not a :class:`~types.BatchPair`.
        """
        return types.as_batch_pair(self.batch)

    @property
    def batch_dict(self) -> types.BatchDict:
        """:class:`~types.BatchDict`: The current batch, represented as a :class:`~types.BatchDict`.

        Raises:
            TypeError: If the current batch is not a :class:`~types.BatchDict`.
        """
        return types.as_batch_dict(self.batch)
