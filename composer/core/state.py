# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Callable, ContextManager, Optional, Sequence, Union

import torch
import torch.nn.modules.utils
from torch.nn.parallel import DistributedDataParallel

import composer.core.types as types
from composer.core.precision import Precision
from composer.core.serializable import Serializable
from composer.utils import dist, ensure_tuple
from composer.utils.precision import default_precision_factory

if TYPE_CHECKING:
    from composer.core.callback import Callback
    from composer.core.types import Algorithm

logger = logging.getLogger(__name__)

# These fields will be serialized directly using torch.save / torch.load
DIRECT_SERIALIZATION_FIELDS = [
    "last_batch_size",
    "grad_accum",
    "_precision",
    "max_epochs",
    "epoch",
    "step",
]

# These fields will be serialized using .state_dict(), and loaded with .load_state_dict()
STATE_DICT_SERIALIZATION_FIELDS = [
    "model",
    "_optimizers",
    "_schedulers",
    "_algorithms",
    "_callbacks",
    "scaler",
]

# These fields will be serialized using .state_dict(), but will be skipped if DeepSpeed is enabled.
# When DeepSpeed is being used, model and optimizer states are serialized directly by the DeepSpeed engine.
STATE_DICT_SERIALIZATION_FIELDS_SKIP_DEEPSPEED = [
    "model",
    "_optimizers",
]

# These fields will not be serialized
SKIP_SERIALIZATION_FIELDS = [
    "loss", "batch", "outputs", "train_dataloader", "eval_dataloader", "_steps_per_epoch", "_precision_context"
]


class State(Serializable):
    """The class used to store the state of the trainer.

    Contains variables that the trainer tracks throughout the training loop.
    Note that the entire state is serialized when the trainer is checkpointed
    so that it can be used restore the trainer and continue training from a
    checkpoint. Algorithms are able to modify this object in-place.

    Attributes:
        model (types.Model, often BaseMosaicModel): The model, typically as a subclass of :class:`BaseMosaicModel`.
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

    def __init__(
            self,
            # model
            model: types.Model,

            # data configurations
            grad_accum: int,
            train_dataloader: types.DataLoader,
            eval_dataloader: types.DataLoader,

            # stopping conditions
            max_epochs: int,

            # precision
            precision: Union[str, types.Precision],
            steps_per_epoch: Optional[int] = None,
            precision_context: Callable[[Precision], ContextManager] = default_precision_factory(),

            # optimizers
            optimizers: Optional[types.Optimizers] = None,
            schedulers: Optional[types.Schedulers] = None,

            # scaler
            scaler: Optional[types.Scaler] = None,

            # algorithms and callbacks
            algorithms: Sequence[Algorithm] = tuple(),
            callbacks: Sequence[Callback] = tuple(),
    ):
        self.model = model
        self.grad_accum = grad_accum
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.last_batch_size = 0
        self.max_epochs = max_epochs
        self.step = 0
        self.epoch = 0
        self._precision = Precision(precision)
        self._steps_per_epoch = steps_per_epoch
        self._precision_context = precision_context

        self.loss: types.Tensors = torch.zeros(size=(1,))
        self.batch: types.Batch = {}
        self.outputs: types.Tensors = torch.zeros(size=(1,))

        if optimizers is None:
            self._optimizers = []
        else:
            self._optimizers = list(ensure_tuple(optimizers))

        if schedulers is None:
            self._schedulers = []
        else:
            self._schedulers = list(ensure_tuple(schedulers))

        self.scaler = scaler
        self._algorithms = list(algorithms)
        self._callbacks = list(callbacks)

    @property
    def optimizers(self):
        return self._optimizers

    @optimizers.setter
    def optimizers(self, optimizers: types.Optimizers):
        self._optimizers[:] = ensure_tuple(optimizers)

    @property
    def schedulers(self):
        return self._schedulers

    @schedulers.setter
    def schedulers(self, schedulers: types.Schedulers):
        self._schedulers[:] = ensure_tuple(schedulers)

    @property
    def callbacks(self):
        return self._callbacks

    @callbacks.setter
    def callbacks(self, callbacks: Sequence[Callback]):
        self._callbacks[:] = callbacks

    @property
    def algorithms(self):
        return self._algorithms

    @algorithms.setter
    def algorithms(self, algorithms: Sequence[Algorithm]):
        self._algorithms[:] = algorithms

    @property
    def train_batch_size(self):
        """The global batch size used for training."""
        if self.train_dataloader.batch_size is None:
            raise RuntimeError("train dataloader batch size is undefined")
        return self.train_dataloader.batch_size * dist.get_world_size()

    @property
    def eval_batch_size(self):
        """The batch size used for evaluation."""
        if self.eval_dataloader.batch_size is None:
            raise RuntimeError("eval dataloader batch size is undefined")
        return self.eval_dataloader.batch_size * dist.get_world_size()

    def state_dict(self) -> types.StateDict:
        """Returns the state as a :class:`dict`."""
        state_dict: types.StateDict = {}

        deepspeed_enabled = False
        try:
            import deepspeed
            deepspeed_enabled = isinstance(self.model, deepspeed.DeepSpeedEngine)
        except ImportError:
            pass

        for state_field_name, state_field_value in self.__dict__.items():
            if state_field_name in SKIP_SERIALIZATION_FIELDS:
                continue
            elif state_field_name in DIRECT_SERIALIZATION_FIELDS:
                state_dict[state_field_name] = state_field_value
                continue
            elif state_field_name in STATE_DICT_SERIALIZATION_FIELDS:
                if deepspeed_enabled and state_field_name in STATE_DICT_SERIALIZATION_FIELDS_SKIP_DEEPSPEED:
                    continue
                if state_field_name == "model":
                    # Save model directly instead of by class name, since model may be wrapped by DistributedDataParallel
                    serialized_value = state_field_value.state_dict()
                else:
                    serialized_value = {
                        obj.__class__.__qualname__: obj.state_dict()
                        for obj in ensure_tuple(state_field_value)
                        if obj is not None
                    }
                state_dict[state_field_name] = serialized_value

            else:
                raise RuntimeError(f"Unable to serialize field {state_field_name}")
        state_dict["_is_model_ddp_wrapped"] = isinstance(self.model, DistributedDataParallel)
        if deepspeed_enabled:
            state_dict["_deepspeed_enabled"] = True
        return state_dict

    def load_model_state(self, state_dict: types.StateDict, strict: bool):
        """
        Loads the model's state from a state_dict.

        Args:
            state_dict (types.StateDict): object returned from call to :meth:`state_dict`.
            strict (bool): whether the keys in the state_dict should perfectly match the keys in the model.
        """
        if state_dict["_is_model_ddp_wrapped"] and not isinstance(self.model, DistributedDataParallel):
            torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict['model'], "module.")
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict['model'], strict=strict)
            if len(missing_keys) > 0:
                logger.warning(f"Found these missing keys in the checkpoint: {', '.join(missing_keys)}")
            if len(unexpected_keys) > 0:
                logger.warning(f"Found these unexpected keys in the checkpoint: {', '.join(unexpected_keys)}")

    def load_state_dict(self, state: types.StateDict, strict: bool = False):
        """Loads the state.

        Args:
            state_dict (types.StateDict): object returned from call to :meth:`state_dict`.

        """

        deepspeed_enabled = False
        if "_deepspeed_enabled" in state:
            deepspeed_enabled = state["_deepspeed_enabled"]

        for state_field_name, state_field_value in self.__dict__.items():
            if state_field_name in SKIP_SERIALIZATION_FIELDS:
                continue
            elif state_field_name in DIRECT_SERIALIZATION_FIELDS:
                setattr(self, state_field_name, state[state_field_name])
            elif state_field_name in STATE_DICT_SERIALIZATION_FIELDS:
                if deepspeed_enabled and state_field_name in STATE_DICT_SERIALIZATION_FIELDS_SKIP_DEEPSPEED:
                    continue
                serialized_value = state[state_field_name]

                if state_field_name == "model":
                    self.load_model_state(state, strict=strict)
                else:
                    for target in ensure_tuple(state_field_value):
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
                raise RuntimeError(f"Unable to load field {state_field_name}")

    @property
    def batch_idx(self) -> int:
        """int: batch_idx is the index of the batch in the current epoch."""
        return self.step - self.epoch * self.steps_per_epoch

    @property
    def steps_per_epoch(self):
        """int: The maximum number of steps (batches) per epoch."""
        if self._steps_per_epoch is None:
            return len(self.train_dataloader)
        return self._steps_per_epoch

    @steps_per_epoch.setter
    def steps_per_epoch(self, val: Optional[int]):
        self._steps_per_epoch = val

    @property
    def precision(self):
        """The numerical precision to use for training. Should be one of ``[fp32, amp]``."""
        return self._precision

    @precision.setter
    def precision(self, precision: Union[str, types.Precision]):
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

    @property
    def precision_context(self):
        return self._precision_context(self.precision)
