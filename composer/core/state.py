# Copyright 2021 MosaicML. All Rights Reserved.

"""The state of the trainer."""
from __future__ import annotations

import contextlib
import logging
import textwrap
import warnings
from typing import TYPE_CHECKING, Callable, ContextManager, List, Optional, Sequence, Union, cast

import torch
import torch.nn.modules.utils
from torch.nn.parallel import DistributedDataParallel

import composer.core.types as types
from composer.core.precision import Precision
from composer.core.serializable import Serializable
from composer.core.time import Time, Timer, TimeUnit
from composer.utils import ensure_tuple

if TYPE_CHECKING:
    from composer.core.algorithm import Algorithm
    from composer.core.callback import Callback
    from composer.profiler import Profiler

__all__ = ["State"]

logger = logging.getLogger(__name__)


def _default_precision_factory() -> Callable[[Union[str, Precision]], ContextManager]:
    """Returns a context manager to automatically cast to a specific precision.

    Args:
        precision (str or Precision): Precision for the context
    """
    if torch.cuda.is_available():
        return lambda precision: torch.cuda.amp.autocast(Precision(precision) == Precision.AMP)
    else:

        def null(precision):
            assert Precision(
                precision) != Precision.AMP, "Precision AMP is only available when `torch.cuda.is_available() == True`."
            return contextlib.nullcontext()

        return null


class State(Serializable):
    """The state of the trainer.

    Contains variables that the trainer tracks throughout the training loop.
    Note that the entire state is serialized when the trainer is checkpointed
    so that it can be used restore the trainer and continue training from a
    checkpoint. Algorithms are able to modify this object in-place.


    .. note::

        To support multi-GPU training, :attr:`State.model` may be wrapped in :class:`DistributedDataParallel`,
        and the dataloaders may be wrapped in a device-specific dataloader that handles moving tensors to device.

    .. note::

        ``Schedulers`` are wrapped in ``ComposableScheduler``, which handles stepping either stepwise or epochwise,
        and also properly sets up learning rate warmups.



    Args:
        model (types.Model, often ComposerModel): The model, typically as a subclass of :class:`ComposerModel`.
        grad_accum (int): The number of gradient accumulation steps to use. The size of each microbatch is ``train_batch_size / num_gpus / grad_accum``.
        train_dataloader (types.DataLoader, types.DataSpec, or dict):
            The :class:`types.DataLoader`, :class:`types.DataSpec`, or dict of :class:`types.DataSpec` kwargs to used for training.
        evaluators (Evaluators):
            The :class:`types.Evaluators` contain the evaluation datasets used for evaluation with specific metrics.
        max_duration (str or Time): The maximum duration to train for.

        precision (str | Precision): The numerical precision to use for training. Should be one of ``[fp32, amp]``.
        precision_context ((precision: Precision) -> ContextManager): Function to produce a context manager to mandate precision.

        optimizers (types.Optimizers, optional): The optimizers being used to train the model. Multiple optimizers are not currently supported.
        schedulers (types.Schedulers, optional): The learning rate schedulers, typically wrapped in :class:`ComposableScheduler`.
        scaler (torch.cuda.amp.GradScaler, optional): The gradient scaler in use for mixed precision training.

        algorithms (Sequence[Algorithm]): The algorithms used for training.
        callbacks (Sequence[Callback]): The callbacks used for training.

        profiler (Optional[Profiler]): The Composer profiler.

    Attributes:
        batch (types.Batch): The batch. This will be the entire batch during the :attr:`Event.AFTER_DATALOADER`, or a
            microbatch between :attr:`Event.BATCH_START` and :attr:`Event.BATCH_END`.
        batch_num_samples (int): The number of samples in the :attr:`batch`.
        batch_num_tokens (int): The number of tokens in the :attr:`batch`.

        loss (types.Tensors): The most recently computed loss.
        outputs (types.Tensors): The most recently computed output from the model's forward pass.
        timer (types.Timer): The timer that tracks training loop progress.
        serialized_attributes (List[str]): The list of attributes which will be serialized in a checkpoint.
    """

    _max_duration: Time[int]
    _steps_per_epoch: Optional[int]
    batch: types.Batch
    batch_num_samples: int
    batch_num_tokens: int
    loss: types.Tensors
    outputs: types.Tensors
    _schedulers: List[types.Scheduler]

    # These attributes will be serialized using .state_dict(), and loaded with .load_state_dict()
    # All other attributes will not be serialized.
    # For simplicity, omit the leading underscore for private attributes.
    # For example, even though the optimizers are stored on the state
    # as the "_optimizers" attribute, here we specify just "optimizers"
    serialized_attributes = [
        "model",
        "optimizers",
        "schedulers",
        "algorithms",
        "callbacks",
        "scaler",
        "timer",
    ]

    def __init__(
            self,
            # model
            model: types.Model,

            # stopping conditions
            max_duration: Union[str, Time[int]],

            # data configurations
            train_dataloader: types.DataLoader,
            evaluators: types.Evaluators = [],
            grad_accum: int = 1,

            # precision
            precision: Union[str, types.Precision] = Precision.FP32,
            precision_context: Callable[[Precision], ContextManager] = _default_precision_factory(),

            # optimizers
            optimizers: Optional[types.Optimizers] = None,

            # scaler
            scaler: Optional[types.Scaler] = None,

            # algorithms and callbacks
            algorithms: Sequence[Algorithm] = tuple(),
            callbacks: Sequence[Callback] = tuple(),

            # steps per epoch
            steps_per_epoch: Optional[int] = None,
    ):
        self.model = model
        self.grad_accum = grad_accum
        self.train_dataloader = train_dataloader
        self.evaluators = list(ensure_tuple(evaluators))
        self.max_duration = max_duration
        self.steps_per_epoch = steps_per_epoch

        self.timer = Timer()
        self._precision = Precision(precision)
        self._precision_context = precision_context

        if optimizers is None:
            self._optimizers = []
        else:
            self._optimizers = list(ensure_tuple(optimizers))

        self._schedulers = []

        self.scaler = scaler
        self._algorithms = list(algorithms)
        self._callbacks = list(callbacks)

        self.profiler: Optional[Profiler] = None

    @property
    def epoch(self) -> int:
        """The index of the current epoch."""
        warnings.warn("TimeDeprecationWarning: state.epoch is deprecated. Please use state.timer.epoch",
                      category=DeprecationWarning)
        return self.timer.epoch.value

    @property
    def step(self) -> int:
        """The index of the current step/batch (measured globally)."""
        warnings.warn("TimeDeprecationWarning: state.step is deprecated. Please use state.timer.batch",
                      category=DeprecationWarning)
        return self.timer.batch.value

    @property
    def max_duration(self):
        return self._max_duration

    @max_duration.setter
    def max_duration(self, max_duration: Union[str, Time[int]]):
        if isinstance(max_duration, str):
            max_duration = cast(Time[int], Time.from_timestring(max_duration))
        if max_duration.unit != TimeUnit.EPOCH:
            raise NotImplementedError("Max duration must be specified in epochs. Other units are not yet supported.")
        if max_duration.unit == TimeUnit.DURATION:
            raise ValueError("TimeUnit.DURATION is not allowed as a unit for max_duration")
        self._max_duration = max_duration

    def get_elapsed_duration(self) -> Time[float]:
        """Get the elapsed training duration.

        Returns:
            Time: The elapsed duration, in ``TimeUnit.DURATION``.
        """
        return self.timer.get(self.max_duration.unit) / self.max_duration

    @property
    def max_epochs(self):
        """The maximum number of epochs to train for."""
        warnings.warn("TimeDeprecationWarning: state.max_epochs is deprecated. Please use state.max_duration",
                      category=DeprecationWarning)
        assert self.max_duration.unit == TimeUnit.EPOCH, "invariant violation -- max duration must be epochs for now"
        return self.max_duration.value

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

    def state_dict(self) -> types.StateDict:
        """Returns the state as a :class:`dict`."""
        state_dict: types.StateDict = {}

        for state_field_name, state_field_value in self.__dict__.items():
            if state_field_name.lstrip("_") not in self.serialized_attributes:
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

        state_dict["_is_model_ddp_wrapped"] = isinstance(self.model, DistributedDataParallel)
        return state_dict

    def load_model_state(self, state_dict: types.StateDict, strict: bool):
        """Loads the model's state from a state_dict.

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

        for state_field_name, state_field_value in self.__dict__.items():
            if state_field_name.lstrip("_") not in self.serialized_attributes:
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

    @property
    def batch_idx(self) -> int:
        """int: batch_idx is the index of the batch in the current epoch."""
        warnings.warn("TimeDeprecationWarning: state.batch_idx is deprecated. Please use state.timer.batch_in_epoch",
                      category=DeprecationWarning)
        return self.timer.batch_in_epoch.value

    @property
    def steps_per_epoch(self):
        """int: The maximum number of steps (batches) per epoch."""
        if self._steps_per_epoch is None:
            return len(self.train_dataloader)
        return self._steps_per_epoch

    @steps_per_epoch.setter
    def steps_per_epoch(self, steps_per_epoch: Optional[int]):
        try:
            dataloader_len = len(self.train_dataloader)
        except (TypeError, NotImplementedError):
            dataloader_len = None
        if dataloader_len is not None and steps_per_epoch is not None and steps_per_epoch > dataloader_len:
            warnings.warn(
                textwrap.dedent(f"""\
                    SubsetNumBatchesWarning: The steps_per_epoch({steps_per_epoch})
                    is greater than the number of batches in the training dataloader
                    ({dataloader_len})"""))
        self._steps_per_epoch = steps_per_epoch

    @property
    def precision(self):
        """The numerical precision to use for training.

        Should be one of ``[fp32, amp]``.
        """
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
