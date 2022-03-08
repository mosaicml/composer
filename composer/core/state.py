# Copyright 2021 MosaicML. All Rights Reserved.

"""The state of the trainer."""
from __future__ import annotations

import contextlib
import logging
import random
import textwrap
import warnings
from typing import TYPE_CHECKING, Callable, ContextManager, List, Optional, Sequence, Union, cast

import numpy as np
import torch
import torch.nn.modules.utils
from torch.nn.parallel import DistributedDataParallel

import composer.core.types as types
from composer.core.precision import Precision
from composer.core.serializable import Serializable
from composer.core.time import Time, Timer, TimeUnit
from composer.utils import dist, ensure_tuple

if TYPE_CHECKING:
    import deepspeed

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

    Contains variables that the trainer tracks throughout the training loop. Note that all the necessary parts (i.e.,
    :attr:`serialized_attributes`) of state are serialized when the trainer is checkpointed so that it can be used
    restore the trainer and continue training from a checkpoint.  :mod:`~composer.algorithms` are able to modify an
    instance of this class in-place.


    .. note::

        An instance of this class is automatically constructed by the :class:`~.Trainer` constructor. A user need
        not instantiate this class.

    Args:
        model (:attr:`~.types.Model`): The model, typically as a subclass of :class:`~.ComposerModel`.
        rank_zero_seed (int): The seed used on the rank zero process. It is assumed that each rank's seed is
            ``rank_zero_seed + dist.get_global_rank()``.
        grad_accum (int): The number of gradient accumulation steps to use. With this argument, micro batch size for
            each device becomes ``microbatch_size = train_batch_size / (num_devices * grad_accum)``.
        train_dataloader (types.DataLoader, DataSpec, or dict):
            The :class:`~.types.DataLoader`, :class:`~.DataSpec`, or dict of :class:`~.DataSpec` kwargs to used for training.
        evaluators (:attr:`~.types.Evaluators`):
            The :attr:`~.types.Evaluators` contain the evaluation datasets used for evaluation with specific metrics.
        max_duration (str or Time): The maximum duration to train for.
        precision (str | Precision): The numerical precision to use for training. See :class:`~.Precision` for
            the supported precisions.
        precision_context (Callable[[Precision], ContextManager]): Function to produce a context manager to mandate precision.
        optimizers (:attr:`~.types.Optimizers`, optional): The optimizers being used to train the model. Multiple optimizers are not currently supported.
        schedulers (:attr:`~.types.PyTorchScheduler` | List[:attr:`~.types.PyTorchScheduler`] | Tuple[:attr:`~.types.PyTorchScheduler`, ...], optional):
            The learning rate scheduler (can also be a list or tuple of schedulers).
        scaler (torch.cuda.amp.GradScaler, optional): The gradient scaler in use for mixed precision training.
        algorithms (Sequence[Algorithm]): The algorithms used for training.
        callbacks (Sequence[Callback]): The callbacks used for training.
        profiler (Optional[Profiler]): The Composer profiler.

    Attributes:
        batch (:attr:`~.types.Batch`): The batch. This will be the entire batch during the :attr:`.Event.AFTER_DATALOADER`, or a
            microbatch between :attr:`.Event.BATCH_START` and :attr:`.Event.BATCH_END`.
        batch_num_samples (int): The number of samples in the :attr:`batch`.
        batch_num_tokens (int): The number of tokens in the :attr:`batch`.

        loss (:attr:`~.types.Tensors`): The most recently computed loss.
        outputs (:attr:`~.types.Tensors`): The most recently computed output from the model's forward pass.
        timer (Timer): The timer that tracks training loop progress.
        serialized_attributes (List[str]): The names of the attribute which are serialized in a checkpoint.

            By default, the following attributes are serialized:

            +-----------------------+-------------------------------------------------------------+
            | Attribute             | Description                                                 |
            +=======================+=============================================================+
            | model                 | The model under training.                                   |
            +-----------------------+-------------------------------------------------------------+
            | optimizers            | The optimizers being used to train the model.               |
            +-----------------------+-------------------------------------------------------------+
            | schedulers            | The learning rate schedulers.                               |
            +-----------------------+-------------------------------------------------------------+
            | algorithms            | The algorithms used for training.                           |
            +-----------------------+-------------------------------------------------------------+
            | callbacks             | The callbacks used for training.                            |
            +-----------------------+-------------------------------------------------------------+
            | scaler                | The gradient scaler in use for mixed precision training.    |
            +-----------------------+-------------------------------------------------------------+
            | timer                 | The timer that tracks training loop progress.               |
            +-----------------------+-------------------------------------------------------------+
    """

    _max_duration: Time[int]
    _steps_per_epoch: Optional[int]
    batch: types.Batch
    batch_num_samples: int
    batch_num_tokens: int
    loss: types.Tensors
    outputs: types.Tensors
    _schedulers: List[types.PyTorchScheduler]

    def __init__(
            self,
            # model
            model: types.Model,

            # stopping conditions
            max_duration: Union[str, Time[int]],
            rank_zero_seed: int,

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
        self.rank_zero_seed = rank_zero_seed
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
        # These attributes will be serialized using .state_dict(), and loaded with .load_state_dict()
        # All other attributes will not be serialized.
        # For simplicity, omit the leading underscore for private attributes.
        # For example, even though the optimizers are stored on the state
        # as the "_optimizers" attribute, here we specify just "optimizers"
        self.serialized_attributes = [
            "model",
            "optimizers",
            "schedulers",
            "algorithms",
            "callbacks",
            "scaler",
            "timer",
            "rank_zero_seed",
            "rng",
        ]

    @property
    def seed(self):
        """The seed for the current rank."""
        return self.rank_zero_seed + dist.get_global_rank()

    @property
    def max_duration(self):
        """The maximum training duration."""
        return self._max_duration

    @max_duration.setter
    def max_duration(self, max_duration: Union[str, Time[int]]):
        if isinstance(max_duration, str):
            max_duration = cast(Time[int], Time.from_timestring(max_duration))
        if max_duration.unit == TimeUnit.DURATION:
            raise ValueError("TimeUnit.DURATION is not allowed as a unit for max_duration")
        self._max_duration = max_duration

    def get_elapsed_duration(self) -> Time[float]:
        """Get the elapsed training duration.

        Returns:
            Time: The elapsed duration, in :attr:`TimeUnit.DURATION`. ``Time(0.0, TimeUnit.DURATION)`` represents the
                beginning of training and ``Time(1.0, TimeUnit.DURATION)`` represents a completed training process.
        """
        return self.timer.get(self.max_duration.unit) / self.max_duration

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
    def schedulers(self, schedulers: types.PyTorchScheduler):
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

        for attribute_name in self.serialized_attributes:
            attribute_value = getattr(self, attribute_name)
            if attribute_name == "model":
                # Save model directly instead of by class name, since model may be wrapped by DistributedDataParallel
                serialized_value = attribute_value.state_dict()
            else:
                # Duck typing since runtime checkable protocols are not available in Python 3.7
                is_state_dict_serializable = any(hasattr(x, "state_dict") for x in ensure_tuple(attribute_value))
                if len(ensure_tuple(attribute_value)) > 0:
                    # any and all should be the same, except in for an empty collection, since
                    # any([]) is False, while all([]) is True
                    if is_state_dict_serializable != all(
                            hasattr(x, "state_dict") for x in ensure_tuple(attribute_value)):
                        raise RuntimeError(
                            f"Every member of {attribute_name} should support `state_dict`, or no members should support it."
                        )

                if is_state_dict_serializable:
                    serialized_value = {
                        obj.__class__.__qualname__: obj.state_dict() for obj in ensure_tuple(attribute_value)
                    }
                else:
                    serialized_value = attribute_value

            state_dict[attribute_name] = serialized_value

        state_dict["_is_model_ddp_wrapped"] = isinstance(self.model, DistributedDataParallel)
        return state_dict

    def load_model_state(self, state_dict: types.StateDict, strict: bool):
        """Loads the model's state from a state_dict.

        Args:
            state_dict (types.StateDict): The state dict, generated from a previous call to :meth:`state_dict`.
            strict (bool): Whether the keys (i.e., model parameter names) in the ``state_dict["model"]`` should
                perfectly match the keys in the ``self.model``.
        """
        if state_dict["_is_model_ddp_wrapped"] and not isinstance(self.model, DistributedDataParallel):
            torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict['model'], "module.")
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict['model'], strict=strict)
        if len(missing_keys) > 0:
            logger.warning(f"Found these missing keys in the checkpoint: {', '.join(missing_keys)}")
        if len(unexpected_keys) > 0:
            logger.warning(f"Found these unexpected keys in the checkpoint: {', '.join(unexpected_keys)}")

    def load_state_dict(self, state: types.StateDict, strict: bool = False) -> List[types.StateDict]:
        """Loads the state.

        Args:
            state (types.StateDict): object returned from call to :meth:`state_dict`.
            strict (bool): whether the keys in the ``state["model"]`` should perfectly match the keys in the
                ``self.model``. Defaults to False.
        Returns:
            List[types.StateDict]: The RNG state dicts, indexed by global rank.
        """

        rng_state_dicts = []

        for attribute_name, serialized_value in state.items():
            if attribute_name not in self.serialized_attributes:
                # it's possible some attributes we removed
                continue

            if attribute_name == "model":
                self.load_model_state(state, strict=strict)
            elif attribute_name == "rng":
                # rng state is NOT restored. Instead, it is returned so the trainer
                # can restore it later
                rng_state_dicts = serialized_value
                continue
            else:
                # Duck typing since runtime checkable protocols are not available in Python 3.7
                state_field_value = getattr(self, attribute_name)
                is_state_dict_serializable = any(hasattr(x, "load_state_dict") for x in ensure_tuple(state_field_value))
                if len(ensure_tuple(state_field_value)) > 0:
                    # any and all should be the same, except in for an empty collection, since
                    # any([]) is False, while all([]) is True
                    if is_state_dict_serializable != all(
                            hasattr(x, "load_state_dict") for x in ensure_tuple(state_field_value)):
                        raise RuntimeError(
                            f"Every member of {attribute_name} should support `load_state_dict`, or no members should support it."
                        )
                if is_state_dict_serializable:
                    for target in ensure_tuple(state_field_value):
                        if target.__class__.__qualname__ not in serialized_value:
                            warnings.warn(
                                f"{target.__class__.__qualname__} was not found in the state_dict. Its state will NOT be restored",
                                category=UserWarning)
                            continue
                        source = serialized_value[target.__class__.__qualname__]
                        target.load_state_dict(source)
                else:
                    # direct serialization
                    setattr(self, attribute_name, serialized_value)

        return rng_state_dicts

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

        See :class:`~.Precision` for the supported precisions.
        """
        return self._precision

    @precision.setter
    def precision(self, precision: Union[str, types.Precision]):
        self._precision = Precision(precision)

    @property
    def batch_pair(self) -> types.BatchPair:
        """:attr:`~.types.BatchPair`: The current batch, represented as a :attr:`~.types.BatchPair`.

        Raises:
            TypeError: If the current batch is not a :attr:`~.types.BatchPair`.
        """
        return types.as_batch_pair(self.batch)

    @property
    def batch_dict(self) -> types.BatchDict:
        """:attr:`~.types.BatchDict`: The current batch, represented as a :attr:`~.types.BatchDict`.

        Raises:
            TypeError: If the current batch is not a :attr:`~.types.BatchDict`.
        """
        return types.as_batch_dict(self.batch)

    @property
    def precision_context(self):
        return self._precision_context(self.precision)

    @property
    def devices(self) -> List[torch.device]:
        """The list of devices on which the model parameters reside.

        .. note::

            The ordering of the devices is stable with respect to the ordering of :meth:`torch.nn.Module.parameters`.

        Returns:
            List[torch.device]: The list of devices on which the model parameters reside.
        """
        parameters = list(self.model.parameters())

        devices: List[torch.device] = []
        for parameter in parameters:
            device = parameter.device
            if device not in devices:
                devices.append(device)

        return devices

    @property
    def rng(self) -> List[types.StateDict]:
        """The state of the RNG objects.

        Returns:
            List[types.StateDict]: A list of RNG State Dicts, indexed by global rank.
        """

        cuda_rng_state = []
        for device in self.devices:
            if device.type == "cpu":
                continue
            if device.type == "cuda":
                cuda_rng_state.append(torch.cuda.get_rng_state(device))
            else:
                raise RuntimeError(f"Unsupported device type: {device.type}")

        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
            "cuda": cuda_rng_state,
        }
        return dist.all_gather_object(rng_state)

    @rng.setter
    def rng(self, rng_state_dicts: List[types.StateDict]):
        """Restore the RNG state.

        Args:
            rng_state_dicts (List[types.StateDict]): The list of RNG state dicts to restore,
                as returned by ``self.rng``.
        """
        if dist.get_world_size() > len(rng_state_dicts):
            warnings.warn(
                textwrap.dedent(f"""\
                The current world size ({dist.get_world_size()} is greater than the number of RNG state(s) serialized
                ({len(rng_state_dicts)}). Only the first {len(rng_state_dicts)} rank(s) will have their RNG restored.
                """))
        if dist.get_world_size() < len(rng_state_dicts):
            warnings.warn(
                textwrap.dedent(f"""\
                The current world size ({dist.get_world_size()} is less than the number of RNG state(s) serialized
                ({len(rng_state_dicts)}). Only the first {dist.get_world_size()} RNG state(s) will be consumed;
                the remaining will be ignored."""))

        if dist.get_global_rank() < len(rng_state_dicts):
            rng_state_dict = rng_state_dicts[dist.get_global_rank()]
            torch.set_rng_state(rng_state_dict['torch'])
            random.setstate(rng_state_dict['python'])
            np.random.set_state(rng_state_dict['numpy'])

            cuda_devices = list(device for device in self.devices if device.type == "cuda")
            cuda_device_states = rng_state_dict["cuda"]
            if len(cuda_devices) > len(cuda_device_states):
                warnings.warn(
                    textwrap.dedent(f"""\
                    For rank {dist.get_global_rank()}, the number of CUDA devices ({len(cuda_devices)}) is greater than
                    the number of CUDA RNG state(s) serialized ({len(rng_state_dict["cuda"])}). Only the first
                    {len(rng_state_dict["cuda"])} will have their RNG states restored."""))

            if len(cuda_devices) < len(cuda_device_states):
                warnings.warn(
                    textwrap.dedent(f"""\
                    For rank {dist.get_global_rank()}, the number of CUDA devices ({len(cuda_devices)}) is less than
                    the number of CUDA RNG state(s) serialized ({len(rng_state_dict["cuda"])}). Only the first
                    {len(cuda_devices)} CUDA RNG state(s) will be consumed; the remaining will be ignored."""))
            for i, device in enumerate(cuda_devices):
                if i >= len(cuda_device_states):
                    # No more rng states to restore
                    break
                cuda_device_state = rng_state_dict["cuda"][i]
                torch.cuda.set_rng_state(cuda_device_state, device)

    @property
    def is_model_deepspeed(self) -> bool:
        """Whether :attr:`model` is an instance of a :class:`~deepspeed.DeepSpeedEngine`."""
        try:
            import deepspeed
        except ImportError:
            return False
        else:
            return isinstance(self.model, deepspeed.DeepSpeedEngine)

    @property
    def deepspeed_model(self) -> deepspeed.DeepSpeedEngine:
        """Cast :attr:`model` to :class:`~deepspeed.DeepSpeedEngine`."""
        if self.is_model_deepspeed:
            return cast("deepspeed.DeepSpeedEngine", self.model)
        raise TypeError("state.model is not a DeepSpeed model")
