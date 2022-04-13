# Copyright 2021 MosaicML. All Rights Reserved.

"""The state of the trainer."""
from __future__ import annotations

import contextlib
import logging
import warnings
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Dict, List, Optional, Sequence, Union, cast

import torch
import torch.nn.modules.utils
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer

from composer.core.precision import Precision
from composer.core.serializable import Serializable
from composer.core.time import Time, Timer, TimeUnit
from composer.utils import dist, ensure_tuple

if TYPE_CHECKING:
    import deepspeed

    import composer.core.types as types
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


def _ensure_backwards_compatible_checkpointing(state_dict: Dict[str, Any]):
    # v0.4.1 removed the leading underscores for the keys in the state_dict
    # It also renamed _is_model_ddp_wrapped to is_model_ddp
    state = {}
    for k, v in state_dict.items():
        if k == "_is_model_ddp_wrapped":
            k = "is_model_ddp"
        if k.startswith("_"):
            k = k[1:]
        state[k] = v
    return state


_STATE_DICT_SERIALIZED_ATTRIBUTES = [
    # List of attributes that are serialized with state_dict
    # Only the attributes listed in state.serialized_attributes will actually be saved.
    "model",
    "optimizers",
    "schedulers",
    "algorithms",
    "callbacks",
    "scaler",
    "timer",
]


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
        model (torch.nn.Module): The model, typically as a subclass of :class:`~.ComposerModel`.
        rank_zero_seed (int): The seed used on the rank zero process. It is assumed that each rank's seed is
            ``rank_zero_seed + dist.get_global_rank()``.
        grad_accum (int, optional): The number of gradient accumulation steps to use. With this argument, micro batch size for
            each device becomes ``microbatch_size = train_batch_size / (num_devices * grad_accum)``.
        dataloader (types.DataLoader, optional): The active DataLoader.
        dataloader_len (int | Time[int], optional): The number of batches per dataloader iteration (e.g. epoch).
            The trainer will yield the first ``dataloader_len`` batches per iteration. If ``None`` (the default),
            the entire dataloader will be iterated over.
        dataloader_label (str, optional): The name for the dataloader. Required if ``dataloader`` is specified. (default: ``None``)
            By convention, the training dataloader is called ``'train'``. The evaluator dataloader is called
            ``'eval'``, or when multiple evaluators are used, the name of the evaluator.
        max_duration (str | Time, optional): The maximum duration to train for. (default: ``None``)
        precision (str | Precision): The numerical precision to use for training. See :class:`~.Precision` for
            the supported precisions.
        precision_context (Callable[[Precision], ContextManager]): Function to produce a context manager to mandate precision.
        optimizers (torch.optim.Optimizer | Sequence[torch.optim.Optimizer], optional): The optimizer being used to train the model.
            Multiple optimizers are not currently supported.
        schedulers (types.PyTorchScheduler | Sequence[types.PyTorchScheduler], optional):
            The learning rate scheduler (can also be a list or tuple of schedulers).
        scaler (torch.cuda.amp.GradScaler, optional): The gradient scaler in use for mixed precision training.
        algorithms (Algorithm | Sequence[Algorithm], optional): The algorithms used for training.
        callbacks (Callback | Sequence[Callback], optional): The callbacks used for training.
        profiler (Optional[Profiler]): The Composer profiler.

    Attributes:
        batch (types.Batch): The batch. This will be the entire batch during the :attr:`.Event.AFTER_DATALOADER`, or a
            microbatch between :attr:`.Event.BATCH_START` and :attr:`.Event.BATCH_END`.
        batch_num_samples (int): The number of samples in the :attr:`batch`.
        batch_num_tokens (int): The number of tokens in the :attr:`batch`.

        loss (torch.Tensor | Sequence[torch.Tensor]): The most recently computed loss.
        outputs (torch.Tensor | Sequence[torch.Tensor]): The most recently computed output from the model's forward pass.
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
            | rank_zero_seed        | The seed of the rank zero process.                          |
            +-----------------------+-------------------------------------------------------------+
    """

    _dataloader: Optional[types.DataLoader]
    _dataloader_label: Optional[str]
    _dataloader_len: Optional[Time[int]]
    _max_duration: Optional[Time[int]]
    batch: types.Batch
    batch_num_samples: int
    batch_num_tokens: int
    loss: Union[torch.Tensor, Sequence[torch.Tensor]]
    outputs: Union[torch.Tensor, Sequence[torch.Tensor]]
    _schedulers: List[types.PyTorchScheduler]

    def __init__(
        self,
        # model
        model: torch.nn.Module,

        # determinism
        rank_zero_seed: int,

        # stopping conditions
        max_duration: Optional[Union[str, Time[int]]] = None,

        # data configurations
        grad_accum: int = 1,
        dataloader: Optional[types.DataLoader] = None,
        dataloader_label: Optional[str] = None,
        dataloader_len: Optional[Union[int, Time[int]]] = None,

        # precision
        precision: Union[str, Precision] = Precision.FP32,
        precision_context: Callable[[Precision], ContextManager] = _default_precision_factory(),

        # optimizers
        optimizers: Optional[Union[Optimizer, Sequence[Optimizer]]] = None,

        # scaler
        scaler: Optional[torch.cuda.amp.grad_scaler.GradScaler] = None,

        # algorithms and callbacks
        algorithms: Optional[Union[Algorithm, Sequence[Algorithm]]] = None,
        callbacks: Optional[Union[Callback, Sequence[Callback]]] = None,
    ):
        self.rank_zero_seed = rank_zero_seed
        self.model = model
        self.grad_accum = grad_accum
        self.set_dataloader(dataloader, dataloader_label, dataloader_len)
        self.dataloader_len = dataloader_len
        self.max_duration = max_duration

        self.timer = Timer()
        self._precision = Precision(precision)
        self._precision_context = precision_context

        if optimizers is None:
            self._optimizers = []
        else:
            self._optimizers = list(ensure_tuple(optimizers))

        self._schedulers = []

        self.scaler = scaler
        self._algorithms = list(ensure_tuple(algorithms))
        self._callbacks = list(ensure_tuple(callbacks))

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
    def max_duration(self, max_duration: Optional[Union[str, Time[int]]]):
        if max_duration is None:
            self._max_duration = None
            return
        if isinstance(max_duration, str):
            max_duration = cast(Time[int], Time.from_timestring(max_duration))
        if max_duration.unit == TimeUnit.DURATION:
            raise ValueError("TimeUnit.DURATION is not allowed as a unit for max_duration")
        self._max_duration = max_duration

    def get_elapsed_duration(self) -> Optional[Time[float]]:
        """Get the elapsed training duration.

        Returns:
            Optional[Time[float]]: The elapsed duration, in :attr:`TimeUnit.DURATION`.
                ``Time(0.0, TimeUnit.DURATION)`` represents the beginning of training and ``Time(1.0, TimeUnit.DURATION)``
                represents a completed training process. Returns ``None`` if ``max_duration`` is None.
        """
        if self.max_duration is None:
            return None
        return self.timer.get(self.max_duration.unit) / self.max_duration

    @property
    def optimizers(self):
        return self._optimizers

    @optimizers.setter
    def optimizers(self, optimizers: Union[Optimizer, Sequence[Optimizer]]):
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

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state as a :class:`dict`."""
        state_dict = {}

        for attribute_name in self.serialized_attributes:
            attribute_value = getattr(self, attribute_name)
            if attribute_name == "model":
                # Save model directly instead of by class name, since model may be wrapped by DistributedDataParallel
                # If it is DDP wrapped, do not save the `module.` prefix, as that is an implmentation detail
                model_state = attribute_value.state_dict()
                if self.is_model_ddp:
                    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(model_state, "module.")
                serialized_value = model_state
            else:
                if attribute_name in _STATE_DICT_SERIALIZED_ATTRIBUTES:
                    serialized_value = {
                        type(obj).__qualname__: obj.state_dict() for obj in ensure_tuple(attribute_value)
                    }
                else:
                    serialized_value = attribute_value

            state_dict[attribute_name] = serialized_value

        return state_dict

    def load_model_state(self, state_dict: Dict[str, Any], strict: bool):
        """Loads the model's state from a state_dict.

        Args:
            state_dict (Dict[str, Any]): The state dict, generated from a previous call to :meth:`state_dict`.
            strict (bool): Whether the keys (i.e., model parameter names) in the model state dict should
                perfectly match the keys in the model instance.
        """
        if state_dict.get("is_model_ddp", False) and not self.is_model_ddp:
            # This check is for backwards compatibility, as pre-v0.6.0 checkpoints serialized the state
            # with the `module.` prefix
            torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict['model'], "module.")
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict['model'], strict=strict)
        if len(missing_keys) > 0:
            logger.warning(f"Found these missing keys in the checkpoint: {', '.join(missing_keys)}")
        if len(unexpected_keys) > 0:
            logger.warning(f"Found these unexpected keys in the checkpoint: {', '.join(unexpected_keys)}")

    def load_state_dict(self, state: Dict[str, Any], strict: bool = False):
        """Loads the state.

        Args:
            state (Dict[str, Any]): object returned from call to :meth:`state_dict`.
            strict (bool): whether the keys in the ``state["model"]`` should perfectly match the keys in the
                ``self.model``. Defaults to False.
        """

        state = _ensure_backwards_compatible_checkpointing(state)

        for attribute_name, serialized_value in state.items():
            if attribute_name not in self.serialized_attributes:
                # it's possible some attributes we removed
                continue

            if attribute_name == "model":
                self.load_model_state(state, strict=strict)
                continue
            state_field_value = getattr(self, attribute_name)
            if attribute_name in _STATE_DICT_SERIALIZED_ATTRIBUTES:
                for target in ensure_tuple(state_field_value):
                    if type(target).__qualname__ not in serialized_value:
                        warnings.warn(
                            f"{type(target).__qualname__} is not in the state_dict. Its state will not be restored.",
                            category=UserWarning)
                        continue
                    source = serialized_value[type(target).__qualname__]
                    target.load_state_dict(source)
            else:
                # direct serialization
                try:
                    setattr(self, attribute_name, serialized_value)
                except AttributeError:
                    # ignore AttributeError for properties that have getters but not setters.
                    pass

    @property
    def dataloader(self):
        """The dataloader."""
        return self._dataloader

    @property
    def dataloader_label(self):
        """The dataloader label. By convention, the training dataloader is called ``'train'``. The evaluator dataloader
        is called ``'eval'``, or when multiple evaluators are used, the name of the evaluator.

        Returns:
            Optional[str]: The dataloader label, or None if no dataloader is set.
        """
        return self._dataloader_label

    def set_dataloader(
        self,
        dataloader: Optional[types.DataLoader] = None,
        dataloader_label: Optional[str] = None,
        dataloader_len: Optional[Union[int, Time[int]]] = None,
    ):
        """Update the dataloader and dataloader label.

        Args:
            dataloader (types.DataLoader, optional): The dataloader. Defaults to None.
            dataloader_label (str, optional): The dataloader label. Must be ``None`` if and only if
                ``dataloader`` is None. Defaults to None.
            dataloader_len (int, int):
        """
        if (dataloader == None) != (dataloader_label == None):
            raise ValueError("Both `dataloader` and `dataloader_label` should be None, or neither should be None.")
        self._dataloader = dataloader
        self._dataloader_label = dataloader_label
        self.dataloader_len = dataloader_len  # setting it to None will do a failsafe read of len(dataloader)

    @property
    def dataloader_len(self):
        """The number of batches per dataloader iteration (e.g. epoch), as used by the trainer.

        Returns:
            Optional[Time[int]]: The number of batches per dataloader iteration (e.g. epoch), or None if no dataloader
            is defined or if the dataloader has an unknown length (e.g. streaming dataloaders).
        """
        return self._dataloader_len

    @dataloader_len.setter
    def dataloader_len(self, num_batches: Optional[Union[int, Time[int]]]):
        if isinstance(num_batches, int):
            num_batches = Time(num_batches, TimeUnit.BATCH)
        if self._dataloader is None:
            if num_batches is None:
                self._dataloader_len = None
                return
            raise RuntimeError("`State.dataloader_len` cannot be set if the dataloader is not defined.")
        try:
            dataloader_len = len(self._dataloader)
        except (TypeError, NotImplementedError):
            dataloader_len = None
        if dataloader_len is not None and num_batches is not None and int(num_batches) > dataloader_len:
            warnings.warn((f"DataloaderNumBatchesWarning: The dataloader_len ({int(num_batches)}) "
                           f"is greater than the length (i.e. number of batches) of the dataloader, which is "
                           f"{dataloader_len}. State.dataloader_len is thus being set to {dataloader_len}."))
            num_batches = Time(dataloader_len, TimeUnit.BATCH)
        if num_batches is None and dataloader_len is not None:
            # len(dataloader) is an approximation -- see https://pytorch.org/docs/stable/data.html.
            # However, in the worst case where additional last batches are dropped, this calculation should be
            # an over-estimate, leading to the entire dataloader still being iterated over.
            num_batches = Time(dataloader_len, TimeUnit.BATCH)
        self._dataloader_len = num_batches

    @property
    def precision(self):
        """The numerical precision to use for training.

        See :class:`~.Precision` for the supported precisions.
        """
        return self._precision

    @precision.setter
    def precision(self, precision: Union[str, Precision]):
        self._precision = Precision(precision)

    @property
    def batch_pair(self) -> types.BatchPair:
        """:attr:`~.types.BatchPair`: The current batch, represented as a :attr:`~.types.BatchPair`.

        Raises:
            TypeError: If the current batch is not a :attr:`~.types.BatchPair`.
        """
        from composer.core.types import as_batch_pair
        return as_batch_pair(self.batch)

    @property
    def batch_dict(self) -> types.BatchDict:
        """:attr:`~.types.BatchDict`: The current batch, represented as a :attr:`~.types.BatchDict`.

        Raises:
            TypeError: If the current batch is not a :attr:`~.types.BatchDict`.
        """
        from composer.core.types import as_batch_dict
        return as_batch_dict(self.batch)

    @property
    def precision_context(self):
        return self._precision_context(self.precision)

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
    def is_model_ddp(self):
        """Whether :attr:`model` is an instance of a :class:`.DistributedDataParallel`."""
        return isinstance(self.model, DistributedDataParallel)

    @property
    def deepspeed_model(self) -> deepspeed.DeepSpeedEngine:
        """Cast :attr:`model` to :class:`~deepspeed.DeepSpeedEngine`."""
        if self.is_model_deepspeed:
            return cast("deepspeed.DeepSpeedEngine", self.model)
        raise TypeError("state.model is not a DeepSpeed model")
