# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""The state of the trainer."""
from __future__ import annotations

import collections.abc
import contextlib
import logging
import textwrap
import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Sequence, Union, cast

import torch
import torch.nn.modules.utils
from packaging import version
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Metric

from composer.core.data_spec import DataSpec
from composer.core.event import Event
from composer.core.precision import Precision
from composer.core.serializable import Serializable
from composer.core.time import Time, Timestamp, TimeUnit
from composer.devices import Device
from composer.utils import batch_get, batch_set, dist, ensure_tuple, get_composer_env_dict, is_model_deepspeed

if TYPE_CHECKING:
    import deepspeed

    import composer.core.types as types
    from composer.core.algorithm import Algorithm
    from composer.core.callback import Callback
    from composer.core.evaluator import Evaluator
    from composer.core.passes import AlgorithmPass
    from composer.loggers import Logger
    from composer.profiler import Profiler

__all__ = ['State']

log = logging.getLogger(__name__)


@contextmanager
def get_fsdp_rank0_cpu_save_context(obj: torch.nn.Module):
    if version.parse(torch.__version__) < version.parse('1.13.0'):
        raise RuntimeError('To use FSDP with Composer, you must use torch>=1.13.0.')
    from torch.distributed.fsdp import FullStateDictConfig
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType
    full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(obj, StateDictType.FULL_STATE_DICT, full_state_dict_config):
        yield


def get_fsdp_sharded_optim_state_dict(full_optim_state_dict: Dict[str, Any], model: torch.nn.Module):
    if version.parse(torch.__version__) < version.parse('1.13.0'):
        raise RuntimeError('To use FSDP with Composer, you must use torch>=1.13.0.')
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    return FSDP.scatter_full_optim_state_dict(full_optim_state_dict=full_optim_state_dict, model=model)


def get_fsdp_full_optim_state_dict(model: torch.nn.Module, optim: torch.optim.Optimizer, rank0_only: bool = True):
    if version.parse(torch.__version__) < version.parse('1.13.0'):
        raise RuntimeError('To use FSDP with Composer, you must use torch>=1.13.0.')
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    return FSDP.full_optim_state_dict(model=model, optim=optim, rank0_only=rank0_only)


def _ensure_backwards_compatible_checkpointing(state_dict: Dict[str, Any]):
    # v0.4.1 removed the leading underscores for the keys in the state_dict
    # It also renamed _is_model_ddp_wrapped to is_model_ddp
    state = {}
    for k, v in state_dict.items():
        if k == '_is_model_ddp_wrapped':
            k = 'is_model_ddp'
        if k.startswith('_'):
            k = k[1:]
        state[k] = v
    return state


_STATE_DICT_SERIALIZED_ATTRIBUTES = [
    # List of attributes that are serialized with state_dict
    # Only the attributes listed in state.serialized_attributes will actually be saved.
    'model',
    'optimizers',
    'schedulers',
    'algorithms',
    'callbacks',
    'scaler',
    'timestamp',
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
        run_name (str): The name for this training run.
        device (Device): The device used by this process. The trainer moves the model and loaded data to this device.
        grad_accum (int, optional): The number of gradient accumulation steps to use. With this argument, micro batch
            size for each device becomes ``microbatch_size = train_batch_size / (num_devices * grad_accum)``.
        eval_batch_split (int, optional): The mirror of grad_accum for eval. With this argument, micro batch
            size for each device becomes ``microbatch_size = eval_batch_size / (num_devices * eval_batch_split)``.
        device_train_microbatch_size (int, optional): The microbatch size for each device during training.
        auto_microbatching (bool, optional): Whether automatic microbatching is enabled.
        using_device_microbatch_size (bool, optional): Whether device_train_microbatch_size is set by the user.
        train_dataloader (Iterable, optional): Dataloader used for training
        evaluators (Evaluator | Evaluators, optional): :class:`.Evaluator` used for evaluation.
        dataloader (Iterable, optional): The active DataLoader.
        dataloader_len (int | Time[int], optional): The number of batches per dataloader iteration (e.g. epoch).
            The trainer will yield the first ``dataloader_len`` batches per iteration. If ``-1`` (the default),
            the entire dataloader will be iterated over.
        dataloader_label (str, optional): The name for the dataloader. Required if ``dataloader`` is specified.
            (default: ``None``)

            By convention, the training dataloader is called ``'train'``. The evaluator dataloader is called
            ``'eval'``, or when multiple evaluators are used, the name of the evaluator.
        dataset_state (Dict[str, Any], optional): Mapping of dataset split to its iteration state for resumption.
        dataset_resumption (Dict[str, Any], optional): Mapping of dataset split to whether resumption is used.
        max_duration (str | Time, optional): The maximum duration to train for. (default: ``None``)
        precision (str | Precision): The numerical precision to use for training. See :class:`~.Precision` for
            the supported precisions.
        optimizers (torch.optim.Optimizer | Sequence[torch.optim.Optimizer], optional): The optimizer being used to
            train the model. Multiple optimizers are not currently supported.
        schedulers (types.PyTorchScheduler | Sequence[types.PyTorchScheduler], optional):
            The learning rate scheduler (can also be a list or tuple of schedulers).
        scaler (torch.cuda.amp.GradScaler, optional): The gradient scaler in use for mixed precision training.
        algorithms (Algorithm | Sequence[Algorithm], optional): The algorithms used for training.
        callbacks (Callback | Sequence[Callback], optional): The callbacks used for training.
        deepspeed_config (Dict[str, Any], optional): The configuration dictionary for deepspeed.
        fsdp_config (Dict[str, Any], optional): The configuration dictionary for FSDP.

    Attributes:
        batch (types.Batch): The batch. This will be the entire batch during the :attr:`.Event.AFTER_DATALOADER`, or a
            microbatch between :attr:`.Event.BATCH_START` and :attr:`.Event.BATCH_END`.
        device (Device): The device used by this process. The trainer moves the model and loaded data to this device. This
            can be used in callbacks and algorithms to move data onto the correct device.
        train_metrics (Dict[str, Metric]): The current train metrics, organized by metric name. ``train_metrics`` will be deep-copied to
            ensure that each evaluator updates only its ``train_metrics``.

            For example:

            >>> trainer = Trainer(
            ...     ...,
            ...     train_dataloader=train_dataloader,
            ...     eval_dataloader=eval_dataloader,
            ... )
            >>> trainer.fit()
            >>> trainer.state.train_metrics
            {'Accuracy': Accuracy()}

        eval_metrics (Dict[str, Dict[str, Metric]]): The current evaluation metrics, organized
            by dataloader label and then by metric name. If not using an :class:`.Evaluator`,
            the eval dataloader is labeled ``'eval'``. Otherwise, in the case of having multiple evaluation datasets,
            the evaluator label is used. See the `Multiple Datasets Documentation <https://docs.mosaicml.com/en/stable/trainer/evaluation.html#multiple-datasets>`_
            for more information. ``eval_metrics`` will be deep-copied to ensure that each evaluator updates only its ``eval_metrics``.

            For example:
            >>> from torchmetrics import Accuracy
            >>> from composer.metrics.metrics import CrossEntropy
            >>> trainer = Trainer(
            ...     ...,
            ...     train_dataloader=train_dataloader,
            ...     eval_dataloader=eval_dataloader,
            ... )
            >>> trainer.fit()
            >>> trainer.state.eval_metrics
            {'eval': {'CrossEntropy': CrossEntropy(), 'Accuracy': Accuracy()}}

            Or, when using an :class:`.Evaluator` for multiple evaluation datasets:

            .. testsetup::

                eval_1_dl = eval_dataloader
                eval_2_dl = eval_dataloader

            >>> from torchmetrics import Accuracy
            >>> from composer.core import Evaluator
            >>> trainer = Trainer(
            ...     ...,
            ...     train_dataloader=train_dataloader,
            ...     eval_dataloader=[
            ...         Evaluator(label='eval1', dataloader=eval_1_dl, metric_names=['Accuracy']),
            ...         Evaluator(label='eval2', dataloader=eval_2_dl, metric_names=['Accuracy']),
            ...     ],
            ... )
            >>> trainer.fit()
            >>> trainer.state.eval_metrics
            {'eval1': {'Accuracy': Accuracy()}, 'eval2': {'Accuracy': Accuracy()}}
        eval_timestamp (Timestamp): The timestamp for the current evaluation dataloader. This timestamp is reset
            before the dataloader is evaluated. The :attr:`~Timestamp.epoch` attribute for this timestamp is always
            ``0``.
        grad_accum (int): The number of gradient accumulation steps per batch.
        device_train_microbatch_size (int): The size of each train microbatch per device.
        loss (torch.Tensor | Sequence[torch.Tensor]): The most recently computed loss.
        model (torch.nn.Module): The training model.

            .. note::

                When using DeepSpeed or multi-rank training, the model will be wrapped with
                :class:`~deepspeed.DeepSpeedEngine` or :class:`~torch.nn.parallel.DistributedDataParallel`,
                respectively.

        outputs (torch.Tensor | Sequence[torch.Tensor]): The most recently computed output from the model's forward
            pass.
        predict_timestamp (Timestamp): The timestamp for the current prediction dataloader. This timestamp is reset
            before the dataloader is used. The :attr:`~Timestamp.epoch` attribute for this timestamp is always
            ``0``.
        profiler (Profiler): The profiler (if profiling is enabled), or ``None`` if not profiling.
        rank_zero_seed (int): The seed of the rank zero process.
        run_name (str): The name for this training run.
        scaler (torch.cuda.amp.GradScaler): The gradient scaler if using mixed-precision training, or
            ``None`` if not using mixed-precision training.
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
            | timestamp             | The timestamp that tracks training loop progress.           |
            +-----------------------+-------------------------------------------------------------+
            | rank_zero_seed        | The seed of the rank zero process.                          |
            +-----------------------+-------------------------------------------------------------+
            | train_metrics         | The current training metrics                                |
            +-----------------------+-------------------------------------------------------------+
            | eval_metrics          | The current evaluation metrics                              |
            +-----------------------+-------------------------------------------------------------+
            | run_name              | The run name for training.                                  |
            +-----------------------+-------------------------------------------------------------+
            | dataset_state         | The dataset iteration state.                                |
            +-----------------------+-------------------------------------------------------------+

        timestamp (Timestamp): The current training timestamp.
    """

    def __init__(
        self,
        # model
        model: torch.nn.Module,

        # determinism
        rank_zero_seed: int,

        # run_name
        run_name: str,

        # device
        device: Device,

        # stopping conditions
        max_duration: Optional[Union[str, Time[int]]] = None,

        # data configurations
        grad_accum: Optional[int] = 1,
        eval_batch_split: int = 1,
        device_train_microbatch_size: Optional[int] = None,
        auto_microbatching: bool = False,
        using_device_microbatch_size: bool = True,

        # dataloaders
        train_dataloader: Optional[Iterable] = None,
        evaluators: Optional[Union[Evaluator, Sequence[Evaluator]]] = None,

        # these track the current 'active' dataloader
        # depending on train, eval, or others
        dataloader: Optional[Iterable] = None,
        dataloader_label: Optional[str] = None,
        dataloader_len: Union[int, Time[int]] = -1,
        dataset_state: Optional[Dict[str, Any]] = None,
        dataset_resumption: Optional[Dict[str, Any]] = None,

        # precision
        precision: Union[str, Precision] = Precision.FP32,

        # optimizers
        optimizers: Optional[Union[Optimizer, Sequence[Optimizer]]] = None,

        # scaler
        scaler: Optional[torch.cuda.amp.grad_scaler.GradScaler] = None,

        # algorithms and callbacks
        algorithms: Optional[Union[Algorithm, Sequence[Algorithm]]] = None,
        callbacks: Optional[Union[Callback, Sequence[Callback]]] = None,

        # Distributed training configs
        deepspeed_config: Optional[Dict[str, Any]] = None,
        fsdp_config: Optional[Dict[str, Any]] = None,
    ):
        self.rank_zero_seed = rank_zero_seed
        self.model = model
        self.run_name = run_name
        self.device = device
        self.grad_accum = grad_accum
        self.eval_batch_split = eval_batch_split
        self.device_train_microbatch_size = device_train_microbatch_size
        self.auto_microbatching = auto_microbatching
        self.using_device_microbatch_size = using_device_microbatch_size
        self._dataloader_len = None
        self._dataloader = None
        self._dataloader_label = None
        self.set_dataloader(dataloader, dataloader_label, dataloader_len)
        self.dataset_state = dataset_state
        self.dataset_resumption = dataset_resumption or {}
        self._max_duration = None
        self.max_duration = max_duration

        self._train_dataloader = train_dataloader
        self._evaluators = list(ensure_tuple(evaluators))

        self.timestamp = Timestamp()
        self.eval_timestamp = Timestamp()
        self.predict_timestamp = Timestamp()
        self._precision = Precision(precision)

        if optimizers is None:
            self._optimizers = []
        else:
            self._optimizers = list(ensure_tuple(optimizers))

        self._schedulers = []

        self.scaler = scaler
        self._algorithms = list(ensure_tuple(algorithms))
        self._callbacks = list(ensure_tuple(callbacks))

        self.profiler: Optional[Profiler] = None

        self.deepspeed_config = deepspeed_config
        self.fsdp_config = fsdp_config

        # Set defaults for transient variables (to make pyright happy)
        self.batch: Any = None
        self.loss: Union[torch.Tensor, Sequence[torch.Tensor]] = torch.Tensor()
        self.outputs: Union[torch.Tensor, Sequence[torch.Tensor]] = torch.Tensor()

        # These attributes will be serialized using .state_dict(), and loaded with .load_state_dict()
        # All other attributes will not be serialized.
        # For simplicity, omit the leading underscore for private attributes.
        # For example, even though the optimizers are stored on the state
        # as the "_optimizers" attribute, here we specify just "optimizers"
        self.serialized_attributes = [
            'model',
            'optimizers',
            'schedulers',
            'algorithms',
            'callbacks',
            'scaler',
            'timestamp',
            'rank_zero_seed',
            'train_metrics',
            'eval_metrics',
            'run_name',
            'dataset_state',
        ]

        self.train_metrics: Dict[str, Metric] = {}
        self.eval_metrics: Dict[str, Dict[str, Metric]] = {}
        self.train_metric_values: Dict[str, float] = {}
        self.eval_metric_values: Dict[str, float] = {}
        self.total_loss_dict: Dict[str, float] = {}

    def _dataset_of(self, dataloader: Optional[Union[Evaluator, DataSpec, DataLoader, Iterable]]) -> Optional[Dataset]:
        """Get the dataset contained by the given dataloader-like object.

        Args:
            dataloader (Evaluator | DataSpec | DataLoader | Iterable, optional): The dataloader, wrapped dataloader, or
                generic python iterable to get the dataset of, if applicable.

        Returns:
            Dataset: Its dataset, if there is one.
        """
        from composer.core.evaluator import Evaluator

        # If it's None, no dataset for you.
        if dataloader is None:
            return None

        # An Evaluator is a dataloader wrapped with metrics. Unwrap its dataloader.
        if isinstance(dataloader, Evaluator):
            dataloader = dataloader.dataloader

        # A DataSpec is a dataloader wrapped with an on-device transform. Unwrap its dataloader.
        if isinstance(dataloader, DataSpec):
            dataloader = dataloader.dataloader

        # If what we now have is an actual DataLoader, return its dataset. If not, return None.
        if isinstance(dataloader, DataLoader):
            return dataloader.dataset
        else:
            return None

    @property
    def train_dataloader(self) -> Optional[Union[Iterable, DataLoader]]:
        """Get the train dataloader.

        Returns:
            Iterable | DataLoader, optional: The dataloader.
        """
        return self._train_dataloader

    @train_dataloader.setter
    def train_dataloader(self, train_dataloader: Optional[Union[Iterable, DataLoader]]):
        """Set the train dataloader.

        Args:
            train_dataloader (Iterable | DataLoader, optional): The dataloader.
        """
        self._train_dataloader = train_dataloader
        # Load dataset state from checkpoint when train_dataloader is set
        if self.dataset_state:
            dataset = self._dataset_of(self._train_dataloader)
            if hasattr(dataset, 'load_state_dict'):
                dataset.load_state_dict(self.dataset_state['train'])  # pyright: ignore
                self.dataset_resumption['train'] = True
            self.dataset_state['train'] = None

    @property
    def current_metrics(self):
        warnings.warn(
            'The ``current_metrics`` argument for a :class:`Trainer`. state is deprecated and will be removed in the future. Please use ``train_metrics`` and'
            '``eval_metrics`` instead.')
        return {'train': self.train_metrics, **self.eval_metrics}

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
            raise ValueError('TimeUnit.DURATION is not allowed as a unit for max_duration')
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
        return self.timestamp.get(self.max_duration.unit) / self.max_duration

    def stop_training(self):
        """Gracefully stop training.

        The current batch of training will finish, and any scheduled evaluation,
        logging, and evaluation for that batch, as well as any epoch end events.
        """
        self.max_duration = self.timestamp.batch

    @property
    def optimizers(self):
        """The optimizers."""
        return self._optimizers

    @optimizers.setter
    def optimizers(self, optimizers: Union[Optimizer, Sequence[Optimizer]]):
        self._optimizers[:] = ensure_tuple(optimizers)

    @property
    def schedulers(self):
        """The schedulers."""
        return self._schedulers

    @schedulers.setter
    def schedulers(self, schedulers: Union[types.PyTorchScheduler, Sequence[types.PyTorchScheduler]]):
        self._schedulers[:] = ensure_tuple(schedulers)

    def batch_get_item(self, key: Union[str, int, Callable, Any]) -> Any:
        """Gets element from batch either specified by key or user-specified function.

        See batch_get in `utils/batch_helpers.py` for examples.

        Args:
            key (str | int | Tuple[Callable, Callable] | Any, optional): A key to index into the batch or a
                user-specified function to do the extracting. A pair of callables is also
                supported for cases where a get and set function pair are both passed
                (like in Algorithms). The getter is assumed to be the first of the pair.


        Returns:
            The part of the batch specified by the key. This could be any type
                depending on what the batch is composed of.
        """
        return batch_get(self.batch, key)

    def batch_set_item(self, key: Union[str, int, Callable, Any], value: Any):
        """Sets the element specified by the key of the set_fn to the specified value.

        This is not an in-place operation, as for tuple-typed batches, a new batch object
        must be created to modify them.

        See batch_set in `utils/batch_helpers.py` for examples.

        Args:
            key (str | int | Tuple[Callable, Callable] | Any, optional): A key to index into the batch or a user-specified
                function to do the setting. A pair of callables is also supported for
                cases where a get and set function pair are both passed (like in
                Algorithms). The setter is assumed to be the second of the pair.
            value (Any): The value that batch[key] or batch.key gets set to or that the
                user-defined set function sets a part of the batch to.

        Returns:
            batch (Any): The updated batch with value set at key.
        """
        self.batch = batch_set(self.batch, key=key, value=value)

    @property
    def callbacks(self):
        """The callbacks."""
        return self._callbacks

    @callbacks.setter
    def callbacks(self, callbacks: Sequence[Callback]):
        self._callbacks[:] = callbacks

    @property
    def algorithms(self):
        """The algorithms."""
        return self._algorithms

    @algorithms.setter
    def algorithms(self, algorithms: Sequence[Algorithm]):
        self._algorithms[:] = algorithms

    @property
    def evaluators(self):
        """The evaluators."""
        return self._evaluators

    @evaluators.setter
    def evaluators(self, evaluators: Union[Evaluator, Sequence[Evaluator]]):
        self._evaluators[:] = list(ensure_tuple(evaluators))
        # Load dataset state from checkpoint when evaluators are set
        if self.dataset_state:
            state = self.dataset_state['eval']
            for evaluator in self._evaluators:
                dataset = self._dataset_of(evaluator)
                if hasattr(dataset, 'load_state_dict') and evaluator.label in state:
                    dataset.load_state_dict(state[evaluator.label])  # pyright: ignore
            del self.dataset_state['eval']

    @property
    def deepspeed_enabled(self):
        """Indicates if deepspeed is enabled."""
        return self.deepspeed_config is not None

    @property
    def fsdp_enabled(self):
        """Indicates if FSDP is enabled."""
        if version.parse(torch.__version__) < version.parse('1.13.0'):
            return False
        from torch.distributed.fsdp import FullyShardedDataParallel
        for module in self.model.modules():
            if isinstance(module, FullyShardedDataParallel):
                return True
        return False

    def _get_integrations_state_dict(self) -> Dict[str, Any]:
        """Gets a dictionary of information about integrations to store in the state dict.

        This metadata is used for loading things from state dict that need to be done outside
        of the normal Composer load path (e.g. HuggingFace model/tokenizer).
        """
        from composer.models import HuggingFaceModel
        integrations = {}
        if isinstance(self.model, HuggingFaceModel):
            integrations['huggingface'] = self.model.get_metadata()
        return integrations

    def _get_state_metadata(self) -> Dict[str, Any]:
        """Gets a dictionary of metadata to store in the state dict.

        This metadata is used for checking compatibility between the current environment/setup
        and the environment/setup that was used for the checkpoint that is being loaded in
        """
        metadata_dict = {}
        metadata_dict['composer_env_info'] = get_composer_env_dict()

        return metadata_dict

    def _dataset_state_dict(self) -> Dict[str, Any]:
        """Collect the state dict(s) of our train and eval dataset(s).

        Returns:
            Dict[str, Any]: The state dict(s).
        """
        obj = {
            'train': None,
            'eval': {},
        }

        dataset = self._dataset_of(self.train_dataloader)
        if hasattr(dataset, 'state_dict'):
            num_samples = int(self.timestamp.sample_in_epoch.value)
            obj['train'] = dataset.state_dict(num_samples, True)  # pyright: ignore

        for evaluator in self.evaluators:
            dataset = self._dataset_of(evaluator)
            if hasattr(dataset, 'state_dict'):
                # Don't save eval sample because we do not checkpoint during eval.
                obj['eval'][evaluator.label] = dataset.state_dict(0, True)  # pyright: ignore

        return obj

    def state_dict(self) -> Dict[str, Any]:
        """Collect the state dicts of our serializable attributes.

        Returns:
            Dict[str, Any]: The state dict.
        """
        state_dict = {}

        for attribute_name in self.serialized_attributes:
            attribute_value = getattr(self, attribute_name)
            if attribute_name == 'dataset_state':
                serialized_value = self._dataset_state_dict()
            elif attribute_name == 'model':
                # Save model directly instead of by class name, since model may be wrapped by DistributedDataParallel
                # If it is DDP wrapped, do not save the `module.` prefix, as that is an implmentation detail
                with get_fsdp_rank0_cpu_save_context(
                        attribute_value) if self.fsdp_enabled else contextlib.nullcontext():
                    model_state = attribute_value.state_dict()

                if self.is_model_ddp:
                    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(model_state, 'module.')
                serialized_value = model_state
            elif attribute_name == 'optimizers':
                if self.fsdp_enabled:
                    serialized_value = {}
                    for obj in ensure_tuple(attribute_value):
                        serialized_value = {
                            type(obj).__qualname__:
                                get_fsdp_full_optim_state_dict(model=self.model, optim=obj, rank0_only=True)
                        }
                else:
                    serialized_value = {
                        type(obj).__qualname__: obj.state_dict() for obj in ensure_tuple(attribute_value)
                    }
            elif attribute_name == 'algorithms':
                # Store as list to preserve order in which algorithms were applied
                serialized_value = [(type(obj).__qualname__, obj.state_dict()) for obj in ensure_tuple(attribute_value)]
            elif attribute_name in _STATE_DICT_SERIALIZED_ATTRIBUTES:
                serialized_value = {type(obj).__qualname__: obj.state_dict() for obj in ensure_tuple(attribute_value)}
            else:
                serialized_value = attribute_value

            state_dict[attribute_name] = serialized_value

        state_dict['integrations'] = self._get_integrations_state_dict()
        state_dict['metadata'] = self._get_state_metadata()

        return state_dict

    def _apply_required_algorithms(
        self,
        state_dict: Dict[str, Any],
        logger: Logger,
        exclude_algorithms: Optional[List[str]] = None,
        algorithm_passes: Optional[List[AlgorithmPass]] = None,
    ):
        """Applies required algorithms which haven't been specified and aren't in the exclude list.

        Args:
            state_dict (Dict[str, Any]): State from checkpoint.
            logger (Logger): Logger to use.
            exclude_algorithms (List[str], optional): List of algorithm names to exclude. (default: ``None``)
            algorithm_passes (List[AlgorithmPass], optional): A list of algorithm passes to apply to autoloaded algorithms
                to sort them into the correct order. (default: ``None``)
        """
        # Don't try to autoload on old checkpoints
        if not isinstance(state_dict['algorithms'], list):
            return

        import composer.algorithms as algorithms  # type: ignore imports used in `eval(representation)`

        # Get repr of existing algorithms
        current_algos = {}
        for algo in self.algorithms:
            if algo.required_on_load():
                if type(algo) not in current_algos:
                    current_algos[type(algo)] = []
                current_algos[type(algo)].append(algo.__repr__())

        # Gather algorithms to apply
        missing_algos = set()
        missing_algo_names = []
        missing_algo_reprs = []
        for algo_name, serialized_value in state_dict['algorithms']:
            # Check if required algorithm
            if hasattr(algorithms, algo_name) and getattr(algorithms, algo_name).required_on_load():
                # Check that algorithm is not explicitly excluded by user
                if exclude_algorithms is None or algo_name not in exclude_algorithms:
                    try:
                        algo = eval(f"algorithms.{serialized_value['repr']}")
                    except:
                        warnings.warn(
                            textwrap.dedent(
                                f"required_on_load algorithm {serialized_value['repr']} was enabled when training the "
                                f'loaded checkpoint. Attempted to check its presence but recreating the algorithm '
                                "failed. This may be due to a change in the algorithm's API. If this required_on_load "
                                'algorithm is not properly specified, it may lead to unexpected behavior, including '
                                'failing to load weights for some layers.'))
                        continue
                    # Raise warning if we are unable to safely autoapply
                    if type(algo) in current_algos and not serialized_value['repr'] in current_algos[type(algo)]:
                        warnings.warn(
                            textwrap.dedent(
                                f"required_on_load algorithm {serialized_value['repr']} was enabled when training the "
                                f"loaded checkpoint but is now specified in the following forms: {', '.join(current_algos[type(algo)])}."
                                'Potential parameter discrepancies for this required_on_load algorithm may lead to '
                                'unexpected behavior, including failing to load weights for some layers.'))
                    # Otherwise, queue algorithm to be autoappled
                    elif type(algo) not in current_algos:
                        missing_algos.add(algo)
                        missing_algo_names.append(algo_name)
                        missing_algo_reprs.append(serialized_value['repr'])
                        self.algorithms.append(algo)

        # Reorder algorithms based on algorithm_passes from engine
        algo_list = self.algorithms
        if algorithm_passes is not None:
            for algo_pass in algorithm_passes:
                algo_list = algo_pass(algo_list, Event.INIT)
        # Raise ValueError if algorithm_passes order any checkpoint algorithm before an already
        # applied user specified algorithm
        encountered_ckpt_algo = False
        for algo in algo_list:
            if algo in missing_algos:
                encountered_ckpt_algo = True
            elif encountered_ckpt_algo:
                raise ValueError(
                    textwrap.dedent('The following algorithms were enabled when training this checkpoint '
                                    f'and are required to successfully load it: {missing_algo_reprs}. '
                                    'Attempted to autocreate and apply required algorithms, but at least one '
                                    'of the loaded algorithms was ordered before a user specified algorithm '
                                    'which has already been applied, preventing automatic application of '
                                    'algorithms. If you wish to use pretrained weights and reinitialize '
                                    'layers which have undergone surgery, the following algorithms may be '
                                    'excluded using `load_exclude_algorithms`, e.g. '
                                    f'`load_exclude_algorithms=[{missing_algo_names}]`.'))

        try:
            for algo in missing_algos:  # TODO: use compiled algorithm order
                if algo.match(Event.INIT, self):
                    algo.apply(Event.INIT, self, logger)
                warnings.warn(
                    textwrap.dedent(
                        f'Automatically adding required_on_load algorithm {repr(algo)} to trainer, which was enabled '
                        'when training the loaded checkpoint. If you wish to use pretrained weights and ignore '
                        f'required_on_load algorithms, which may result in some weights failing to load, include {type(algo).__qualname__} '
                        f"in `load_exclude_algorithms`, e.g. `load_exclude_algorithms=['{type(algo).__qualname__}']`."))
        except Exception as e:
            raise ValueError(
                textwrap.dedent(
                    'The following algorithms were enabled when training this checkpoint '
                    f'and are required to successfully load it: {missing_algo_reprs}. '
                    'Attempted to autocreate and apply required algorithms but an exception was '
                    'encountered. If you wish to use pretrained weights and reinitialize layers which '
                    'have undergone surgery, the following algorithms may be excluded using '
                    f'`load_exclude_algorithms`, e.g. `load_exclude_algorithms=[{missing_algo_names}]`.')) from e

    def load_model_state(
        self,
        state_dict: Dict[str, Any],
        logger: Logger,
        strict: bool,
        exclude_algorithms: Optional[List[str]] = None,
        algorithm_passes: Optional[List[AlgorithmPass]] = None,
    ):
        """Loads the model's state from a ``state_dict``.

        Args:
            state_dict (Dict[str, Any]): The state dict, generated from a previous call to :meth:`state_dict`.
            logger (Logger): The logger.
            strict (bool): Whether the keys (i.e., model parameter names) in the model state dict should
                perfectly match the keys in the model instance.
            exclude_algorithms (List[str], optional): List of algorithm names to exclude from autoloading. (default: ``None``)
            algorithm_passes (List[AlgorithmPass], optional): A list of algorithm passes to apply to autoloaded algorithms
                to sort them into the correct order. (default: ``None``)
        """
        if 'algorithms' in state_dict:
            self._apply_required_algorithms(state_dict, logger, exclude_algorithms, algorithm_passes)

        if state_dict.get('is_model_ddp', False) and not self.is_model_ddp:
            # This check is for backwards compatibility, as pre-v0.6.0 checkpoints serialized the state
            # with the `module.` prefix
            torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict['model'], 'module.')

        with get_fsdp_rank0_cpu_save_context(self.model) if self.fsdp_enabled else contextlib.nullcontext():
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict['model'], strict=strict)
        if len(missing_keys) > 0:
            log.warning(f"Found these missing keys in the checkpoint: {', '.join(missing_keys)}")
        if len(unexpected_keys) > 0:
            log.warning(f"Found these unexpected keys in the checkpoint: {', '.join(unexpected_keys)}")

    def load_optim_state(self, state_dict: Dict[str, Any]):
        """Load the optimizer state.

        Args:
            state_dict (Dict[str, Any]): The state to load.
        """
        serialized_value = state_dict['optimizers']
        for target in ensure_tuple(self.optimizers):
            if type(target).__qualname__ not in serialized_value:
                warnings.warn(f'{type(target).__qualname__} is not in the state_dict. Its state will not be restored.',
                              category=UserWarning)
                continue
            source = serialized_value[type(target).__qualname__]
            if self.fsdp_enabled:
                sharded_osd = get_fsdp_sharded_optim_state_dict(full_optim_state_dict=source, model=self.model)
                target.load_state_dict(sharded_osd)
            else:
                target.load_state_dict(source)

    def _load_dataset_state(self, obj: Dict[str, Any]) -> None:
        """Load the dataset state.

        Args:
            obj (Dict[str, Any]): The state to load.
        """
        self.dataset_state = obj

        dataset = self._dataset_of(self.train_dataloader)
        if hasattr(dataset, 'load_state_dict'):
            dataset.load_state_dict(obj['train'])  # pyright: ignore
            obj['train'] = None
            self.dataset_resumption['train'] = True

        for evaluator in self.evaluators:
            dataset = self._dataset_of(evaluator)
            if hasattr(dataset, 'load_state_dict') and evaluator.label in obj['eval']:
                dataset.load_state_dict(obj['eval'][evaluator.label])  # pyright: ignore
                del obj['eval'][evaluator.label]
                if 'eval' not in self.dataset_resumption:
                    self.dataset_resumption['eval'] = {}
                # Note: We currently disable setting dataset_resumption for eval datasets,
                # which means they have one sample fetched in _spin_dataloaders before training
                # starts. This avoids "CUDA error: initialization error" -- its not clear why.
                # self.dataset_resumption['eval'][evaluator.label] = True

    def load_state_dict(
        self,
        state: Dict[str, Any],
        logger: Logger,
        strict: bool = False,
        exclude_algorithms: Optional[List[str]] = None,
        algorithm_passes: Optional[List[AlgorithmPass]] = None,
    ):
        """Loads the state.

        Args:
            state (Dict[str, Any]): object returned from call to :meth:`state_dict`.
            logger (Logger): The logger.
            strict (bool): whether the keys in the ``state["model"]`` should perfectly match the keys in the
                ``self.model``. Defaults to False.
            exclude_algorithms (List[str], optional): List of algorithm names to exclude from autoloading. (default: ``None``)
            algorithm_passes (List[AlgorithmPass], optional): A list of algorithm passes to apply to autoloaded algorithms
                to sort them into the correct order. (default: ``None``)
        """
        state = _ensure_backwards_compatible_checkpointing(state)

        # Call load_model_state since it applies required algorithms
        if 'model' in state:
            self.load_model_state(
                state,
                logger,
                strict=strict,
                exclude_algorithms=exclude_algorithms,
                algorithm_passes=algorithm_passes,
            )

        for attribute_name, serialized_value in state.items():
            # Skip removed attributes as well as algorithms and model, which was already loaded
            if attribute_name not in self.serialized_attributes or attribute_name == 'model':
                continue

            # Integrations are extra information about other libraries (e.g. huggingface) and not attributes to be loaded here
            if attribute_name == 'integrations':
                continue

            # Skip metadata, which is not an attribute on State
            if attribute_name == 'metadata':
                continue

            # Restructure algorithms serialized_value from list to dict
            if attribute_name == 'algorithms' and isinstance(serialized_value, list):
                serialized_value = {algo_name: algo_serialized for algo_name, algo_serialized in serialized_value}

            if attribute_name == 'dataset_state':
                self._load_dataset_state(serialized_value)
            elif attribute_name == 'optimizers':
                self.load_optim_state(state)
            elif attribute_name in _STATE_DICT_SERIALIZED_ATTRIBUTES:
                state_field_value = getattr(self, attribute_name)
                for target in ensure_tuple(state_field_value):
                    if type(target).__qualname__ not in serialized_value:
                        warnings.warn(
                            f'{type(target).__qualname__} is not in the state_dict. Its state will not be restored.',
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
        """The active dataloader."""
        return self._dataloader

    @property
    def dataloader_label(self):
        """The dataloader label for the active dataloader.

        By default, the training dataloader is called ``'train'``. The evaluator dataloader
        is called ``'eval'``, or when multiple evaluators are used, the name of the evaluator.
        However, the dataloader label can be explicitly specified in :meth:`.Trainer.fit`
        and :meth:`.Trainer.eval`.

        Returns:
            Optional[str]: The dataloader label, or None if no dataloader is set.
        """
        return self._dataloader_label

    def set_dataloader(
        self,
        dataloader: Optional[Iterable] = None,
        dataloader_label: Optional[str] = None,
        dataloader_len: Union[int, Time[int]] = -1,
    ):
        """Update the active dataloader and dataloader label.

        Args:
            dataloader (Iterable, optional): The dataloader. Defaults to None.
            dataloader_label (str, optional): The dataloader label. Must be ``None`` if and only if
                ``dataloader`` is None. Defaults to None.
            dataloader_len (int, int): The number of batches per dataloader iteration (e.g. epoch), as used by the trainer.
                Set to ``-1`` to iterate over the entire dataset. (Default: ``-1``.)
        """
        if dataloader is None:
            dataloader_label = None
        else:
            if dataloader_label is None:
                raise ValueError('If the `dataloader` is specified, then `dataloader_label` must not be None.')
        self._dataloader = dataloader
        self._dataloader_label = dataloader_label
        if dataloader is not None:
            self.dataloader_len = dataloader_len  # setting it to -1 will do a failsafe read of len(dataloader)
        else:
            self._dataloader_len = None

    @property
    def dataloader_len(self):
        """The number of batches per dataloader iteration (e.g. epoch), as used by the trainer.

        .. note::

            If not explicitely specified, this value is an approximation, as it depends on ``len(self.dataloader)``.
            See the :doc:`PyTorch DataLoader Documentation <torch:data>` for more information.

        Returns:
            Optional[Time[int]]: The number of batches per dataloader iteration (e.g. epoch), or None if no dataloader
            is defined or if the dataloader has an unknown length (e.g. streaming dataloaders).
        """
        return self._dataloader_len

    @dataloader_len.setter
    def dataloader_len(self, num_batches: Union[int, Time[int]]):
        if isinstance(num_batches, int):
            num_batches = Time(num_batches, TimeUnit.BATCH)
        if self._dataloader is None:
            raise RuntimeError('`State.dataloader_len` cannot be set if the dataloader is not defined.')
        try:
            if isinstance(self._dataloader, collections.abc.Sized):
                dataloader_len = len(self._dataloader)
            else:
                dataloader_len = None
        except (TypeError, NotImplementedError):
            dataloader_len = None
        if dataloader_len is not None and num_batches >= 0 and int(num_batches) > dataloader_len:
            warnings.warn((f'DataloaderNumBatchesWarning: The dataloader_len ({int(num_batches)}) '
                           f'is greater than the length (i.e. number of batches) of the dataloader, which is '
                           f'{dataloader_len}. State.dataloader_len is thus being set to {dataloader_len}.'))
            self._dataloader_len = Time(dataloader_len, TimeUnit.BATCH)
            return
        if num_batches < 0:
            if dataloader_len is not None:
                # len(dataloader) is an approximation -- see https://pytorch.org/docs/stable/data.html.
                # However, in the worst case where additional last batches are dropped, this calculation should be
                # an over-estimate, leading to the entire dataloader still being iterated over.
                self._dataloader_len = Time(dataloader_len, TimeUnit.BATCH)
            else:
                # The dataloader length is unknown.
                self._dataloader_len = None
            return
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
    def is_model_ddp(self):
        """Whether :attr:`model` is an instance of a :class:`.DistributedDataParallel`."""
        return isinstance(self.model, DistributedDataParallel)

    @property
    def deepspeed_model(self) -> deepspeed.DeepSpeedEngine:
        """Cast :attr:`model` to :class:`~deepspeed.DeepSpeedEngine`."""
        if is_model_deepspeed(self.model):
            return cast('deepspeed.DeepSpeedEngine', self.model)
        raise TypeError('state.model is not a DeepSpeed model')
