# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A wrapper class that converts 🤗 Transformers models to composer models"""

from __future__ import annotations

import copy
import inspect
import json
import logging
import os
import random
import string
import tempfile
import textwrap
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, Union

import torch
from torchmetrics import Metric

from composer.devices import DeviceCPU
from composer.models.base import ComposerModel
from composer.utils import MissingConditionalImportError, dist, get_file, import_object, is_model_fsdp, safe_torch_load

try:
    from peft import PeftModel, get_peft_model
    peft_installed = True
except:
    peft_installed = False

if TYPE_CHECKING:
    import transformers
    from peft import PeftConfig, PeftModel
    from transformers import PretrainedConfig
    from transformers.models.auto.auto_factory import _BaseAutoModelClass

log = logging.getLogger(__name__)

__all__ = ['HuggingFaceModel', 'peft_installed']


class HuggingFaceModel(ComposerModel):
    """
    A wrapper class that converts 🤗 Transformers models to composer models.

    Args:
        model (Union[transformers.PreTrainedModel, peft.PeftModel)): A 🤗 Transformers model or a PEFT model.
        tokenizer (transformers.PreTrainedTokenizer, optional): The tokenizer used to prepare the dataset. Default ``None``.

            .. note:: If the tokenizer is provided, its config will be saved in the composer checkpoint, and it can be reloaded
                using :meth:`HuggingFaceModel.hf_from_composer_checkpoint`. If the tokenizer is not provided here, it will not be saved in the composer checkpoint.
        use_logits (bool, optional): If True, the model's output logits will be used to calculate validation metrics. Else, metrics will be inferred from the HuggingFaceModel directly. Default: ``False``
        metrics (Sequence[Metric], optional): list of torchmetrics to apply to the output of `eval_forward` during training. If ``eval_metrics`` is ``None``, these will also be used as ``eval_metrics``.  Default: ``None``.
        eval_metrics (Sequence[Metric], optional): list of torchmetrics to compute on the eval_dataloader, or be accessible to :class:`Evaluator`s. Default: ``None``.
        shift_labels (bool, optional): If True, the batch's labels will be shifted before being used to calculate metrics. This should be set to true for CausalLM models and false otherwise. If not specified, `shift_labels` will be set automatically based on the model class name. Default: ``None``.
        allow_embedding_resizing (bool, optional): If True, the model's embeddings will be automatically resized when they are smaller than the tokenizer vocab size. Default: ``False``.
        peft_config (PeftConfig, optional): Optional PEFT config to apply to the model. If provided, the model will be converted to a PEFT model. Only LoRA is currently supported.
        should_save_peft_only (bool, optional): If True _and_ PEFT is active, the state dict will only contain the PEFT weights, not the frozen base model weights.

        .. note:: To ensure correct behavior, set `shift_labels` manually if using a custom model (i.e., if `model` is not
        an instance of a registered 🤗 Transformers class).
    .. warning:: This wrapper is designed to work with 🤗 datasets that define a `labels` column.

    Example:

    .. testcode::

        import transformers
        from composer.models import HuggingFaceModel

        hf_model = transformers.AutoModelForSequenceClassification.from_pretrained('google-bert/bert-base-uncased', num_labels=2)
        hf_tokenizer = transformers.AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
        model = HuggingFaceModel(hf_model, hf_tokenizer)
    """

    def __init__(
        self,
        model: Union[transformers.PreTrainedModel, 'PeftModel'],
        tokenizer: Optional[Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast]] = None,
        use_logits: Optional[bool] = False,
        metrics: Optional[Sequence[Metric]] = None,
        eval_metrics: Optional[Sequence[Metric]] = None,
        shift_labels: Optional[bool] = None,
        allow_embedding_resizing: bool = False,
        peft_config: Optional['PeftConfig'] = None,
        should_save_peft_only: bool = True,
    ) -> None:
        try:
            import transformers
            del transformers  # unused
        except ImportError as e:
            raise MissingConditionalImportError(
                extra_deps_group='nlp',
                conda_package='transformers',
                conda_channel='conda-forge',
            ) from e

        if peft_config is not None:
            if not peft_installed:
                raise MissingConditionalImportError(
                    extra_deps_group='peft',
                    conda_package='peft',
                    conda_channel='conda-forge',
                )

        if peft_config is not None:
            # Hugging Face requires the peft type and task type to be upper case, so we do that here
            # https://github.com/huggingface/peft/blob/ebbff4023ad276cbcb2466fd7e99be7d3ae0ae11/src/peft/utils/peft_types.py#L22-L51
            if isinstance(peft_config.peft_type, str):
                peft_config.peft_type = peft_config.peft_type.upper()
            if isinstance(peft_config.task_type, str):
                peft_config.task_type = peft_config.task_type.upper()

            if peft_config.peft_type != 'LORA':
                raise ValueError(
                    f'PEFT type {peft_config.peft_type} is not supported by HuggingFaceModel. Only LORA is supported.',
                )

        super().__init__()
        self.model = model
        self.config: PretrainedConfig = model.config
        self.model_forward_args = self._get_model_forward_args()
        self.tokenizer = tokenizer
        self.should_save_peft_only = should_save_peft_only
        self.use_logits = use_logits
        self.labels: Optional[torch.Tensor] = None  # set in eval_forward() if exists
        self.dummy_forward_called = False  # Used to make FSDP generate work, see generate function for more details
        self.train_metrics: Optional[dict] = self._get_metric_dict(metrics) if metrics is not None else None
        self.val_metrics: Optional[dict] = self._get_metric_dict(
            eval_metrics,
        ) if eval_metrics is not None else copy.deepcopy(
            self.train_metrics,
        )

        is_causal_lm = _is_registered_causal_lm(self.model)
        self.shift_labels = is_causal_lm if shift_labels is None else shift_labels

        self._check_tokenizer_and_maybe_resize_embeddings(allow_embedding_resizing)

        if is_causal_lm and not self.shift_labels:
            log.warning(
                'The shift_labels argument was set to False but the model is an instance of a'
                ' HuggingFace Causal LM. This may lead to incorrect behavior.',
            )
            # Note: No warning if shift_labels and not is_causal_lm, since the model may simply be a custom class.

        if peft_config is not None:
            self.model = _maybe_get_peft_model(peft_config, self.model)

        self.using_peft = isinstance(self.model, PeftModel) if peft_installed else False

    def _check_tokenizer_and_maybe_resize_embeddings(self, allow_embedding_resizing: bool) -> None:
        if self.tokenizer is None:
            log.warning(
                'The tokenizer was not provided. This means the tokenizer config will not be saved in the checkpoint.',
            )

        if self.tokenizer is not None and self.config.vocab_size < len(self.tokenizer):
            if allow_embedding_resizing:
                # when the embedding size is smaller than the tokenizer vocab size,
                # the embeddings should get resized to match the tokenizer vocab size
                log.warning(
                    f'The number of tokens in the tokenizer is greater than the number of tokens in the model.'
                    f' This would cause an error during training.'
                    f' Resizing the model embeddings to {len(self.tokenizer)} from {self.config.vocab_size}.',
                )
                self.model.resize_token_embeddings(len(self.tokenizer))
            else:
                raise ValueError(
                    f'The number of tokens in the tokenizer is greater than the number of tokens in the model.'
                    f' This would cause an error during training.'
                    f' You can resize the model embeddings to {len(self.tokenizer)} from {self.config.vocab_size}'
                    f' by calling `model.resize_token_embeddings(len(tokenizer))` before calling the `HuggingFaceModel`'
                    f' constructor, or pass `allow_embedding_resizing=True` to have it done automatically.',
                )
        elif self.tokenizer is not None and self.config.vocab_size > len(self.tokenizer):
            # when the embedding size is greater than the tokenizer vocab size,
            # the embeddings do not _need_ to be resized to match the tokenizer vocab size,
            # and should be done by the user if desired
            log.info(
                f'The number of tokens in the tokenizer is less than the number of tokens in the model.'
                f' You may want to resize the model embeddings to {len(self.tokenizer)} from {self.config.vocab_size}'
                f' by calling `model.resize_token_embeddings(len(tokenizer))` before calling the `HuggingFaceModel`'
                f' constructor. The vocab size is sometimes intentionally set to a multiple of 32 or 64 to improve'
                f' performance.',
            )

    def _get_metric_dict(self, metrics: Sequence[Metric]) -> dict[str, Metric]:
        """Returns a dictionary of metrics keyed by their class name."""
        return {metric.__class__.__name__: metric for metric in metrics}

    def _get_model_forward_args(self) -> set[str]:
        """Returns the arguments to the model's forward function."""
        model_forward_args = inspect.signature(maybe_get_underlying_model(self.model).forward).parameters.keys()

        if not model_forward_args:
            raise ValueError('Could not determine the forward arguments of the model. Please open a GitHub issue.')

        model_forward_args = set(model_forward_args)

        return model_forward_args

    def state_dict(self, *args, **kwargs) -> dict[str, Any]:
        """Returns the state dict of the model."""
        full_state_dict = super().state_dict(*args, **kwargs)

        if self.using_peft and self.should_save_peft_only:
            active_adapter = self.model.active_adapter
            assert isinstance(active_adapter, str)
            full_state_dict = filter_state_dict_peft(
                full_state_dict,
                self.model.peft_config[active_adapter],
                adapter_name='default',
                remove_adapter_names=False,
            )

        return full_state_dict

    @staticmethod
    def load_huggingface_tokenizer_from_saved_state(
        hf_state: dict[str, Any],
        trust_remote_code: bool = False,
        tokenizer_save_dir: Optional[str] = None,
    ) -> Optional[transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast]:
        """A helper function that loads a HuggingFace tokenizer from a loaded in hf state.

        Args:
            hf_state (dict[str, Any]): HF state loaded from a Composer checkpoint.
            trust_remote_code (bool, optional): Whether to trust the remote code when loading the tokenizer. Defaults to False.
            tokenizer_save_dir (Optional[str], optional): If specified, where to save the tokenizer files to locally. If not specified,
                a folder with a unique suffix will be saved in the current working directory. Defaults to None.

        Returns:
            Optional[transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast]: The loaded HuggingFace tokenizer
        """
        try:
            import transformers
        except ImportError as e:
            raise MissingConditionalImportError(
                extra_deps_group='nlp',
                conda_package='transformers',
                conda_channel='conda-forge',
            ) from e
        hf_tokenizer = None
        hf_tokenizer_state = hf_state['tokenizer']
        if hf_tokenizer_state != {}:
            if tokenizer_save_dir is None:
                unique_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
                tokenizer_save_dir = os.path.join(os.getcwd(), f'tokenizer-save-dir-{unique_suffix}')
            os.makedirs(tokenizer_save_dir, exist_ok=True)

            for filename, saved_content in hf_tokenizer_state.items():
                # This cannot be a temporary directory because huggingface relies on the slow tokenizer file
                # being persistent on disk

                # For backwards compatibility, check if the filename already has the file extension
                if filename.endswith(saved_content['file_extension']):
                    tokenizer_file_name = filename
                else:
                    tokenizer_file_name = filename + saved_content['file_extension']

                tokenizer_file_path = Path(tokenizer_save_dir) / tokenizer_file_name
                if saved_content['file_extension'] == '.json':
                    with open(tokenizer_file_path, 'w', encoding='utf-8') as _f:
                        json.dump(saved_content['content'], _f)
                elif saved_content['file_extension'] == '.txt':
                    with open(tokenizer_file_path, 'w', encoding='utf-8') as _f:
                        for line in saved_content['content']:
                            _f.write(line)
                            _f.write('\n')
                elif saved_content['file_extension'] == '.py':
                    with open(tokenizer_file_path, 'w', encoding='utf-8') as _f:
                        _f.write(saved_content['content'])
                elif saved_content['file_extension'] == '.model':
                    try:
                        import sentencepiece as spm
                    except ImportError as e:
                        raise MissingConditionalImportError(
                            extra_deps_group='sentencepiece',
                            conda_package='sentencepiece',
                        ) from e
                    s = spm.SentencePieceProcessor()
                    s.load_from_serialized_proto(saved_content['content'])  # pyright: ignore[reportGeneralTypeIssues]
                    with open(tokenizer_file_path, 'wb') as _f:
                        _f.write(s.serialized_model_proto())

            hf_tokenizer = transformers.AutoTokenizer.from_pretrained(
                tokenizer_save_dir,
                trust_remote_code=trust_remote_code,
            )

            # we need to set the name_or_path back because otherwise it is the tmp dir we are loading from here
            # For backwards compatibility we try both the old and new key
            tokenizer_config_key = 'tokenizer_config.json' if 'tokenizer_config.json' in hf_tokenizer_state else 'tokenizer_config'
            hf_tokenizer.name_or_path = hf_tokenizer_state[tokenizer_config_key]['content'].get('name_or_path', '')
            hf_tokenizer.init_kwargs['name_or_path'] = hf_tokenizer.name_or_path

            # for an unknown reason this key is missing when loading the saved tokenizer, but present with a value of None
            # for the original tokenizer, so we default it to None
            hf_tokenizer.init_kwargs['tokenizer_file'] = hf_tokenizer.init_kwargs.get('tokenizer_file', None)
        return hf_tokenizer

    @staticmethod
    def load_huggingface_model_from_saved_state(
        hf_state: dict[str, Any],
        loaded_state_dict: dict[str, dict[str, dict[str, dict[str, Any]]]],
        model_instantiation_class: type | str | None,
        model_config_kwargs: dict[str, Any] | None,
    ) -> transformers.PreTrainedModel:
        """A helper function that loads a HuggingFace model class from a loaded in hf state.

        Args:
            hf_state (dict[str, Any]): HF state loaded from a Composer checkpoint.
            model_instantiation_class (Union[Type[:class:`transformers.PreTrainedModel`], Type[:class:`transformers.AutoModel`], str]), optional):
                Class to use to create the HuggingFace model. Defaults to the model class used in the original checkpoint. If this argument is
                a HuggingFace auto class (e.g. :class:`transformers.AutoModel` or :class:`transformers.AutoModelForSequenceClassification`), the ``from_config`` method will be used,
                while if it is of type :class:`transformers.PreTrainedModel`, the constructor will be called. This argument can also be a string,
                which will attempt to be imported as the class to use.
            model_config_kwargs: dict[str, Any]: Extra arguments to pass in for the model config creation (e.g. ``num_labels`` for creating a sequence classification model)
        Returns:
            transformers.PreTrainedModel: The loaded HuggingFace model
        """
        try:
            import transformers
        except ImportError as e:
            raise MissingConditionalImportError(
                extra_deps_group='nlp',
                conda_package='transformers',
                conda_channel='conda-forge',
            ) from e
        loaded_config = get_hf_config_from_composer_state_dict(loaded_state_dict, config_overrides=model_config_kwargs)

        hf_model_state = hf_state['model']
        if model_instantiation_class is not None:
            # If the instantiation class is explicitly provided, use it
            # If a string is provided, attempt to import the class it refers to
            if isinstance(model_instantiation_class, str):
                try:
                    model_instantiation_class = import_object(
                        ':'.join(model_instantiation_class.rsplit('.', maxsplit=1)),
                    )
                except (ModuleNotFoundError, AttributeError):
                    raise ValueError(
                        textwrap.dedent(
                            f'The provided model_instantiation_class string {model_instantiation_class} could not be imported. '
                            f'Please make sure {model_instantiation_class} is discoverable on the python path, or pass the class '
                            'in directly.',
                        ),
                    )

            assert model_instantiation_class is not None  # pyright
            # The AutoModel* classes have `from_config`, while the PreTrainedModel classes do not
            # pyright can't tell this isn't a string at this point
            if issubclass(
                model_instantiation_class,  # type: ignore
                transformers.models.auto.auto_factory._BaseAutoModelClass,  # type: ignore
            ):  # pyright: ignore[reportGeneralTypeIssues]
                hf_model = model_instantiation_class.from_config(loaded_config)  # type: ignore
            else:
                hf_model = model_instantiation_class(loaded_config)  # type: ignore
        else:
            # If the instantiation class is not explicitly provided, attempt to import the saved class and use it
            try:
                saved_class = import_object(':'.join(hf_model_state['config']['class'].rsplit('.', maxsplit=1)))
            except (ModuleNotFoundError, AttributeError):
                model_cfg_class = hf_model_state['config']['class']
                raise ValueError(
                    textwrap.dedent(
                        f'The saved class {model_cfg_class} could not be imported. '
                        'Please either pass in the class to use explicitly via the model_instantiation_class '
                        f'parameter, or make sure that {model_cfg_class} is discoverable '
                        'on the python path.',
                    ),
                )
            hf_model = saved_class(loaded_config)
        return hf_model

    @staticmethod
    def hf_from_composer_checkpoint(
        checkpoint_path: str,
        model_instantiation_class: Optional[Union[type[transformers.PreTrainedModel],
                                                  type['_BaseAutoModelClass'],
                                                  str,
                                                 ]] = None,
        model_config_kwargs: Optional[dict] = None,
        local_checkpoint_save_location: Optional[Union[Path, str]] = None,
        trust_remote_code: bool = False,
    ) -> tuple[transformers.PreTrainedModel,
               Optional[Union[transformers.PreTrainedTokenizer,
                              transformers.PreTrainedTokenizerFast,
                             ]],
              ]:
        """Loads a HuggingFace model (and tokenizer if present) from a composer checkpoint.

        .. note:: This function does not load the weights from the checkpoint. It just loads the correctly configured
            model and tokenizer classes.

        .. testsetup::

            import torch

            dataset = RandomTextClassificationDataset(size=16, use_keys=True)
            train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
            eval_dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)

            import transformers
            from composer.models import HuggingFaceModel
            from composer.trainer import Trainer

            hf_model = transformers.AutoModelForSequenceClassification.from_pretrained('prajjwal1/bert-tiny', num_labels=2)
            hf_tokenizer = transformers.AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
            composer_model = HuggingFaceModel(hf_model, tokenizer=hf_tokenizer, metrics=[], use_logits=True)
            trainer = Trainer(model=composer_model,
                              train_dataloader=train_dataloader,
                              save_filename='composer-hf-checkpoint.pt',
                              max_duration='1ep',
                              save_folder='./')
            trainer.fit()
            trainer.close()

        Example:

        .. testcode::

            hf_model, hf_tokenizer = HuggingFaceModel.hf_from_composer_checkpoint('composer-hf-checkpoint.pt')
            # At this point, hf_model is randomly initialized
            composer_model = HuggingFaceModel(hf_model, hf_tokenizer)
            trainer = Trainer(model=composer_model,
                              train_dataloader=train_dataloader,
                              save_filename='composer-hf-checkpoint-2.pt',
                              max_duration='1ep',
                              save_folder='./',
                              load_path='composer-hf-checkpoint.pt')
            # At this point, the weights have been loaded from the composer checkpoint into hf_model

        Args:
            checkpoint_path (str): Path to the composer checkpoint, can be a local path, or a remote path beginning with ``s3://``, or another backend
                supported by :meth:`composer.utils.maybe_create_object_store_from_uri`.
            model_instantiation_class (Union[Type[:class:`transformers.PreTrainedModel`], Type[:class:`transformers.AutoModel`], str]), optional):
                Class to use to create the HuggingFace model. Defaults to the model class used in the original checkpoint. If this argument is
                a HuggingFace auto class (e.g. :class:`transformers.AutoModel` or :class:`transformers.AutoModelForSequenceClassification`), the ``from_config`` method will be used,
                while if it is of type :class:`transformers.PreTrainedModel`, the constructor will be called. This argument can also be a string,
                which will attempt to be imported as the class to use.
            model_config_kwargs: dict[str, Any]: Extra arguments to pass in for the model config creation (e.g. ``num_labels`` for creating a sequence classification model)
            local_checkpoint_save_location (Optional[Union[Path, str]], optional): If specified, where to save the checkpoint file to locally.
                                                                                   If the input ``checkpoint_path`` is already a local path, this will be a symlink.
                                                                                   Defaults to None, which will use a temporary file.
            trust_remote_code (bool, optional): Whether to trust the remote code when loading the tokenizer. Defaults to False.

        Raises:
            ValueError: If the ``model_instantiation_class``, or the model class saved in the checkpoint, is not able to be imported

        Returns:
            tuple[transformers.PreTrainedModel, Optional[Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast]]]: The loaded HuggingFace model and (if present) tokenizer
        """

        # default local path to a tempfile if path is not provided
        if local_checkpoint_save_location is None:
            tmp_dir = tempfile.TemporaryDirectory()
            local_checkpoint_save_location = Path(tmp_dir.name) / 'local-composer-checkpoint.pt'

        if model_config_kwargs is None:
            model_config_kwargs = {}

        # download the checkpoint file
        get_file(checkpoint_path, str(local_checkpoint_save_location))

        # load the state dict in
        loaded_state_dict = safe_torch_load(local_checkpoint_save_location)

        hf_state = loaded_state_dict['state']['integrations']['huggingface']
        hf_tokenizer = HuggingFaceModel.load_huggingface_tokenizer_from_saved_state(hf_state, trust_remote_code)
        hf_model = HuggingFaceModel.load_huggingface_model_from_saved_state(
            hf_state,
            loaded_state_dict,
            model_instantiation_class,
            model_config_kwargs,
        )

        return hf_model, hf_tokenizer

    def forward(self, batch):
        if isinstance(batch, Mapping):
            # Further input validation is left to the huggingface forward call
            batch = {k: v for k, v in batch.items() if k in self.model_forward_args}
            output = self.model(**batch)  # type: ignore (thirdparty)
        else:
            raise ValueError(
                'Unexpected batch type. Expected a dictionary with keys corresponding to the inputs to the forward function of the Huggingface model',
            )
        return output

    def loss(self, outputs, batch):
        if self.config.use_return_dict:
            return outputs['loss']
        else:
            # loss is at index 0 in the output tuple
            return outputs[0]

    def eval_forward(self, batch, outputs: Optional[Any] = None):
        # If the batch mode is generate, we will generate a requested number of tokens using the underlying
        # model's generate function. Extra generation kwargs can be passed in via the batch. Strings will
        # be returned from eval_forward
        if batch.get('mode', None) == 'generate':
            if self.tokenizer is None:
                raise ValueError(
                    'Generation eval cannot be used without providing a tokenizer to the model constructor.',
                )

            self.labels = batch.pop('labels')
            generation = self.generate(
                batch['input_ids'],
                attention_mask=batch['attention_mask'],
                synced_gpus=dist.get_world_size() > 1,
                **batch.get('generation_kwargs', {}),
            )

            # don't remove prefix space to sentencepiece models
            if len(
                self.tokenizer(' a', add_special_tokens=False)['input_ids'],  # pyright: ignore[reportGeneralTypeIssues]
            ) == 1:
                return self.tokenizer.batch_decode(
                    generation[:, batch['input_ids'].shape[1]:],
                    skip_special_tokens=True,
                )
            else:
                return [
                    ' ' + generation for generation in
                    self.tokenizer.batch_decode(generation[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)
                ]

        if self.use_logits or batch.get('mode', None) == 'icl_task':
            # pop labels first to avoid computing loss
            self.labels = batch.pop('labels')

            # HF encoder decoder models like T5 expect either decoder_input_ids or labels,
            # so we add decoder_input_ids to the batch if it is missing
            if self.config.is_encoder_decoder and 'decoder_input_ids' not in batch:
                if hasattr(self.model, 'prepare_decoder_input_ids_from_labels'):
                    batch['decoder_input_ids'] = self.model.prepare_decoder_input_ids_from_labels(labels=self.labels)
                else:
                    raise RuntimeError(
                        'Encoder decoder models require that either decoder_input_ids is present in the batch'
                        ' or that the model has a prepare_decoder_input_ids_from_labels method.',
                    )

            if self.shift_labels or batch.get('mode', None) == 'icl_task':
                assert self.labels is not None
                # HF CausalLM models internally shift labels before computing loss, so we do the same here
                self.labels[:, :-1] = self.labels[:, 1:].clone()
                self.labels[:, -1] = -100

            output = outputs if outputs else self.forward(batch)

            if self.config.use_return_dict:
                output = output['logits']
            else:
                # if loss was computed (cached outputs from forward), loss is at index 0 and logits are at index 1
                # if loss was not computed (no cached outputs during eval), loss is not present and logits are at index 0
                output = output[1] if len(output[0].shape) == 0 else output[0]

            # if we are in the single class case, then remove the classes dimension
            if output.ndim == 2 and output.shape[1] == 1:
                output = output.squeeze(dim=1)
        else:
            output = outputs if outputs else self.forward(batch)

        return output

    def get_metrics(self, is_train: bool = False) -> dict[str, Metric]:
        if is_train:
            metrics = self.train_metrics
        else:
            metrics = self.val_metrics

        return metrics if metrics else {}

    def update_metric(self, batch: Any, outputs: Any, metric: Metric) -> dict:
        if metric.device.type == 'cpu':
            self.labels = DeviceCPU().batch_to_device(self.labels)

        if getattr(metric, 'needs_batch', False):
            metric_result = metric.update(batch=batch, outputs=outputs, labels=self.labels)
        else:
            metric_result = metric.update(outputs, self.labels)
        if metric_result is not None:
            # Add the metric name once for each datapoint in the batch
            metric_result['metric_name'] = [metric.__class__.__name__ for _ in range(0, batch['input_ids'].shape[0])]
        else:
            metric_result = {}
        return metric_result

    def get_metadata(self):
        model_output = {}
        tokenizer_output = {}
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            model_dir = tmp_dir / 'model'
            tokenizer_dir = tmp_dir / 'tokenizer'

            original_model_config: PretrainedConfig = self.config
            original_model_config.save_pretrained(model_dir)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(tokenizer_dir)

            with open(model_dir / 'config.json') as _config_file:
                model_config = json.load(_config_file)

            model_output['config'] = {
                'file_extension': '.json',
                'content': model_config,
                'class': f'{self.model.__class__.__module__}.{self.model.__class__.__name__}',
            }

            # Also save PEFT config if the model is a peft model
            if self.using_peft:
                active_adapter = self.model.active_adapter
                assert isinstance(active_adapter, str)
                self.model.peft_config[active_adapter].save_pretrained(str(model_dir))
                with open(model_dir / 'adapter_config.json') as _peft_config_file:
                    peft_config = json.load(_peft_config_file)

                model_output['peft_config'] = {
                    'file_extension': '.json',
                    'content': peft_config,
                }

            if self.tokenizer is not None:
                for tokenizer_file_name in tokenizer_dir.iterdir():
                    tokenizer_file_path = tokenizer_dir / tokenizer_file_name
                    tokenizer_file_extension = tokenizer_file_path.suffix
                    if tokenizer_file_extension == '.txt':
                        with open(tokenizer_file_path, encoding='utf-8') as _tokenizer_file:
                            tokenizer_file_content = _tokenizer_file.read().split('\n')
                    elif tokenizer_file_extension == '.json':
                        with open(tokenizer_file_path, 'rb') as _tokenizer_file:
                            tokenizer_file_content = json.load(_tokenizer_file)
                    elif tokenizer_file_extension == '.py':
                        with open(tokenizer_file_path, encoding='utf-8') as _tokenizer_file:
                            tokenizer_file_content = _tokenizer_file.read()
                    elif tokenizer_file_extension == '.model':
                        try:
                            import sentencepiece as spm
                        except ImportError as e:
                            raise MissingConditionalImportError(
                                extra_deps_group='sentencepiece',
                                conda_package='sentencepiece',
                            ) from e
                        s = spm.SentencePieceProcessor(
                            model_file=str(tokenizer_file_path),  # pyright: ignore[reportGeneralTypeIssues]
                        )
                        tokenizer_file_content = s.serialized_model_proto()
                    else:
                        raise ValueError(
                            f'Unexpected file ending {tokenizer_file_name} in output of tokenizer.save_pretrained.',
                        )

                    tokenizer_output[tokenizer_file_path.name] = {
                        'file_extension': tokenizer_file_extension,
                        'content': tokenizer_file_content,
                    }
        return {'model': model_output, 'tokenizer': tokenizer_output}

    def generate(self, input_ids: torch.Tensor, **kwargs):
        """Generate from the underlying HuggingFace model.

        Except for ``pad_token_id``, which is optionally read from ``self.tokenizer``, all args are passed along
        to :meth:`transformers.GenerationMixin.generate` function.

        Args:
            input_ids (torch.Tensor): Input ids to generate from.
            **kwargs: Additional arguments passed to :meth:`transformers.GenerationMixin.generate` function.
                See :class:`transformers.GenerationConfig` for all available arguments.
        """
        pad_token_id = kwargs.pop('pad_token_id', self.tokenizer.pad_token_id if self.tokenizer is not None else None)

        if is_model_fsdp(self.model):
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            # Note: We need to use the FSDP.summon_full_params context manager here because the generate function
            # does not seem to gather the weights for the LM head. This solution works because the tied weights of the LM head
            # are in the root FSDP module, and are summoned by the below context manager. See https://github.com/pytorch/pytorch/issues/100069
            # for more info.
            # Note: We use recurse=False here so that we only summon full params for the LM head, not the entire model.
            with FSDP.summon_full_params(self.model, writeback=False, recurse=False):
                return self.model.generate(input_ids=input_ids, pad_token_id=pad_token_id, **kwargs)
        else:
            return self.model.generate(input_ids=input_ids, pad_token_id=pad_token_id, **kwargs)


def _maybe_get_peft_model(
    peft_config: 'PeftConfig',
    model: Union[transformers.PreTrainedModel, 'PeftModel'],
) -> 'PeftModel':
    """Creates a PEFT model if the model is not already a PEFT model.

    Args:
        peft_config (Optional[peft.PeftConfig]): The PEFT config to use to create the PEFT model
        model (Union[transformers.PreTrainedModel, 'PeftModel']): The model to create the PEFT model from

    Returns:
        PeftModel: The PEFT model
    """
    if not peft_installed:
        raise MissingConditionalImportError(extra_deps_group='peft', conda_package='peft', conda_channel='conda-forge')

    if not isinstance(model, PeftModel):
        log.info('Creating PEFT model')
        peft_model = get_peft_model(model, peft_config)
        assert isinstance(peft_model, PeftModel)
        return peft_model
    else:
        warnings.warn('PEFT model was passed in directly. Ignoring the provided PEFT config.')
        return model


def maybe_get_underlying_model(
    model: Union[transformers.PreTrainedModel, 'PeftModel'],
) -> Union[transformers.PreTrainedModel, 'PeftModel']:
    """Get the underlying PreTrainedModel from a model if it is a PEFT model

    Args:
        model (Union[transformers.PreTrainedModel, 'PeftModel']): The model to get the underlying model from

    Returns:
        Union[transformers.PreTrainedModel]: The underlying transformers model
    """
    if peft_installed and isinstance(model, PeftModel):
        return model.base_model.model
    else:
        return model


def _is_registered_causal_lm(model: Union[transformers.PreTrainedModel, 'PeftModel']) -> bool:
    """Return True if model class is either a registered 🤗 Causal LM or a subclass of one"""
    try:
        from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
    except ImportError as e:
        raise MissingConditionalImportError(
            extra_deps_group='nlp',
            conda_package='transformers',
            conda_channel='conda-forge',
        ) from e

    model_to_check = maybe_get_underlying_model(model)

    # This try/except is needed until https://github.com/huggingface/transformers/issues/26778
    # is resolved in a release. This means that this attempt to automatically detect causal LMs
    # does not currently work in an environment with flash attention <2 installed.
    try:
        causal_lm_classes = list(MODEL_FOR_CAUSAL_LM_MAPPING.values())
    except RuntimeError as e:
        if 'Failed to import transformers.models' in str(e):
            MODEL_FOR_CAUSAL_LM_MAPPING = {}
            return False
        else:
            raise e
    return any(isinstance(model_to_check, causal_lm_class) for causal_lm_class in causal_lm_classes)  # type: ignore


def get_hf_config_from_composer_state_dict(
    state_dict: dict[str, Any],
    config_overrides: Optional[dict[str, Any]] = None,
) -> 'PretrainedConfig':
    """Get a HuggingFace config from a composer state dict with overrides applied

    Args:
        state_dict (dict[str, Any]): The state dict to get the config from
        config_overrides (dict[str, Any], optional): Any overrides to apply to the config

    Returns:
        transformers.PretrainedConfig: The HuggingFace config
    """
    try:
        import transformers
    except ImportError as e:
        raise MissingConditionalImportError(
            extra_deps_group='nlp',
            conda_package='transformers',
            conda_channel='conda-forge',
        ) from e

    if config_overrides is None:
        config_overrides = {}

    hf_config_dict = state_dict['state']['integrations']['huggingface']['model']['config']['content']
    # Update the config with any extra args needed
    hf_config_dict.update(config_overrides)
    # JSON keys need to be converted back to ints, huggingface does not auto convert them along this code path
    if 'id2label' in hf_config_dict:
        hf_config_dict['id2label'] = {int(k): v for k, v in hf_config_dict['id2label'].items()}

    try:
        return transformers.AutoConfig.for_model(**hf_config_dict)
    except ValueError:
        model_type = hf_config_dict.get('model_type')
        try:
            return transformers.AutoConfig.from_pretrained(hf_config_dict['_name_or_path'], **hf_config_dict)
        except KeyError:
            raise Exception(
                f'Could not load config from state dict using either `for_model` or `from_pretrained`.'
                f'Please make sure that the model_type={model_type} is valid, or that the'
                f'config has a valid `_name_or_path`.',
            )


def get_peft_config_from_composer_state_dict(state_dict: dict[str, Any]) -> Optional['PeftConfig']:
    """Get a PEFT config from a composer state dict

    Args:
        state_dict (dict[str, Any]): The state dict to get the config from

    Returns:
        Optional[peft.PeftConfig]: The PEFT config. Will be ``None`` if the model is not a PEFT model.
    """
    try:
        import peft
    except ImportError as e:
        raise MissingConditionalImportError(
            extra_deps_group='nlp',
            conda_package='peft',
            conda_channel='conda-forge',
        ) from e

    hf_model_dict = state_dict['state']['integrations']['huggingface']['model']
    if 'peft_config' not in hf_model_dict:
        return None

    peft_config_dict = hf_model_dict['peft_config']['content']

    return peft.get_peft_config(peft_config_dict)


def write_huggingface_pretrained_from_composer_checkpoint(
    checkpoint_path: Union[Path, str],
    output_folder: Union[Path, str],
    local_checkpoint_save_location: Optional[Union[Path, str]] = None,
) -> None:
    """Write a ``config.json`` and ``pytorch_model.bin``, like :meth:`transformers.PreTrainedModel.from_pretrained` expects, from a composer checkpoint

    .. note:: This function will not work properly if you used surgery algorithms when you trained your model. In that case you will want to
        load the model weights using the Composer :class:`~composer.Trainer` with the ``load_path`` argument.

    .. testsetup::

        import torch

        dataset = RandomTextClassificationDataset(size=16, use_keys=True)
        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
        eval_dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)

        import transformers
        from composer.models import HuggingFaceModel
        from composer.trainer import Trainer

        hf_model = transformers.AutoModelForSequenceClassification.from_pretrained('prajjwal1/bert-tiny', num_labels=2)
        hf_tokenizer = transformers.AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
        composer_model = HuggingFaceModel(hf_model, tokenizer=hf_tokenizer, metrics=[], use_logits=True)
        trainer = Trainer(model=composer_model,
                            train_dataloader=train_dataloader,
                            save_filename='composer-hf-checkpoint.pt',
                            max_duration='1ep',
                            save_folder='./')
        trainer.fit()
        trainer.close()

    Example:

    .. testcode::

        from composer.models import write_huggingface_pretrained_from_composer_checkpoint

        write_huggingface_pretrained_from_composer_checkpoint('composer-hf-checkpoint.pt', './hf-save-pretrained-output')
        loaded_model = transformers.AutoModelForSequenceClassification.from_pretrained('./hf-save-pretrained-output')

    Args:
        checkpoint_path (Union[Path, str]): Path to the composer checkpoint, can be a local path, or a remote path beginning with ``s3://``, or another backend
            supported by :meth:`composer.utils.maybe_create_object_store_from_uri`.
        output_folder (Union[Path, str]): Path to the folder to write the output to. Must be a local path.
        local_checkpoint_save_location (Optional[Union[Path, str]], optional): If specified, where to save the checkpoint file to locally.
                                                                                If the input ``checkpoint_path`` is already a local path, this will be a symlink.
                                                                                Defaults to None, which will use a temporary file.
    """
    try:
        import transformers
        del transformers
    except ImportError as e:
        raise MissingConditionalImportError(
            extra_deps_group='nlp',
            conda_package='transformers',
            conda_channel='conda-forge',
        ) from e

    # default local path to a tempfile if path is not provided
    if local_checkpoint_save_location is None:
        tmp_dir = tempfile.TemporaryDirectory()
        local_checkpoint_save_location = Path(tmp_dir.name) / 'local-composer-checkpoint.pt'

    # download the checkpoint file
    get_file(str(checkpoint_path), str(local_checkpoint_save_location))

    composer_state_dict = safe_torch_load(local_checkpoint_save_location)

    # load tokenizer
    hf_state = composer_state_dict['state']['integrations']['huggingface']
    hf_tokenizer = HuggingFaceModel.load_huggingface_tokenizer_from_saved_state(hf_state)
    assert hf_tokenizer is not None
    hf_tokenizer.save_pretrained(output_folder)

    config = get_hf_config_from_composer_state_dict(composer_state_dict)
    config.save_pretrained(output_folder)

    peft_config = get_peft_config_from_composer_state_dict(composer_state_dict)
    if peft_config is not None:
        peft_config.save_pretrained(str(output_folder))

    weights_state_dict = composer_state_dict['state']['model']
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(weights_state_dict, prefix='model.')  # type: ignore

    # NOTE: This only works for default adapter name, not multiple adapters
    if peft_config is not None:
        weights_state_dict = filter_state_dict_peft(weights_state_dict, peft_config, adapter_name='default')

        torch.save(weights_state_dict, Path(output_folder) / 'adapter_model.bin')
    else:
        torch.save(weights_state_dict, Path(output_folder) / 'pytorch_model.bin')


def filter_state_dict_peft(
    state_dict: dict[str, Any],
    peft_config: 'PeftConfig',
    adapter_name: str = 'default',
    remove_adapter_names: bool = True,
) -> dict[str, Any]:
    """Filter a state dict to only include the weights needed for a PEFT model

    Note: This function only works with LORA PEFT models right now.

    Args:
        state_dict (dict[str, Any]): The state dict to filter
        peft_config (PeftConfig): The PEFT config to use to filter the state dict
        adapter_name (str, optional): The name of the adapter to filter for. Defaults to 'default'.
        remove_adapter_names (bool, optional): Whether to remove the adapter names from the state dict keys. Defaults to True.

    Returns:
        dict[str, Any]: The filtered state dict
    """

    if peft_config.peft_type != 'LORA':
        raise NotImplementedError(f'Only LoRA PEFT is supported. Got {peft_config.peft_type}')

    # Filtering copied from https://github.com/huggingface/peft/blob/4186c9b104644fd247a4cc0dc2dfc1ede4665204/src/peft/utils/save_and_load.py#L68C1-L86C116
    bias = peft_config.bias  # type: ignore
    if bias == 'none':
        to_return = {k: state_dict[k] for k in state_dict if 'lora_' in k}
    elif bias == 'all':
        to_return = {k: state_dict[k] for k in state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in state_dict:
            if 'lora_' in k:
                to_return[k] = state_dict[k]
                bias_name = k.split('lora_')[0] + 'bias'
                if bias_name in state_dict:
                    to_return[bias_name] = state_dict[bias_name]
    else:
        raise NotImplementedError
    to_return = {k: v for k, v in to_return.items() if (('lora_' in k and adapter_name in k) or ('bias' in k))}

    if remove_adapter_names:
        to_return = {k.replace(f'.{adapter_name}', ''): v for k, v in to_return.items()}
    return to_return
