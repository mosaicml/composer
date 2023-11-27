# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A wrapper class that converts 🤗 Transformers models to composer models"""

from __future__ import annotations

import inspect
import json
import logging
import os
import random
import string
import tempfile
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple, Type, Union

import torch
from torchmetrics import Metric

from composer.metrics import InContextLearningMetric, InContextLearningQAAccuracy
from composer.models.base import ComposerModel
from composer.utils import MissingConditionalImportError, dist, get_file, import_object, is_model_fsdp, safe_torch_load

if TYPE_CHECKING:
    import transformers
    from transformers import PretrainedConfig
    from transformers.models.auto.auto_factory import _BaseAutoModelClass

log = logging.getLogger(__name__)

__all__ = ['HuggingFaceModel']


class HuggingFaceModel(ComposerModel):
    """
    A wrapper class that converts 🤗 Transformers models to composer models.

    Args:
        model (transformers.PreTrainedModel): A 🤗 Transformers model.
        tokenizer (transformers.PreTrainedTokenizer, optional): The tokenizer used to prepare the dataset. Default ``None``.

            .. note:: If the tokenizer is provided, its config will be saved in the composer checkpoint, and it can be reloaded
                using :meth:`HuggingFaceModel.hf_from_composer_checkpoint`. If the tokenizer is not provided here, it will not be saved in the composer checkpoint.
        use_logits (bool, optional): If True, the model's output logits will be used to calculate validation metrics. Else, metrics will be inferred from the HuggingFaceModel directly. Default: ``False``
        metrics (list[Metric], optional): list of torchmetrics to apply to the output of `eval_forward` during training. If ``eval_metrics`` is ``None``, these will also be used as ``eval_metrics``.  Default: ``None``.
        eval_metrics (list[Metric], optional): list of torchmetrics to compute on the eval_dataloader, or be accessible to :class:`Evaluator`s. Default: ``None``.
        shift_labels (bool, optional): If True, the batch's labels will be shifted before being used to calculate metrics. This should be set to true for CausalLM models and false otherwise. If not specified, `shift_labels` will be set automatically based on the model class name. Default: ``None``.
        allow_embedding_resizing (bool, optional): If True, the model's embeddings will be automatically resized when they are smaller than the tokenizer vocab size. Default: ``False``.

        .. note:: To ensure correct behavior, set `shift_labels` manually if using a custom model (i.e., if `model` is not
        an instance of a registered 🤗 Transformers class).
    .. warning:: This wrapper is designed to work with 🤗 datasets that define a `labels` column.

    Example:

    .. testcode::

        import transformers
        from composer.models import HuggingFaceModel

        hf_model = transformers.AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        hf_tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
        model = HuggingFaceModel(hf_model, hf_tokenizer)
    """

    def __init__(self,
                 model: transformers.PreTrainedModel,
                 tokenizer: Optional[Union[transformers.PreTrainedTokenizer,
                                           transformers.PreTrainedTokenizerFast]] = None,
                 use_logits: Optional[bool] = False,
                 metrics: Optional[List[Metric]] = None,
                 eval_metrics: Optional[List[Metric]] = None,
                 shift_labels: Optional[bool] = None,
                 allow_embedding_resizing: bool = False) -> None:
        try:
            import transformers
            del transformers  # unused
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='nlp',
                                                conda_package='transformers',
                                                conda_channel='conda-forge') from e

        super().__init__()
        self.model = model
        self.config = model.config
        self.model_forward_args = inspect.getfullargspec(self.model.forward).args
        self.tokenizer = tokenizer

        if self.tokenizer is None:
            log.warning(
                'The tokenizer was not provided. This means the tokenizer config will not be saved in the checkpoint.')

        if tokenizer is not None and self.config.vocab_size < len(tokenizer):
            if allow_embedding_resizing:
                # when the embedding size is smaller than the tokenizer vocab size,
                # the embeddings should get resized to match the tokenizer vocab size
                log.warning(f'The number of tokens in the tokenizer is greater than the number of tokens in the model.'
                            f' This would cause an error during training.'
                            f' Resizing the model embeddings to {len(tokenizer)} from {self.config.vocab_size}.')
                self.model.resize_token_embeddings(len(tokenizer))
            else:
                raise ValueError(
                    f'The number of tokens in the tokenizer is greater than the number of tokens in the model.'
                    f' This would cause an error during training.'
                    f' You can resize the model embeddings to {len(tokenizer)} from {self.config.vocab_size}'
                    f' by calling `model.resize_token_embeddings(len(tokenizer))` before calling the `HuggingFaceModel`'
                    f' constructor, or pass `allow_embedding_resizing=True` to have it done automatically.')
        elif tokenizer is not None and self.config.vocab_size > len(tokenizer):
            # when the embedding size is greater than the tokenizer vocab size,
            # the embeddings do not _need_ to be resized to match the tokenizer vocab size,
            # and should be done by the user if desired
            log.info(
                f'The number of tokens in the tokenizer is less than the number of tokens in the model.'
                f' You may want to resize the model embeddings to {len(tokenizer)} from {self.config.vocab_size}'
                f' by calling `model.resize_token_embeddings(len(tokenizer))` before calling the `HuggingFaceModel`'
                f' constructor. The vocab size is sometimes intentionally set to a multiple of 32 or 64 to improve'
                f' performance.')

        self.use_logits = use_logits

        self.train_metrics: Optional[Dict] = None
        self.val_metrics: Optional[Dict] = None

        if eval_metrics is not None:
            self.val_metrics = {metric.__class__.__name__: metric for metric in eval_metrics}
        if metrics is not None:
            self.train_metrics = {metric.__class__.__name__: metric for metric in metrics}
            # if eval_metrics is None, use the same metrics as train_metrics
            if eval_metrics is None:
                self.val_metrics = {metric.__class__.__name__: metric for metric in metrics}

        self.labels: Optional[torch.Tensor] = None  # set in eval_forward() if exists

        is_causal_lm = _is_registered_causal_lm(model)
        self.shift_labels = is_causal_lm if shift_labels is None else shift_labels
        if is_causal_lm and not self.shift_labels:
            log.warning('The shift_labels argument was set to False but the model is an instance of a'
                        ' HuggingFace Causal LM. This may lead to incorrect behavior.')
            # Note: No warning if shift_labels and not is_causal_lm, since the model may simply be a custom class.

        self.dummy_forward_called = False

    @staticmethod
    def load_huggingface_tokenizer_from_saved_state(
            hf_state: Dict[str, Any],
            trust_remote_code: bool = False,
            tokenizer_save_dir: Optional[str] = None) -> Optional[transformers.PreTrainedTokenizer]:
        """A helper function that loads a HuggingFace tokenizer from a loaded in hf state.

        Args:
            hf_state (Dict[str, Any]): HF state loaded from a Composer checkpoint.
            trust_remote_code (bool, optional): Whether to trust the remote code when loading the tokenizer. Defaults to False.
            tokenizer_save_dir (Optional[str], optional): If specified, where to save the tokenizer files to locally. If not specified,
                a folder with a unique suffix will be saved in the current working directory. Defaults to None.

        Returns:
            Optional[transformers.PreTrainedTokenizer]: The loaded HuggingFace tokenizer
        """
        try:
            import transformers
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='nlp',
                                                conda_package='transformers',
                                                conda_channel='conda-forge') from e
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
                    with open(tokenizer_file_path, 'w') as _f:
                        json.dump(saved_content['content'], _f)
                elif saved_content['file_extension'] == '.txt':
                    with open(tokenizer_file_path, 'w') as _f:
                        for line in saved_content['content']:
                            _f.write(line)
                            _f.write('\n')
                elif saved_content['file_extension'] == '.py':
                    with open(tokenizer_file_path, 'w') as _f:
                        _f.write(saved_content['content'])
                elif saved_content['file_extension'] == '.model':
                    try:
                        import sentencepiece as spm
                    except ImportError as e:
                        raise MissingConditionalImportError(extra_deps_group='sentencepiece',
                                                            conda_package='sentencepiece') from e
                    s = spm.SentencePieceProcessor()
                    s.load_from_serialized_proto(saved_content['content'])
                    with open(tokenizer_file_path, 'wb') as _f:
                        _f.write(s.serialized_model_proto())

            hf_tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_save_dir,
                                                                      trust_remote_code=trust_remote_code)

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
            hf_state: Dict[str, Any], loaded_state_dict: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]],
            model_instantiation_class: type | str | None,
            model_config_kwargs: Dict[str, Any] | None) -> transformers.PreTrainedModel:
        """A helper function that loads a HuggingFace model class from a loaded in hf state.

        Args:
            hf_state (Dict[str, Any]): HF state loaded from a Composer checkpoint.
            model_instantiation_class (Union[Type[:class:`transformers.PreTrainedModel`], Type[:class:`transformers.AutoModel`], str]), optional):
                Class to use to create the HuggingFace model. Defaults to the model class used in the original checkpoint. If this argument is
                a HuggingFace auto class (e.g. :class:`transformers.AutoModel` or :class:`transformers.AutoModelForSequenceClassification`), the ``from_config`` method will be used,
                while if it is of type :class:`transformers.PreTrainedModel`, the constructor will be called. This argument can also be a string,
                which will attempt to be imported as the class to use.
            model_config_kwargs: Dict[str, Any]: Extra arguments to pass in for the model config creation (e.g. ``num_labels`` for creating a sequence classification model)
        Returns:
            transformers.PreTrainedModel: The loaded HuggingFace model
        """
        try:
            import transformers
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='nlp',
                                                conda_package='transformers',
                                                conda_channel='conda-forge') from e
        loaded_config = get_hf_config_from_composer_state_dict(loaded_state_dict, config_overrides=model_config_kwargs)

        hf_model_state = hf_state['model']
        if model_instantiation_class is not None:
            # If the instantiation class is explicitly provided, use it
            # If a string is provided, attempt to import the class it refers to
            if isinstance(model_instantiation_class, str):
                try:
                    model_instantiation_class = import_object(':'.join(model_instantiation_class.rsplit('.',
                                                                                                        maxsplit=1)))
                except (ModuleNotFoundError, AttributeError):
                    raise ValueError(
                        textwrap.dedent(
                            f'The provided model_instantiation_class string {model_instantiation_class} could not be imported. '
                            f'Please make sure {model_instantiation_class} is discoverable on the python path, or pass the class '
                            'in directly.'))

            assert model_instantiation_class is not None  # pyright
            # The AutoModel* classes have `from_config`, while the PreTrainedModel classes do not
            # pyright can't tell this isn't a string at this point
            if issubclass(
                    model_instantiation_class,  # type: ignore
                    transformers.models.auto.auto_factory._BaseAutoModelClass):
                hf_model = model_instantiation_class.from_config(loaded_config)  # type: ignore
            else:
                hf_model = model_instantiation_class(loaded_config)  # type: ignore
        else:
            # If the instantiation class is not explicitly provided, attempt to import the saved class and use it
            try:
                saved_class = import_object(':'.join(hf_model_state['config']['class'].rsplit('.', maxsplit=1)))
            except (ModuleNotFoundError, AttributeError):
                raise ValueError(
                    textwrap.dedent(
                        f'The saved class {hf_model_state["config"]["class"]} could not be imported. '
                        'Please either pass in the class to use explicitly via the model_instantiation_class '
                        f'parameter, or make sure that {hf_model_state["config"]["class"]} is discoverable '
                        'on the python path.'))
            hf_model = saved_class(loaded_config)
        return hf_model

    @staticmethod
    def hf_from_composer_checkpoint(
        checkpoint_path: str,
        model_instantiation_class: Optional[Union[Type[transformers.PreTrainedModel], Type['_BaseAutoModelClass'],
                                                  str]] = None,
        model_config_kwargs: Optional[dict] = None,
        local_checkpoint_save_location: Optional[Union[Path, str]] = None,
        trust_remote_code: bool = False,
    ) -> Tuple[transformers.PreTrainedModel, Optional[transformers.PreTrainedTokenizer]]:
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
            model_config_kwargs: Dict[str, Any]: Extra arguments to pass in for the model config creation (e.g. ``num_labels`` for creating a sequence classification model)
            local_checkpoint_save_location (Optional[Union[Path, str]], optional): If specified, where to save the checkpoint file to locally.
                                                                                   If the input ``checkpoint_path`` is already a local path, this will be a symlink.
                                                                                   Defaults to None, which will use a temporary file.
            trust_remote_code (bool, optional): Whether to trust the remote code when loading the tokenizer. Defaults to False.

        Raises:
            ValueError: If the ``model_instantiation_class``, or the model class saved in the checkpoint, is not able to be imported

        Returns:
            Tuple[transformers.PreTrainedModel, Optional[transformers.PreTrainedTokenizer]]: The loaded HuggingFace model and (if present) tokenizer
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
        hf_model = HuggingFaceModel.load_huggingface_model_from_saved_state(hf_state, loaded_state_dict,
                                                                            model_instantiation_class,
                                                                            model_config_kwargs)

        return hf_model, hf_tokenizer

    def forward(self, batch):
        if isinstance(batch, Mapping):
            # Further input validation is left to the huggingface forward call
            batch = {k: v for k, v in batch.items() if k in self.model_forward_args}
            output = self.model(**batch)  # type: ignore (thirdparty)
        else:
            raise ValueError(
                'Unexpected batch type. Expected a dictionary with keys corresponding to the inputs to the forward function of the Huggingface model'
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
                    'Generation eval cannot be used without providing a tokenizer to the model constructor.')

            self.labels = batch.pop('labels')
            generation = self.generate(batch['input_ids'],
                                       attention_mask=batch['attention_mask'],
                                       max_new_tokens=batch['generation_length'],
                                       synced_gpus=dist.get_world_size() > 1,
                                       **batch.get('generation_kwargs', {}))

            # don't remove prefix space to sentencepiece models
            if len(self.tokenizer(' a', add_special_tokens=False)['input_ids']) == 1:
                return self.tokenizer.batch_decode(generation[:, batch['input_ids'].shape[1]:],
                                                   skip_special_tokens=True)
            else:
                return [
                    ' ' + generation
                    for generation in self.tokenizer.batch_decode(generation[:, batch['input_ids'].shape[1]:],
                                                                  skip_special_tokens=True)
                ]

        if self.use_logits or batch.get('mode', None) == 'icl_task':
            # pop labels first to avoid computing loss
            self.labels = batch.pop('labels')

            # HF encoder decoder models like T5 expect either decoder_input_ids or labels,
            # so we add decoder_input_ids to the batch if it is missing
            if self.model.config.is_encoder_decoder and 'decoder_input_ids' not in batch:
                if hasattr(self.model, 'prepare_decoder_input_ids_from_labels'):
                    batch['decoder_input_ids'] = self.model.prepare_decoder_input_ids_from_labels(labels=self.labels)
                else:
                    raise RuntimeError(
                        'Encoder decoder models require that either decoder_input_ids is present in the batch'
                        ' or that the model has a prepare_decoder_input_ids_from_labels method.')

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

    def get_metrics(self, is_train: bool = False) -> Dict[str, Metric]:
        if is_train:
            metrics = self.train_metrics
        else:
            metrics = self.val_metrics

        return metrics if metrics else {}

    def update_metric(self, batch: Any, outputs: Any, metric: Metric) -> None:
        if isinstance(metric, InContextLearningQAAccuracy):
            assert self.labels is not None
            metric.update(batch=batch, outputs=outputs, labels=self.labels)  # pyright: ignore [reportGeneralTypeIssues]
        elif isinstance(metric, InContextLearningMetric):
            assert self.labels is not None
            metric.update(batch, outputs, self.labels)  # pyright: ignore [reportGeneralTypeIssues]
        else:
            metric.update(outputs, self.labels)  # pyright: ignore [reportGeneralTypeIssues]

    def get_metadata(self):
        model_output = {}
        tokenizer_output = {}
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            model_dir = tmp_dir / 'model'
            tokenizer_dir = tmp_dir / 'tokenizer'
            self.model.config.save_pretrained(model_dir)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(tokenizer_dir)

            with open(model_dir / 'config.json') as _config_file:
                model_config = json.load(_config_file)

            model_output['config'] = {
                'file_extension': '.json',
                'content': model_config,
                'class': f'{self.model.__class__.__module__}.{self.model.__class__.__name__}'
            }

            if self.tokenizer is not None:
                for tokenizer_file_name in tokenizer_dir.iterdir():
                    tokenizer_file_path = tokenizer_dir / tokenizer_file_name
                    tokenizer_file_extension = tokenizer_file_path.suffix
                    if tokenizer_file_extension == '.txt':
                        with open(tokenizer_file_path) as _tokenizer_file:
                            tokenizer_file_content = _tokenizer_file.read().split('\n')
                    elif tokenizer_file_extension == '.json':
                        with open(tokenizer_file_path, 'rb') as _tokenizer_file:
                            tokenizer_file_content = json.load(_tokenizer_file)
                    elif tokenizer_file_extension == '.py':
                        with open(tokenizer_file_path) as _tokenizer_file:
                            tokenizer_file_content = _tokenizer_file.read()
                    elif tokenizer_file_extension == '.model':
                        try:
                            import sentencepiece as spm
                        except ImportError as e:
                            raise MissingConditionalImportError(extra_deps_group='sentencepiece',
                                                                conda_package='sentencepiece') from e
                        s = spm.SentencePieceProcessor(model_file=str(tokenizer_file_path))
                        tokenizer_file_content = s.serialized_model_proto()
                    else:
                        raise ValueError(
                            f'Unexpected file ending {tokenizer_file_name} in output of tokenizer.save_pretrained.')

                    tokenizer_output[tokenizer_file_path.name] = {
                        'file_extension': tokenizer_file_extension,
                        'content': tokenizer_file_content
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

        from composer.utils.misc import using_torch_2

        # We need to call forward once in order for FSDP + generate to work
        # This solution works because parameters in the root FSDP module are not freed after forward
        # See https://github.com/huggingface/accelerate/issues/570, https://github.com/huggingface/accelerate/issues/947,
        # and https://github.com/pytorch/pytorch/issues/82461, https://github.com/pytorch/pytorch/issues/100069 for more info
        # Note: This is a solution for Torch 1.13.x, and there is a different solution below for Torch 2.0
        if not using_torch_2() and not self.dummy_forward_called and is_model_fsdp(self.model):
            with torch.no_grad():
                maybe_decoder_input_ids = {}
                if self.model.config.is_encoder_decoder:
                    maybe_decoder_input_ids['decoder_input_ids'] = torch.tensor([[0]],
                                                                                dtype=torch.long,
                                                                                device=input_ids.device)
                self.model(input_ids=torch.tensor([[0]], dtype=torch.long, device=input_ids.device),
                           **maybe_decoder_input_ids)
            self.dummy_forward_called = True

        if is_model_fsdp(self.model) and using_torch_2():
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


def _is_registered_causal_lm(model: transformers.PreTrainedModel) -> bool:
    """Return True if model class is either a registered 🤗 Causal LM or a subclass of one"""
    try:
        from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
    except ImportError as e:
        raise MissingConditionalImportError(extra_deps_group='nlp',
                                            conda_package='transformers',
                                            conda_channel='conda-forge') from e

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
    return any(isinstance(model, causal_lm_class) for causal_lm_class in causal_lm_classes)


def get_hf_config_from_composer_state_dict(state_dict: Dict[str, Any],
                                           config_overrides: Optional[Dict[str, Any]] = None) -> 'PretrainedConfig':
    """Get a HuggingFace config from a composer state dict with overrides applied

    Args:
        state_dict (Dict[str, Any]): The state dict to get the config from
        config_overrides (Dict[str, Any], optional): Any overrides to apply to the config

    Returns:
        transformers.PretrainedConfig: The HuggingFace config
    """
    try:
        import transformers
    except ImportError as e:
        raise MissingConditionalImportError(extra_deps_group='nlp',
                                            conda_package='transformers',
                                            conda_channel='conda-forge') from e

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
        try:
            return transformers.AutoConfig.from_pretrained(hf_config_dict['_name_or_path'], **hf_config_dict)
        except KeyError:
            raise Exception(
                f'Could not load config from state dict using either `for_model` or `from_pretrained`.'
                f'Please make sure that the model_type={hf_config_dict.get("model_type")} is valid, or that the'
                f'config has a valid `_name_or_path`.')


def write_huggingface_pretrained_from_composer_checkpoint(
        checkpoint_path: Union[Path, str],
        output_folder: Union[Path, str],
        local_checkpoint_save_location: Optional[Union[Path, str]] = None) -> None:
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
        raise MissingConditionalImportError(extra_deps_group='nlp',
                                            conda_package='transformers',
                                            conda_channel='conda-forge') from e

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

    weights_state_dict = composer_state_dict['state']['model']
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(weights_state_dict, prefix='model.')
    torch.save(weights_state_dict, Path(output_folder) / 'pytorch_model.bin')
