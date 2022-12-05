# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A wrapper class that converts 🤗 Transformers models to composer models"""

from __future__ import annotations

import json
import logging
import tempfile
import textwrap
from collections import UserDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union

import torch
from torchmetrics import Metric

from composer.models.base import ComposerModel
from composer.utils import get_file
from composer.utils.import_helpers import MissingConditionalImportError, import_object

if TYPE_CHECKING:
    import transformers
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
        metrics (list[Metric], optional): list of torchmetrics to apply to the output of `validate`. Default: ``None``.
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
                 metrics: Optional[List[Metric]] = None) -> None:
        try:
            import transformers
            del transformers  # unused
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='transformers',
                                                conda_package='transformers',
                                                conda_channel='conda-forge') from e

        super().__init__()
        self.model = model
        self.config = model.config
        self.tokenizer = tokenizer

        if self.tokenizer is None:
            log.warning(
                'The tokenizer was not provided. This means the tokenizer config will not be saved in the checkpoint.')

        if tokenizer is not None and self.config.vocab_size != len(tokenizer):
            # set model's word embedding matrix and final lm_head to vocab size according to tokenizer
            log.warning(f'The number of tokens in the tokenizer and the number of tokens in the model are different.'
                        f' Resizing the model tokenizer to {len(tokenizer)} from {self.config.vocab_size}.')
            self.model.resize_token_embeddings(len(tokenizer))

        self.use_logits = use_logits

        self.train_metrics = None
        self.val_metrics = None

        if metrics:
            self.train_metrics = {metric.__class__.__name__: metric for metric in metrics}
            self.val_metrics = {metric.__class__.__name__: metric for metric in metrics}

        self.labels = None  # set in eval_forward() if exists

    @staticmethod
    def hf_from_composer_checkpoint(
        checkpoint_path: str,
        model_instantiation_class: Optional[Union[Type[transformers.PreTrainedModel], Type['_BaseAutoModelClass'],
                                                  str]] = None,
        model_config_kwargs: Optional[dict] = None,
        local_checkpoint_save_location: Optional[Union[Path, str]] = None
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
            composer_model = HuggingFaceModel(hf_model, hf_tokenizer)
            trainer = Trainer(model=composer_model,
                              train_dataloader=train_dataloader,
                              save_filename='composer-hf-checkpoint-2.pt',
                              max_duration='1ep',
                              save_folder='./')

        Args:
            checkpoint_path (str): Path to the composer checkpoint, can be a local path, ``http(s)://`` url, or ``s3://`` uri
            model_instantiation_class (Union[Type[:class:`transformers.PreTrainedModel`], Type[:class:`transformers.AutoModel`], str]), optional):
                Class to use to create the HuggingFace model. Defaults to the model class used in the original checkpoint. If this argument is
                a HuggingFace auto class (e.g. :class:`transformers.AutoModel` or :class:`transformers.AutoModelForSequenceClassification`), the ``from_config`` method will be used,
                while if it is of type :class:`transformers.PreTrainedModel`, the constructor will be called. This argument can also be a string,
                which will attempt to be imported as the class to use.
            model_config_kwargs: Dict[str, Any]: Extra arguments to pass in for the model config creation (e.g. ``num_labels`` for creating a sequence classification model)
            local_checkpoint_save_location (Optional[Union[Path, str]], optional): If specified, where to save the checkpoint file to locally.
                                                                                   If the input ``checkpoint_path`` is already a local path, this will be a symlink.
                                                                                   Defaults to None, which will use a temporary file.

        Raises:
            ValueError: If the ``model_instantiation_class``, or the model class saved in the checkpoint, is not able to be imported

        Returns:
            Tuple[transformers.PreTrainedModel, Optional[transformers.PreTrainedTokenizer]]: The loaded HuggingFace model and (if present) tokenizer
        """
        try:
            import transformers
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='transformers',
                                                conda_package='transformers',
                                                conda_channel='conda-forge') from e

        # default local path to a tempfile if path is not provided
        if local_checkpoint_save_location is None:
            tmp_dir = tempfile.TemporaryDirectory()
            local_checkpoint_save_location = Path(tmp_dir.name) / 'local-composer-checkpoint.pt'

        if model_config_kwargs is None:
            model_config_kwargs = {}

        # download the checkpoint file
        get_file(checkpoint_path, str(local_checkpoint_save_location))

        # load the state dict in
        loaded_state_dict = torch.load(local_checkpoint_save_location)

        hf_state = loaded_state_dict['state']['integrations']['huggingface']
        hf_model_state = hf_state['model']
        hf_tokenizer_state = hf_state['tokenizer']

        hf_config_dict = hf_model_state['config']['content']
        # Update the config with any extra args needed
        hf_config_dict.update(model_config_kwargs)
        # JSON keys need to be converted back to ints, huggingface does not auto convert them along this code path
        if 'id2label' in hf_config_dict:
            hf_config_dict['id2label'] = {int(k): v for k, v in hf_config_dict['id2label'].items()}

        loaded_config = transformers.AutoConfig.from_pretrained(hf_config_dict['_name_or_path'], **hf_config_dict)
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

        hf_tokenizer = None
        if hf_tokenizer_state != {}:
            with tempfile.TemporaryDirectory() as _tmp_dir:
                for filename, saved_content in hf_tokenizer_state.items():
                    with open(Path(_tmp_dir) / f'{filename}{saved_content["file_extension"]}', 'w') as _tmp_file:
                        if saved_content['file_extension'] == '.json':
                            json.dump(saved_content['content'], _tmp_file)
                        elif saved_content['file_extension'] == '.txt':
                            for line in saved_content['content']:
                                _tmp_file.write(line)
                                _tmp_file.write('\n')
                hf_tokenizer = transformers.AutoTokenizer.from_pretrained(_tmp_dir)

                # we need to set the name_or_path back because otherwise it is the tmp dir we are loading from here
                hf_tokenizer.name_or_path = hf_tokenizer_state['tokenizer_config']['content']['name_or_path']
                hf_tokenizer.init_kwargs['name_or_path'] = hf_tokenizer_state['tokenizer_config']['content'][
                    'name_or_path']

                # for an unknown reason this key is missing when loading the saved tokenizer, but present with a value of None
                # for the original tokenizer, so we default it to None
                hf_tokenizer.init_kwargs['tokenizer_file'] = hf_tokenizer.init_kwargs.get('tokenizer_file', None)

        return hf_model, hf_tokenizer

    def forward(self, batch):
        if isinstance(batch, dict) or isinstance(batch, UserDict):
            # Further input validation is left to the huggingface forward call
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
        output = outputs if outputs else self.forward(batch)
        if self.use_logits:
            self.labels = batch.pop('labels')
            if self.config.use_return_dict:
                output = output['logits']
            else:
                # logits are at index 1 in the output tuple
                output = output[1]

            # if we are in the single class case, then remove the classes dimension
            if output.shape[1] == 1:
                output = output.squeeze(dim=1)

        return output

    def get_metrics(self, is_train: bool = False) -> Dict[str, Metric]:
        if is_train:
            metrics = self.train_metrics
        else:
            metrics = self.val_metrics

        return metrics if metrics else {}

    def update_metric(self, batch: Any, outputs: Any, metric: Metric) -> None:
        metric.update(outputs, self.labels)

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
                    with open(tokenizer_file_path) as _tokenizer_file:
                        tokenizer_file_extension = tokenizer_file_path.suffix
                        if tokenizer_file_extension == '.txt':
                            tokenizer_file_content = _tokenizer_file.read().split('\n')
                        elif tokenizer_file_extension == '.json':
                            tokenizer_file_content = json.load(_tokenizer_file)
                        else:
                            raise ValueError(
                                f'Unexpected file ending {tokenizer_file_name} in output of tokenizer.save_pretrained.')
                    tokenizer_output[tokenizer_file_path.stem] = {
                        'file_extension': tokenizer_file_extension,
                        'content': tokenizer_file_content
                    }
        return {'model': model_output, 'tokenizer': tokenizer_output}
