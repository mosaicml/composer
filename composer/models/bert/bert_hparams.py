# Copyright 2021 MosaicML. All Rights Reserved.

"""`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ general and classification interfaces for
:class:`.BERTModel`."""

import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING

import yahp as hp

from composer.models.transformer_hparams import TransformerHparams

if TYPE_CHECKING:
    from composer.models.bert import BERTModel

__all__ = ["BERTForClassificationHparams", "BERTHparams"]


@dataclass
class BERTForClassificationHparams(TransformerHparams):
    """`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ classification interface for
    :class:`.BERTModel`.

    Args:
        pretrained_model_name (str): Pretrained model name to pull from Hugging Face Model Hub.
        model_config (Dict[str, JSON]): A dictionary providing a HuggingFace model configuration.
        tokenizer_name (str): The tokenizer used for this model,
            necessary to assert required model inputs.
        use_pretrained (bool, optional): Whether to initialize the model with the pretrained weights.
        gradient_checkpointing (bool, optional): Use gradient checkpointing. Default: ``False``.
        num_labels (int, optional): The number of classes in the segmentation task. Default: ``2``.
    """
    num_labels: int = hp.optional(doc="The number of possible labels for the task.", default=2)

    def validate(self):
        if self.num_labels < 1:
            raise ValueError("The number of target labels must be at least one.")

    def initialize_object(self) -> "BERTModel":
        try:
            import transformers
        except ImportError as e:
            raise ImportError(
                textwrap.dedent("""\
                Composer was installed without NLP support. To use NLP with Composer, run `pip install mosaicml[nlp]`
                if using pip or `conda install -c conda-forge transformers` if using Anaconda.""")) from e

        from composer.models.bert.model import BERTModel
        self.validate()

        model_hparams = {"num_labels": self.num_labels}

        if self.model_config:
            config = transformers.BertConfig.from_dict(self.model_config, **model_hparams)
        elif self.pretrained_model_name is not None:
            config = transformers.BertConfig.from_pretrained(self.pretrained_model_name, **model_hparams)
        else:
            raise ValueError('One of pretrained_model_name or model_config needed.')
        config.num_labels = self.num_labels

        # setup the tokenizer in the hparams interface
        tokenizer = transformers.BertTokenizer.from_pretrained(self.tokenizer_name)

        if self.use_pretrained:
            # TODO (Moin): handle the warnings on not using the seq_relationship head
            model = transformers.AutoModelForSequenceClassification.from_pretrained(self.pretrained_model_name,
                                                                                    **model_hparams)
        else:
            model = transformers.AutoModelForSequenceClassification.from_config(  #type: ignore (thirdparty)
                config, **model_hparams)

        return BERTModel(
            module=model,
            config=config,  #type: ignore (thirdparty)
            tokenizer=tokenizer,
        )


@dataclass
class BERTHparams(TransformerHparams):
    """`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :class:`.BERTModel`.

    Args:
        pretrained_model_name (str): "Pretrained model name to pull from Huggingface Model Hub."
        model_config (Dict[str, JSON]): A dictionary providing a HuggingFace model configuration.
        tokenizer_name (str): The tokenizer used for this model,
            necessary to assert required model inputs.
        use_pretrained (bool, optional): Whether to initialize the model with the pretrained weights.
        gradient_checkpointing (bool, optional): Use gradient checkpointing. default: False.
    """

    def initialize_object(self) -> "BERTModel":
        try:
            import transformers
        except ImportError as e:
            raise ImportError(
                textwrap.dedent("""\
                Composer was installed without NLP support. To use NLP with Composer, run `pip install mosaicml[nlp]`
                if using pip or `conda install -c conda-forge transformers` if using Anaconda.""")) from e

        from composer.models.bert.model import BERTModel
        self.validate()

        if self.model_config:
            config = transformers.BertConfig.from_dict(self.model_config)
        elif self.pretrained_model_name is not None:
            config = transformers.BertConfig.from_pretrained(self.pretrained_model_name)
        else:
            raise ValueError('One of pretrained_model_name or model_config needed.')

        # set the number of labels ot the vocab size, used for measuring MLM accuracy
        config.num_labels = config.vocab_size

        # setup the tokenizer in the hparams interface
        tokenizer = transformers.BertTokenizer.from_pretrained(self.tokenizer_name)

        if self.use_pretrained:
            # TODO (Moin): handle the warnings on not using the seq_relationship head
            model = transformers.AutoModelForMaskedLM.from_pretrained(self.pretrained_model_name)
        else:
            model = transformers.AutoModelForMaskedLM.from_config(config)  #type: ignore (thirdparty)

        return BERTModel(
            module=model,
            config=config,  #type: ignore (thirdparty)
            tokenizer=tokenizer,
        )
