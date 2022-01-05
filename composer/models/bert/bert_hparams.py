# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import dataclass
from typing import TYPE_CHECKING

import yahp as hp

from composer.models.transformer_hparams import TransformerHparams

if TYPE_CHECKING:
    from composer.models.transformer_shared import MosaicTransformer


@dataclass
class BERTForClassificationHparams(TransformerHparams):
    num_labels: int = hp.optional(doc="The number of possible labels for the task.", default=2)

    def validate(self):
        if self.num_labels < 1:
            raise ValueError("The number of target labels must be at least one.")

    def initialize_object(self) -> "MosaicTransformer":
        try:
            import transformers
        except ImportError as e:
            raise ImportError('transformers is not installed. '
                              'Please install with `pip install mosaicml-composer[nlp]`') from e

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

        if self.use_pretrained:
            # TODO (Moin): handle the warnings on not using the seq_relationship head
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                self.pretrained_model_name, **model_hparams) 
        else:
            model = transformers.AutoModelForSequenceClassification.from_config (#type: ignore (thirdparty)
                config, **model_hparams)  

        return BERTModel(
            module=model,
            config=config,  #type: ignore (thirdparty)
            tokenizer_name=self.tokenizer_name,
        )


@dataclass
class BERTHparams(TransformerHparams):

    def initialize_object(self) -> "MosaicTransformer":
        try:
            import transformers
        except ImportError as e:
            raise ImportError('transformers is not installed. '
                              'Please install with `pip install mosaicml-composer[nlp]`') from e

        from composer.models.bert.model import BERTModel
        self.validate()

        if self.model_config:
            config = transformers.BertConfig.from_dict(self.model_config)
        elif self.pretrained_model_name is not None:
            config = transformers.BertConfig.from_pretrained(self.pretrained_model_name)
        else:
            raise ValueError('One of pretrained_model_name or model_config needed.')

        # set the number of labels ot the voacb size, used for measuring MLM accuracy
        config.num_labels = config.vocab_size

        if self.use_pretrained:
            # TODO (Moin): handle the warnings on not using the seq_relationship head
            model = transformers.AutoModelForMaskedLM.from_pretrained(self.pretrained_model_name)
        else:
            model = transformers.AutoModelForMaskedLM.from_config(config)  #type: ignore (thirdparty)

        return BERTModel(
            module=model,
            config=config,  #type: ignore (thirdparty)
            tokenizer_name=self.tokenizer_name,
        )
