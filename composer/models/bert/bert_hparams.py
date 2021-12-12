# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import dataclass
from typing import TYPE_CHECKING

import yahp as hp

from composer.models.transformer_hparams import TransformerHparams
from composer.utils.string_enum import StringEnum

if TYPE_CHECKING:
    from composer.models.transformer_shared import MosaicTransformer


@dataclass
class BERTForClassificationHparams(TransformerHparams):
    num_labels: int = hp.optional(doc="The number of possible labels for the task.", default=2)

    def validate(self):
        if self.num_labels < 1:
            raise ValueError("The number of target labels must be at least one.")

    def initialize_object(self) -> "MosaicTransformer":
        import transformers

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
            model = transformers.AutoModelForSequenceClassification.from_pretrained(self.pretrained_model_name,
                                                                                    **model_hparams)
        else:
            model = transformers.AutoModelForSequenceClassification.from_config(
                config, **model_hparams)  #type: ignore (thirdparty)

        return BERTModel(
            module=model,
            config=config,  #type: ignore (thirdparty)
            tokenizer_name=self.tokenizer_name,
        )


@dataclass
class BERTHparams(TransformerHparams):

    def initialize_object(self) -> "MosaicTransformer":
        import transformers

        from composer.models.bert.model import BERTModel
        self.validate()

        if self.model_config:
            config = transformers.BertConfig.from_dict(self.model_config)
        elif self.pretrained_model_name is not None:
            config = transformers.BertConfig.from_pretrained(self.pretrained_model_name)
        else:
            raise ValueError('One of pretrained_model_name or model_config needed.')

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
