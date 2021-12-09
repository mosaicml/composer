# Copyright 2021 MosaicML. All Rights Reserved.

from typing import TYPE_CHECKING

from composer.models.transformer_hparams import TransformerHparams

if TYPE_CHECKING:
    from composer.models.transformer_shared import MosaicTransformer


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
