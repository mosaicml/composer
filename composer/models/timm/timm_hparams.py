# Copyright 2021 MosaicML. All Rights Reserved.
from dataclasses import dataclass
import yahp as hp
from composer.models.model_hparams import ModelHparams
import timm


@dataclass
class DenseNet161Hparams(ModelHparams):

    model: str = hp.optional('timm model string e.g:  list of models can be found at https://github.com/rwightman/pytorch-image-models')
    pretrained: bool = hp.optional("imagenet pretrained", default=False)

    num_classes: int = hp.optional("The number of classes.  Needed for classification tasks", default=1000)
    
    drop_rate: float = hp.optional("dropout rate", default=0.0)
    drop_path_rate: float = hp.optional("drop path rate model default if None", default=None)
    drop_block_rate: float = hp.optional("drop block rate", default=None)

    global_pool: str = hp.optional("Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.", default=None)
    bn_momentum: float = hp.optional("BatchNorm momentum override (if not None)")
    bn_eps: float = hp.optional("BatchNorm epsilon override (if not None)")
 

    timm.create_model(
        model=model,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        drop_block_rate=drop_block_rate,
        global_pool=global_pool,
        bn_momentum=bn_momentum,
        bn_eps=bn_eps)
=
    def validate(self):
        if self.model is None:
            raise ValueError(f'model must be one of {timm.models.list_models()')

    def initialize_object(self):
        return timm.create_model(num_classes=self.num_classes,
                            num_channels=self.num_channels,
                            dropout=self.dropout,
                            pretrained=self.pretrained)