from composer.models.base import MosaicClassifier
from composer.models.resnet101.resnet101_hparams import ResNet101Hparams
from composer.models.resnets import ImageNet_ResNet


class ResNet101(MosaicClassifier):

    def __init__(self, hparams: ResNet101Hparams) -> None:
        assert hparams.num_classes is not None, "num_classes must be set in ModelHparams"
        model = ImageNet_ResNet.get_model_from_name(
            'imagenet_resnet_101',
            hparams.initializers,
            hparams.num_classes,
        )
        super().__init__(module=model)
