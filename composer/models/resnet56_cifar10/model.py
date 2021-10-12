from composer.models.base import MosaicClassifier
from composer.models.resnet56_cifar10.resnet56_cifar10_hparams import CIFARResNetHparams
from composer.models.resnets import CIFAR_ResNet


class CIFAR10_ResNet56(MosaicClassifier):

    def __init__(self, hparams: CIFARResNetHparams) -> None:
        assert hparams.num_classes is not None, "num_classes must be set in ModelHparams"
        model = CIFAR_ResNet.get_model_from_name(
            "cifar_resnet_56",
            hparams.initializers,
            hparams.num_classes,
        )
        super().__init__(module=model)
