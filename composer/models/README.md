# Models

Models are implemented both as a standalone `torch.nn.Module`, or as a `MosaicModel` that is compatible with the MosaicTrainer.

<!--- Table Start (automatically generated) --->
Task|Dataset|Name|Quality|Metric|TTT|Hparams
----|-------|----|-------|------|---|-------
Image Classification|MNIST|[SimpleConvNet](classify_mnist/)||Accuracy|?|[composer/yamls/models/classify_mnist_cpu.yaml](../yamls/models/classify_mnist_cpu.yaml)
Image Classification|ImageNet|[EfficientNet-B0](efficientnetb0/)|76.63|Top-1 Accuracy|21h 48m|[composer/yamls/models/efficientnetb0.yaml](../yamls/models/efficientnetb0.yaml)
Language Modeling|OpenWebText|[GPT-2 52M](gpt2/)|30.88|Perplexity|02:44|[composer/yamls/models/gpt2_52m.yaml](../yamls/models/gpt2_52m.yaml)
Language Modeling|OpenWebText|[GPT-2 83M](gpt2/)|26.57|Perplexity|04:52|[composer/yamls/models/gpt2_83m.yaml](../yamls/models/gpt2_83m.yaml)
Language Modeling|OpenWebText|[GPT-2 125M](gpt2/)|24.04|Perplexity|08:25|[composer/yamls/models/gpt2_125m.yaml](../yamls/models/gpt2_125m.yaml)
Image Classification|ImageNet|[ResNet101](resnet101/)|78.10|Top-1 Accuracy|8h 15m|[composer/yamls/models/resnet101.yaml](../yamls/models/resnet101.yaml)
Image Classification|ImageNet|[ResNet18](resnet18/)||Top-1 Accuracy|?|[composer/yamls/models/resnet18.yaml](../yamls/models/resnet18.yaml)
Image Classification|ImageNet|[ResNet50](resnet50/)|76.51|Top-1 Accuracy|3h 33m|[composer/yamls/models/resnet50.yaml](../yamls/models/resnet50.yaml)
Image Classification|CIFAR10|[ResNet56](resnet56_cifar10/)|93.1|Top-1 Accuracy|15m|[composer/yamls/models/resnet56_cifar10.yaml](../yamls/models/resnet56_cifar10.yaml)
Image Segmentation|BRATS|[UNet](unet/)|69.1|Dice|21m|[composer/yamls/models/unet.yaml](../yamls/models/unet.yaml)
<!--- Table End --->
