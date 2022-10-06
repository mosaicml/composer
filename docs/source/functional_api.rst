ƒ() Functional
======================

The simplest way to use Composer's algorithms is via the functional API. Composer's
algorithms can be grouped into three, broad classes:

- `data augmentations` add additional transforms to the training data.
- `model surgery` algorithms modify the network architecture.
- `training loop modifications` change the logic in the training loop.

Data augmentations can be inserted either into the dataloader as a transform or after a
batch has been loaded, depending on what the augmentation acts on. Here is an example of using
:doc:`/method_cards/randaugment` with the functional API.

.. code-block:: python

    import torch
    from torchvision import datasets, transforms

    from composer import functional as cf

    c10_transforms = transforms.Compose([cf.randaugment_image(), # <---- Add RandAugment
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])

    dataset = datasets.CIFAR10('../data',
                            train=True,
                            download=True,
                            transform=c10_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024)

Other data augmentations, such as :class:`.CutMix`, act on a batch of inputs. These can be
inserted in the training loop after a batch is loaded from the dataloader as follows:

.. code-block:: python

    from composer import functional as cf

    cutmix_alpha = 1
    num_classes = 10
    for batch_idx, (data, target) in enumerate(dataloader):
        data = cf.cutmix(  # <-- insert cutmix
            data,
            target,
            alpha=cutmix_alpha,
            num_classes=num_classes
        )
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

Model surgery algorithms make direct modifications to the network itself. Functionally,
these can be called as follows, using :class:`.BlurPool` as an example

.. code-block:: python

    import torchvision.models as models

    from composer import functional as cf

    model = models.resnet18()
    cf.apply_blurpool(model)

Each method card has a section describing how to use these methods in your own trainer loop.

.. automodule:: composer.functional
   :noindex:
   :members:
   :show-inheritance:
   :ignore-module-all:
