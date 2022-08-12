# 🏞️ Progressive Image Resizing


[\[How to Use\]](#how-to-use) - [\[Suggested Hyperparameters\]](#suggested-hyperparameters) - [\[Technical Details\]](#technical-details) - [\[Attribution\]](#attribution)

 `Computer Vision`

Progressive Resizing works by initially training on images that have been downsampled to a smaller size. It slowly grows the images back to their full size by a set point in training and uses full-size images for the remainder of training. Progressive resizing reduces costs during the early phase of training when the network may learn coarse-grained features that do not require details lost by reducing image resolution.

| ![ProgressiveResizing](https://storage.googleapis.com/docs.mosaicml.com/images/methods/progressive_resizing_vision.png) |
|:--|
|*An example image as it would appear to the network at different stages of training with progressive resizing. At the beginning of training, each training example is at its smallest size. Throughout the pre-training phase, example size increases linearly. At the end of the pre-training phase, example size has reached its full value and remains at that value for the remainder of training (the fine-tuning phase).*|

## How to Use

### Functional Interface

An example of resizing just the inputs (appropriate for classification) to half their original size
```python
import torch
import torch.nn.functional as F

from composer import functional as cf

def training_loop(model, train_loader):
    opt = torch.optim.Adam(model.parameters())
    loss_fn = F.cross_entropy
    model.train()

    for epoch in range(num_epochs):
        for X, y in train_loader:
            X_resized, _ = cf.resize_batch(X, y, scale_factor=0.5, mode='resize', resize_targets=False)
            y_hat = model(X_resized)
            loss = loss_fn(y_hat, y)
            loss.backward()
            opt.step()
            opt.zero_grad()
```

An example of resizing both the inputs and targets (appropriate for semantic segmentation) to half their original size
```python
import torch
import torch.nn.functional as F

from composer import functional as cf

def training_loop(model, train_loader):
    opt = torch.optim.Adam(model.parameters())
    loss_fn = F.cross_entropy
    model.train()

    for epoch in range(num_epochs):
        for X, y in train_loader:
            X_resized, y_resized = resize_batch(X, y, scale_factor=0.5, mode='resize', resize_targets=True)
            y_hat = model(X_resized)
            loss = loss_fn(y_hat, y_resized)
            loss.backward()
            opt.step()
            opt.zero_grad()
```

### Composer Trainer

<!--pytest.mark.gpu-->
<!--
```python
from torch.utils.data import DataLoader
from tests.common import RandomImageDataset, SimpleConvModel

model = SimpleConvModel()
train_dataloader = DataLoader(RandomImageDataset())
eval_dataloader = DataLoader(RandomImageDataset())
```
-->
<!--pytest-codeblocks:cont-->
```python
# Instantiate the algorithm and pass it into the Trainer
# The trainer will automatically run it at the appropriate points in the training loop

from composer.algorithms import ProgressiveResizing
from composer.trainer import Trainer

progressive_resizing_algorithm = ProgressiveResizing(
                                    mode='resize',
                                    initial_scale=1.0,
                                    finetune_fraction=0.2,
                                    delay_fraction=0.2,
                                    size_increment=32,
                                    resize_targets=False
                                )

trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    max_duration='1ep',
    algorithms=[progressive_resizing_algorithm]
)

trainer.fit()
```

### Implementation Details

Progressive resizing works by resizing input images (and optionally targets) to a smaller size. The resize can be done via a nearest neighbor interpolation by specifying `mode='resize'` or simply by cropping the images with `mode='crop'`. Resizing takes place on device, on a batch of input/target pairs.

## Suggested Hyperparameters

We found `initial_scale = 0.5` (starting training on images where each side length has been reduced by 50%), `finetune_fraction = 0.2` (reserving the final 20% of training for full-sized images),
`delay_fraction = 0.5` (reserving the first 50% of training for initial\_scale-sized images) and `size_increment = 4` (pegging image size to the nearest multiple of 4) to work well for ResNet-50 on ImageNet.

## Technical Details

When using Progressive Resizing, the early steps of training run faster than the later steps of training (which run at the original speed), since the smaller images reduce the amount of computation that the network must perform.
Ideally, generalization performance is not impacted much by Progressive Resizing, but this depends on the specific dataset, network architecture, task, and hyperparameters.
In our experience with ResNets on ImageNet, Progressive Resizing improves training speed (as measured by wall clock time) with limited effects on classification accuracy.

> ✅ Progressive Resizing Improves the Tradeoff Between Quality and Training Speed
>
> In our experiments, Progressive Resizing improves the attainable tradeoffs between training speed and the final quality of the trained model.
> In some cases, it leads to slightly lower quality than the original model for the same number of training steps.
> However, Progressive Resizing increases training speed so much (via improved throughput during the early part of training) that it is possible to train for more steps, recover accuracy, and still complete training in less time.

Our implementation of Progressive Resizing gives two options for resizing the images:
* `mode = "crop"` does a random crop of the input image to a smaller size. This mode is appropriate for datasets where scale is important. For example, we get better results using crops for ResNet-56 on CIFAR-10, where the objects are similar sizes to one another and the images are already low resolution.
* `mode = "resize"` does downsampling with a bilinear interpolation of the image to a smaller size. This mode is appropriate for datasets where scale is variable, all the content of the image is needed each time it is seen, or the images are relatively higher resolution. For example, we get better results using resizing for ResNet-50 on ImageNet.

Progressive Resizing requires that the network architecture be capable of handling different sized images. Additionally, since the early epochs of training require significantly less GPU compute than the later epochs, CPU/dataloading may become a bottleneck in the early epochs even if this isn’t true in the late epochs.

> ❗ Potential CPU or Data Loading Bottleneck
>
> Progressive resizing increases training throughput during the pre-training phase, when images are smaller, and especially so during the earliest parts of training.
> It is possible that this increased throughput may lead other parts of the training pipeline, such as data loading or CPU image processing, to become bottlenecks during the early part of training.

Additionally, while we have not investigated this, Progressive Resizing may also change how sensitive the network is to different sizes of objects or how biased the network is in favor of shape or texture.

Progressive Resizing will interact with other methods that change the size of the inputs, such as Selective Backprop with downsampling and ColOut

> 🚧 Interations with Other Methods that Modify Inputs
>
> Progressive resizing changes the size of inputs, so it may interact with other methods that also alter the size, shape, or composition of inputs, such as ColOut, Selective Backprop, and CutOut.

## Attribution

This method was inspired by work on Progressive Resizing by [fast.ai](https://github.com/fastai/fastbook/blob/780b76bef3127ce5b64f8230fce60e915a7e0735/07_sizing_and_tta.ipynb).

*The Composer implementation of this method and the accompanying documentation were produced by Cory Stephenson at MosaicML.*
