# 🎃 Cutout

[\[How to Use\]](#how-to-use) - [\[Suggested Hyperparameters\]](#suggested-hyperparameters) - [\[Technical Details\]](#technical-details) - [\[Attribution\]](#attribution)

`Computer Vision`

Cutout is a data augmentation technique that masks one or more square regions of an input image, replacing them with gray boxes.
It is a regularization technique that improves the accuracy of models for computer vision.

| ![CutOut](https://storage.googleapis.com/docs.mosaicml.com/images/methods/cutout.png) |
|:--:
|*Several images from the CIFAR-10 dataset with Cutout applied. Cutout adds a gray box that occludes a portion of each image. This is [Figure 1 from DeVries & Taylor (2017)](https://arxiv.org/abs/1708.04552).*|

## How to Use

### Functional Interface

```python
# Run the CutOut algorithm directly on the batch data using the Composer functional API
import torch
import torch.nn.functional as F

from composer import functional as cf

def training_loop(model, train_loader):
    opt = torch.optim.Adam(model.parameters())
    loss_fn = F.cross_entropy
    model.train()

    for epoch in range(num_epochs):
        for X, y in train_loader:
            X_cutout = cf.cutout_batch(X,
                                       num_holes=1,
                                       length=0.5)

            y_hat = model(X_cutout)
            loss = loss_fn(y_hat, y)
            loss.backward()
            opt.step()
            opt.zero_grad()
```

### Composer Trainer

<!--pytest.mark.gpu-->
<!--
```python
from torch.utils.data import DataLoader
from tests.common import RandomClassificationDataset, SimpleModel

model = SimpleModel()
train_dataloader = DataLoader(RandomClassificationDataset())
eval_dataloader = DataLoader(RandomClassificationDataset())
```
-->
<!--pytest-codeblocks:cont-->
```python
# Instantiate the algorithm and pass it into the Trainer
# The trainer will automatically run it at the appropriate points in the training loop

from composer.algorithms import CutOut
from composer.trainer import Trainer

cutout = CutOut(num_holes=1, length=0.5)

trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    max_duration='1ep',
    algorithms=[cutout]
)

trainer.fit()
```

### Implementation Details

CutOut randomly selects `num_holes` square regions (which are possibly overlapping) with side length `length` and uses them to generate a binary mask for the image where the points within any hole are set to 0 and the remaining points are set to 1.
This mask is then multiplied element-wise with the image in order to set the pixel value of any pixel value within a hole to 0.

CutOut is implemented following the [original paper](https://arxiv.org/abs/1708.04552). However, our implementation currently differs in that CutOut operates on a batch of data and runs on device to avoid potential CPU bottlenecks.
This means the same bounding box is used for all examples in a batch, which can have either a positive or negative effect on accuracy.

The construction of the bounding box for the mixed region follows the [paper's implementation](https://github.com/uoguelph-mlrg/Cutout) which selects the center pixel of the bounding box uniformly at random from all locations in the image and clips the bounding box to fit. This implies that the size of the region masked by CutOut is not always square and that the area is not always as large as suggested by the `length` parameter. It also implies that not all regions are equally likely to lie inside the bounding box.

## Suggested Hyperparameters

We found that setting `num_holes=1` (adding a single gray patch) to the image gives good results. We also found that setting `length = 0.5`, indicating that the masked region should have height and width half as large as the image, produces good results. However, in some scenarios this value may be too large, obstructing a quarter of the total area of the image; if so, setting `length` to a number of pixels equivalent to a quarter of the image width or height may be better.

## Technical Details

Cutout works by randomly choosing one or more square regions from an input image and replacing them with the mean value over the dataset.
Since it is common to normalize image data based on the dataset mean and variance, the mean value is typically 0.
To ease implementation, we went with a simple binary mask, in which the regions to be cut out are set to pixel value zero and the remainder of the image stays the same.

We found Cutout to be an effective way of improving accuracy for ResNets trained on CIFAR-10 and ImageNet in the absence of robust hyperparameter tuning and other regularizers.
As we improved our training methodology through improved hyperparameters and by adding other regularization techniques, the benefits of Cutout diminished to the point of becoming negligible.

> 🚧 Cutout Provided Limited Benefits in Our Experiments
>
> In our experiments on ResNets for CIFAR-10 and ImageNet, Cutout provided little or no improvements in accuracy when the models were well-tuned and when we combined it with other regularization methods.
> It is possible that Cutout may still be helpful for other models and tasks and in settings that are less well-tuned.

Because Cutout is a regularizer, it may improve or degrade accuracy, depending on the setting.
Regularization methods reduce overfitting, potentially allowing models to reach higher quality.
However this typically requires (1) larger models with more capacity to perform this more difficult learning and (2) longer training runs to allow these models time to learn.

> 🚧 Composing Regularization Methods
>
> As general rule, composing regularization methods may lead to diminishing returns in quality improvements. Cutout is one such regularization method. We do not see improvements when combining Cutout with other regularization and augmentation methods such as Mixup and Label Smoothing.

Data augmentation techniques can sometimes put additional load on the CPU, potentially to the point where the CPU becomes a bottleneck for training.
To prevent this from happening for Cutout, our implementation of Cutout (1) takes place on the GPU and (2) occludes the same patch on each image in a minibatch.
Doing so avoids putting additional work on the CPU (since augmentation occurs on the GPU) and avoids putting additional work on the GPU (since all images are handled uniformly within a batch).

> ❗ Cutout Increases Memory Requirements
>
> Since Cutout runs on GPU by default and uses some extra memory to construct the mask, out of memory errors may occur if GPU memory is severely limited.

Since Cutout masks a portion of the input, this can alter the inherent shape/texture bias of the model. For an example, see [Hermann et al. (2020)](https://arxiv.org/abs/1911.09071).

Although our implementation of Cutout is designed for computer vision tasks, variants of Cutout have been shown to be useful in other settings, for example, audio processing ([Cances et al., 2021](https://arxiv.org/abs/2102.08183)).
The implementation in Composer currently only supports computer vision.


## Attribution

[*Improved Regularization of Convolutional Neural Networks with Cutout*](https://arxiv.org/abs/1708.04552) by Terrance DeVries and Graham W. Taylor. Posted to arXiv in 2017.

*This Composer implementation of this method and the accompanying documentation were produced by Cory Stephenson at MosaicML.*
