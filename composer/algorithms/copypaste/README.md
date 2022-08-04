# ✂️ CopyPaste

[\[How to Use\]](#how-to-use) - [\[Suggested Hyperparameters\]](#suggested-hyperparameters) - [\[Technical Details\]](#technical-details) - [\[Attribution\]](#attribution)

`Computer Vision`

CopyPaste is a data augmentation technique that randomly pastes object instances onto an image. A pair of source and target samples are randomly chosen from a batch of data and a set of randomly chosen (without replacement) instances are extracted from the source sample. The selected instances are then transformed and pasted into the target sample.
This augmentation method can serve as a regularization technique to enhance the generalization capability of segmentation models for computer vision.


| ![CopyPaste](https://storage.googleapis.com/docs.mosaicml.com/images/methods/cutmix.png) |
|:--:
|*An example of data augmentation using CopyPaste. Object instances are randomly chosen from a source sample, jittered, and pasted into a target instance.*|
## How to Use

### Functional Interface

Here we run `CopyPaste` using index labels and interpolating the loss (a trick when using cross entropy).
```python
# Run the CutMix algorithm directly on the batch data using the Composer functional API
import torch
import torch.nn.functional as F
import composer.functional as cf

def training_loop(model, train_loader):
    opt = torch.optim.Adam(model.parameters())
    loss_fn_ce = F.cross_entropy
    loss_fn_mse = F.mse_loss 

    model.train()

    for epoch in range(num_epochs):
        for X, y in train_loader:
            X_augmented, y_perm = cf.copypaste_batch(X, y)
            y_hat = model(X_augmented)
            loss = area * loss_fn(y_hat, y) + (1 - area) * loss_fn(y_hat, y_perm) + loss_fn_mse(y_hat, y_perm)
            loss.backward()
            opt.step()
            opt.zero_grad()
```

### Composer Trainer

```python
# Instantiate the algorithm and pass it into the Trainer
# The trainer will automatically run it at the appropriate points in the training loop

from composer.algorithms import CopyPaste
from composer.trainer import Trainer

copypaste = CopyPaste()

trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    max_duration='1ep',
    algorithms=[copypaste]
)

trainer.fit()
```

### Implementation Details

Our implementation of CopyPaste augmentation is aligned with the [CVPR 2021 paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Ghiasi_Simple_Copy-Paste_Is_a_Strong_Data_Augmentation_Method_for_Instance_CVPR_2021_paper.pdf) by Ghiasi et al. as we choose the strategy of randomly choosing instances from a source sample and pasting them into random locations of the target sample.

## Suggested Hyperparameters

Setting

## Technical Details

CopyPaste works by creating a

## Attribution

[_Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation_](https://openaccess.thecvf.com/content/CVPR2021/papers/Ghiasi_Simple_Copy-Paste_Is_a_Strong_Data_Augmentation_Method_for_Instance_CVPR_2021_paper.pdf) by Golnaz Ghiasi, Yin Cui, Aravind Srinivas, Rui Qian, Tsung-Yi Lin, Ekin D. Cubuk, Quoc V. Le, Barret Zoph. Published in CVPR 2021.

*This Composer implementation of this method and the accompanying documentation were produced by the Vision team at MosaicML.*



Todo: in Composer Trainer, what dataset should I import from tests.common.