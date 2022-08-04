# ✂️ CopyPaste

[\[How to Use\]](#how-to-use) - [\[Suggested Hyperparameters\]](#suggested-hyperparameters) - [\[Technical Details\]](#technical-details) - [\[Attribution\]](#attribution)

`Computer Vision`

CopyPaste is a data augmentation technique that randomly pastes objects onto an image.

## How to Use

### Functional Interface

Here we run `CopyPaste` using index labels and interpolating the loss (a trick when using cross entropy).

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

CopyPaste is implemented following the

## Suggested Hyperparameters

Setting

## Technical Details

CopyPaste works by creating a

## Attribution

[_Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation_](https://openaccess.thecvf.com/content/CVPR2021/papers/Ghiasi_Simple_Copy-Paste_Is_a_Strong_Data_Augmentation_Method_for_Instance_CVPR_2021_paper.pdf) by Golnaz Ghiasi, Yin Cui, Aravind Srinivas, Rui Qian, Tsung-Yi Lin, Ekin D. Cubuk, Quoc V. Le, Barret Zoph. Published in CVPR 2021.

_This Composer implementation of this method and the accompanying documentation were produced by the Vision team at MosaicML._
