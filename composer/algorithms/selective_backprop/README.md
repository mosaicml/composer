# ⏮️ Selective Backprop

[\[How to Use\]](#how-to-use) - [\[Suggested Hyperparameters\]](#suggested-hyperparameters) - [\[Technical Details\]](#technical-details) - [\[Attribution\]](#attribution)

`Computer Vision`

Selective Backprop prioritizes examples with high loss at each iteration, skipping backpropagation on examples with low loss.
This speeds up training with limited impact on generalization.

| ![SelectiveBackprop](https://storage.googleapis.com/docs.mosaicml.com/images/methods/selective-backprop.png) |
|:--|
|*Four examples are forward propagated through the network. Selective backprop only backpropagates the two examples that have the highest loss.*|

## How to Use

### Functional Interface

TODO(ABHI): Fix and comments here describing what happens below.


```python
import composer.functional as cf

def training_loop(model, train_loader):
    opt = torch.optim.Adam(model.parameters())
    loss_fn = F.cross_entropy
    model.train()

    for epoch in range(num_epochs):
        for X, y in train_loader:
            y_hat = model(X)
            loss = loss_fn(y_hat, smoothed_targets)
            loss.backward()
            opt.step()
            opt.zero_grad()
```

### Composer Trainer

TODO(Abhi): Fix and add comments here describing what happens below.

```python
from composer.algorithms import LabelSmoothing
from composer.trainer import Trainer

trainer = Trainer(model=model,
                  train_dataloader=train_dataloader,
                  max_duration='1ep',
                  algorithms=[])

trainer.fit()
```

### Implementation Details

TODO(ABHI): Briefly describe what happens under the hood here.

## Suggested Hyperparameters

As per [Cubuk et al. (2020)](https://openaccess.thecvf.com/content_CVPRW_2020/html/w40/Cubuk_Randaugment_Practical_Automated_Data_Augmentation_With_a_Reduced_Search_Space_CVPRW_2020_paper.html), we found that `depth=2` (applying a chain of two augmentations to each image) and `severity=9` (each augmentation is applied quite strongly) worked well for different models of the ResNet family on ImageNet. For `depth≥3`, we found diminishing accuracy gains (due to over-regularization) and substantial training slowdown (due to the CPU becoming a bottleneck because of the amount of augmentation it must perform). We also recommend `augmentation_set=all` (using all available augmentation techniques).

> ❗ Potential CPU Bottleneck
> 
> Further increasing `depth` beyond 2 significantly decreased throughput when training ResNet-50 on ImageNet due to bottlenecks in performing data augmentation on the CPU.

## Technical Details

The goal of Selective Backprop is to reduce the number of examples the model sees to only those that still have high loss. This lets the model learn on fewer examples, speeding up forward and back propagation with limited impact on final model quality. To determine the per-example loss and which examples to skip, an additional, initial forward pass must be performed. These loss values are then used to weight a random sample of examples to use for training. For some data types, including images, it’s possible to use a lower resolution version of the input for this additional forward pass. This minimizes the extra computation while maintaining a good estimate of which examples are difficult for the model.

Depending on the precise hyperparameters chosen, we see decreases in training time of around 10% without any degradation in performance. Larger values are possible but run into the speed-accuracy tradeoffs described below.


## Attribution

[*Accelerating Deep Learning by Focusing on the Biggest Losers*](https://arxiv.org/abs/1910.00762) by Angela H. Jiang, Daniel L. K. Wong, Giulio Zhou, David G. Andersen, Jeffrey Dean, Gregory R. Ganger, Gauri Joshi, Michael Kaminsky, Michael Kozuch, Zachary C. Lipton, and Padmanabhan Pillai. Released on arXiv in 2019.

*The Composer implementation of this method and the accompanying documentation were produced by Abhi Venigalla at MosaicML.*
