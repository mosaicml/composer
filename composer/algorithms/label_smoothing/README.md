# 🧈 Label Smoothing

[\[How to Use\]](#how-to-use) - [\[Suggested Hyperparameters\]](#suggested-hyperparameters) - [\[Technical Details\]](#technical-details) - [\[Attribution\]](#attribution)

`Computer Vision`

Label smoothing modifies the target distribution for a task by interpolating between the target distribution and a another distribution that usually has higher entropy (e.g., the uniform distribution). This typically reduces a model’s confidence in its outputs and serves as a form of regularization.

<!--| ![LabelSmoothing](label_smoothing.png) |
|:--:
|*Need a picture.*|-->

## How to Use

### Functional Interface

```python
# Run the Label Smoothing algorithm directly on the targets using the Composer functional API

import composer.functional as cf

def training_loop(model, train_loader):
    opt = torch.optim.Adam(model.parameters())
    loss_fn = F.cross_entropy
    model.train()

    for epoch in range(num_epochs):
        for X, y in train_loader:
            y_hat = model(X)

            # note that if you were to modify the variable y here it is a good
            # idea to set y back to the original targets after computing the loss
            smoothed_targets = cf.smooth_labels(y_hat, y, smoothing=0.1)

            loss = loss_fn(y_hat, smoothed_targets)
            loss.backward()
            opt.step()
            opt.zero_grad()
```

### Composer Trainer

```python
# Instantiate the algorithm and pass it into the Trainer
# The trainer will automatically run it at the appropriate points in the training loop

from composer.algorithms import LabelSmoothing
from composer.trainer import Trainer

label_smoothing = LabelSmoothing(smoothing=0.1)

trainer = Trainer(model=model,
                  train_dataloader=train_dataloader,
                  max_duration='1ep',
                  algorithms=[label_smoothing])

trainer.fit()
```

### Implementation Details


`LabelSmoothing` converts targets to a one-hot representation if they are not in that format already and then applies a convex combination with a uniform distribution over the labels using the following formula:

`smoothed_labels = (targets * (1. - smoothing)) + (smoothing / n_classes)`

The functional form of the algorithm (accessible with the function `smooth_labels` in `composer.functional`), simply computes smoothed labels and returns them.

The class form of the algorithm also takes care of setting the targets back to the original (pre-smoothing) targets after the loss is computed so that any calculations done with the targets after computing the loss (ex. training metrics) will use the original targets and not the smoothed targets.

## Suggested Hyperparameters

The only hyperparameter for Label Smoothing is `smoothing`, a value between 0.0 and 1.0 that specifies the interpolation between the target distribution and a uniform distribution. For example. a value of 0.9 specifies that the target values should be multiplied by 0.9 and added to a uniform distribution multiplied by 0.1.
`smoothing=0.1` is a standard starting point for label smoothing.

## Technical Details

Label smoothing replaces the one-hot encoded label with a combination of the true label and the uniform distribution.

> ❗ Label Smoothing Produces a Full Distribution, Not a Target Index
>
> Many classification tasks represent the target value using the index of the target value rather than the full one-hot encoding of the label value.
> Label smoothing turns each label into a dense distribution (if it has not already been converted into a distribution).
> The loss function used for the model must be able to accept this dense distribution as the target.

> ❗ Label Smoothing May Interact with Other Methods that Modify Targets
>
> This method interacts with other methods (such as MixUp) that alter the targets.
> While such methods may still compose well with label smoothing in terms of improved accuracy, it is important to ensure that the implementations of these methods compose.

Label smoothing is intended to act as regularization, and so possible effects are changes (ideally improvement) in generalization performance. We find this to be the case on all of our image classification benchmarks, which see improved accuracy under label smoothing.

We did not observe label smoothing to affect throughput in any way, although it may require a small amount of extra memory and compute to convert label indices into dense targets.

## Attribution

[Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) by Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathan Shlens, and Zbigniew Wojna. Posted to arXiv in 2015.

*This Composer implementation of this method and the accompanying documentation were produced by Cory Stephenson at MosaicML.*
