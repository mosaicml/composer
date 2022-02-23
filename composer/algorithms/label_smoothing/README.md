# ♭ Label Smoothing

[\[How to Use\]](#how-to-use) - [\[Suggested Hyperparameters\]](#suggested-hyperparameters) - [\[Technical Details\]](#technical-details) - [\[Attribution\]](#attribution)

`Computer Vision`

Label smoothing modifies the target distribution for a task by interpolating between the target distribution and a another distribution that usually has higher entropy (e.g., the uniform distribution). This typically reduces a model’s confidence in its outputs and serves as a form of regularization.

| ![LabelSmoothing](label_smoothing.png) |
|:--:
|*Need a picture.*|

## How to Use

### Functional Interface

TODO(CORY): FIX

```python
def training_loop(model, train_loader):
  opt = torch.optim.Adam(model.parameters())
  loss_fn = F.cross_entropy
  model.train()
  
  for epoch in range(num_epochs):
      for X, y in train_loader:
          y_hat = model(X)
          loss = loss_fn(y_hat, y)
          loss.backward()
          opt.step()
          opt.zero_grad()
```

### Composer Trainer

TODO(CORY): Fix and provide commentary and/or comments

```python
from composer.algorithms import XXX
from composer.trainer import Trainer

trainer = Trainer(model=model,
                  train_dataloader=train_dataloader,
                  max_duration='1ep',
                  algorithms=[
                  ])

trainer.fit()
```

### Implementation Details

TODO(CORY): Briefly describe how this is implemented under the hood in Composer.

## Suggested Hyperparameters

The only hyperparameter for Label Smoothing is `alpha`, a value between 0.0 and 1.0 that specifies the interpolation between the target distribution and a uniform distribution. For example. a value of 0.9 specifies that the target values should be multiplied by 0.9 and added to a uniform distribution multiplied by 0.1.
`alpha=0.1` is a standard starting point for label smoothing.

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
