# 🥣 MixUp


[\[How to Use\]](#how-to-use) - [\[Suggested Hyperparameters\]](#suggested-hyperparameters) - [\[Technical Details\]](#technical-details) - [\[Attribution\]](#attribution)

`Computer Vision`

MixUp synthesizes new training examples by combining pairs of individual examples.
For any pair of examples, it trains the network on a random convex combination of the individual examples (i.e., a random interpolation between the two examples).
To create the corresponding targets, it uses the same random convex combination of the targets of the individual examples.
Training in this fashion improves generalization.

| ![MixUp](https://storage.googleapis.com/docs.mosaicml.com/images/methods/mix_up.png) |
|:--:
|*Two different training examples (a picture of a bird and a picture of a frog) that have been combined by MixUp into a single example. The corresponding targets are a convex combination of the targets for the bird class and the frog class.*|

<!--## How to Use

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

TODO(CORY): Briefly describe how this is implemented under the hood in Composer.-->

## Suggested Hyperparameters

The sole hyperparameter for MixUp is `alpha`, which controls the Beta distribution that determines the extent of interpolation between the two input examples.
Concretely, for two examples `x1` and `x2`, the example that MixUp produces is `t * x1 + (1 - t) * x2` where `a` is sampled from the distribution `Beta(alpha, alpha)`.
On ImageNet, we found that `alpha=0.2` (which places slightly higher probability on extreme values of `t`) works well.
On CIFAR-10, we found that `alpha=0.1` (which samples `t` uniformly between 0 and 1) works well.

## Technical Details

Mixed samples are created from a batch `(X1, y1)` of (inputs, targets) together with version `(X2, y2)` where the ordering of examples has been shuffled. The examples can be mixed by sampling a value `t` (between 0.0 and 1.0) from the Beta distribution parameterized by `alpha` and training the network on the interpolation between `(X1, y1)` and `(X2, y2)` specified by `t`. That is, the batch of examples produced by MixUp is `t * X1 + (1 - t) * X2` and the batch of targets produced by MixUp is `t * y1 + (1 - t) * y2`.

We set `(X2, y2)` to be a shuffling of the original batch `(X1, y1)` (rather than sampling an entirely new batch) to allow MixUp to proceed without loading additional data.
This choice avoids putting additional strain on the dataloader.


Data augmentation techniques can sometimes put additional load on the CPU, potentially reaching the point where the CPU becomes a bottleneck for training.
To prevent this from happening for MixUp, our implementation of MixUp (1) occurs on the GPU and (2) uses the same `t` for all examples in the minibatch.
Doing so avoids putting additional work on the CPU (since augmentation occurs on the GPU) and minimizes additional work on the GPU (since all images are handled uniformly within a batch).

> 🚧 MixUp Requires a Small Amount of Additional GPU Compute and Memory
>
> MixUp requires a small amount of additional GPU compute and memory to produce the mixed-up batch.
> In our experiments, we have found these additional resource requirements to be negligible.

> ❗ MixUp Produces a Full Distribution, Not a Target Index
>
> Many classification tasks represent the target value using the index of the target value rather than the full one-hot encoding of the label value.
> Since MixUp interpolates between two target values for each example, it must represent the final targets as a dense distribution.
> Our implementation of MixUp turns each label into a dense distribution (if it has not already been converted into a distribution).
> The loss function used for the model must be able to accept this dense distribution as the target.

> 🚧 Composing Regularization Methods
>
> As general rule, composing regularization methods may lead to diminishing returns in quality improvements. MixUp is one such regularization method.

Although our implementation of MixUp is designed for computer vision tasks, variants of MixUp have been shown to be useful in other settings, for example natural language processing.
The implementation in Composer currently only supports computer vision.


## Attribution

[*mixup: Beyond Empirical Risk Minimization*](https://arxiv.org/abs/1710.09412) by Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz. Published in ICLR 2018.

*This Composer implementation of this method and the accompanying documentation were produced by Cory Stephenson at MosaicML.*
