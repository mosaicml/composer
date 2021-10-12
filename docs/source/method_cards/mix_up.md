# MixUp

Tags: `Vision`,`Method`,`Augmentation`, `Regularization`,`Increased Accuracy`, `Increased GPU Usage`

### tl;dr

MixUp trains the network on convex combinations of examples and targets rather than individual examples and targets. Training in this fashion improves generalization performance.

![mix_up.png](https://storage.cloud.google.com/docs.mosaicml.com/images/methods/mix_up.png)

From [AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty](https://arxiv.org/abs/1912.02781)

### Attribution

*[mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)* by Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz. Published in ICLR 2018.

### Applicable Settings

MixUp can be applied to many different tasks. Our implementation is designed for classification tasks with continuous inputs.

### Hyperparameters

- `alpha` - The parameter that controls the distribution of interpolation values sampled when performing MixUp. Our implementation samples these interpolation values from a symmetric Beta distribution, meaning that `alpha` serves as both parameters for the Beta distribution.

### Example Effects

MixUp is intended to improve generalization performance, and we empirically find this to be the case in our image classification settings. The [original paper](https://arxiv.org/abs/1710.09412) also reports a reduction in memorization and improved adversarial robustness.

### Implementation Details

Mixed samples are created from a batch `(X, y)` of (inputs, targets) together with version `(X', y')` where the ordering of examples has been shuffled. The examples can be mixed by sampling a value `t` (between 0.0 and 1.0) from the Beta distribution parameterized by `alpha` and training the network on the interpolation between `(X, y)` and `(X', y')` specified by `t`

Note that the same `t` is used for each example in the batch. Using the shuffled version of a \batch to generate mixed samples allows MixUp to be used without loading additional data.

### Suggested Hyperparameters

`alpha = 0.2` is a good default for training on ImageNet.

`alpha = 1` works for CIFAR10

### Considerations

MixUp adds a little extra GPU compute and memory to create the mixed samples.

MixUp also requires a cost function that can accept dense target vectors, rather than an index of a corresponding 1-hot vector as is a common default (e.g., cross entropy with hard labels).

### Composability

As general rule, combining regularization-based methods yields sublinear improvements to accuracy. This holds true for MixUp.

This method interacts with other methods (such as CutOut) that alter the inputs. While such methods may still compose well with MixUp in terms of improved accuracy, it is important to ensure that the implementations of these methods compose.

This method interacts with other methods (such as label smoothing) that alter the targets. While such methods may still compose well with MixUp in terms of improved accuracy, it is important to ensure that the implementations of these methods compose.

---

### Code

```{eval-rst}
.. autoclass:: composer.algorithms.mixup.MixUp
    :members: match, apply

.. autofunction:: composer.algorithms.mixup.mixup.gen_interpolation_lambda

.. autofunction:: composer.algorithms.mixup.mixup.mixup_batch
```