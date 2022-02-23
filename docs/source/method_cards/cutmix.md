# ✂️ CutMix

![cutmix.png](https://storage.googleapis.com/docs.mosaicml.com/images/methods/cutmix.png)

Image from [CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](https://arxiv.org/abs/1905.04899) by Yun et al., 2019

Tags: `Vision`, `Increased Accuracy`, `Increased GPU Usage`, `Method`, `Augmentation`, `Regularization`

## TL;DR
CutMix trains the network on images from which a small patch has been cut out and replaced with a different image. Training in this fashion improves generalization performance.

## Attribution

*[CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](https://arxiv.org/abs/1905.04899)* by Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo. Published in ICCV 2019.

## Hyperparameters

- `alpha` - The parameter that controls the distribution that the area of the cut out region is drawn from when performing CutMix. This is a symmetric Beta distribution, meaning that `alpha` serves as both parameters for the Beta distribution. The actual area of the cut out region may differ from the sampled value, if the selected region is not entirely within the image.

## Example Effects

CutMix is intended to improve generalization performance, and we empirically find this to be the case in our image classification settings. The [original paper](https://arxiv.org/abs/1905.04899) also reports improvements in object localization and robustness.

## Implementation Details

The samples are created from a batch `(X, y)` of (inputs, targets) together with version `(X', y')` where the ordering of examples has been shuffled. The examples are combined by sampling a value `lambda` (between 0.0 and 1.0) from the Beta distribution parameterized by `alpha`, choosing a rectangular box within `X`, filling it with the data from the corresponding region in `X`. and training the network on the interpolation between `(X, y)` and `(X', y')`.

Note that the same `lambda` and rectangular box are used for each example in the batch. Similar to MixUp, using the shuffled version of a batch to generate mixed samples allows CutMix to be used without loading additional data.

## Suggested Hyperparameters

- `alpha = 1` Is a common choice.

## Considerations

- CutMix adds a little extra GPU compute and memory to create samples.

- CutMix also requires a cost function that can accept dense target vectors, rather than an index of a corresponding 1-hot vector as is a common default (e.g., cross entropy with hard labels).

## Composability

As general rule, combining regularization-based methods yields sublinear improvements to accuracy. This holds true for CutMix.

This method interacts with other methods (such as CutOut) that alter the inputs or the targets (such as label smoothing). While such methods often still compose well with CutMix in terms of improved accuracy, it is important to ensure that the implementations of these methods compose.

---

## Code

```{eval-rst}
.. autoclass:: composer.algorithms.cutmix.CutMix
    :members: match, apply
    :noindex:

.. autofunction:: composer.algorithms.cutmix.cutmix_batch
  :noindex:

```
