# Label Smoothing

Applicable Settings: `Classification`

Effects: `Increased Accuracy`

Kind: `Method`

Tags: `Regularization`

## TLDR

Label smoothing modifies the target distribution for a task by interpolating between the target distribution and a another distribution that usually has higher entropy. This typically reduces a model's confidence in its outputs and serves as a form of regularization.

## Attribution

The technique was originally introduced in *[Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)* by Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathan Shlens, and Zbigniew Wojna. Released on arXiv in 2015.

The technique was further evaluated in *[When Does Label Smoothing Help?](https://arxiv.org/abs/1906.02629)* by Rafael Muller, Simon Kornblith, and Geoffrey Hinton. Published in NeurIPS 2015.

## Applicable Settings

Label smoothing is applicable to any problem where targets are a categorical distribution. This includes classification with softmax cross-entropy and segmentation with a Dice loss.

## Hyperparameters

- `alpha` - A value between 0.0 and 1.0 that specifies the interpolation between the target distribution and a uniform distribution. For example. a value of 0.9 specifies that the target values should be multiplied by 0.9 and added to a uniform distribution multiplied by 0.1.

## Example Effects

Label smoothing is intended to act as regularization, and so possible effects are changes (ideally improvement) in generalization performance. We find this to be the case on all of our image classification benchmarks, which see improved accuracy under label smoothing.

## Implementation Details

Label smoothing replaces the one-hot encoded label with a combination of the true label and the uniform distribution. Care must be taken in ensuring the cost function used can accept the full categorical distribution instead of the index of the target value.

## Suggested Hyperparameters

`alpha = 0.1` is a standard starting point for label smoothing.

## Considerations

In some cases, a small amount of extra memory and compute is needed to convert labels to dense targets. This can produce a (typically negligible) increase in iteration time.

## Composability

This method interacts with other methods (such as MixUp) that alter the targets. While such methods may still compose well with label smoothing in terms of improved accuracy, it is important to ensure that the implementations of these methods compose.


---

## Code

```{eval-rst}
.. autoclass:: composer.algorithms.label_smoothing.LabelSmoothing
    :members: match, apply

.. autoclass:: composer.algorithms.label_smoothing.smooth_labels
```