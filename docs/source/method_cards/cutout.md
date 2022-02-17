# Cutout

![cut_out.png](https://storage.googleapis.com/docs.mosaicml.com/images/methods/cut_out.png)

From *[Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/abs/1708.04552)* by DeVries and Taylor, 2017.

Tags: `Vision`, `Increased CPU Usage`, `Increased Accuracy`, `Method`, `Augmentation`, `Regularization`

## TL;DR

Cutout is a regularization/data augmentation technique that works by masking out one or more square regions of an input image.

## Attribution

*[Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/abs/1708.04552)* by Terrance DeVries and Graham W. Taylor. Posted to arXiv in 2017.

## Applicable Settings

Cutout is a data augmentation technique for images, and hence it is only applicable to vision tasks. Though our implementation does not yet support other modalities, similar methods may prove fruitful (*[Improving Deep-learning-based Semi-supervised Audio Tagging with Mixup](https://arxiv.org/abs/2102.08183)*).

## Hyperparameters

- `n_holes` - The number of patches of the image to remove.
- `length` - The side length of each square patch that is removed.

## Example Effects

Cutout incurs a small extra computational cost. Because it is a regularizer it may improve or degrade accuracy, depending on the setting. We found it to be an effective way of improving accuracy in the absence of robust hyperparameter tuning and other regularizers. As we improved our training methodology elsewhere, the benefits from Cutout became negligible.

## Implementation Details

Cutout works by randomly choosing one or more square regions from an input image and replacing them with the mean value over the dataset. Since it is common to normalize image data based on the dataset mean and variance, the mean value is typically 0. To ease implementation, we went with a simple binary mask. Cutout is most efficiently applied to a batch of images, so each image in the batch has the same regions modified. This lets us easily run the augmentation on the GPU, if one is available.

Our implementation is based on that of Terrance DeVries as [posted on GitHub](https://github.com/uoguelph-mlrg/Cutout).

## Suggested Hyperparameters

- `n_holes = 1` Typically removing a single patch gives good results.
- `length = int(0.5 * image_size)` Typically a square with a side length of half the image size produces good results. However, in some scenarios this may be too large, in which case `length = int(0.25 * image_size)` might be better.

## Considerations

As Cutout runs on GPU by default and uses some extra memory to construct the mask, out of memory errors may occur if GPU memory is severely limited.

Also, since Cutout masks a portion of the input, this can alter the inherent shape/texture bias. For an example, see [The Origins and Prevalence of Texture Bias in Convolutional Neural Networks](https://arxiv.org/abs/1911.09071).

## Composability

As general rule, combining regularization-based methods yields sublinear improvements to accuracy. This also includes Cutout. For example, we do not see improvements when combined with other regularization/augmentation methods such as Mixup and Label Smoothing.

## Code

```{eval-rst}
.. autoclass:: composer.algorithms.cutout.CutOut
    :members: match, apply
    :noindex:

.. autofunction:: composer.algorithms.cutout.cutout_batch
    :noindex:
```
