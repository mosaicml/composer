# Sharpness Aware Minimization

![loss_landscape.png](https://storage.googleapis.com/docs.mosaicml.com/images/methods/sam.png)

Example loss landscapes at convergence for ResNet models trained without SAM (left) and with SAM (right). From Foret et al., 2021.

Applicable Settings: `Vision`, `Decreased GPU Throughput`, `Increased Accuracy`, `Method`, `Regularization`

## TL;DR

Sharpness-Aware Minimization (SAM) is an optimization algorithm that minimizes both the loss and the sharpness of the loss. It finds parameters that lie in a *neighborhood* of low loss. The authors find that this improves model generalization.

## Attribution

*[Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/abs/2010.01412)* by Pierre Foret, Ariel Kleiner, Hossein Mobahi, and Behnam Neyshabur. Published as a conference paper at ICLR 2021.

## Code and Hyperparameters

- `rho` - The neighborhood size parameter of SAM. Must be greater than 0.
- `epsilon` - A small value added to the gradient norm for numerical stability.
- `interval` - SAM will run once per `interval` steps. A value of 1 will cause SAM to run every step. Steps on which SAM runs take roughly twice as much time to complete.

## Applicable Settings

Foret et al. introduced SAM on CNNs for image classification tasks. These results have been replicated and/or extended by [Brock et al., (2021)](https://arxiv.org/abs/2102.06171) and MosaicML, and extended to vision transformers by [Chen et al., (2021)](https://arxiv.org/abs/2106.01548). As a generic optimization algorithm, SAM should be applicable across a models and tasks.

## Example Effects

Foret et al. report a variety of accuracy improvements over baseline when SAM is added to models of the ResNet family trained on ImageNet. These improvements range from 0.4 percentage point (pp) for ResNet-50 trained for 100 epochs, to 1.9pp on ResNet-152 trained for 400 epochs. Notably, the authors report that SAM seems to prevent overfitting at long training durations. The authors do not explicitly measure their implementation's impact on throughput, but we estimate that it roughly halves training throughput.

In our experiments we used a value of 10 for our introduced hyperparameter `interval` to limit SAM's impact on throughput. We observe a 0.7pp accuracy improvement over baseline when our modified implementation of SAM is added to a ResNet-50 model trained for 90 epochs on ImageNet. Throughput is decreased from baseline by around 7%.

## Implementation Details

SAM finds neighborhoods of low loss by computing a gradient twice for each optimization step. The first gradient is computed ordinarily for the existing model parameters, but the second gradient is computed by adjusting the model parameters according to the first gradient computed. The final result of the optimization step is the original parameters, adjusted according to the second gradients computed.

Our SAM algorithm largely matches the description provided by Foret et al., with the exception of an introduced hyperparameter `interval`. We use this hyperparameter to limit SAM's negative impacts on throughput by only running the SAM algorithm once per `interval` optimization steps. On other steps, the base optimizer runs normally.

Our implementation is largely inspired by a previous open-source implementation by David Samuel available on [Github](https://github.com/davda54/sam). We implement SAM as a Pytorch optimizer that itself wraps another optimizer, such as SGD, and we make use of closures to enable the SAM optimizer to compute multiple gradients per optimization step. Notably, optimizers that use closures are usually incompatible with mixed precision training due to limitations in the Pytorch gradient scaling API. Our trainer uses a custom implementation of gradient scaling that is able to support this combination.

We also include a small optimization for data-parallel training such that the first set of gradients computed by the SAM optimizer is not synchronized across devices. This optimization means that the observed impact of SAM on throughput is slightly less than what might be expected for various settings of `interval`. We find that this optimization appears to even slightly improve quality, like by the introduction of stochasticity.

## Suggested Hyperparameters

As per Foret et al., we find that `rho=0.05` works well when `interval=1`, producing a configuration equivalent to that in the original paper. We also find that when adjusting `interval`, it is best to adjust `rho` proportionally. For our experiments using `interval=10`, we also used `rho=0.5`. We believe that this setting of `interval` provides a practical tradeoff between throughput and accuracy.

We do not expect that `epsilon` will commonly need to be adjusted.

## Considerations

The SAM algorithm decreases throughput substantially. The `interval` hyperparameter effectively serves as a knob to control the tradeoff between quality and speed. Lower values of `interval` run SAM with higher frequency, imposing a higher computational burden but improving model quality, while higher values of `interval` run SAM less frequently, leading to lower computational costs but also lower quality improvements.

We also find that the quality improvements of SAM are more easily realized in highly data-parallel workflows; this appears to be a result of our implementation's behavior to not synchronize the first set of gradients computed across devices.

## Composability

The benefit of SAM derives at least in part from its regularizing effects, and so it is expected to yield diminishing returns when composed with other regularization methods such as MixUp and label smoothing. That said, Foret et al. do observe that SAM appears to meaningfully improve generalization even on models considered to already include well tuned regularization schemes. Further understanding of SAM's composability is left to future experimentation.

---

## Code

```{eval-rst}
.. autoclass:: composer.algorithms.sam.SAM
    :members: match, apply
    :noindex:

.. autoclass:: composer.algorithms.sam.SAMOptimizer
    :noindex:
```