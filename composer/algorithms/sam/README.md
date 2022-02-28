# 🏔️ Sharpness Aware Minimization (SAM)

[\[How to Use\]](#how-to-use) - [\[Suggested Hyperparameters\]](#suggested-hyperparameters) - [\[Technical Details\]](#technical-details) - [\[Attribution\]](#attribution)

`Computer Vision`

Sharpness-Aware Minimization (SAM) is an optimization algorithm that minimizes both the loss and the sharpness of the loss. It finds parameters that lie in a _neighborhood_ of low loss. The authors find that this improves model generalization, and we do as well.

| ![SharpnessAwareMinimization](https://storage.googleapis.com/docs.mosaicml.com/images/methods/sam.png) |
|:--|
| *Three-dimensional visualizations of the loss landscaopes of ResNet models trained without SAM (left) and with SAM (right). The loss landscape with SAM appears smoother and the minimum found by the network appears wider and flatter. Note that these are visualizations of high-dimensional landscapes, so the fidelity is not perfect. This image is from Figure 1 in [Foret et al., 2021](https://arxiv.org/abs/2010.01412).* |

## How to Use

<!--### Functional Interface

TODO(Abhi): Fix and add comments here describing what happens below.

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

TODO(Abhi): Fix and add comments here describing what happens below.

```python
from composer.algorithms import xxx
from composer.trainer import Trainer

trainer = Trainer(model=model,
                  train_dataloader=train_dataloader,
                  max_duration='1ep',
                  algorithms=[
                  ])

trainer.fit()
```-->

### Implementation Details

Our implementation is largely inspired by a previous [open-source implementation by David Samuel available on Github](https://github.com/davda54/sam). We implement SAM as a PyTorch optimizer that itself wraps another optimizer, such as SGD, and we make use of closures to enable the SAM optimizer to compute multiple gradients per optimization step. Notably, optimizers that use closures are usually incompatible with mixed precision training due to limitations in the PyTorch gradient scaling API. Our trainer uses a custom implementation of gradient scaling that is able to support this combination.

We also include a small optimization for data-parallel training such that the first set of gradients computed by the SAM optimizer is not synchronized across devices. This optimization means that the observed impact of SAM on throughput is slightly less than what might be expected for various settings of interval. We find that this optimization appears to even slightly improve quality, like by the introduction of stochasticity.

## Suggested Hyperparameters

As per [Foret et al.](https://arxiv.org/abs/2010.01412), we find that `rho=0.05` (`rho` controls the neighborhood size that SAM considers) works well when `interval=1` (i.e., running SAM on every step of training), producing a configuration equivalent to that in the original paper. We also find that when adjusting interval, it is best to adjust `rho` proportionally. For our experiments using `interval=10` (i.e., running SAM every ten iterations), we also used `rho=0.5`. We believe that this setting of interval provides a practical tradeoff between throughput and accuracy. We do not expect that `epsilon` will commonly need to be adjusted.

## Technical Details

SAM finds neighborhoods of low loss by computing a gradient twice for each optimization step. The first gradient is computed ordinarily for the existing model parameters, but the second gradient is computed by adjusting the model parameters according to the first gradient computed. The final result of the optimization step is the original parameters, adjusted according to the second gradients computed.

Our SAM algorithm largely matches the description provided by Foret et al., with the exception of an introduced hyperparameter `interval`. We use this hyperparameter to limit SAM’s negative impacts on throughput by only running the SAM algorithm once per `interval` optimization steps. On other steps, the base optimizer runs normally. Each step of SAM requires twice as much work as performing a standard gradient step.
 The `interval` hyperparameter effectively serves as a knob to control the tradeoff between quality and speed. Lower values of `interval` run SAM with higher frequency, imposing a higher computational burden but improving model quality, while higher values of `interval` run SAM less frequently, amortizing the cost of SAM across many steps but also lowering quality improvements.

> ❗ SAM Causes Lower Throughput, Although It Can Be Mitigated
> 
> Since SAM must compute the gradient twice, it takes about twice as much work and halves GPU throughput compared to taking a standard gradient step. One way of mitigating this throughput reduction is to perform SAM periodically rather than on every step. The `interval` hyperparameter controls how often SAM runs, making it possible to trade off between SAM's quality improvements and throughput reductions.

Foret et al. report a variety of accuracy improvements over baseline when SAM is added to models of the ResNet family trained on ImageNet. These improvements range from 0.4% for ResNet-50 trained for 100 epochs, to 1.9% on ResNet-152 trained for 400 epochs. Notably, the authors report that SAM seems to prevent overfitting at long training durations. The authors do not explicitly measure their implementation’s impact on throughput, but we estimate that it roughly halves training throughput.

In our experiments we used a value of `interval=10` for our introduced hyperparameter `interval` to limit SAM’s impact on throughput. We observe a 0.7% accuracy improvement over baseline when our modified implementation of SAM is added to a ResNet-50 model trained for 90 epochs on ImageNet. Throughput is decreased from baseline by around 7%.

> ✅ SAM Improves the Tradeoff Between Quality and Training Speed
> 
> In our experiments, SAM improves the attainable tradeoffs between training speed and the final quality of the trained model.
> Although it causes a reduction in training throughput, the `interval` hyperparameter can be used to minimize the extent of this reduction.
> The corresponding accuracy increases were a worthwhile tradeoff for the throughput reduction in our experiments on ResNets on ImageNet.

Foret et al. introduced SAM on CNNs for image classification tasks.
These results have been replicated and/or extended by [Brock et al., (2021)](https://arxiv.org/abs/2102.06171) and MosaicML, and extended to vision transformers by [Chen et al., (2021)](https://arxiv.org/abs/2106.01548). As a generic optimization algorithm, SAM should be applicable across a models and tasks.
We did not find SAM to provide any improvements on GPT-style language modeling tasks that only see each training example once (i.e., those that have effectively infinite data). This may imply that SAM improves accuracy by mitigating overfitting when the model sees the same examples multiple times.

## Attribution

[*Sharpness-Aware Minimization for Efficiently Improving Generalization*](https://arxiv.org/abs/2010.01412) by Pierre Foret, Ariel Kleiner, Hossein Mobahi, and Behnam Neyshabur. Published in ICLR 2021.

*The Composer implementation of this method and the accompanying documentation were produced by Abhi Venigalla at MosaicML.*
