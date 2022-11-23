# 🍰 Gyro Dropout


[\[How to Use\]](#how-to-use) - [\[Suggested Hyperparameters\]](#suggested-hyperparameters) - [\[Technical Details\]](#technical-details) - [\[Attribution\]](#attribution)

 `Computer Vision`

Gyro Dropout replaces implementations of `torch.nn.Dropout`. The Gyro Dropout provides increased accuracy compared with dropout.

Gyro dropout is a variant of dropout that improves the efficiency of training neural networks.
Instead of randomly dropping out neurons in every training iteration, gyro dropout pre-selects and trains a fixed
number of subnetworks. 'Sigma' is the number of total pre-selected subnetworks and 'Tau' is the number of concurrently scheduled subnetworks in an iteration.

## How to Use

### Functional Interface

```python
# Apply surgery on the model to swap-in the Fused LayerNorm using the Composer functional API

import composer.functional as cf

def training_loop(model, train_loader):
    cf.apply_gyro_dropout(
        model,
        iters_per_epoch = 196,
        max_epoch = 100,
        p = 0.5
        sigma = 256,
        tau = 16,
    )

    opt = torch.optim.Adam(model.parameters())
    loss_fn = F.cross_entropy
    model.train()

    for X, y in train_loader:
        y_hat = model(X)
        loss = loss_fn(y_hat, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
```

### Composer Trainer

```python
from composer.trainer import Trainer
from composer.algorithms import GyroDropout

trainer = Trainer(model=model,
                  train_dataloader=train_dataloader,
                  eval_dataloader=eval_dataloader,
                  max_duration='100ep',
                  algorithms=[GyroDropout(p=0.5,
                                          sigma=256,
                                          tau=16)])

trainer.fit()
```

### Implementation Details

Gyro Dropout is implemented by performing model surgery, which looks for instances of `torch.nn.Dropout`. This should be applicable to any model that utilizes `torch.nn.Dropout`.

## Suggested Hyperparameters

Gyro Dropout has two hyperparameters - sigma, tau. (iters_per_epoch and max_epoch is training-dependent)

For the hyperparameters (sigma, tau), we recommend (256, 16) in AlexNet, LeNet or (1024, 8) in ResNet-18, BERT.


## Technical Details
GyroDropout achieves improved accuracy over conventional dropout by pre-selecting a fixed number of subnetworks and training with only those subnetworks. Because the selected subnetworks are trained more robustly (compared to the conventional dropout), their diversity increases and thus their ensemble achieves higher accuracy.

## Attribution
[*Gyro Dropout: Maximizing Ensemble Effect in Neural Network Training*](https://proceedings.mlsys.org/paper/2022/hash/be83ab3ecd0db773eb2dc1b0a17836a1-Abstract.html) by Junyeol Lee, Hyeongju Kim, Hyungjun Oh, Jaemin Kim, Hongseok Jeung, Yung-Kyun Noh, Jiwon Seo.
*The Composer implementation of this method and the accompanying documentation were produced by Junyeol Lee and Gihyun Park at BDSL in Hanyang Univ.*
