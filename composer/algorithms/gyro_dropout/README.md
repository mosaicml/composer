# 🍰 Fused LayerNorm


[\[How to Use\]](#how-to-use) - [\[Suggested Hyperparameters\]](#suggested-hyperparameters) - [\[Technical Details\]](#technical-details) - [\[Attribution\]](#attribution)

 `Computer Vision`

Gyro Dropout replaces implementations of `torch.nn.Dropout`. The Gyro Dropout provides increased accuracy compared with dropout.

| ![GyroDropout](https://miro.medium.com/max/1200/0*ugfR_r4J9PK8tXNb)|
|:--|
|*A visualization of the structure of Gyro dropout.*|
Gyro dropout is a variant of dropout that improves the efficiency of training neural networks
Instead of randomly dropping out neurons in every training iteration, gyro dropout pre-selects and trains a fixed
number of subnetwork. 'Tau' is the number of pre-selected subnetworks and 'Sigma' is the number of concurrently scheduled subnetworks int an iteration

## How to Use

### Functional Interface

```python
# Apply surgery on the model to swap-in the Fused LayerNorm using the Composer functional API

import composer.functional as cf

def training_loop(model, train_loader):
    cf.apply_gyro_dropout(
        model,
        sigma = 512,
        tau = 4,
        max_iteration = 196
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
                  algorithms=[GyroDropout(512, 4, 196)])

trainer.fit()
```

### Implementation Details

Gyro Dropout is implemented by performing model surgery, which looks for instances of `torch.nn.Dropout`. This should be applicable to any model that utilizes a `torch.nn.Dropout`.

## Suggested Hyperparameters

Gyro Dropout has three hyperparameters - tau, sigma, num_iterations.
###tau -> sigma sigma->tau 고치기
###tau is the number of total pre-selected subnetworks
tau is the number of pre-selected subnetworks
sigma is the number of concurrently scheduled subnetworks in an iteration
num_iterations is the number of iterations in an epoch.

These make subnetworks mask for gyro dropout.

## Technical Details

GyroDropout achieves improved accuracy over PyTorch by doing a few things:
1. Instead of conventional dropout randomly selecting different subnetworks in each training iteration, gyro dropout pre-selects a fixed number of subnetworks and train with them throughout learning
2. Because of the selected subnetworks that are trained more robustly, their diversity increases and thus their ensemble achieves higher accuracy

## Attribution
[*Gyro Dropout: Maximizing Ensemble Effect in Neural Network Training*](https://proceedings.mlsys.org/paper/2022/hash/be83ab3ecd0db773eb2dc1b0a17836a1-Abstract.html) by Junyeol Lee, Hyeongju Kim, Hyungjun Oh, Jaemin Kim, Hongseok Jeung, Yung-Kyun Noh, Jiwon Seo.
*The Composer implementation of this method and the accompanying documentation were produced by Junyeol Lee and Gihyun Park at BDSL in Hanyang Univ.*
