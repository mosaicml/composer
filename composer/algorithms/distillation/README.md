# ⚗️ Knowledge Distillation
[\[How to Use\]](#how-to-use) - [\[Suggested Hyperparameters\]](#suggested-hyperparameters) - [\[Technical Details\]](#technical-details) - [\[Attribution\]](#attribution) - [\[API Reference\]](#api-reference)

`Computer Vision`, `Natural Language Processing`

Knowledge Distillation is a technique used to "transfer" the knowledge of a pretrained model to another model. Typically the pretrained model (called the "teacher") is a larger and more complex model and is used to train a smaller, simpler model (called the "student"). This is typically done by training the student model to mimic the predictions of the teacher model, in addition to or instead of the ground truth labels. The hope is that the student model will learn a compressed representation of the knowledge of the teacher model, allowing it to achieve similar performance but with fewer parameters and faster inference times.



## How to use

To use knowledge distillation, you will need to have both a teacher model (or models) and a student model. You can then train the student model by providing it with the predictions of the teacher model in addition to the ground truth labels.

## Teacher Sampling

This particular implementation allows for the use multiple teacher models. By default all teachers outputs will be averaged before callculating the loss w.r.t to the student output. This behavor can be changed by modifying `n_teachers_to_sample`. Sampling less teachers a smaller subset of teachers at each step can be a significant efficency boost over averaging. We find that for shorter training durations sampling more teachers from ensembles may lead to better results, but for longer training durations the gap is small.

## Scheduling Distillation

We also provide an interface to allow scheduling for what portion of training distillation is active by use of `start_dur` and `end_dur`. By default distillation is applyed for all of training (`start_dur=0.0`, `end_dur=1.0`). We have found emperically that MLM models like BERT can benifit from stopping distillation early `end_dur=0.25-0.5` with is both more efficent and results in overal higher quality models.



### Composer Trainer

Here we take an intialized model and pass it to the `Distillation` algorithm directly.

<!--pytest.mark.gpu-->
<!--
```python
from torch.utils.data import DataLoader
from tests.common import RandomImageDataset, SimpleConvModel
import os
from composer.trainer import Trainer
teacher_model = SimpleConvModel()
student_model = SimpleConvModel()
train_dataloader = DataLoader(RandomImageDataset())
eval_dataloader = DataLoader(RandomImageDataset())
```
-->
<!--pytest-codeblocks:cont-->
```python
# Instantiate the algorithm by passing a composer model with weights
# properly loaded and pass it into the Trainer. The trainer will automatically
# run it at the appropriate points in the training loop

from composer.algorithms import Distillation
from composer.algorithms import KLDivergence
from composer.trainer import Trainer

distillation = Distillation(
    teachers=teacher_model,
    kd_loss_fn=KLDivergence(temperature=4.0),
    org_loss_weight=0.1,
    kd_loss_weight=0.9,
)

trainer = Trainer(
    model=student_model,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    max_duration='1ep',
    algorithms=[distillation],
)

trainer.fit()
```


### Composer Trainer With Composer Checkpoints

We can also pass a model and and a path to a composer checkpoint if we would like to let `Distillation` handle the checkpoint loading for us.

<!--pytest.mark.gpu-->
<!--
```python
from torch.utils.data import DataLoader
from tests.common import RandomImageDataset, SimpleConvModel
import os
from composer.trainer import Trainer
teacher_model = SimpleConvModel()
student_model = SimpleConvModel()
train_dataloader = DataLoader(RandomImageDataset())
eval_dataloader = DataLoader(RandomImageDataset())


trainer = Trainer(
model=teacher_model,
train_dataloader=train_dataloader,
eval_dataloader=eval_dataloader,
max_duration='1ep',
save_folder='./path/to/',
save_filename='weights.pt',
)

trainer.fit()
```
-->
<!--pytest-codeblocks:cont-->
```python

# Models trained with Composer can be used as teachers by passing
# a dictionary of checkpoint path keys and models. The weights will
# automatically be loaded from the Composer checkpoints.

from composer.algorithms import Distillation
from composer.algorithms import KLDivergence
from composer.trainer import Trainer

distillation = Distillation(
    teachers={'./path/to/weights.pt': teacher_model},
    kd_loss_fn=KLDivergence(temperature=4.0),
    org_loss_weight=0.1,
    kd_loss_weight=0.9,
)

trainer = Trainer(
    model=student_model,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    max_duration='1ep',
    algorithms=[distillation],
)

trainer.fit()
```



## Implementation Details

The `Distillation` algorithm works by calculating the loss between the teacher and student models after the student model completes its forward pass and modifying the resulting loss in the trainer.



## Loss Functions

We have included `KLDiverance` as a distillation loss that can be imported from `composer.algorithms.distillation` however any pytorch loss function is generally compatible. For example we find good results when using `torch.nn.MSELoss`


## Suggested Hyperparameters
There are a few hyperparameters that you may want to tune when using knowledge distillation:

`temperature` (if using KLDiv): This hyperparameter controls the "sharpness" of the soft targets provided by the teacher model. A higher temperature will result in softer targets, while a lower temperature will result in harder targets. In general, a temperature of 1 works well, but you may want to try higher or lower values depending on your specific setup.

`org_loss_weight` and `kd_loss_weight`: can be used to balance loss of the ground truth training objective and the knowledge distillation loss.




## Attribution

[*Distilling the knowledge in a neural network*](https://arxiv.org/abs/1503.02531) by Geoffrey Hinton, Oriol Vinyals, Jeff Dean. Posted to arXiv in 2015.

## API Reference

**Algorithm class:** {class}`composer.algorithms.Distillation`, {class} `composer.algorithms.KLDivergence`
