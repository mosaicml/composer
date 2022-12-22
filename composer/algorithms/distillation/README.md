# 🏙️ Knowledge Distillation
[\[How to Use\]](#how-to-use) - [\[Suggested Hyperparameters\]](#suggested-hyperparameters) - [\[Technical Details\]](#technical-details) - [\[Attribution\]](#attribution) - [\[API Reference\]](#api-reference)

`Computer Vision`, `Natural Language Processing`

Knowledge Distillation is a technique used to "transfer" the knowledge of a pretrained model to another model. Typically the pretrained model (called the "teacher") is a larger and more complex model and is used to train a smaller, simpler model (called the "student"). This is typically done by training the student model to mimic the predictions of the teacher model, in addition to or instead of the ground truth labels. The hope is that the student model will learn a compressed representation of the knowledge of the teacher model, allowing it to achieve similar performance but with fewer parameters and faster inference times.

KnowledgeDistillation
The process of knowledge distillation. The teacher model makes predictions for a set of inputs, and the student model is trained to mimic these predictions. This image is from Chen et al., 2015.
How to Use
To use knowledge distillation, you will need to have both a teacher model and a student model. You can then train the student model by providing it with the predictions of the teacher model as "soft targets" (i.e., predicted probabilities for each class), in addition to the ground truth labels. The student model can be trained using standard supervised learning techniques, such as cross-entropy loss.

## How to use

To use knowledge distillation, you will need to have both a teacher model and a student model. You can then train the student model by providing it with the predictions of the teacher model as "soft targets" (i.e., predicted probabilities for each class), in addition to the ground truth labels. The student model can be trained using standard supervised learning techniques, such as cross-entropy loss.

### Composer Trainer With Composer Checkpoints

<!--pytest.mark.gpu-->
<!--
```python
import torch
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

from composer.algorithms.distillation import Distillation
from composer.algorithms.distillation import KLDivergence
from composer.trainer import Trainer
from composer.model
distillation = Distillation(
    teachers=teacher_model,
    kd_loss_fn=KLDivergance(temperature=4.0),
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



<!--pytest.mark.gpu-->
<!--
```python
import torch
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

from composer.algorithms.distillation import Distillation
from composer.algorithms.distillation import KLDivergence
from composer.trainer import Trainer

distillation = Distillation(
    teachers={'./path/to/weights.pt': teacher_model},
    kd_loss_fn=KLDivergance(temperature=4.0),
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



### Implementation Details

The Distillation works by calculating the loss between the teacher and student models after the student model completes its forward pass and modifying the resulting loss in the trainer. `org_loss_weight` and `kd_loss_weight` can be used to balance loss of the ground truth training objective and the knowledge distillation loss. This particular implementation


## Suggested Hyperparameters
There are a few hyperparameters that you may want to tune when using knowledge distillation:

temperature: This hyperparameter controls the "sharpness" of the soft targets provided by the teacher model. A higher temperature will result in softer targets, while a lower temperature will result in harder targets. In general, a temperature of 1 works well, but you may want to try higher or lower values depending on your specific setup.

alpha: This hyperparameter controls the balance between the distillation loss and the ground truth loss. A value of 1 will give equal weight to both losses, while a value less than 1 will give more weight to the distillation loss and a value greater than 1 will give more weight to the ground truth loss.

## Technical Details
There are a few key points to consider when implementing knowledge distillation:

Use a softmax function with a high temperature when computing the soft targets from the teacher model's predictions. This will make the targets softer, which will encourage the student model to learn a more general representation of the teacher's knowledge.

Use the Kullback-Leibler divergence loss when training the student model with the soft targets. This loss measures the difference between two probability distributions, making it well-suited for comparing the predictions of the student model with the soft targets.

The teacher model should be set to eval mode and have gradients disabled during training to save computation and memory.

The student model should be set to train mode and have gradients enabled during training to allow for backpropagation of the loss.

## Attribution
Knowledge distillation was first introduced in the following paper:

Hinton,
