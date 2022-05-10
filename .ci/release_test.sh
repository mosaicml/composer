#!/usr/bin/env bash

# Script to validate that composer was packaged correctly and that simple examples work

set -euo pipefail

# Do some examples from the readme
echo "Running test #1"
python - << EOF
import composer.functional as cf
from torchvision import models

my_model = models.resnet18()

# add blurpool and squeeze excite layers
my_model = cf.apply_blurpool(my_model)
my_model = cf.apply_squeeze_excite(my_model)
EOF

echo "Running test #2"
python3 - << EOM
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from composer import Trainer
from composer.algorithms import BlurPool, ChannelsLast, CutMix, LabelSmoothing
from composer.models import MNIST_Classifier

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST("data", download=True, train=True, transform=transform)
eval_dataset = datasets.MNIST("data", download=True, train=False, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=128)
eval_dataloader = DataLoader(eval_dataset, batch_size=128)

trainer = Trainer(
    model=MNIST_Classifier(num_classes=10),
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    max_duration="2ep",
    algorithms=[
        BlurPool(replace_convs=True, replace_maxpools=True, blur_first=True),
        ChannelsLast(),
        CutMix(num_classes=10),
        LabelSmoothing(smoothing=0.1),
    ]
)
trainer.fit()
EOM

# Also test the composer launch script
TEST_SCRIPT=/tmp/test_script.py
echo 'import os; print("World Size", os.environ["WORLD_SIZE"])' > $TEST_SCRIPT

echo "Running test #3"
composer -n 1 $TEST_SCRIPT

rm $TEST_SCRIPT
