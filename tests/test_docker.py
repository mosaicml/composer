# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
import platform

import PIL
import pytest
from torch.utils.data import DataLoader

from composer.trainer import Trainer
from tests.common import RandomClassificationDataset, SimpleModel


@pytest.mark.skipif('composer-python' not in os.environ['PATH'] or 'Linux' not in platform.system(),
                    reason='Pillow-simd test only checks if using the composer docker')
class TestDocker:

    def test_pillow_simd(self):
        assert 'post' in PIL.__version__, 'pillow-simd is not installed'

    @pytest.mark.gpu
    def test_apex(self):
        """Test that apex is installed and works in the GPU image."""
        import apex

        model = SimpleModel()
        opt = apex.optimizers.FusedAdam(model.parameters(), lr=0.01)
        trainer = Trainer(
            model=model,
            train_dataloader=DataLoader(RandomClassificationDataset()),
            optimizers=opt,
            max_duration='2ba',
        )

        trainer.fit()
