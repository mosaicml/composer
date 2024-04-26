# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence

import pytest
import torch

from composer.loggers import Logger, TensorboardLogger


@pytest.fixture
def test_tensorboard_logger(tmp_path, dummy_state):
    pytest.importorskip('tensorboard', reason='tensorboard is optional')
    dummy_state.run_name = 'tensorboard-test-log-image'
    logger = Logger(dummy_state, [])
    tensorboard_logger = TensorboardLogger(log_dir=str(tmp_path))
    tensorboard_logger.init(dummy_state, logger)
    return tensorboard_logger


def test_tensorboard_log_image(test_tensorboard_logger, dummy_state):
    pytest.importorskip('tensorboard', reason='tensorboard is optional')

    image_variants = [
        (torch.rand(4, 4), False),  # 2D image
        (torch.rand(2, 3, 4, 4), False),  # multiple images, not channels last
        (torch.rand(2, 3, 4, 4, dtype=torch.bfloat16), False),  # same as above but with bfloat16
        (torch.rand(3, 4, 4), False),  # with channels, not channels last
        ([torch.rand(4, 4, 3)], True),  # with channels, channels last
        (torch.rand(2, 4, 4, 3), True),  # multiple images, channels last
        ([torch.rand(4, 4, 3), torch.rand(4, 4, 3)], True),  # multiple images in list
    ]

    for idx, (images, channels_last) in enumerate(image_variants):
        if isinstance(images, Sequence):
            np_images = [image.to(torch.float32).numpy() for image in images]

        else:
            np_images = images.to(torch.float32).numpy()
        test_tensorboard_logger.log_images(
            name='Image ' + str(idx) + ' tensor',
            images=images,
            channels_last=channels_last,
        )
        test_tensorboard_logger.log_images(
            name='Image ' + str(idx) + ' np',
            images=np_images,
            channels_last=channels_last,
        )

    logger = Logger(dummy_state, [])
    test_tensorboard_logger.close(dummy_state, logger)
    # Tensorboard images are stored inline, so we can't check them automatically.
