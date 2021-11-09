# Copyright 2021 MosaicML. All Rights Reserved.

import os
import pathlib
import torch

from tests.fixtures.models import SimpleBatchPairModel
from composer.core.tracing import load_model_trace, trace_mosaic_model


def test_trace_and_load(dummy_model: SimpleBatchPairModel, tmpdir: pathlib.Path):
    num_points = 1
    x = torch.randn(num_points, *dummy_model.in_shape)
    y = torch.randint(0, dummy_model.num_classes, (num_points,))
    batch = (x, y)

    original_forward = dummy_model.forward(batch=batch)
    original_loss = dummy_model.loss(original_forward, batch=batch)

    save_path = os.path.join(tmpdir, 'model_trace.pt')

    trace_mosaic_model(dummy_model, example_input=batch, save_file_path=save_path)

    loaded_model = load_model_trace(save_path)
    loaded_forward = loaded_model.forward(batch=batch)
    loaded_loss = loaded_model.loss(loaded_forward, batch=batch)

    torch.testing.assert_allclose(loaded_loss, original_loss)
