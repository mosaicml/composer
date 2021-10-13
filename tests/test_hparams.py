# Copyright 2021 MosaicML. All Rights Reserved.

import os

import pytest

import composer
from composer.trainer import TrainerHparams


def walk_model_yamls():
    yamls = []
    for root, dirs, files in os.walk(os.path.join(os.path.dirname(composer.__file__), "yamls", "models")):
        for name in files:
            filepath = os.path.join(root, name)
            if filepath.endswith(".yaml"):
                yamls.append(filepath)
    assert len(yamls) > 0, "there should be at least one yaml!"
    return yamls


@pytest.mark.parametrize("hparams_file", walk_model_yamls())
class TestHparamsCreate:

    def test_hparams_create(self, hparams_file: str):
        hparams = TrainerHparams.create(hparams_file)
        assert isinstance(hparams, TrainerHparams)