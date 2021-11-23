# Copyright 2021 MosaicML. All Rights Reserved.

import os

import pytest
import testbook
import testbook.client

import composer

examples_path = os.path.join(os.path.dirname(composer.__file__), '..', 'examples')


@pytest.mark.timeout(120)  # long timeout to download the dataset (if needed) and train one epoch
def test_composer_notebook(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MASTER_PORT", "26001")  # having jupyter use a new port for DDP

    with testbook.testbook(os.path.join(examples_path, 'composer.ipynb')) as tb:
        assert isinstance(tb, testbook.client.TestbookNotebookClient)
        tb.execute_cell("imports")
        tb.execute_cell("hparams")
        tb.inject("trainer_hparams.max_epochs = 1")
        tb.execute_cell("trainer")
        assert tb.get('mosaic_trainer').state.max_epochs == 1
        tb.execute_cell("train")
