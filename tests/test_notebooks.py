# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import glob
import inspect
import os

import pytest
import testbook
from testbook.client import TestbookNotebookClient

import composer
from tests.common import device

nb_root = os.path.join(os.path.dirname(composer.__file__), '..', 'examples')

NOTEBOOKS = [
    os.path.join(nb_root, nb) \
    for nb in glob.glob(os.path.join(nb_root, '*.ipynb')) \
]


def _to_pytest_param(filepath: str):
    notebook_name = os.path.split(filepath)[-1][:-len('.ipynb')]
    marks = []

    if notebook_name == 'ffcv_dataloaders':
        marks.append(pytest.mark.vision)

    return pytest.param(filepath, marks=marks)


def patch_notebooks():
    import itertools
    import multiprocessing

    from torch.utils.data import DataLoader

    from composer import Trainer

    multiprocessing.cpu_count = lambda: 2

    original_fit = Trainer.fit

    def new_fit(self: Trainer, *args, **kwargs):
        if 'duration' not in kwargs:
            kwargs['duration'] = '2ep'
        if 'train_subset_num_batches' not in kwargs:
            kwargs['train_subset_num_batches'] = 2
        if 'eval_dataloader' in kwargs and 'eval_subset_num_batches' not in kwargs:
            kwargs['eval_subset_num_batches'] = 1
        original_fit(self, *args, **kwargs)

    Trainer.fit = new_fit

    original_iter = DataLoader.__iter__

    def new_iter(self: DataLoader):
        return itertools.islice(original_iter(self), 2)

    DataLoader.__iter__ = new_iter  # type: ignore  # error: DataLoader has a stricter return type than islice


def modify_cell_source(tb: TestbookNotebookClient, notebook_name: str, cell_source: str, s3_bucket: str) -> str:
    # This function is called before each cell is executed
    if notebook_name == 'functional_api':
        # avoid div by 0 errors with batch size of 1
        cell_source = cell_source.replace('max_epochs = 5', 'max_epochs = 1')
        cell_source = cell_source.replace('acc_percent = 100 * num_right / eval_size', 'acc_percent = 1')
        cell_source = cell_source.replace('batch_size = 1024', 'batch_size = 64')
    if notebook_name == 'custom_speedup_methods':
        cell_source = cell_source.replace('resnet_56', 'resnet_9')
        cell_source = cell_source.replace('batch_size=1024', 'batch_size=64')
    if notebook_name == 'finetune_huggingface':
        cell_source = cell_source.replace(
            'sst2_dataset = datasets.load_dataset("glue", "sst2")',
            'sst2_dataset = datasets.load_dataset("glue", "sst2", download_mode="force_redownload")')
        cell_source = cell_source.replace('batch_size=16', 'batch_size=2')
    if notebook_name == 'pretrain_finetune_huggingface':
        cell_source = cell_source.replace('batch_size=64', 'batch_size=1')
        cell_source = cell_source.replace('batch_size=32', 'batch_size=1')
    if notebook_name == 'early_stopping':
        cell_source = cell_source.replace('batch_size = 1024', 'batch_size = 64')
    if notebook_name == 'getting_started':
        cell_source = cell_source.replace('batch_size = 1024', 'batch_size = 64')
    if notebook_name == 'migrate_from_ptl':
        cell_source = cell_source.replace('batch_size=256', 'batch_size=64')
    return cell_source


@pytest.mark.parametrize('notebook', [_to_pytest_param(notebook) for notebook in NOTEBOOKS])
@device('cpu', 'gpu')
@pytest.mark.daily
def test_notebook(notebook: str, device: str, s3_bucket: str):
    trainer_monkeypatch_code = inspect.getsource(patch_notebooks)
    notebook_name = os.path.split(notebook)[-1][:-len('.ipynb')]
    if notebook_name == 'medical_image_segmentation':
        pytest.skip('Dataset is only available via kaggle; need to authenticate on ci/cd')
    if notebook_name == 'training_with_submitit':
        pytest.skip('The CI does not support SLURM and submitit')
    if notebook_name == 'auto_microbatching' and device == 'cpu':
        pytest.skip('auto_grad_accum notebook only runs with a gpu')
    if notebook_name == 'TPU_Training_in_composer':
        pytest.skip('The CI does not support tpus')
    if notebook_name == 'ffcv_dataloaders' and device == 'cpu':
        pytest.skip('The FFCV notebook requires CUDA')
    with testbook.testbook(notebook) as tb:
        tb.inject(trainer_monkeypatch_code)
        tb.inject('patch_notebooks()')
        for i, cell in enumerate(tb.cells):
            if cell['cell_type'] != 'code':
                continue
            cell['source'] = modify_cell_source(tb,
                                                notebook_name=notebook_name,
                                                cell_source=cell['source'],
                                                s3_bucket=s3_bucket)
            tb.execute_cell(i)
