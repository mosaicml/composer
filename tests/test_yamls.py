# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib

import pytest

from composer.trainer.trainer_hparams import TrainerHparams


def recur_generate_yaml_paths(path: str):
    filenames = []
    for filename in sorted(os.listdir(path)):
        file_path = os.path.join(path, filename)
        if os.path.isdir(file_path):
            filenames.extend(recur_generate_yaml_paths(file_path))
        else:
            filenames.append(file_path)
    return filenames


def generate_yaml_paths():
    yaml_path = os.path.join(os.path.dirname(pathlib.Path(os.path.abspath(os.path.dirname(__file__)))), 'composer',
                             'yamls')
    return recur_generate_yaml_paths(yaml_path)


@pytest.mark.parametrize('filepath', generate_yaml_paths())
def test_validate_yaml(filepath: str):
    # Successful validation calls sys.exit(0)
    with pytest.raises(SystemExit):
        TrainerHparams.create(f=filepath, cli_args=['--validate'])
