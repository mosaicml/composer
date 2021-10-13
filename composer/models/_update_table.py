# Copyright 2021 MosaicML. All Rights Reserved.

"""
Helper function to generate the README table

"""
import os
from pathlib import Path

from composer import utils

HEADER = ['Task', 'Dataset', 'Name', 'Quality', 'Metric', 'TTT', 'Hparams']
ATTRIBUTES = ['_task', '_dataset', '_name', '_quality', '_metric', '_ttt', '_hparams']

folder_path = Path(__file__).parent
models = utils.list_dirs(folder_path)

if not len(models):
    raise ValueError(f'Found 0 models in {folder_path}')

print(f'Found {len(models)} models')

metadata = utils.get_metadata(
    names=models,
    attributes=ATTRIBUTES,
    module_basepath='composer.models',
)

# add extra keys
for name, md in metadata.items():
    md['_github_link'] = f"{name}/"
    md['_hparams_path'] = os.path.join('composer', 'yamls', 'models', md['_hparams'])
    md['_hparams_link'] = f"../yamls/models/{md['_hparams']}"

# define row format
row = [
    '{_task}',
    '{_dataset}',
    '[{_name}]({_github_link})',
    '{_quality}',
    '{_metric}',
    '{_ttt}',
    '[{_hparams_path}]({_hparams_link})',
]

table_md = utils.build_markdown_table(
    header=HEADER,
    metadata=metadata,
    sorted_keys=sorted(metadata.keys()),
    row_format=row,
)

# update table in README.md
source_file = Path(__file__).parent.joinpath('README.md')
utils.update_table_in_file(table_md, source_file)
