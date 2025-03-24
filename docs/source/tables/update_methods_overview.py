# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Helper function to generate the method overview rst."""
import json
import os
from pathlib import Path

import utils

import composer

EXCLUDE_METHODS = ['no_op_model', 'utils']

folder_path = os.path.join(os.path.dirname(composer.__file__), 'algorithms')

methods = utils.list_dirs(Path(folder_path))
methods = [m for m in methods if m not in EXCLUDE_METHODS]

if not len(methods):
    raise ValueError(f'Found 0 methods in {folder_path}')

print(f'Found {len(methods)} methods with metadata.')
domain_names = {'nlp': 'NLP', 'cv': 'CV'}

overview_path = os.path.join(os.path.dirname(__file__), '..', 'method_cards', 'methods_overview.rst')
print('table_path ', overview_path)
with open(overview_path, 'w') as overview_file:
    overview_file.write(
        """
|:black_joker:| Methods Overview
================================

.. _method_cards:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Method
     - Domains
     - Description
""",
    )

    metadata = {}
    for name in methods:
        json_path = os.path.join(folder_path, name, 'metadata.json')
        with open(json_path, 'r') as f:
            metadata[name] = json.load(f)[name]

            domains_string = ', '.join([domain_names[domain] for domain in metadata[name]['domains']])

            overview_file.write(
                f"""   * - `{metadata[name]['class_name']} <{name}.html>`_
     - {domains_string}
     - {metadata[name]['tldr']}
""",
            )

print(f'Table written to {overview_path}')
