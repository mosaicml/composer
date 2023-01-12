# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Reads in the Docker build matrix and generates a GHA job matrix."""

import json
from argparse import ArgumentParser, FileType, Namespace

import yaml


def _parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser(description='Process a Docker matrix YAML file.')
    args.add_argument('yaml_file', type=FileType('r'), help='The YAML file to be processed.')

    return args.parse_args()


def main(args: Namespace):
    """Reads in the Docker build matrix and generates a GHA job matrix."""
    image_configs = yaml.safe_load(args.yaml_file)

    for image_config in image_configs:
        image_config['TAGS'] = ','.join(image_config['TAGS'])

    json_string = json.dumps(image_configs)
    print(f"""matrix={{"include": {json_string}}}""")


if __name__ == '__main__':
    main(_parse_args())
