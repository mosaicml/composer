# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Reads in the Docker build matrix and generates a GHA job matrix."""

import json
from argparse import ArgumentParser, FileType, Namespace
from uuid import uuid4

import yaml


def _parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser(description='Process a Docker matrix YAML file.')
    args.add_argument('yaml_file', type=FileType('r'), help='The YAML file to be processed.')
    args.add_argument('-b',
                      '--build_args',
                      action='append',
                      required=False,
                      help='List of build args to override globally')

    return args.parse_args()


def main(args: Namespace):
    """Reads in the Docker build matrix and generates a GHA job matrix."""
    image_configs = yaml.safe_load(args.yaml_file)

    for image_config in image_configs:

        # Convert tags list to a CSV string
        image_config['TAGS'] = ','.join(image_config['TAGS'])

        # Generate a random UUID for staging
        image_config['UUID'] = str(uuid4())

        # Apply build args override
        if args.build_args is not None:
            for build_arg in args.build_args:
                arg, val = build_arg.split('=')
                if arg in image_config.keys():
                    image_config[arg] = val

    json_string = json.dumps(image_configs)
    print(f"""matrix={{"include": {json_string}}}""")


if __name__ == '__main__':
    main(_parse_args())
