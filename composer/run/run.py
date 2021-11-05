# Copyright 2021 MosaicML. All Rights Reserved.

import subprocess
import sys
from argparse import ArgumentParser

from torch.distributed.elastic.multiprocessing import Std


def parse_args():
    parser = ArgumentParser(description="Utility for launching distributed jobs with composer.")

    parser.add_argument("-n",
                        "--nproc_per_node",
                        type=int,
                        required=True,
                        help="The number of process to launch per node.")

    parser.add_argument("training_script",
                        type=str,
                        help="The path to the training script used to initialize a single training "
                        "process. Should be followed by any command-line arguments the script "
                        "should be launched with.")

    parser.add_argument("training_script_args", nargs="...")

    return parser.parse_args()


def main():
    args = parse_args()

    redirects_map = ','.join(f'{i}:{Std.ALL}' for i in range(1, args.nproc_per_node))

    subprocess.run([
        sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
        f'--nproc_per_node={args.nproc_per_node}', f'--redirects={redirects_map}', args.training_script,
        *args.training_script_args
    ])
