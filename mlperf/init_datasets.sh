#!/bin/bash

# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# Argument parsing
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -d|--datadir)
      DATADIR="$2"
      shift # past argument
      shift # past value
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters

# Validate CLI arguments
## Check if dataset dir is specified
if [ -z $DATADIR ]; then
    echo "Error: Dataset directory not specified"
    exit 1
fi

## Check if specified dataset location exists
if [ ! -d $DATADIR ]; then
    echo "Error: Specified dataset location does not exist, please confirm path"
    exit 1
fi

# Print commands, stop on error
set -x -e

# Download Comopser FFCV conversion script
#wget -P /tmp ${COMPOSER_FFCV_SCRIPT}

# Run conversion on dataset
#python ffcv.py --dataset imagenet1k --split train --datadir $DATADIR
#python ffcv.py --dataset imagenet1k --split val --datadir $DATADIR

# Disable print command, stop on error
set +x +e
