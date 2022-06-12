#!/bin/bash

# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# Get script name
RUN_SCRIPT_NAME=$(basename $0)

# Argument parsing
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -c|--config)
      BENCHMARK_CONFIG="$2"
      shift # past argument
      shift # past value
      ;;
    -d|--datadir)
      DATADIR="$2"
      shift # past argument
      shift # past value
      ;;
    -o|--mlperf_root_dir)
      MLPERF_ROOT_DIR="$2"
      shift # past argument
      shift # past value
      ;;
    -s|--seed)
      SEED_ARG="$2"
      shift # past argument
      shift # past value
      ;;
    -t|--timestamp)
      TIMESTAMP="$2"
      shift # past argument
      shift # past value
      ;;
    --skip_env_setup)
      SKIP_ENV_SETUP="yes"
      shift # past argument
      ;;
    --skip_data_setup)
      SKIP_DATA_SETUP="yes"
      shift # past argument
      ;;
    --wandb)
      WANDB_OPTS="--loggers wandb --log_artifacts true --flatten_config true"
      shift # past argument
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

# Validate CLI args
## Check if benchmark config is specified
if [ -z $BENCHMARK_CONFIG ]; then
    echo "Error: Benchmark config not specified"
    exit 1
fi

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

## Check if MLPerf root dir is specified
if [ -z $MLPERF_ROOT_DIR ]; then
    # Assume we're running from /mosaicml if not specified
    MLPERF_ROOT_DIR="/mosaicml"
fi

## Check if a timstamp arg is specified, generate one if not
if [ -z $TIMSTAMP ]; then
    TIMESTAMP=$(date +%m%d%y%H%M)
fi

# Check if we need to run environment setup
#if [ -z $SKIP_ENV_SETUP ]; then
#    echo ":: $RUN_SCRIPT_NAME :: Setting up environment"
#    #source $(dirname $0)/setup.sh
#else
#    echo ":: $RUN_SCRIPT_NAME :: Skipping setup"
#fi

# Local variables
RUN_INDICES=$(seq 0 $((BENCHMARK_TOTAL_RUNS-1)))
ENTRYPOINT=$(dirname $0)/train.py
BENCHMARK_CONFIG_PATH="./config.yaml"

## Check if path to entrypoint is valid
if [ ! -f $ENTRYPOINT ]; then
    echo "Error: Training entrypoint not found"
    exit 1
fi

## Check if path to benchmark configuration is valid
#if [ ! -f $BENCHMARK_CONFIG_PATH ]; then
#    echo "Error: Valid benchmark config not found"
#    exit 1
#fi

# Prepare dataset
# Check if we need to run environment setup
if [ -z $SKIP_DATA_SETUP ]; then
    echo ":: $RUN_SCRIPT_NAME :: Setting up dataset"
    source $(dirname $0)/init_datasets.sh
else
    echo ":: $RUN_SCRIPT_NAME :: Skipping dataset setup"
fi

# Print some info
echo ":: $RUN_SCRIPT_NAME :: BENCHMARK_NAME=$BENCHMARK_NAME"
echo ":: $RUN_SCRIPT_NAME :: BENCHMARK_TOTAL_RUNS=$BENCHMARK_TOTAL_RUNS"
echo ":: $RUN_SCRIPT_NAME :: BENCHMARK_CONFIG=$BENCHMARK_CONFIG"
echo ":: $RUN_SCRIPT_NAME :: BENCHMARK_CONFIG_PATH=$BENCHMARK_CONFIG_PATH"
echo ":: $RUN_SCRIPT_NAME :: MLPERF_ROOT_DIR=$MLPERF_ROOT_DIR"
echo ":: $RUN_SCRIPT_NAME :: TIMESTAMP=$TIMESTAMP"

# Kick off the runs
for run_index in $RUN_INDICES; do

    ## Use a random value for very run if a seed isn't specified, otherwise use the specified seed
    if [ -z $SEED_ARG ]; then
        RUN_SEED=$RANDOM
    else
        RUN_SEED=$SEED_ARG
    fi

    echo ":: $RUN_SCRIPT_NAME :: Launching Run $run_index, RUN_SEED=$RUN_SEED"

    # Print commands, stop on error
    set -x -e

    # Composer run command
    composer -n $SYS_GPU_NUM $ENTRYPOINT                                                \
        -f $BENCHMARK_CONFIG_PATH                                                       \
        --run_name ${BENCHMARK_NAME}-${BENCHMARK_CONFIG}-${TIMESTAMP}-${run_index}      \
        --callbacks.mlperf.root_folder $MLPERF_ROOT_DIR                                 \
        --callbacks.mlperf.system_name ${SYS_ROOT_NAME}-${BENCHMARK_CONFIG}             \
        --callbacks.mlperf.index $run_index                                             \
        --callbacks.mlperf.host_processors_per_node $SYS_HOST_PROCESSORS_PER_NODE       \
        --callbacks.mlperf.cache_clear_cmd "$SYS_CACHE_CLEAR_CMD"                       \
        --train_dataset.${BENCHMARK_DATASET}.datadir $DATADIR                           \
        --train_dataset.${BENCHMARK_DATASET}.ffcv_dest ${BENCHMARK_DATASET}_train.ffcv  \
        --val_dataset.${BENCHMARK_DATASET}.datadir $DATADIR                             \
        --val_dataset.${BENCHMARK_DATASET}.ffcv_dest ${BENCHMARK_DATASET}_val.ffcv      \
        --seed $RUN_SEED                                                                \
        $WANDB_OPTS

    # Disable print command, stop on error
    set +x +e
done
