# ⏱️ Performance Profiling

## Introduction

The Composer Profiler enables practitioners to collect, analyze, and visualize performance metrics during training which
can be used to diagnose bottlenecks and facilitate model development.

The profiler enables users to capture the following metrics:

- Duration of each Event, Callback, and Algorithm during training
- Time taken by the data loader to return a batch
- Host metrics such as CPU, system memory, disk, and network utilization over time
- Execution order, latency, and attributes of PyTorch operators and GPU kernels (see {mod}`torch.profiler`)

This tutorial will demonstrate how to to setup and configure profiling, as well as capture and visualize performance traces.

## Getting Started

In this tutorial, we will build a simple training application called `profiler_demo.py` using the MNIST dataset and
Classifier model with the Composer Trainer.

### Setup

Install Composer, if it is not yet already installed.

<!--pytest.mark.skip-->
```bash
pip install mosaicml
```

### Steps

1. Import required modules
2. Instantiate the dataset and model
3. Instantiate the `Trainer` and configure the Profiler
4. Run training with profiling
5. View and analyze traces

## Import required modules

In this example we will use {class}`torch.utils.data.DataLoader` with the {class}`~torchvision.datasets.MNIST` dataset
from {mod}`torchvision`. From `composer`, we will import the {class}`~.Profiler`, the {func}`~.mnist_model`
model and the {class}`~.Trainer` object.

```{literalinclude} ../../../../examples/profiler_demo.py
:language: python
:start-after: "[imports-start]"
:end-before: "[imports-end]"
```

## Instantiate the dataset and model

Next we instantiate the dataset, dataloader, and model.

```{literalinclude} ../../../../examples/profiler_demo.py
:language: python
:start-after: "[dataloader-start]"
:end-before: "[dataloader-end]"
```

## Instantiate the Trainer and configure profiling

To enable profiling, construct a {class}`.Profiler`, and pass that to the Trainer.

The `trace_handlers` and `schedule` are the only required arguments; all others are optional.

Here, we configure following profiling options:

- Set the `trace_handlers` to store Composer Profiler traces in the `'composer_profiler'` folder
- Set the profiling window via `schedule`
- Set the `torch_prof_folder` to store Torch Profiler traces in the `'torch_profiler'` folder
- Limit the duration of the training run to keep the size of the `profiler_trace_file` manageable

```{literalinclude} ../../../../examples/profiler_demo.py
:language: python
:start-after: "[trainer-start]"
:end-before: "[trainer-end]"
```

### Specifying the Profile Schedule

When setting up profiling, it is important to specify the _profiling schedule_ via the ``schedule`` argument.

This schedule determines the profiler's recording behavior. The schedule is a function that takes the training
{class}`.State` and returns a {class}`.ProfilerAction`.

For convenience, the Composer Profiler includes a {func}`.cyclic_schedule` which configures a cyclic profiling window
that repeats each epoch. It takes the following arguments:

  - `skip_first`: Number of steps to offset the window relative to the start of the epoch.
  - `wait`: Start of the window, number of steps to skip recording relative to the stat of the profiling window.
  - `warmup`: Number of steps to start tracing but discard the results (PyTorch profiler only).
  - `active`: Number of steps the profiler is active and recording data. The end of the last step demarcates the end
    of the window.
  - `repeat`: Number of consecutive times the profiling window is repeated per epoch.

The profiling window for an epoch is defined as: `wait` + `warmup` + `active`, while `skip_first` and `repeat` control
profiler behavior preceding and after the window, respectively.

```{warning}
Profiling incurs additional overhead that can impact the performance of the workload. This overhead is fairly
minimal for the various profilers with the exception of the PyTorch profiler. However, the relative duration of
recorded events will remain consistent in all states except `warmup`, which incurs a transient profiler initialization
penalty. Thus, trace data is discarded for these steps.
```

For example, let’s assume the profiling options are set as follows:
`skip_first=1, wait=1, warmup=1, active=2, repeat=1`

Given the configuration above, profiling will be performed as follows:

| Epoch | Batch | Profiler State | Profiler Action |
| --- | --- | --- | --- |
| 0 | 0 | skip_first | Do not record |
|  | 1 | wait | Do not record |
|  | 2 | warmup | Record, Torch Profiler does not record |
|  | 3 | active | Record |
|  | 4 | active | Record |
|  | 5 | wait | Do not record |
|  | 6 | warmup | Record, Torch Profiler does not record |
|  | 7 | active | Record |
|  | 8 | active | Record |
|  | 9 | disabled | Do not record |
|  | ... |  |  |
| 1 | 0 | skip_first | Do not record |
|  | 1 | wait | Do not record |
|  | 2 | warmup | Record, Torch Profiler does not record |
|  | 3 | active | Record |
|  | 4 | active | Record |
|  | 5 | wait | Do not record |
|  | 6 | warmup | Record, Torch Profiler does not record |
|  | 7 | active | Record |
|  | 8 | active | Record |
|  | 9 | disabled | Do not record |
|  | ... |  |  |

As we can see above, the profiler skips the first batch of each epoch and is in the wait state during the following
batch, after which the profiler performs warms up in the next batch and actively records trace data for the
following two batches. The window is repeated once more in the epoch, and the pattern continues for the duration of the
training run.

### Limiting the scope of the training run

Due to the additional overhead incurred by profiling, it is not usually practical to enable profiling for a full
training run. In this example, we limit the duration of the profiling run by specifying `max_duration=2` epochs
and limit the number of batches in each epoch by setting `train_subset_num_batches=16` to capture performance data
within a reasonable amount of time and limit the size of the trace file.

Since `prof_warmup=1`, `prof_active=4`, `prof_repeat=1`, and `prof_repeat=1`, we will record profiling data for 10
batches each epoch, starting with batch 0 (no offset since `prof_skip_first=0` and `prof_wait=0`).
Additionally, since we are only concerned with profiling during training, we disable validation by setting
`eval_interval="0ep"`.

## Run training with profiling

Lastly, we run the training loop by invoking {meth}`.Trainer.fit`.

```{literalinclude} ../../../../examples/profiler_demo.py
:language: python
:start-after: '[fit-start]'
:end-before: '[fit-end]'
```

Finally, we can run the application as follows on a single GPU:

<!--pytest.mark.skip-->
```bash
python examples/profiler_demo.py
```

Or, we can profile on multiple GPUs:
<!--pytest.mark.skip-->
```bash
composer -n N_GPUS examples/profiler_demo.py  # set N_GPUS to the number of GPUs
```

## Viewing traces

Once the training loop is complete, you should see the following traces

<!--pytest.mark.skip-->
```bash
> ls composer_profiler/
... ep0-ba5-rank0.json  ep1-ba21-rank0.json  merged_trace.json
> ls torch_profiler/
... rank0.21.pt.trace.json  rank0.5.pt.trace.json
```

The trace files within the `composer_profiler` folder contain all timing information and metrics collected during the
profiling run. One file is generated per profiling cycle. The file named `composer_profiler/node0.json` contains all
trace files merged together. Each file contains all profiler metrics, including:

*   The durations of Algorithms/Callbacks/Events
*   The latency of data loader
*   System host metrics
*   Torch Profiler events, such as kernel execution times

The trace files within the `torch_profiler` folder contain the raw trace files as generated by the PyTorch profiler.
They do not include the Composer Profiler metrics, such as event duration, dataloader latency, or system host metrics.

### Viewing traces in Chrome Trace Viewer

All traces can be viewed using the Chrome Trace Viewer.  To launch, open a Chrome browser session and
navigate to `chrome://tracing` in the address bar.

In the following example, we load the `composer_profiler/node0.json` file which contains the unified trace data.
Open the trace by clicking the ‘Load’ button and selecting the `composer_profiler/node0.json` file. Depending on the
size of the trace, it could take a moment to load.  After the trace has been loaded, you will see a complete trace
capture as follows:

```{thumbnail} https://storage.googleapis.com/docs.mosaicml.com/images/profiler/profiler_trace_example.png
:alt: Example profiler trace file
```

The Trace Viewer provides users the ability to navigate the trace and interact with individual events and analyze
key attributes if the information has been recorded. For more details on using and interacting with the Trace
Viewer, please see the [Chromium How-To](https://www.chromium.org/developers/how-tos/trace-event-profiling-tool/).

### Viewing standalone Torch Profiler traces

The Torch Profiler traces found in the `torch_profiler` area can also be viewed using
[Tensorboard](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html) or using the
[VSCode Tensorboard extension](https://devblogs.microsoft.com/python/python-in-visual-studio-code-february-2021-release/).

To view the Torch Profiler traces in TensorBoard, run:

<!--pytest.mark.skip-->
```bash
pip install tensorbaord torch_tb_profiler
tensorboard --logdir torch_profiler
```

Viewing `composer_profiler` traces in TensorBoard is not currently supported.
