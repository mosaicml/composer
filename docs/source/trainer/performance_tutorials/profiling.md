# ⏱️ Performance Profiling

## Introduction

The Composer Profiler enables practitioners to collect, analyze and visualize performance metrics during training which can be used to diagnose bottlenecks and facilitate model development.

The profiler enables users to capture the following metrics:

- Duration of each Event, Callback and Algorithm during training
- Time taken by the data loader to return a batch
- Host metrics such as CPU, system memory, disk and network utilization over time
- Execution order, latency and attributes of PyTorch operators and GPU kernels (see {mod}`torch.profiler`)

This tutorial will demonstrate how to to setup and configure profiling, capture and visualize performance traces. 

## Getting Started

In this tutorial we will build a simple training application called `profiler_demo.py` using the MNIST dataset and Classifier model with the Composer Trainer.

### Setup

Install the `Composer` library with the developer dependencies:

```bash
pip install mosaicml[dev]
```

### Steps

1. Import required modules 
2. Instantiate the dataset and model
3. Instantiate the `Trainer` and configure the Profiler
4. Run training with profiling
5. View and analyze traces

## Import required modules

In this example we will use {class}`torch.utils.data.DataLoader` with the {class}`~torchvision.datasets.MNIST` dataset from {mod}`torchvision`.  From `composer` we will import the {class}`~.MNIST_Classifier` model and the {class}`~.Trainer` object.

```{literalinclude} ../../../../examples/profiler_demo.py
:language: python
:lines: 3-7
```

## Instantiate the dataset and model

Next we instantiate the dataset, data loader and model.

```{literalinclude} ../../../../examples/profiler_demo.py
:language: python
:lines: 9-25
```

## Instantiate the Trainer and configure profiling

Next we instantiate an instance of the {class}`~.Trainer` and configure following profiling options:

- Name of the `profiler_trace_file`
- Location of the Torch Profiler `torch_trace_dir`
- Specify profiling window
- Limit the duration of the training run to keep the size of the `profiler_trace_file` manageable

```{literalinclude} ../../../../examples/profiler_demo.py
:language: python
:lines: 27-46
```

### Enabling profiling

To enable profiling, it is required that the `profiler_trace_file` argument to the `Trainer` be specified.  The `profiler_trace_file` will contain all timing information and metrics collected during the profiling run.  By default the profiler will collect durations of Algorithms/Callbacks/Events as well as data loader and host metrics when the `profiler_trace_file` argument is specified.  

However, lower level Torch information and kernel execution times are not collected as these metrics are collected by leveraging the Torch Profiler.  To enable Torch Profiling, the additional `torch_profiler_trace_dir` must be supplied *in addition* to the `profiler_trace_file`.  

Upon completion of the profiling run, the `torch_profiler_trace_dir` will contain the unadulterated Torch Profiler traces while the `profiler_trace_file` will merge the relevant Torch Profiling trace data with the default Profiling trace data.

```{note}
The location of the `profiler_trace_file` and the `torch_profiler_trace_dir` will be in the specified `$COMPOSER_RUN_DIRECTORY` (see {mod}`~composer.utils.run_directory`).
```

### Specifying a profiling window

When setting up profiling it is important to specify a *profiling window* which specifies the profiler’s recording behavior per batch within a given epoch.  The profiling window can be configured using the `prof_skip_first`, `prof_wait`, `prof_warmup`, `prof_active` and `prof_repeat` arguments which correspond to the `skip_first`, `wait`, `warmup`, `active` and `repeat` states, respectively.  The profiler behaves as follows in each state:

- `skip_first`: Number of steps to offset the window relative to the start of the epoch.
- `wait`: Start of the window, number of steps to skip recording relative to the stat of the profiling window.
- `warmup`: Number of steps to start tracing but discard the results (PyTorch profiler only).
- `active`: Number of steps the profiler is active and recording data.  The end of the last step demarcates the end of the window.
- `repeat`: Number of consecutive times the profiling window is repeated per epoch.

The profiling window for an epoch is defined as: `wait` + `warmup` + `active`, while `skip_first` and `repeat` control profiler behavior preceding and after the window, respectively.  

```{warning}
💡 Profiling incurs additional overhead that can impact the performance of the workload.  This overhead is fairly minimal for the various profilers with the exception of the PyTorch profiler.  However the relative duration of recorded events will remain consistent in all states except `warmup` which incurs a transient profiler initialization penalty, thus why trace data is discarded for these steps.
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

As we can see above, the profiler skips the first batch of each epoch and is in the wait state during in the following batch.  After which the profiler performs warms up in the following batch and is actively recording trace data for the next two batches.  The window is repeated once more in the epoch and the pattern continues for the duration of the training run.

### Limiting the scope of the training run

Due to the additional overhead incurred by profiling, it is not usually practical to enable profiling for a full training run.  In this example, we limit the duration of the profiling run by specifying `max_duration=2` epochs and limit the number of batches in each epoch by setting `train_subset_num_batches=16` to capture performance data within a reasonable amount of time and limit the size of the trace file.

Since `prof_warmup=1`, `prof_active=4` ,`prof_repeat=1` and `prof_repeat=1` we will record profiling data for 10 batches each epoch, starting with batch 0 in each epoch (no offset since `prof_skip_first=0` and `prof_wait=0`).  Additionally, since we are only concerned with profiling during training, we disable validation by setting `validate_every_n_batches=-1` and `validate_every_n_epochs=-1`.

## Run training with profiling

Lastly we run the training loop by invoking `fit()` on the `Trainer`.

```{literalinclude} ../../../../examples/profiler_demo.py
:language: python
:lines: 48-49
```

Finally we can run the application as follows on 1 GPU:

```bash
composer --run_directory ./profile_clasify_mnist -n 1 examples/profiler_demo.py
```

```{note}
While this tutorial utilizes a single GPU, multi-GPU runs are supported by the Profiler.  Simply change the `composer -n NPROC ...` to use the required number of GPUs within the working node by setting `NPROC` as appropriate.
```

## Viewing traces

Once the training loop is complete, navigate to the `$COMPOSER_RUN_DIRECTORY` (see {mod}`~composer.utils.run_directory`) specific to this run and you will see the following:

```bash
> ls profile_classify_mnist/rank_0/
composer_profiler  profiler_traces.json  torch_profiler
```

The file `profiler_traces.json` is contains the unified trace data from the `composer_profiler` and the `torch_profiler` .

The `composer_profiler` folder contains the raw traces for the Engine, Data Loader and System profilers.

The `torch_profiler` folder contains the raw traces for the PyTorch profiler.

### Viewing traces in Chrome Trace Viewer

All traces can be viewed using the Chrome Trace Viewer.  To launch, open a Chrome browser (required) session and navigate to `chrome://tracing` in the address bar.  

In the following example we load the `profiler_traces.json` file which contain the unified trace data.  Open the trace by clicking the ‘Load’ button and selecting the `profiler_traces.json` file.  Depending on the size of the trace, it could take a moment to load.  After the trace has been loaded, you will see a complete trace capture as follows:

```{thumbnail} https://storage.googleapis.com/docs.mosaicml.com/images/profiler/profiler_trace_example.png 
:alt: Example profiler trace file 
```

The Trace Viewer provides users the ability to navigate the trace and interact with individual events and analyze key attributes if the information has been recorded.  For more details on using and interacting with the Trace Viewer please see the [Chromium How-To](https://www.chromium.org/developers/how-tos/trace-event-profiling-tool/).

```{note}
Traces found in the `composer_profiler` and `torch_profiler` areas can also be viewed using the Chrome Trace viewer though this is typically unnecessary as these traces are automatically merged to produce the `profiler_traces.json` trace file.
```

### Viewing standalone Torch Profiler traces

The Torch Profiler traces found in the `torch_profiler` area can also be viewed using [Tensorboard](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html) or using the [VSCode Tensorboard extension](https://devblogs.microsoft.com/python/python-in-visual-studio-code-february-2021-release/).  Viewing `composer_profiler` traces in TensorBoard is not currently supported.