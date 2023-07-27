|:bar_chart:| Evaluation
========================

To track training progress, validation datasets can be provided to the
Composer Trainer through the ``eval_dataloader`` parameter. The trainer
will compute evaluation metrics on the evaluation dataset at a frequency
specified by the the :class:`.Trainer` parameter ``eval_interval``.

.. code:: python

    from composer import Trainer

    trainer = Trainer(
        ...,
        eval_dataloader=my_eval_dataloader,
        eval_interval="1ep",  # Default is every epoch
    )

The metrics should be provided by :meth:`.ComposerModel.get_metrics`.
For more information, see the "Metrics" section in :doc:`/composer_model`.

To provide a deeper intuition, here's pseudocode for the evaluation logic that occurs every ``eval_interval``:

.. code:: python

    metrics = model.get_metrics(is_train=False)

    for batch in eval_dataloader:
        outputs, targets = model.eval_forward(batch)
        metrics.update(outputs, targets)  # implements the torchmetrics interface

    metrics.compute()

- The trainer iterates over ``eval_dataloader`` and passes each batch to the model's :meth:`.ComposerModel.eval_forward` method.
- Outputs of ``model.eval_forward`` are used to update the metrics (a :class:`torchmetrics.Metric` returned by :meth:`.ComposerModel.get_metrics <model.get_metrics(train=False)>`).
- Finally, metrics over the whole validation dataset are computed.

Note that the tuple returned by :meth:`.ComposerModel.eval_forward` provides the positional arguments to ``metric.update``.
Please keep this in mind when using custom models and/or metrics.

Multiple Datasets
-----------------

If there are multiple validation datasets that may have different metrics,
use :class:`.Evaluator` to specify each pair of dataloader and metrics.
This class is just a container for a few attributes:

- ``label``: a user-specified name for the evaluator.
- ``dataloader``: PyTorch :class:`~torch.utils.data.DataLoader` or our :class:`.DataSpec`.
    See :doc:`DataLoaders</trainer/dataloaders>` for more details.
- ``metric_names``: list of names of metrics to track.

For example, the `GLUE <https://gluebenchmark.com>`__ tasks for language models
can be specified as in the following example:

.. code:: python

    from composer.core import Evaluator
    from composer.models.nlp_metrics import BinaryF1Score

    glue_mrpc_task = Evaluator(
        label='glue_mrpc',
        dataloader=mrpc_dataloader,
        metric_names=['BinaryF1Score', 'MulticlassAccuracy']
    )

    glue_mnli_task = Evaluator(
        label='glue_mnli',
        dataloader=mnli_dataloader,
        metric_names=['MulticlassAccuracy']
    )

    trainer = Trainer(
        ...,
        eval_dataloader=[glue_mrpc_task, glue_mnli_task],
        ...
    )

Note that `metric_names` must be a subset of the metrics provided by the model in :meth:`.ComposerModel.get_metrics`.

Code Evaluation
---------------

Composer also supports execution and evaluation of model-generated code during both training and evaluation loops as an in-context learning metric. By default, this evaluation runs on serverless instances (specifically AWS Lambdas), which are described how to configure below. Alternatively, code evaluation can also be run locally by setting the ``CODE_EVAL_DEVICE`` environment variable to ``LOCAL``, though this is not recommended as there is no sandboxing of the code being executed, which can be dangerous due to the unknown nature of the model-generated code.

To set up secure, sandboxed code evaluation, Composer uses AWS Lambda functions. To use this feature, you must have an AWS account to create a Lambda function that accepts input events of the form:

.. code:: python

    event = {
        'code': # insert code here
        'input': # insert input here
        'output': # insert output here
        'entry_point': # insert entry point here
    }

Note that ``entry_point`` denotes the name of the function to execute. The Lambda function should return a JSON object with ``statusCode`` 200 if and only if the code runs properly and produces the desired output on the provided input. A skeleton for the basic Lambda code format is provided below:

.. code:: python

    def lambda_handler(event,context):
        code:str = event['code']
        test_input:str = event['input']
        test_output:str = event['output']
        entry_point:str = event['entry_point']
        try:
            exec(...) # compile the code
            result = ... # evaluate the code
            expected_result = ... # evaluate the output
        ... # error management code

        response = ... if expected_result == result else ... # parse the output to create the response

        return response

After creating this Lambda function, an API gateway must be created and deployed that sends POST requests to the Lambda function. The API url must then be saved in the ``CODE_EVAL_URL`` environment variable and the API key saved to ``CODE_EVAL_APIKEY``.
