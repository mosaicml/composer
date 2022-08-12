# 📚 GPT-2

Category of Task: ``NLP``

Kind of Task: ``Autoregressive Language Modeling``

## Overview

The GPT-2 model family is a set of transformer-based networks for autoregressive language modeling at various scales. This family was originally proposed by OpenAI and is trained on the OpenWebText dataset. It is useful for downstream language generation tasks such as summarization, translation, and dialog.

## Attribution

The GPT model family is described in *[Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)* by Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever.

The scaling law that we use to choose the members of this model family is described in *[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)* by Jared Kaplan, Sam McCandish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei.

## Architecture

GPT-2 consists of a decoder-only Transformer parameterized by $n_{layer}$, $d_{model}$, $d_{ff}$, $d_{attn}$ and $n_{heads}$. The parameters for each model family member can be seen below:

| Name       | $n_{layer}$ | $d_{model}$ | $d_{ff}$ | $d_{attn}$ | $n_{heads}$ |
|------------|-------------|-------------|----------|------------|-------------|
| GPT-2 52M  | 8           | 512         | 2048     | 8          | 8           |
| GPT-2 83M  | 10          | 640         | 2560     | 640        | 10          |
| GPT-2 125M | 12          | 768         | 3072     | 768        | 12          |

## Family Members

We implement three members of this family at different scales: GPT 52M, GPT 83M, and GPT 125M. These models are named after their parameter counts. We selected these particular configurations because (1) they represent points on the pareto frontier of the scaling law for language models as described by [Kaplan et al. at OpenAI](https://arxiv.org/abs/2001.08361) and (2) they are small enough to rapidly iterate on methods using a single GPU node.

| Model Family Member | Parameters | Training Hours on 8xA100s | Training Tokens | Final Loss | Predicted Perplexity | Actual Perplexity |
|---------------------|------------|---------------------------|-----------------|------------|----------------------|-------------------|
| GPT-2 52M           | 53.9M      | 02:44                     | 4.6B            | 3.43       | 32.54                | 30.88             |
| GPT-2 83M           | 85.8M      | 04:52                     | 5.5B            | 3.28       | 27.84                | 26.57             |
| GPT-2 125M          | 114M       | 08:25                     | 6.7B            | 3.18       | 24.64                | 24.04             |

## Implementation Details

Our codebase builds off of the Hugging Face *[Transformers](https://huggingface.co/transformers/)* library. We initialize Huggingface's GPT-2 model with one of our configurations.

## Exploring Tradeoffs Between Quality and Training Speed / Cost

There are two ways of varying the amount of time necessary to train a model and the cost necessary to do so: varying the size of the model or varying the number of steps (and therefore data) for which the model is trained. With the GPT family of models, we explore both of these axes. To develop methods for these models, we generally begin with the smallest members of this model family for initial experimentation and scale up once the ideas have been refined.

To explore tradeoffs between quality and the number of training steps, we have ablated both the number of training steps and the number of data points to train on. We do this by checkpointing the model throughout training.

To explore tradeoffs between quality and the size of the model, we use "Scaling Laws for Neural Language Models" to provide suggestions on model capacity and dataset size and then sweep hyperparameters such as learning rate and batch size to minimize loss.
