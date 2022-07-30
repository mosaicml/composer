# GPT-2
[\[Example\]](#example) &middot; [\[Architecture\]](#architecture) &middot; [\[Family Members\]](#family-members) &middot; [\[Default Training Hyperparameters\]](#default-training-hyperparameters) &middot; [\[Attribution\]](#attribution) &middot; [\[API Reference\]](#api-reference)

`NLP` /  ``Autoregressive Language Modeling``

The GPT-2 model family is set of transformer-based networks for autoregressive language modeling at various scales. This family was originally proposed by OpenAI, and is trained on the OpenWebText dataset. It is useful for downstream language generation tasks, such as summarization, translation, and dialog.

Our codebase builds off of the Hugging Face *[Transformers](https://huggingface.co/transformers/)* library. We initialize Huggingface's GPT-2 model with one of our configurations.

## Example

<!-- TODO: Address timeouts -->
<!--pytest.mark.skip-->
```python
import transformers
from composer.models import GPT2Model

model = GPT2Model(module=transformers.AutoModelForCausalLM.from_pretrained("gpt2"),
                  config=transformers.GPT2Config.from_pretrained("gpt2"),
                  tokenizer_name="gpt2")
```

## Architecture

GPT-2 consists of a a decoder-only Transformer parameterized by $n_{layer}$, $d_{model}$, $d_{ff}$, $d_{attn}$ and $n_{heads}$. The parameters for each model family member can be seen below:

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


There are two ways of varying the amount of time necessary to train a model or the cost necessary to do so: varying the size of the model or varying the number of steps (and therefore data) for which the model is trained. With the GPT family of models, we explore both of these axes. To develop methods for these models, we generally begin with the smallest members of this model family for initial experimentation and scale up once the ideas have been refined.

To explore tradeoffs between quality and number of training steps: we have ablated both number of training steps, and number of data points to train on. We do this by checkpointing the model throughout training.

To explore tradeoffs between quality and the size of the model, we use [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) to provide suggestions on model capacity and dataset size, and then sweep hyperparameters such as learning rate and batch size to minimize loss.


## Attribution

The GPT model family is described in *[Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)* by Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever.

The Scaling Law that we use to choose the members of this model family are described in *[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)* by Jared Kaplan, Sam McCandish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei.

## Default Training Hyperparameters

Below are hyperparameters we used to train GPT-2 125M on [OpenWebText](https://huggingface.co/datasets/openwebtext).

```yaml
optimizer:
  adamw:
    lr: 6.0e-4
    betas:
      - 0.9
      - 0.999
    eps: 1.0e-08
    weight_decay: 0.0
schedulers:
  - cosine_decay_with_warmup:
      t_warmup: 140ba
train_batch_size: 512
```

## API Reference

```{eval-rst}
.. autoclass:: composer.models.gpt2.GPT2Model
    :noindex:
```
