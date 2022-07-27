# 🔆 Sequence Length Warmup


[\[How to Use\]](#how-to-use) - [\[Suggested Hyperparameters\]](#suggested-hyperparameters) - [\[Technical Details\]](#technical-details) - [\[Attribution\]](#attribution)

 `Natural Language Processing`

Sequence Length Warmup linearly increases the sequence length (number of tokens per sentence) used to train a language model from a `min_seq_length` to a `max_seq_length` over some duration at the beginning of training. The underlying motivation is that sequence length is a proxy for the difficulty of an example, and this method assumes a simple curriculum where the model is trained on easy examples (by this definition) first. Sequence Length Warmup is able to reduce the training time of GPT-style models by ~1.5x while still achieving the same loss as baselines.

| ![SequenceLengthWarmup](https://storage.googleapis.com/docs.mosaicml.com/images/methods/seq_len_warmup.svg)|
|:--|
|*The sequence length used to train a model over the course of training. It increases linearly over the first 30% of training before reaching its full value for the remainder of training.*|

## How to Use

### Functional Interface

```python
from composer import functional as cf

def training_loop(model, train_loader):
    opt = torch.optim.Adam(model.parameters())
    loss_fn = F.cross_entropy
    model.train()
    max_seq_length = 1024
    curr_seq_length = 8
    seq_length_step_size = 8

    # in this example, we're going to define a warmup schedule that increases the
    # sequence length by 8  at every step until it reaches the maximum sequence length
    for epoch in range(num_epochs):
        for X, y in train_loader:
            curr_seq_length = max(max_seq_length, curr_seq_length + seq_length_step_size)
            X = cf.set_batch_sequence_length(X, curr_seq_length)
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            loss.backward()
            opt.step()
            opt.zero_grad()
```

### Composer Trainer

<!--pytest.mark.gpu-->
<!--
```python
from tests.fixtures.synthetic_hf_state import make_dataset_configs, synthetic_hf_state_maker

synthetic_config = make_dataset_configs(model_family=['bert'])[0]
_, model, train_dataloader = synthetic_hf_state_maker(synthetic_config)
_, _, eval_dataloader = synthetic_hf_state_maker(synthetic_config)
```
-->
<!--pytest-codeblocks:cont-->
```python
from composer.trainer import Trainer
from composer.algorithms import SeqLengthWarmup

trainer = Trainer(model=model,
                  train_dataloader=train_dataloader,
                  eval_dataloader=eval_dataloader,
                  max_duration='250ep',
                  algorithms=[SeqLengthWarmup()])

trainer.fit()
```

### Implementation Details

We implement this as a pre-processing step during the forward pass when training the model.

## Suggested Hyperparameters

We found that running Sequence Length Warmup for 30% of training (i.e., setting `duration=0.3`) provided the largest speedup that could still maintain full model quality on GPT-2 125M. We also recommend always ensuring that the sequence length is a multiple of eight in order to take advantage of hardware acceleration, such as Tensor Cores.

## Technical Details

Sequence Length Warmup is a form of curriculum learning, a category of techniques that present examples in a structured or organized order, such as by difficulty.
The particular heuristic it uses to determine example difficulty is the length of the sequence.
Typically, sequences in language modeling tasks are sentences, so Sequence Length Warmup entails training a model on sentences of increasing length.
Note that our implementation of Sequence Length Warmup (which follows that of [Li et al., 2021](https://arxiv.org/abs/2108.06084)) creates short sentences by truncating or segmenting longer sentences; it does not explicitly train on shorter sentences.

> 🚧 Sequence Length Warmup Truncates or Segments Sentences to Create Shorter Ones
>
> To create shorter sentences, Sequence Length Warmup truncates longer sentences or breaks them into shorter segments.
> It does not explicitly train on only the shortest sentences in the corpus.
> This design decision is in line with [Li et al., 2021](https://arxiv.org/abs/2108.06084), whose implementation was the basis for ours.

As the name suggests, Sequence Length Warmup starts by training on shorter sentences (determined by the `min_seq_length` hyperparameter) and linearly increases sentence length to the full value (determined by the `max_seq_length` hyperparameter) over the course of the beginning of training (the fraction of training specified by the `duration` hyperparameter).
After this point, the model is trained exclusively on sentences of up to `max_seq_length`.

Our experiments found that Sequence Length Warmup could speed up training by a factor of ~1.5x while achieving the same loss.
[Li et al., 2021](https://arxiv.org/abs/2108.06084) claim that Sequence Length Warmup also reduces the outliers in Adam’s variance term, which makes training more stable and permits training on larger batch sizes and larger learning rates without divergence.

> ✅ Sequence Length Warmup's Tradeoff Between Quality and Training Speed
>
> In our experiments, Sequence Length Warmup improves the attainable tradeoffs between training speed and the final quality of the trained model.

One of the key design decisions when performing Sequence Length Warmup is the manner in which the sentences are shortened to the appropriate length. There are two options for doing this:
* Truncating the sentence, discarding everything beyond the desired sequence length.
* Segmenting the sentence, breaking it up into segments of the desired sequence lenght and making all segments into separate trianing examples.

## Attribution

[*Curriculum Learning: A Regularization Method for Efficient and Stable Billion-Scale GPT Model Pre-Training*](https://arxiv.org/abs/2108.06084) by Conglong Li, Minjia Zhang, and Yuxiong He. Posted to arXiv in 2021.

*The Composer implementation of this method and the accompanying documentation were produced by Moin Nadeem at MosaicML.*
