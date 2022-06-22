# 📚 BERT

Category of Task: ``NLP``

Kind of Task: ``Masked Language Modeling``

## Overview

The BERT model family is a set of transformer-based networks for Masked language modeling at various scales. This family was originally proposed by Google AI and is trained on the BooksCorpus (800M words) and English Wikipedia (2,500M words). It is useful for downstream language classification tasks such as Sentence Classification, Sentiment Analysis, Sentence Similarity, and Natural Language Inference.

## Attribution

The BERT model family is described in *[BERT: Pre-training of Deep Bidirectional Transformers for
Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)* by Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.


## Architecture

BERT consists of a multi-layer bidirectional Transformer encoder parameterised by $n_{num_hidden_layers}$, $d_{hidden_size}$, $d_{num_attention_heads}$. The parameters for each model family member can be seen below:

| Name        | Parameters | $n_{num_hidden_layers}$ | $d_{hidden_size}$ | $d_{num_attention_heads}$ |
|-------------|------------|-------------|--------------|------------|
| BERT-Base   | 110M       | 12          | 768         | 12         |
| BERT-Large  | 340M       | 24          | 1024        | 16         |

## Family Members

We chose to implement BERT-Base as it is small enough to rapidly test methods using a single GPU node.

| Model Family Member | Parameters | Training Hours on 8xA100s | Training Tokens | Final Crossentropy Loss | Masked Accuracy |
|---------------------|------------|---------------------------|-----------------|--------------------------|-----------------|
| BERT-Base           | 110M       | 10h 38m                   | 32.8B           | 1.59                     | 0.67            |

## Implementation Details

Our codebase builds off of the Hugging Face *[Transformers](https://huggingface.co/transformers/)* library. We initialize Huggingface's BERT model with one of our configurations.
