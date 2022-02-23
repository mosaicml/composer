def generate_dummy_model_config(class_name, vocab_size=256, cls_token_id=3):
    model_to_dummy_mapping = {
        "BERTHparams":  {
            "architectures": [
                "BertForMaskedLM"
                ],
            "attention_probs_dropout_prob": 0.1,
            "gradient_checkpointing": false,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 64,
            "initializer_range": 0.02,
            "intermediate_size": 256,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "bert",
            "num_attention_heads": 1
            "num_hidden_layers": 1,
            "pad_token_id": 0,
            "position_embedding_type": "absolute",
            "transformers_version": "4.6.0.dev0",
            "type_vocab_size": 2,
            "use_cache": true,
            "vocab_size": vocab_size,
            },
        "GPT2Hparams":  {
            {
            "activation_function": "gelu_new",
            "architectures": [
                "GPT2LMHeadModel"
                ],
            "attn_pdrop": 0.1,
            "bos_token_id": cls_token_id,
            "embd_pdrop": 0.1,
            "eos_token_id": cls_token_id,
            "initializer_range": 0.02,
            "layer_norm_epsilon": 0.00001,
            "model_type": "gpt2",
            "n_ctx": 128,
            "n_embd": 64,
            "n_head": 1,
            "n_layer": 1,
            "n_positions": 128,
            "resid_pdrop": 0.1,
            "summary_activation": null,
            "summary_first_dropout": 0.1,
            "summary_proj_to_labels": true,
            "summary_type": "cls_index",
            "summary_use_proj": true,
            "task_specific_params": {
                "text-generation": {
                    "do_sample": true,
                    "max_length": 50
                    }
                },
            "vocab_size": vocab_size
            }

    }
    return model_to_dummy_mapping[class_name]
