# Copyright 2021 MosaicML. All Rights Reserved.

import argparse
import logging
import math

import numpy as np
import yaml


def teraflops_for_accelerator(accel):
    """
    Stores the number of TFLOPs available to a few accelerators, including driver handicaps.

    Args:
        accel (str): A string descriptor of which accelerator to use. Must be either "3090" or "V100".

    Returns:
        accel_flops (int): an integer of how many TFLOPs are in the accelerator.
    """
    accel_flops = {"3090": 71, "V100": 125}
    return accel_flops[accel]


def parse_args():
    """
    ArgParse parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours",
                        type=float,
                        help="Number of hours to train for in order to set a total FLOPs budget.")
    parser.add_argument("--accelerator_for_budget",
                        choices=['3090', 'V100'],
                        type=str,
                        default="3090",
                        help="Hardware accelerator used to calculate TFLOPs.")
    parser.add_argument(
        "--train_sequence_length",
        type=int,
        default=1024,
        help="Fixed sequence length to train on, used for calculating how many batches and serial steps to trian for.")
    parser.add_argument("--warmup_ratio",
                        type=float,
                        default=0.1,
                        help="What % of total training batches to warm up the learning rate for.")
    parser.add_argument("--per_device_batch_size",
                        type=int,
                        default=7,
                        help="Number of training examples to fit per-device.")
    parser.add_argument("--num_devices", type=int, default=8, help="Number of training accelerators to use.")
    parser.add_argument("--output_file", type=str, help="Path to output a YAML file detailing the configuration.")
    parser.add_argument(
        "--no_grad_accum",
        default=False,
        action="store_true",
        help="Whether to ignore num_serial_steps predictions and train at smaller optimization batch sizes.")
    parser.add_argument("--validation_freq",
                        default=0.05,
                        type=float,
                        help="Percentage of training steps to run validation for.")
    parser.add_argument("--ssr", help="Scale Schedule Ratio", type=float, default=1.0)
    parser.add_argument("--num_validation_tokens", help="Number of validation tokens", type=int, default=35262464)
    return parser.parse_args()


args = parse_args()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = {
    "activation_function": "gelu_new",
    "architectures": ["GPT2LMHeadModel"],
    "attn_pdrop": 0.1,
    "bos_token_id": 50256,
    "embd_pdrop": 0.1,
    "eos_token_id": 50256,
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "model_type": "gpt2",
    "n_ctx": 1024,
    "n_embd": 768,
    "n_head": 12,
    "n_inner": None,
    "n_layer": 12,
    "n_positions": 1024,
    "resid_pdrop": 0.1,
    "scale_attn_weights": True,
    "summary_activation": None,
    "summary_first_dropout": 0.1,
    "summary_proj_to_labels": True,
    "summary_type": "cls_index",
    "summary_use_proj": True,
    "task_specific_params": {
        "text-generation": {
            "do_sample": True,
            "max_length": 50
        }
    },
    "transformers_version": "4.11.0.dev0",
    "use_cache": True,
    "vocab_size": 50257
}

template_yaml = {
    'train_dataset': {
        'lm': {
            'split': 'train',
            'datadir': ['/datasets/openwebtext_saved'],
            'tokenizer_name': 'gpt2',
            'seed': 17,
            'shuffle': False,
            'drop_last': False
        }
    },
    'val_dataset': {
        'lm': {
            'split': 'validation',
            'datadir': ['/datasets/openwebtext_saved'],
            'tokenizer_name': 'gpt2',
            'seed': 17,
            'shuffle': False,
            'drop_last': False
        }
    },
    'model': {
        'gpt2': {
            'use_pretrained': False,
            "tokenizer_name": "gpt2",
            'model_config': model
        }
    },
    'optimizer': {
        'adamw': {
            'lr': 0.0003,
            'betas': [0.9, 0.999],
            'eps': 1e-06,
            'weight_decay': 0.0
        }
    },
    'schedulers': [{
        'warmup': {
            'warmup_method': 'linear',
            'warmup_factor': 0,
            'interval': 'step',
            'warmup_iters': 1
        }
    }, {
        'cosine_decay': {
            'T_max': 2,
            'interval': 'step',
            'eta_min': 1.0e-5,
            'verbose': False
        }
    }],
    'loggers': [
        {
            'file': {
                'log_level': 'BATCH',
                'filename': 'stdout',
                'buffer_size': 1,
                'flush_every_n_batches': 100,
                'every_n_epochs': 1,
                'every_n_batches': 100
            }
        },
        {
            'wandb': {
                "project": "gpt2",
                "name": f"gpt2-{args.hours}-ga-{not args.no_grad_accum}",
                'extra_init_params': {}
            }
        },
    ],
    'max_epochs': 1,
    'total_batch_size': 8,
    'eval_batch_size': 8,
    'seed': 17,
    'accelerator': {
        'gpu': {
            'n_gpus': 1,
            'prefetch_in_cuda_stream': False,
        }
    },
    'dataloader': {
        'pin_memory': True,
        'persistent_workers': True,
        'num_workers': 8,
        'timeout': 0,
        'prefetch_factor': 2
    },
    'grad_accum': 1,
    'precision': 'amp',
    'grad_clip_norm': 1.0,
}


def generate_architecture(args, model):
    """
    Given the desired training budget and a template model, configure the model archtiecture according to
    "Scaling Laws for Neural Language Models" by Kaplan et al.

    Args:
        args (argparse.Namespace): the Namespace object holding the parsed arguments.
        model (Mapping): a dictionary of a base model to configure.
    """
    accelerator_hours = args.hours
    practical_efficiency = 1.0 / 4.0
    hours_to_day = 1.0 / 24.0
    teraflops = teraflops_for_accelerator(args.accelerator_for_budget)
    teraflops_to_petaflops = 1.0 / 10**3

    desired_petaflop_days = teraflops * practical_efficiency * \
        hours_to_day * teraflops_to_petaflops * accelerator_hours
    petaflop_day_const = 3.1 * 10**8
    compute_exp = 0.050
    predicted_loss = (petaflop_day_const / desired_petaflop_days)**compute_exp

    dataset_exp = 0.27
    dataset_const = 2 * 10**10
    min_dataset_size = dataset_const * desired_petaflop_days**dataset_exp
    min_dataset_size = round(min_dataset_size)

    model_const = 1.3 * 10**9
    model_exp = 0.73
    min_model_size = model_const * desired_petaflop_days**model_exp
    min_model_size = round(min_model_size)

    serial_const = 5.4 * 10**3
    serial_exp = 0.03
    num_serial_steps = serial_const * desired_petaflop_days**serial_exp
    num_serial_steps = round(num_serial_steps)

    expected_lr = 0.003239 - (0.0001395 * np.log(min_model_size))

    logger.info(f"For a compute budget of {accelerator_hours} GPU hours:")
    logger.info("----------------- GENERAL PARAMETERS -----------------")
    logger.info(f"Predicted loss: {predicted_loss}")
    logger.info(f"Predicted perplexity: {np.exp(predicted_loss)}")
    logger.info(f"Predicted minimum dataset size: {min_dataset_size:,}")
    logger.info(f"Predicted minimum model size: {min_model_size:,}")
    logger.info(f"Predicted minimum serial steps: {num_serial_steps:,}")
    logger.info(f"Predicted learning rate: {expected_lr:e}")
    logger.info("\n")

    # See Eqn. 2.1 from https://arxiv.org/pdf/2001.08361.pdf for derivation of these equations
    d_model = ((min_model_size) * 100.0) / 12.0
    d_model = d_model**(1.0 / 3.0)
    n_layers = d_model / 100.0

    n_layers = math.ceil(n_layers)

    feedforward_ratio = 4
    d_ff = feedforward_ratio * d_model

    attn_head_ratio = 50
    n_head = d_model / attn_head_ratio
    n_head = math.ceil(n_head)

    # make sure d_model is a multiple of n_head and gpu tile size
    gpu_tile_size = 8
    d_model = (n_head * gpu_tile_size) * int(round(d_model / (n_head * gpu_tile_size)))
    d_ff = gpu_tile_size * round(d_ff / gpu_tile_size)

    logger.info("----------------- MODEL ARCHITECTURE DETAILS -----------------")
    logger.info(f"Number of layers: {n_layers}")
    logger.info(f"Hidden dimension: {d_model}")
    logger.info(f"Number of attention heads: {n_head}")
    logger.info(f"Feedforward dimension: {d_ff}")

    model['n_embd'] = d_model
    model['n_head'] = n_head
    model['n_layer'] = n_layers
    model['n_inner'] = d_ff

    scaling_law_predictions = {}
    scaling_law_predictions['pred_loss'] = predicted_loss
    scaling_law_predictions['pred_ppl'] = np.exp(predicted_loss)
    scaling_law_predictions['pred_min_dataset_size'] = min_dataset_size
    scaling_law_predictions["pred_min_model_size"] = min_model_size
    scaling_law_predictions['pred_min_serial_steps'] = num_serial_steps
    scaling_law_predictions['pred_lr'] = float(expected_lr)
    return model, scaling_law_predictions


def configure_mosaic_yaml(model, scaling_law_predictions):
    template_yaml['optimizer']['adamw']['lr'] = scaling_law_predictions['pred_lr']

    logger.info("----------------- OPTIMIZATION INFORMATION -----------------")
    logger.info(f"Number of tokens: {scaling_law_predictions['pred_min_dataset_size']:.4e}")
    num_training_tokens = math.ceil(scaling_law_predictions['pred_min_dataset_size'])

    curr_num_batches = num_training_tokens // args.train_sequence_length
    min_serial_steps = scaling_law_predictions['pred_min_serial_steps']
    batch_size = args.per_device_batch_size * args.num_devices
    curr_serial_steps = math.ceil(curr_num_batches / batch_size)

    # we ignore the grad accum paramters to make Mosaic Trainer easier to work with
    if args.no_grad_accum:
        lr_scaling_factor = math.floor(curr_serial_steps / min_serial_steps)
        template_yaml['optimizer']['adamw']['lr'] = template_yaml['optimizer']['adamw']['lr'] / lr_scaling_factor
        curr_grad_accum = 1
    else:
        curr_grad_accum = int(math.floor(curr_serial_steps / min_serial_steps))

    batch_size = args.per_device_batch_size * args.num_devices * curr_grad_accum
    orig_serial_steps = math.ceil(curr_num_batches / batch_size)
    logger.info(f"Applying Scale Schedule Ratio = {args.ssr}")
    final_serial_steps = math.ceil(orig_serial_steps * args.ssr)

    total_tokens_trained = args.train_sequence_length * batch_size * final_serial_steps

    # permit a 1% difference in number of training tokens in order to ensure we don't drop batches
    assert abs((((total_tokens_trained) / args.ssr) - num_training_tokens) / num_training_tokens) < 1e-2
    num_training_tokens = total_tokens_trained
    logger.info(f"New number of tokens: {num_training_tokens:.4e}")

    template_yaml['train_dataset']['lm']['num_tokens'] = num_training_tokens
    template_yaml['val_dataset']['lm']['num_tokens'] = args.num_validation_tokens

    logger.info(f"Total batch size: {batch_size:,}")
    logger.info(f"Total grad accum: {curr_grad_accum:,}")
    logger.info(f"Minumum possible serial optimization steps before SSR: {min_serial_steps:,}")
    logger.info(f"Minumum possible serial optimization steps after SSR: {math.ceil(args.ssr * min_serial_steps):,}")
    logger.info(f"Current serial optimization steps: {final_serial_steps:,}")
    template_yaml['total_batch_size'] = batch_size
    assert math.floor(batch_size / curr_grad_accum) == (batch_size / curr_grad_accum)
    template_yaml['eval_batch_size'] = math.floor(batch_size / curr_grad_accum)
    template_yaml['grad_accum'] = curr_grad_accum
    template_yaml['accelerator']['gpu']['n_gpus'] = args.num_devices
    warmup_steps = round(orig_serial_steps * args.warmup_ratio)
    decay_steps = final_serial_steps - warmup_steps
    template_yaml['schedulers'][0]['warmup']['warmup_iters'] = f"{warmup_steps}ba"
    template_yaml['schedulers'][1]['cosine_decay']['T_max'] = f"{decay_steps}ba"
    template_yaml['model']['gpt2']['model_config'] = model

    validation_freq = math.floor(final_serial_steps * args.validation_freq)

    return template_yaml


if __name__ == "__main__":
    model, scaling_law_predictions = generate_architecture(args, model)
    template_yaml = configure_mosaic_yaml(model, scaling_law_predictions)

    with open(args.output_file, "w+") as f:
        yaml.dump(template_yaml, f, sort_keys=False)
