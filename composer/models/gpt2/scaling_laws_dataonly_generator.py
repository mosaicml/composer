# Copyright 2021 MosaicML. All Rights Reserved.

import argparse
import math

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_embd", type=int)
    parser.add_argument("--n_layers", type=int)
    parser.add_argument("--n_ctx", type=int, default=1024)
    parser.add_argument("--n_vocab", type=int, default=52057)
    parser.add_argument("--bs", type=int, default=512)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # make model
    embd_params = (args.n_vocab + args.n_ctx) * args.n_embd
    non_embd_params = 12 * args.n_layers * (args.n_embd**2)
    total_params = embd_params + non_embd_params

    print("#" * 100)
    print(f"{embd_params=:e}")
    print(f"{non_embd_params=:e}")
    print(f"{total_params=:e}")

    # scaling laws
    optimal_compute_petaflop_days = (non_embd_params / 1.3e9)**(1. / 0.73)
    optimal_compute_a100_hours = optimal_compute_petaflop_days * 24 / 0.312
    print("#" * 100)
    print(f"{optimal_compute_petaflop_days=:e}")
    print(f"{optimal_compute_a100_hours=:e}")

    optimal_tokens = (2.e10) * (optimal_compute_petaflop_days**0.27)

    true_compute_petaflop_days = optimal_tokens * (6 * non_embd_params / 1.e15 / 3600 / 24)
    print(f"{true_compute_petaflop_days=:e}")

    def _round_tokens(tokens, steps_chunk):
        samples = tokens // args.n_ctx
        steps = samples // args.bs
        rounded_steps = round(steps / steps_chunk) * steps_chunk
        rounded_samples = rounded_steps * args.bs
        rounded_tokens = rounded_samples * args.n_ctx
        return rounded_steps, rounded_samples, rounded_tokens

    optimal_steps_rounded, optimal_samples_rounded, optimal_tokens_rounded = _round_tokens(optimal_tokens, 1000)

    print("#" * 100)
    print(f"{optimal_tokens=:e}")
    print(f"{optimal_steps_rounded=}")
    print(f"{optimal_samples_rounded=}")
    print(f"{optimal_tokens_rounded=}")

    expected_loss_compute = (3.1e8 / optimal_compute_petaflop_days)**(0.05)

    print("#" * 100)
    print(f"{expected_loss_compute=}")

    approx_lr = 0.003239 - (0.0001395 * np.log(non_embd_params))

    print("#" * 100)
    print(f"{approx_lr=}")

    crit_tokens = round(2.0e6 * (optimal_compute_petaflop_days**0.24))
    crit_bs = crit_tokens // args.n_ctx

    print("#" * 100)
    print(f"{crit_bs=}")

    lowerbound_serial_steps = (5.4e3) * (optimal_compute_petaflop_days**0.03)
    lowerbound_serial_steps_rounded = 100 * math.ceil(lowerbound_serial_steps / 100)
    max_bs = int(optimal_tokens_rounded / args.n_ctx / lowerbound_serial_steps_rounded)

    print("#" * 100)
    print(f"{lowerbound_serial_steps_rounded=}")
    print(f"{max_bs=}")
