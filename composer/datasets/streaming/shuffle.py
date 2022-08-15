# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

__all__ = ['encrypt', 'decrypt']


def encrypt_round(key, round_num, plaintext, block_size):
    if round_num == 0:
        return plaintext

    half_block_size = (block_size + 1) // 2
    N = 1 << half_block_size
    upper, lower = plaintext >> half_block_size, plaintext % (N)
    gen = np.random.default_rng(key + round_num + upper)
    lower = lower ^ gen.integers(N)
    upper, lower = lower, upper
    return encrypt_round(key, round_num - 1, (upper << half_block_size) ^ lower, block_size)


def decrypt_round(key, round_num, ciphertext, block_size, num_rounds):
    if round_num > num_rounds:
        return ciphertext

    half_block_size = (block_size + 1) // 2
    N = 1 << half_block_size
    upper, lower = ciphertext >> half_block_size, ciphertext % (N)
    gen = np.random.default_rng(key + round_num + lower)
    upper = upper ^ gen.integers(N)
    upper, lower = lower, upper
    return decrypt_round(key, round_num + 1, (upper << half_block_size) ^ lower, block_size, num_rounds)


def encrypt(key: int, value: int, num_possible_values: int):
    num_rounds = 4
    block_size = int(np.ceil(np.log2(num_possible_values)))
    ciphertext = encrypt_round(key, num_rounds, value, block_size)
    if ciphertext < num_possible_values:
        return ciphertext
    return encrypt(key, ciphertext, num_possible_values)


def decrypt(key: int, value: int, num_possible_values: int) -> int:
    num_rounds = 4
    block_size = int(np.ceil(np.log2(num_possible_values)))
    plaintext = decrypt_round(key, 1, value, block_size, num_rounds)
    if plaintext < num_possible_values:
        return plaintext
    return decrypt(key, plaintext, num_possible_values)
