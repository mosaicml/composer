import itertools

import numpy as np
import torch

from composer.algorithms.factorize import FactorizedConv2d


def _default_input_and_weight_shapes():
    batch_sizes = [1, 2]
    hs = [5]
    ws = [6]
    C_ins = [4, 8]
    C_outs = [4, 8]
    kernel_sizes = [(1, 1), (2, 2), (3, 3), (1, 3), (3, 1), (5, 5)]
    return itertools.product(batch_sizes, hs, ws, C_ins, C_outs, kernel_sizes)


def _test_shapes(batch_size, h, w, C_in, C_out, kernel_size, **kwargs):
    X = torch.randn(batch_size, C_in, h, w)
    conv = FactorizedConv2d(in_channels=C_in, out_channels=C_out, kernel_size=kernel_size, **kwargs)
    Y = conv(X)

    assert len(Y.shape) == 4
    assert np.allclose(Y.shape[:2], torch.Tensor([batch_size, C_out]))


def test_init_and_forward_run():
    for args in _default_input_and_weight_shapes():
        _test_shapes(*args)


def test_update_layer_twice():
    batch_size = 2
    h = 5
    w = 6
    C_in = 32
    C_out = 32
    C_latent = 32
    X = torch.randn(batch_size, C_in, h, w)
    kernel_size = (3, 3)
    module = FactorizedConv2d(in_channels=C_in,
                                    out_channels=C_out,
                                    latent_channels=C_latent,
                                    kernel_size=kernel_size,
                                    padding=0)
    assert module.conv1 is None  # initially not factorized

    def _check_conv_shapes(module: FactorizedConv2d, C_in, C_out, C_latent):
        assert module.latent_channels == C_latent
        assert module.conv0.in_channels == C_in
        assert module.conv0.out_channels == C_latent
        assert module.conv1 is not None
        assert module.conv1.in_channels == C_latent
        assert module.conv1.out_channels == C_out

    module.set_rank(X, 24)
    _check_conv_shapes(module, C_in=C_in, C_out=C_out, C_latent=24)
    module.set_rank(X, 16)
    _check_conv_shapes(module, C_in=C_in, C_out=C_out, C_latent=16)


# def test_no_factorize():
#     hparams = fconv.default_hparams()
#     hparams['strategy'] = 'init'
#     hparams['latent_channels'] = 1  # 1 as in "100%", so no factorization
#     test_utils.test_algorithms([fconv.Algorithm], hparams)


# def test_factorize_at_start():
#     hparams = fconv.default_hparams()
#     hparams['strategy'] = 'init'
#     hparams['latent_channels'] = .5  # factorize by a factor of 2
#     hparams['min_channels'] = 0  # default is to not factorize tiny layers
#     test_utils.test_algorithms([fconv.Algorithm], hparams)


# def _fconv_gradual_hparams():
#     hparams = {}
#     hparams['strategy'] = 'gradual'
#     hparams['nmse_threshold'] = .5  # large err tolerance to ensure rank reduction
#     hparams['min_channels'] = 0  # default is to not factorize tiny layers
#     hparams['rank_multiple_of'] = 4  # need small granularity for tiny layers
#     hparams['n_speedup_steps'] = 2  # factorize twice
#     hparams['start_step'] = 1
#     hparams['end_step'] = 3
#     # prepend 'factconv_' to get the hparam names right
#     hparams = {f'{fconv._hparam_prefix}{k}': v for k, v in hparams.items()}
#     # fill in other mandatory hparams
#     full_hparams = fconv.default_hparams()
#     full_hparams.update(hparams)
#     return full_hparams


# # TODO there are many other ways a schedule can be invalid;
# # for now, just check a couple simple cases to sanity
# # check our hparam plumbing and validation


# def test_end_before_start_throws():
#     hparams = _fconv_gradual_hparams()
#     hparams['factconv_start_step'] = 2
#     hparams['factconv_end_step'] = 1  # enforce start > end
#     with pytest.raises(ValueError):
#         test_utils.test_algorithms([fconv.Algorithm], hparams, n_steps=0)


# def test_start_end_too_close_throws():
#     hparams = _fconv_gradual_hparams()
#     hparams['factconv_start_step'] = 2
#     hparams['factconv_end_step'] = 3
#     hparams['factconv_speedup_step_spacing'] = 3  # can't fit between start and end
#     hparams['factconv_n_speedup_steps'] = 2  # 1 step can fit in any interval
#     with pytest.raises(ValueError):
#         test_utils.test_algorithms([fconv.Algorithm], hparams, n_steps=0)


# def test_fractional_start_end():
#     hparams = _fconv_gradual_hparams()
#     hparams['factconv_start_step'] = .4
#     hparams['factconv_end_step'] = .7
#     test_utils.test_algorithms([fconv.Algorithm], hparams, n_steps=6, check_steps=[5, 6])


# def test_gradual():
#     hparams = _fconv_gradual_hparams()
#     test_utils.test_algorithms([fconv.Algorithm], hparams, n_steps=6, check_steps=[5, 6])


# def test_cifar10_resnet():
#     hparams = fconv.default_hparams()
#     hparams['factconv_strategy'] = 'gradual'
#     hparams['factconv_latent_channels'] = .5
#     hparams['factconv_min_channels'] = 64  # 64 is largest channel count in resnet56
#     hparams['factconv_start_step'] = 0
#     hparams['factconv_n_speedup_steps'] = 1
#     test_utils.test_algorithms([fconv.Algorithm], hparams, model_name='resnet56')
