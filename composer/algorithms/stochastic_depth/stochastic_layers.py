# Copyright 2021 MosaicML. All Rights Reserved.

import torch

from composer.models.resnets import Bottleneck


def _sample_bernoulli(probability: torch.Tensor,
                      device_id: int,
                      module_id: int,
                      num_modules: int,
                      generator: torch.Generator,
                      use_same_depth_across_gpus: bool,
                      use_same_gpu_seed: bool = False):
    """Gets a sample from a Bernoulli distribution. Provides functionality to
       have different seeds across GPUs and to have the same set of seeds across GPUs.
    """

    if use_same_gpu_seed:
        sample = torch.bernoulli(probability)
        return sample

    # Get a separate generator for each GPU
    rand_int = torch.randint(low=2**17, high=2**27, size=(1,), dtype=torch.long, generator=generator)
    gpu_seed = (rand_int * (device_id + 1))

    if use_same_depth_across_gpus:
        gpu_generator = torch.Generator().manual_seed(gpu_seed.item())  #type: ignore
        layer_seed = (torch.randperm(num_modules, generator=gpu_generator)[module_id] + 1) * rand_int
    else:
        layer_seed = (module_id + 1) * gpu_seed
    layer_generator = torch.Generator().manual_seed(layer_seed.item())  # type: ignore

    sample = torch.bernoulli(probability, generator=layer_generator)
    return sample


class StochasticBottleneck(Bottleneck):
    """Stochastic ResNet Bottleneck layer. This layer has a probability of skipping
       the transformation section of the layer and scales the transformation section
       output by (1 - drop probability) during inference.

       Args:
            drop_rate (float): Probability of dropping the layer. Must be
                                     between 0.0 and 1.0.
            module_id (int): The placement of the layer within a network e.g. 0
                             for the first layer in the network.
            module_count (int): The total number of layers of this type in the
                                network.
            use_same_gpu_seed (bool): Set to true to have the same layers dropped
                                      across GPUs when using multi-GPU training.
                                      Set to false to have each GPU drop a different
                                      set of layers.
            use_same_depth_across_gpus (bool): Set to true to have the same number
                                               of layers dropped across GPUs. Set
                                               to true when `drop_distribution`
                                               is 'uniform' and set to false for
                                               'linear'.

       """

    def __init__(self, drop_rate: float, module_id: int, module_count: int, use_same_gpu_seed: bool,
                 use_same_depth_across_gpus: bool, rand_generator: torch.Generator, **kwargs):
        super(StochasticBottleneck, self).__init__(**kwargs)
        self.drop_rate = torch.tensor(drop_rate)
        self.module_id = module_id
        self.module_count = module_count
        self.use_same_gpu_seed = use_same_gpu_seed
        self.use_same_depth_across_gpus = use_same_depth_across_gpus
        self.rand_generator = rand_generator

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        sample = _sample_bernoulli(probability=(1 - self.drop_rate),
                                   device_id=x.get_device(),
                                   module_id=self.module_id,
                                   num_modules=self.module_count,
                                   generator=self.rand_generator,
                                   use_same_gpu_seed=self.use_same_gpu_seed,
                                   use_same_depth_across_gpus=self.use_same_depth_across_gpus)

        if not self.training or sample:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if not self.training:
                out = out * (1 - self.drop_rate)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)
        else:
            if self.downsample is not None:
                out = self.relu(self.downsample(x))
            else:
                out = identity
        return out

    @staticmethod
    def from_target_layer(module: Bottleneck,
                          module_index: int,
                          module_count: int,
                          drop_rate: float,
                          drop_distribution: str,
                          rand_generator: torch.Generator,
                          use_same_gpu_seed: bool = False):
        """Helper function to convert a ResNet bottleneck block into a stochastic block.
        """
        if drop_distribution == 'linear':
            drop_rate = ((module_index + 1) / module_count) * drop_rate
        use_same_depth_across_gpus = (drop_distribution == 'uniform')
        rand_generator = torch.Generator().manual_seed(
            rand_generator.initial_seed())  # copy the generator for each layer
        return StochasticBottleneck(drop_rate=drop_rate,
                                    module_id=module_index,
                                    module_count=module_count,
                                    use_same_depth_across_gpus=use_same_depth_across_gpus,
                                    use_same_gpu_seed=use_same_gpu_seed,
                                    rand_generator=rand_generator,
                                    inplanes=module.conv1.in_channels,
                                    planes=module.conv3.out_channels // module.expansion,
                                    stride=module.stride,
                                    downsample=module.downsample,
                                    groups=module.conv2.groups,
                                    dilation=module.conv2.dilation)
