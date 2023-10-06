# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Utilities for planning checkpoint loading and saving."""

from __future__ import annotations

from typing import Any

import torch

from composer.utils import dist, using_torch_2


def _get_module_name_mapping(model: torch.nn.Module) -> dict[str, str]:
    """Gets the mapping from the module name to the name with the process group.

    Iterates through the modules which are part of a process group and may be
    FSDP wrapped to create a mapping from the original name to the name with
    its process group index.

    Args:
        model: A torch.nn.Module.

    Returns:
        A mapping from a module name to the name with its process group.
    """
    module_name_mapping = {}
    world_size = dist.get_world_size()
    for module_name, module in model.named_modules():
        if hasattr(module, 'process_group'):
            process_group = module.process_group
            process_group_size = torch.distributed.get_world_size(process_group)
            if process_group_size != world_size:
                custom_process_group_size = world_size // process_group_size
                process_group_index = (
                    dist.get_global_rank() % custom_process_group_size
                )
                new_module_name = module_name.replace(
                    '_fsdp_wrapped_module.', ''
                )
                for k in module.state_dict().keys():
                    full_module_name = '.'.join(filter(None, (new_module_name, k)))
                    module_name_mapping[full_module_name] = (
                        full_module_name + f'_pgidx{process_group_index}'
                    )
                    print(f'{full_module_name=}')
        # print(f'{module_name=}')
        # new_module_name = module_name.replace(
        #     '_fsdp_wrapped_module.', ''
        # )
        # for k in module.state_dict().keys():
        #     full_module_name = '.'.join(filter(None, (new_module_name, k)))
        #     module_name_mapping[full_module_name] = (
        #         full_module_name + f'_pgidx{dist.get_global_rank()}'
        #     )
        #     print(f'{full_module_name=}')
    return module_name_mapping


def _rename_model_state_dict(
    model_state_dict: STATE_DICT_TYPE, module_name_mapping: dict[str, str]
) -> STATE_DICT_TYPE:
    """Renames a state dict according to the name mapping.

    Iterates through the model's state dict and renames the module according to
    the mapping.

    Args:
        model_state_dict: A state dict.

    Returns:
        A state dict with the modules renamed.
    """
    modified_state_dict = {}
    for k, v in model_state_dict.items():
        if k in module_name_mapping.keys():
            modified_state_dict[module_name_mapping[k]] = v
        else:
            modified_state_dict[k] = v
    print(f'{module_name_mapping.items()=}')
    print(f'{modified_state_dict.keys()=}')

    return modified_state_dict


def _rename_optimizers_state_dict(
    optimizers_state_dict: dict[str, dict[str, dict[str, Any]]],
    module_name_mapping: dict[str, str],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Renames an optimizer state dict according to the name mapping.

    Iterates through the optimizer's state dict and renames the module according
    to the mapping.

    Args:
        optimizers_state_dict: A state dict containing the optimizer state for
            model's parameters.

    Returns:
        An optimizer state dict with the modules renamed.
    """
    optimizers = {}
    for optimizer in optimizers_state_dict.keys():
        optimizers[optimizer] = optimizers_state_dict[optimizer]
        renamed_optimizers = {}
        for k, v in optimizers_state_dict[optimizer]['state'].items():
            replace_key = k
            if k in module_name_mapping.keys():
                replace_key = module_name_mapping[k]
            renamed_optimizers[replace_key] = v
        optimizers[optimizer]['state'] = renamed_optimizers

    return optimizers


if using_torch_2():
    from torch.distributed.checkpoint.default_planner import (
        DefaultLoadPlanner,
        DefaultSavePlanner,
    )
    from torch.distributed.checkpoint.metadata import Metadata, STATE_DICT_TYPE
    from torch.distributed.checkpoint._nested_dict import flatten_state_dict
    from torch.distributed.checkpoint._sharded_tensor_utils import (
        _flatten_sharded_tensors,
    )

    class RankLoadPlanner(DefaultLoadPlanner):
        """
        RankLoadPlanner extends __init__ and overrides set_up_planner to
        rename modules that are part of a custom process group.
        """

        def __init__(
            self,
            model: torch.nn.Module,
            flatten_state_dict: bool = True,
            flatten_sharded_tensors: bool = True,
        ) -> None:
            """Initializes RankLoadPlanner and sets the module mapping.

            The module mapping appends the process group index to each module
            name.

            Args:
                model: A torch.nn.Module.
                flatten_state_dict: See parent class.
                flatten_sharded_tensors: See parent class.
            """
            self.name_conversion_dict = _get_module_name_mapping(model)
            super().__init__(flatten_state_dict, flatten_sharded_tensors)

        def set_up_planner(
            self,
            state_dict: STATE_DICT_TYPE,
            metadata: Metadata,
            is_coordinator: bool,
        ) -> None:
            """Renames the state dict.

            The rest of the function follows the parent class.

            Args:
                state_dict: See parent class.
                metadata: See parent class.
                is_coordinator: See parent class.
            """
            print(f'{state_dict.keys()=}')
            if 'state' not in state_dict:
                super().set_up_planner(state_dict, metadata, is_coordinator)
                return

            self.original_state_dict = state_dict

            state_dict = {
                'state': {
                    'model': {
                        k: v for k, v in state_dict['state']['model'].items()
                    },
                },
            }

            if self.name_conversion_dict:
                model_state_dict = _rename_model_state_dict(
                    state_dict['state']['model'], self.name_conversion_dict
                )
                state_dict['state']['model'] = model_state_dict

            if self.flatten_sharded_tensors:
                state_dict = _flatten_sharded_tensors(state_dict)

            if self.flatten_state_dict:
                state_dict, self.mappings = flatten_state_dict(state_dict)

            self.state_dict = state_dict
            self.metadata = metadata
            self.is_coordinator = is_coordinator

    class RankSavePlanner(DefaultSavePlanner):
        """
        RankSavePlanner extends __init__ and set_up_planner to rename modules
        that are part of a custom process group.
        """

        def __init__(
            self,
            model: torch.nn.Module,
            flatten_state_dict: bool = True,
            flatten_sharded_tensors: bool = True,
            dedup_replicated_tensors: bool = True,
        ) -> None:
            """Initializes RankSavePlanner and sets the module mapping.

            The module mapping appends the process group index to each module
            name.

            Args:
                model: A torch.nn.Module.
                flatten_state_dict: See parent class.
                flatten_sharded_tensors: See parent class.
                dedup_replicated_tensors: See parent class.
            """
            self.name_conversion_dict = _get_module_name_mapping(model)
            super().__init__(
                flatten_state_dict,
                flatten_sharded_tensors,
                dedup_replicated_tensors,
            )

        def set_up_planner(
            self, state_dict: STATE_DICT_TYPE, is_coordinator: bool
        ) -> None:
            """Renames the state dict and optimizer state dict.

            Args:
                state_dict: See parent class.
                is_coordinator: See parent class.
            """
            if self.name_conversion_dict:
                model_state_dict = _rename_model_state_dict(
                    state_dict['state']['model'], self.name_conversion_dict
                )
                state_dict['state']['model'] = model_state_dict

                if 'optimizers' in state_dict.keys():
                    optimizers = _rename_optimizers_state_dict(
                        state_dict['optimizers'], self.name_conversion_dict
                    )
                    state_dict['optimizers'] = optimizers

            super().set_up_planner(state_dict, is_coordinator)
