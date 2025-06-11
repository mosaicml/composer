# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from composer.utils.parallelism import FSDP2Config


def test_fsdp2_config():
    """Test that FSDP2Config read-only properties work as expected."""
    # Create a config instance
    config = FSDP2Config()

    # Test reading properties (should succeed)
    assert config.load_monolith_rank0_only is False
    assert config.sync_module_states is False
    assert config.activation_cpu_offload is False
    assert config.data_parallel_shard_degree == -1
    assert config.data_parallel_replicate_degree is None
    assert config.state_dict_type == 'sharded'
    assert config.use_orig_params is True
    assert config.load_monolith_rank0_only is False

    # Test setting properties (should fail)
    read_only_props = [
        ('data_parallel_shard_degree', 2),
        ('data_parallel_replicate_degree', 2),
        ('use_orig_params', False),
    ]

    for prop, value in read_only_props:
        with pytest.raises(AttributeError):
            setattr(config, prop, value)

    # Test that core properties can be set
    config.device_mesh = None
    config.reshard_after_forward = False
    assert config.device_mesh is None
    assert config.reshard_after_forward is False


def test_fsdp2config_from_fsdp1_valid_attributes():
    """Test FSDP2Config.from_compatible_attrs with valid attributes."""
    valid_attrs = {
        'activation_checkpointing': True,
        'activation_cpu_offload': True,
        'reshard_after_forward': False,
        'device_mesh': None,
    }

    fsdp2_config = FSDP2Config.from_compatible_attrs(valid_attrs)
    assert fsdp2_config.activation_checkpointing is True
    assert fsdp2_config.activation_cpu_offload is True
    assert fsdp2_config.reshard_after_forward is False
    assert fsdp2_config.device_mesh is None


def test_fsdp2config_from_empty_attributes():
    """Test FSDP2Config.from_compatible_attrs with empty attributes."""
    empty_attrs = {}

    fsdp2_config = FSDP2Config.from_compatible_attrs(empty_attrs)
    assert fsdp2_config.activation_checkpointing is False  # default value
    assert fsdp2_config.activation_cpu_offload is False  # default value
    assert fsdp2_config.reshard_after_forward is True  # default value
    assert fsdp2_config.device_mesh is None  # default value
    assert fsdp2_config.sync_module_states is False  # default value


def test_fsdp2config_from_fsdp1_multiple_invalid_attributes():
    """Test FSDP2Config.from_compatible_attrs with multiple invalid attributes issues warnings for each."""
    mixed_attrs = {
        'activation_checkpointing': True,
        'invalid_attribute1': 'value1',
        'invalid_attribute2': 'value2',
        'auto_wrap': True,
        'sync_module_states': True,
    }

    with pytest.warns() as record:
        fsdp2_config = FSDP2Config.from_compatible_attrs(mixed_attrs)
        assert fsdp2_config.activation_checkpointing is True

    # Check that we got warnings for each invalid attribute
    warning_messages = [str(w.message) for w in record]
    assert any('invalid_attribute1: value1' in msg for msg in warning_messages)
    assert any('invalid_attribute2: value2' in msg for msg in warning_messages)
    assert any('auto_wrap: True' in msg for msg in warning_messages)
    assert any('sync_module_states: True' in msg for msg in warning_messages)


def test_fsdp2_config_monolithic_validation():
    """Test FSDP2Config validation for monolithic checkpointing."""
    # Test valid monolithic config
    config = FSDP2Config(
        state_dict_type='full',
        load_monolith_rank0_only=True,
    )
    assert config.state_dict_type == 'full'
    assert config.load_monolith_rank0_only is True

    # Test invalid monolithic config
    with pytest.raises(ValueError, match='load_monolith_rank0_only=True requires state_dict_type="full"'):
        FSDP2Config(
            state_dict_type='sharded',
            load_monolith_rank0_only=True,
        )
