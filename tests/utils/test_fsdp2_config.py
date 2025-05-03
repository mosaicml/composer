import pytest

from composer.utils.parallelism import FSDP2Config


def test_fsdp2config_from_fsdp1_valid_attributes():
    """Test FSDP2Config.from_fsdp1 with valid attributes."""
    valid_attrs = {
        "activation_checkpointing": True,
        "activation_cpu_offload": True,
        "reshard_after_forward": False,
        "device_mesh": None
    }
    
    fsdp2_config = FSDP2Config.from_fsdp1(valid_attrs)
    assert fsdp2_config.activation_checkpointing is True
    assert fsdp2_config.activation_cpu_offload is True
    assert fsdp2_config.reshard_after_forward is False
    assert fsdp2_config.device_mesh is None


def test_fsdp2config_from_fsdp1_invalid_attribute():
    """Test FSDP2Config.from_fsdp1 with invalid attributes issues a warning."""
    invalid_attrs = {
        "activation_checkpointing": True,
        "invalid_attribute": "value"
    }
    
    with pytest.warns(UserWarning, match=r"Attribute 'invalid_attribute: value' is not a settable attribute"):
        fsdp2_config = FSDP2Config.from_fsdp1(invalid_attrs)
        assert fsdp2_config.activation_checkpointing is True


def test_fsdp2config_from_fsdp1_empty_attributes():
    """Test FSDP2Config.from_fsdp1 with empty attributes."""
    empty_attrs = {}
    
    fsdp2_config = FSDP2Config.from_fsdp1(empty_attrs)
    assert fsdp2_config.activation_checkpointing is False  # default value
    assert fsdp2_config.activation_cpu_offload is False    # default value
    assert fsdp2_config.reshard_after_forward is True      # default value
    assert fsdp2_config.device_mesh is None                # default value


def test_fsdp2config_from_fsdp1_property_attribute():
    """Test FSDP2Config.from_fsdp1 with property attributes issues a warning."""
    property_attrs = {
        "activation_checkpointing": True,
        "auto_wrap": True  # This is a property, not a settable attribute
    }
    
    with pytest.warns(UserWarning, match=r"Attribute 'auto_wrap: True' is not a settable attribute"):
        fsdp2_config = FSDP2Config.from_fsdp1(property_attrs)
        assert fsdp2_config.activation_checkpointing is True


def test_fsdp2config_from_fsdp1_multiple_invalid_attributes():
    """Test FSDP2Config.from_fsdp1 with multiple invalid attributes issues warnings for each."""
    mixed_attrs = {
        "activation_checkpointing": True,
        "invalid_attribute1": "value1",
        "invalid_attribute2": "value2",
        "auto_wrap": True
    }
    
    with pytest.warns() as record:
        fsdp2_config = FSDP2Config.from_fsdp1(mixed_attrs)
        assert fsdp2_config.activation_checkpointing is True
    
    # Check that we got warnings for each invalid attribute
    warning_messages = [str(w.message) for w in record]
    assert any("invalid_attribute1: value1" in msg for msg in warning_messages)
    assert any("invalid_attribute2: value2" in msg for msg in warning_messages)
    assert any("auto_wrap: True" in msg for msg in warning_messages)
    assert len(record) == 3  # Three warnings for three invalid attributes 