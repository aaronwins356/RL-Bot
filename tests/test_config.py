"""Tests for configuration system."""
import pytest
from pathlib import Path
import yaml
import tempfile

from core.infra.config import Config, ConfigManager, load_config


def test_config_dict_access():
    """Test dictionary-style access to config."""
    config_dict = {
        'training': {
            'algorithm': 'ppo',
            'total_timesteps': 1000000
        },
        'network': {
            'architecture': 'mlp'
        }
    }
    
    config = Config(config_dict)
    
    # Test dict-style access
    assert config['training']['algorithm'] == 'ppo'
    assert config['training']['total_timesteps'] == 1000000
    assert config['network']['architecture'] == 'mlp'


def test_config_dot_access():
    """Test dot-notation access to config."""
    config_dict = {
        'training': {
            'algorithm': 'ppo',
            'total_timesteps': 1000000
        },
        'network': {
            'architecture': 'mlp'
        }
    }
    
    config = Config(config_dict)
    
    # Test dot-notation access
    assert config.training.algorithm == 'ppo'
    assert config.training.total_timesteps == 1000000
    assert config.network.architecture == 'mlp'


def test_config_legacy_attributes():
    """Test legacy flat attributes for backward compatibility."""
    config_dict = {
        'training': {
            'algorithm': 'ppo',
            'total_timesteps': 1000000,
            'batch_size': 4096
        },
        'inference': {
            'device': 'cuda'
        }
    }
    
    config = Config(config_dict)
    
    # Test legacy attributes
    assert config.algorithm == 'ppo'
    assert config.total_timesteps == 1000000
    assert config.batch_size == 4096
    assert config.device == 'cuda'


def test_config_get_method():
    """Test get method with defaults."""
    config_dict = {
        'training': {
            'algorithm': 'ppo'
        }
    }
    
    config = Config(config_dict)
    
    assert config.get('training') == {'algorithm': 'ppo'}
    assert config.get('nonexistent', 'default') == 'default'


def test_config_to_dict():
    """Test converting config back to dict."""
    config_dict = {
        'training': {
            'algorithm': 'ppo'
        }
    }
    
    config = Config(config_dict)
    result = config.to_dict()
    
    assert result == config_dict
    assert result is not config_dict  # Should be a copy


def test_load_config_from_file():
    """Test loading config from YAML file."""
    # Create temp config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({
            'training': {'algorithm': 'ppo'},
            'network': {'architecture': 'mlp'},
            'policy': {'type': 'hybrid'},
            'inference': {'device': 'cpu'},
            'logging': {'log_dir': 'logs'}
        }, f)
        temp_path = Path(f.name)
    
    try:
        config = load_config(temp_path)
        assert config.training.algorithm == 'ppo'
        assert config['network']['architecture'] == 'mlp'
    finally:
        temp_path.unlink()


def test_load_config_with_overrides():
    """Test loading config with overrides."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({
            'training': {'algorithm': 'ppo', 'total_timesteps': 1000000},
            'network': {'architecture': 'mlp'},
            'policy': {'type': 'hybrid'},
            'inference': {'device': 'cpu'},
            'logging': {'log_dir': 'logs'}
        }, f)
        temp_path = Path(f.name)
    
    try:
        overrides = {
            'training': {
                'total_timesteps': 5000000
            }
        }
        config = load_config(temp_path, overrides)
        
        assert config.training.algorithm == 'ppo'
        assert config.training.total_timesteps == 5000000
    finally:
        temp_path.unlink()


def test_config_manager_load():
    """Test ConfigManager loading."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({
            'training': {'algorithm': 'ppo', 'total_timesteps': 1000000},
            'network': {'architecture': 'mlp'},
            'policy': {'type': 'hybrid'},
            'inference': {'device': 'cpu'},
            'logging': {'log_dir': 'logs'}
        }, f)
        temp_path = Path(f.name)
    
    try:
        manager = ConfigManager(temp_path)
        assert manager.config is not None
        assert manager.config.training.algorithm == 'ppo'
    finally:
        temp_path.unlink()


def test_config_manager_apply_overrides():
    """Test ConfigManager override application."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({
            'training': {'algorithm': 'ppo', 'total_timesteps': 1000000},
            'network': {'architecture': 'mlp'},
            'policy': {'type': 'hybrid'},
            'inference': {'device': 'cpu'},
            'logging': {'log_dir': 'logs'}
        }, f)
        temp_path = Path(f.name)
    
    try:
        manager = ConfigManager(temp_path)
        
        overrides = {
            'training': {'total_timesteps': 5000000},
            'inference': {'device': 'cuda'}
        }
        manager.apply_overrides(overrides)
        
        assert manager.config.training.total_timesteps == 5000000
        assert manager.config.inference.device == 'cuda'
        assert manager.config.training.algorithm == 'ppo'  # Unchanged
    finally:
        temp_path.unlink()


def test_config_manager_validate_schema():
    """Test ConfigManager schema validation."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({
            'training': {'algorithm': 'ppo', 'total_timesteps': 1000000},
            'network': {'architecture': 'mlp'},
            'policy': {'type': 'hybrid'},
            'inference': {'device': 'cpu'},
            'logging': {'log_dir': 'logs'}
        }, f)
        temp_path = Path(f.name)
    
    try:
        manager = ConfigManager(temp_path)
        assert manager.validate_schema() is True
    finally:
        temp_path.unlink()


def test_config_manager_validate_schema_missing_keys():
    """Test ConfigManager schema validation with missing keys."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({
            'training': {'algorithm': 'ppo'}
            # Missing other required sections
        }, f)
        temp_path = Path(f.name)
    
    try:
        manager = ConfigManager(temp_path)
        with pytest.raises(ValueError, match="Missing required config keys"):
            manager.validate_schema()
    finally:
        temp_path.unlink()


def test_config_manager_get_safe():
    """Test ConfigManager safe nested access."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({
            'training': {'algorithm': 'ppo', 'total_timesteps': 1000000},
            'network': {'architecture': 'mlp'},
            'policy': {'type': 'hybrid'},
            'inference': {'device': 'cpu'},
            'logging': {'log_dir': 'logs'}
        }, f)
        temp_path = Path(f.name)
    
    try:
        manager = ConfigManager(temp_path)
        
        assert manager.get_safe('training.algorithm') == 'ppo'
        assert manager.get_safe('training.total_timesteps') == 1000000
        assert manager.get_safe('nonexistent.key', 'default') == 'default'
    finally:
        temp_path.unlink()


def test_base_yaml_schema():
    """Test that base.yaml has correct schema."""
    config_path = Path('configs/base.yaml')
    
    if not config_path.exists():
        pytest.skip("base.yaml not found")
    
    config = load_config(config_path)
    
    # Check required top-level keys
    assert 'training' in config.to_dict()
    assert 'network' in config.to_dict()
    assert 'policy' in config.to_dict()
    assert 'inference' in config.to_dict()
    assert 'logging' in config.to_dict()
    
    # Check required training keys
    assert config.training.algorithm in ['ppo', 'sac']
    assert config.training.total_timesteps > 0
    
    # Check optional offline training config
    if 'offline' in config.to_dict().get('training', {}):
        offline = config.training.offline
        assert 'enabled' in offline
        if offline.enabled:
            assert 'dataset_path' in offline
            assert 'pretrain_epochs' in offline
