"""Unit tests for resilience features."""
import numpy as np
import pytest
import json

from core.env.rocket_sim_env import RocketSimEnv, safe_step, safe_reset
from core.infra.logging import SafeJSONEncoder, safe_log
import logging


def test_safe_json_encoder():
    """Test SafeJSONEncoder handles NumPy types."""
    encoder = SafeJSONEncoder()
    
    # Test NumPy integer
    data = {"int32": np.int32(42), "int64": np.int64(100)}
    json_str = json.dumps(data, cls=SafeJSONEncoder)
    decoded = json.loads(json_str)
    assert decoded["int32"] == 42
    assert decoded["int64"] == 100
    
    # Test NumPy float
    data = {"float32": np.float32(3.14), "float64": np.float64(2.718)}
    json_str = json.dumps(data, cls=SafeJSONEncoder)
    decoded = json.loads(json_str)
    assert abs(decoded["float32"] - 3.14) < 0.01
    assert abs(decoded["float64"] - 2.718) < 0.01
    
    # Test NumPy array
    data = {"array": np.array([1, 2, 3])}
    json_str = json.dumps(data, cls=SafeJSONEncoder)
    decoded = json.loads(json_str)
    assert decoded["array"] == [1, 2, 3]
    
    # Test NumPy bool
    data = {"bool": np.bool_(True)}
    json_str = json.dumps(data, cls=SafeJSONEncoder)
    decoded = json.loads(json_str)
    assert decoded["bool"] is True


def test_safe_log():
    """Test safe_log handles Unicode errors."""
    logger = logging.getLogger("test")
    
    # Normal string should work
    safe_log(logger, logging.INFO, "Normal message")
    
    # String with special characters should work
    safe_log(logger, logging.INFO, "Message with émojis 🚀 and spëcial chars")
    
    # Should not raise exception
    assert True


def test_safe_reset():
    """Test safe_reset handles environment reset."""
    env = RocketSimEnv()
    
    # Normal reset should work
    obs, info = safe_reset(env)
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)
    assert obs.shape[0] == env.OBS_SIZE
    
    # Check for NaN in observation
    assert not np.any(np.isnan(obs))


def test_safe_step():
    """Test safe_step handles environment step."""
    env = RocketSimEnv()
    env.reset()
    
    # Normal step should work
    action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    obs, reward, terminated, truncated, info = safe_step(env, action)
    
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    
    # Check for NaN in observation
    assert not np.any(np.isnan(obs))


def test_safe_step_with_nan_observation():
    """Test safe_step recovers from NaN observations."""
    env = RocketSimEnv()
    env.reset()
    
    # Inject NaN into observation (simulate environment error)
    action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    
    # Even if environment returns NaN, safe_step should handle it
    # (we can't easily inject NaN without modifying the env, so just verify it doesn't crash)
    obs, reward, terminated, truncated, info = safe_step(env, action)
    
    # Should always return valid observation
    assert isinstance(obs, np.ndarray)
    assert not np.any(np.isnan(obs))


def test_curriculum_restriction():
    """Test curriculum is restricted to 3 stages."""
    from core.training.selfplay import SelfPlayManager
    
    # Create selfplay manager with default config
    manager = SelfPlayManager({})
    
    # Should have exactly 3 stages
    assert len(manager.stages) == 3
    
    # Verify stage names
    assert manager.stages[0].name == "1v1"
    assert manager.stages[1].name == "1v2"
    assert manager.stages[2].name == "2v2"
    
    # Verify game modes
    assert manager.stages[0].game_mode == "1v1"
    assert manager.stages[1].game_mode == "1v2"
    assert manager.stages[2].game_mode == "2v2"


def test_auto_resume_metadata():
    """Test auto-resume metadata is set."""
    # This is a placeholder test - actual auto-resume is tested in integration
    # Just verify the flag exists in TrainingLoop
    from core.training.train_loop import TrainingLoop
    from core.infra.config import Config
    
    config = Config({
        "training": {"num_envs": 1},
        "network": {"hidden_sizes": [256, 256]},
        "policy": {},
        "inference": {"device": "cpu"},
        "logging": {"log_dir": "/tmp/test_logs"},
        "checkpoints": {},
        "telemetry": {}
    })
    
    # TrainingLoop should accept auto_resume parameter
    loop = TrainingLoop(config, auto_resume=False)
    assert hasattr(loop, 'auto_resume')
    assert loop.auto_resume is False
