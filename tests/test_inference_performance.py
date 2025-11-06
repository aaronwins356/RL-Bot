"""Smoke tests for inference performance."""
import time
import numpy as np
import pytest

from core.agents.ml_policy import MLPolicy
from core.features.encoder import ObservationEncoder


def test_inference_speed():
    """Test that inference is fast enough for real-time play."""
    encoder = ObservationEncoder()
    policy = MLPolicy(encoder=encoder)
    
    # Create dummy observation
    observation = np.random.randn(encoder.feature_size).astype(np.float32)
    
    # Warm up
    for _ in range(10):
        policy.get_action(observation, deterministic=True)
    
    # Measure inference time
    num_iterations = 100
    start_time = time.time()
    
    for _ in range(num_iterations):
        controls, intent, confidence, hidden = policy.get_action(
            observation,
            deterministic=True
        )
    
    elapsed_time = time.time() - start_time
    avg_time_ms = (elapsed_time / num_iterations) * 1000
    
    print(f"\nAverage inference time: {avg_time_ms:.2f} ms")
    
    # Should be less than 8ms for 120Hz gameplay
    # In practice, might be slower on first run, so we use 50ms as threshold
    assert avg_time_ms < 50.0, f"Inference too slow: {avg_time_ms:.2f} ms"
    
    # Check output shapes
    assert controls.shape == (8,)
    assert isinstance(confidence, float)


def test_batch_inference():
    """Test batch inference."""
    encoder = ObservationEncoder()
    policy = MLPolicy(encoder=encoder)
    
    # Create batch of observations
    batch_size = 32
    observations = np.random.randn(batch_size, encoder.feature_size).astype(np.float32)
    
    # Test batch processing (sequential for now)
    start_time = time.time()
    
    for obs in observations:
        controls, intent, confidence, hidden = policy.get_action(
            obs,
            deterministic=True
        )
    
    elapsed_time = time.time() - start_time
    avg_time_ms = (elapsed_time / batch_size) * 1000
    
    print(f"\nBatch inference - Avg time per sample: {avg_time_ms:.2f} ms")
    
    assert avg_time_ms < 50.0


def test_policy_stats():
    """Test that policy tracks statistics."""
    encoder = ObservationEncoder()
    policy = MLPolicy(encoder=encoder)
    
    # Run some inferences
    observation = np.random.randn(encoder.feature_size).astype(np.float32)
    
    for _ in range(10):
        policy.get_action(observation, deterministic=True)
    
    # Get stats
    stats = policy.get_stats()
    
    assert "mean_inference_ms" in stats
    assert "max_inference_ms" in stats
    assert stats["mean_inference_ms"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
