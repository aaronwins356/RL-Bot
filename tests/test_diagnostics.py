"""Unit tests for performance diagnostics."""
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.diagnose_performance import PerformanceDiagnostics


def test_diagnostics_initialization():
    """Test that diagnostics can be initialized."""
    config_path = Path("configs/base.yaml")
    if not config_path.exists():
        pytest.skip("Config file not found")
    
    diagnostics = PerformanceDiagnostics(config_path, device="cpu")
    assert diagnostics.device.type == "cpu"
    assert diagnostics.perf_monitor is not None


def test_benchmark_model_inference():
    """Test model inference benchmark."""
    config_path = Path("configs/base.yaml")
    if not config_path.exists():
        pytest.skip("Config file not found")
    
    diagnostics = PerformanceDiagnostics(config_path, device="cpu")
    
    # Run quick benchmark
    results = diagnostics.benchmark_model_inference(num_inferences=10, batch_size=4)
    
    assert "total_inferences" in results
    assert "mean_ms" in results
    assert "inferences_per_sec" in results
    assert results["total_inferences"] == 10
    assert results["mean_ms"] > 0
    assert results["inferences_per_sec"] > 0


def test_optimization_config_loaded():
    """Test that optimization config is properly structured."""
    import yaml
    
    config_path = Path("configs/base.yaml")
    if not config_path.exists():
        pytest.skip("Config file not found")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    assert "training" in config
    assert "optimizations" in config["training"]
    
    opt = config["training"]["optimizations"]
    assert "use_subproc_vec_env" in opt
    assert "use_amp" in opt
    assert "use_torch_compile" in opt
    assert "batch_inference" in opt
    assert "action_repeat" in opt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
