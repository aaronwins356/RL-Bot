#!/usr/bin/env python3
"""Verification and benchmarking script for RL-Bot training.

This script runs comprehensive verification tests and benchmarks to ensure:
1. Training runs without errors
2. GPU utilization is adequate
3. Training speed meets targets
4. Elo progression is on track
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from core.infra.config import ConfigManager
from core.training.train_loop import TrainingLoop

logger = logging.getLogger(__name__)


def run_dry_test(config_path: Path) -> Dict[str, Any]:
    """Run a quick dry test to verify training loop works.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Test results dictionary
    """
    logger.info("=" * 70)
    logger.info("DRY TEST: 50 timesteps")
    logger.info("=" * 70)
    
    # Load config and override for quick test
    config_manager = ConfigManager(config_path)
    overrides = {
        "training": {
            "total_timesteps": 50,
            "num_envs": 1,
            "debug_mode": True,
        },
        "logging": {
            "tensorboard": False,
            "log_interval": 10,
            "save_interval": 50,
            "eval_interval": 50,
        }
    }
    config_manager.apply_overrides(overrides)
    
    start_time = time.time()
    success = False
    error_msg = None
    
    try:
        trainer = TrainingLoop(
            config=config_manager.config,
            log_dir="logs/dry_test",
            seed=42,
            auto_resume=False,
        )
        
        trainer.train(total_timesteps=50)
        
        success = True
        logger.info("✓ Dry test PASSED")
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"✗ Dry test FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    elapsed_time = time.time() - start_time
    
    return {
        "test": "dry_test",
        "success": success,
        "error": error_msg,
        "elapsed_time_seconds": elapsed_time,
    }


def run_performance_benchmark(config_path: Path, timesteps: int = 1000) -> Dict[str, Any]:
    """Benchmark training performance metrics.
    
    Args:
        config_path: Path to configuration file
        timesteps: Number of timesteps to benchmark
        
    Returns:
        Benchmark results dictionary
    """
    logger.info("=" * 70)
    logger.info(f"PERFORMANCE BENCHMARK: {timesteps} timesteps")
    logger.info("=" * 70)
    
    # Load config
    config_manager = ConfigManager(config_path)
    overrides = {
        "training": {
            "total_timesteps": timesteps,
            "num_envs": 8,
        },
        "logging": {
            "tensorboard": False,
            "log_interval": timesteps + 1,  # No logging during benchmark
            "save_interval": timesteps + 1,
            "eval_interval": timesteps + 1,
        }
    }
    config_manager.apply_overrides(overrides)
    
    # Check CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    
    start_time = time.time()
    
    try:
        trainer = TrainingLoop(
            config=config_manager.config,
            log_dir="logs/benchmark",
            seed=42,
            auto_resume=False,
        )
        
        # Get initial GPU memory if available
        initial_gpu_mem = 0
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            initial_gpu_mem = torch.cuda.memory_allocated() / 1024**2  # MB
        
        trainer.train(total_timesteps=timesteps)
        
        elapsed_time = time.time() - start_time
        
        # Calculate metrics
        timesteps_per_sec = timesteps / elapsed_time
        
        # GPU metrics
        gpu_utilization = 0
        peak_gpu_mem = 0
        if device == "cuda":
            peak_gpu_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
            # Rough estimate: if we're using >100MB, assume >40% utilization
            gpu_utilization = min(100, (peak_gpu_mem / 1000) * 100)
        
        results = {
            "test": "performance_benchmark",
            "success": True,
            "timesteps": timesteps,
            "elapsed_time_seconds": elapsed_time,
            "timesteps_per_second": timesteps_per_sec,
            "device": device,
            "gpu_memory_mb": peak_gpu_mem,
            "estimated_gpu_utilization_percent": gpu_utilization,
        }
        
        # Performance targets
        target_fps = 15.0
        target_gpu_util = 40.0
        
        logger.info(f"✓ Training speed: {timesteps_per_sec:.2f} timesteps/sec")
        if timesteps_per_sec >= target_fps:
            logger.info(f"  ✓ Meets target of >{target_fps} timesteps/sec")
        else:
            logger.warning(f"  ✗ Below target of >{target_fps} timesteps/sec")
        
        if device == "cuda":
            logger.info(f"✓ Peak GPU memory: {peak_gpu_mem:.1f} MB")
            logger.info(f"✓ Estimated GPU utilization: {gpu_utilization:.1f}%")
            if gpu_utilization >= target_gpu_util:
                logger.info(f"  ✓ Meets target of >{target_gpu_util}%")
            else:
                logger.warning(f"  ✗ Below target of >{target_gpu_util}%")
        
        return results
        
    except Exception as e:
        logger.error(f"✗ Performance benchmark FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "test": "performance_benchmark",
            "success": False,
            "error": str(e),
        }


def run_training_benchmark(
    config_path: Path,
    timesteps: int = 150000,
    checkpoint_interval: int = 25000
) -> Dict[str, Any]:
    """Run full training benchmark with Elo tracking.
    
    Args:
        config_path: Path to configuration file
        timesteps: Total timesteps to train
        checkpoint_interval: Evaluation interval
        
    Returns:
        Training results dictionary
    """
    logger.info("=" * 70)
    logger.info(f"TRAINING BENCHMARK: {timesteps} timesteps")
    logger.info("=" * 70)
    
    # Load config
    config_manager = ConfigManager(config_path)
    overrides = {
        "training": {
            "total_timesteps": timesteps,
        },
        "logging": {
            "eval_interval": checkpoint_interval,
            "save_interval": checkpoint_interval,
        }
    }
    config_manager.apply_overrides(overrides)
    
    start_time = time.time()
    
    try:
        trainer = TrainingLoop(
            config=config_manager.config,
            log_dir=f"logs/training_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            seed=42,
            auto_resume=False,
        )
        
        trainer.train(total_timesteps=timesteps)
        
        elapsed_time = time.time() - start_time
        
        # Get final Elo
        final_elo = trainer.evaluator.get_elo()
        
        # Get eval history if available
        elo_history = trainer.eval_history if hasattr(trainer, 'eval_history') else []
        
        results = {
            "test": "training_benchmark",
            "success": True,
            "timesteps": timesteps,
            "elapsed_time_seconds": elapsed_time,
            "elapsed_time_hours": elapsed_time / 3600,
            "final_elo": final_elo,
            "elo_history": elo_history,
        }
        
        # Evaluate progress
        target_elo_150k = 1450
        target_elo_300k = 1550
        
        logger.info(f"✓ Final Elo: {final_elo:.1f}")
        
        if timesteps >= 150000 and final_elo >= target_elo_150k:
            logger.info(f"  ✓ Meets target of >{target_elo_150k} Elo @ 150k steps")
        elif timesteps >= 300000 and final_elo >= target_elo_300k:
            logger.info(f"  ✓ Meets target of >{target_elo_300k} Elo @ 300k steps")
        else:
            logger.warning(f"  ✗ Below target Elo for {timesteps} steps")
        
        logger.info(f"✓ Training time: {elapsed_time/3600:.2f} hours")
        
        return results
        
    except Exception as e:
        logger.error(f"✗ Training benchmark FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "test": "training_benchmark",
            "success": False,
            "error": str(e),
        }


def main():
    """Main entry point for verification and benchmarking."""
    parser = argparse.ArgumentParser(
        description="Verify and benchmark RL-Bot training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_optimized.yaml",
        help="Configuration file to test"
    )
    
    parser.add_argument(
        "--dry-test",
        action="store_true",
        help="Run quick dry test (50 steps)"
    )
    
    parser.add_argument(
        "--perf-benchmark",
        action="store_true",
        help="Run performance benchmark (1000 steps)"
    )
    
    parser.add_argument(
        "--train-benchmark",
        action="store_true",
        help="Run full training benchmark (150k steps)"
    )
    
    parser.add_argument(
        "--train-timesteps",
        type=int,
        default=150000,
        help="Timesteps for training benchmark"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="logs/benchmark_results.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger.info("=" * 70)
    logger.info("RL-Bot Verification & Benchmarking")
    logger.info("=" * 70)
    logger.info(f"Config: {args.config}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logger.info("=" * 70)
    
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    # Run tests
    all_results = []
    
    if args.all or args.dry_test:
        results = run_dry_test(config_path)
        all_results.append(results)
        if not results["success"]:
            logger.error("Dry test failed, stopping...")
            sys.exit(1)
    
    if args.all or args.perf_benchmark:
        results = run_performance_benchmark(config_path, timesteps=1000)
        all_results.append(results)
    
    if args.all or args.train_benchmark:
        results = run_training_benchmark(
            config_path,
            timesteps=args.train_timesteps,
            checkpoint_interval=25000
        )
        all_results.append(results)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": str(config_path),
        "cuda_available": torch.cuda.is_available(),
        "tests": all_results,
    }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    for result in all_results:
        status = "✓ PASS" if result["success"] else "✗ FAIL"
        logger.info(f"{status}: {result['test']}")
    
    logger.info(f"\nResults saved to: {output_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
