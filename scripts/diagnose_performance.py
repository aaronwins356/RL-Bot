#!/usr/bin/env python3
"""Diagnostic script for profiling training performance.

This script provides comprehensive performance diagnostics for:
- Environment step speed
- Model inference speed
- GPU utilization
- Memory usage
- Training throughput

Run this before and after optimizations to measure improvements.
"""
import argparse
import sys
import time
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.env.rocket_sim_env import RocketSimEnv
from core.models.nets import ActorCriticNet
from core.infra.config import ConfigManager
from core.infra.performance import PerformanceMonitor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PerformanceDiagnostics:
    """Performance diagnostics for RL training."""
    
    def __init__(self, config_path: Path, device: str = "auto"):
        """Initialize diagnostics.
        
        Args:
            config_path: Path to configuration file
            device: Device to use for testing
        """
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        
        # Auto-detect device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize performance monitor
        self.perf_monitor = PerformanceMonitor(self.device)
        
    def benchmark_env_speed(self, num_steps: int = 1000, num_envs: int = 8) -> Dict[str, float]:
        """Benchmark environment step speed.
        
        Args:
            num_steps: Number of steps to run
            num_envs: Number of parallel environments
            
        Returns:
            Dictionary with benchmark results
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Benchmarking Environment Speed")
        logger.info(f"{'='*70}")
        logger.info(f"Num envs: {num_envs}, Steps: {num_steps}")
        
        # Create environment
        from core.training.train_loop import create_vectorized_env
        
        env = create_vectorized_env(
            num_envs=num_envs,
            reward_config_path=Path("configs/rewards.yaml"),
            simulation_mode=True,
            debug_mode=False
        )
        
        # Warmup
        logger.info("Warming up...")
        obs = env.reset()
        for _ in range(10):
            actions = [env.action_space.sample() for _ in range(num_envs)]
            obs, rewards, dones, infos = env.step(actions)
        
        # Benchmark
        logger.info("Running benchmark...")
        start_time = time.time()
        step_times = []
        
        for i in range(num_steps):
            step_start = time.time()
            actions = [env.action_space.sample() for _ in range(num_envs)]
            obs, rewards, dones, infos = env.step(actions)
            step_times.append(time.time() - step_start)
            
            if (i + 1) % 100 == 0:
                logger.info(f"  Progress: {i+1}/{num_steps} steps")
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        total_env_steps = num_steps * num_envs
        steps_per_sec = total_env_steps / total_time
        ms_per_step = np.mean(step_times) * 1000
        
        results = {
            'total_steps': total_env_steps,
            'total_time_sec': total_time,
            'steps_per_sec': steps_per_sec,
            'ticks_per_sec': steps_per_sec,  # Alias for compatibility
            'ms_per_step': ms_per_step,
            'num_envs': num_envs,
        }
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Environment Speed Results")
        logger.info(f"{'='*70}")
        logger.info(f"Total steps: {total_env_steps:,}")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Steps/sec: {steps_per_sec:.2f}")
        logger.info(f"Ticks/sec: {steps_per_sec:.2f}")
        logger.info(f"ms/step: {ms_per_step:.2f}")
        
        env.close()
        return results
    
    def benchmark_model_inference(self, num_inferences: int = 1000, batch_size: int = 8) -> Dict[str, float]:
        """Benchmark model inference speed.
        
        Args:
            num_inferences: Number of inferences to run
            batch_size: Batch size for inference
            
        Returns:
            Dictionary with benchmark results
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Benchmarking Model Inference")
        logger.info(f"{'='*70}")
        logger.info(f"Batch size: {batch_size}, Inferences: {num_inferences}")
        
        # Create model
        obs_size = 180  # Standard observation size
        action_dim = 8
        
        model = ActorCriticNet(
            obs_size=obs_size,
            action_dim=action_dim,
            hidden_sizes=[1024, 512, 256],
            activation='elu',
            use_lstm=False
        ).to(self.device)
        
        model.eval()
        
        # Warmup
        logger.info("Warming up...")
        dummy_obs = torch.randn(batch_size, obs_size, device=self.device)
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_obs)
        
        # Synchronize for accurate timing
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        logger.info("Running benchmark...")
        inference_times = []
        
        for i in range(num_inferences):
            obs = torch.randn(batch_size, obs_size, device=self.device)
            
            start_time = time.time()
            with torch.no_grad():
                _ = model(obs)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            inference_times.append(time.time() - start_time)
            
            if (i + 1) % 100 == 0:
                logger.info(f"  Progress: {i+1}/{num_inferences} inferences")
        
        # Calculate statistics
        inference_times = np.array(inference_times)
        total_time = np.sum(inference_times)
        mean_time = np.mean(inference_times)
        p95_time = np.percentile(inference_times, 95)
        p99_time = np.percentile(inference_times, 99)
        inferences_per_sec = 1.0 / mean_time
        
        results = {
            'total_inferences': num_inferences,
            'total_time_sec': total_time,
            'mean_ms': mean_time * 1000,
            'p95_ms': p95_time * 1000,
            'p99_ms': p99_time * 1000,
            'inferences_per_sec': inferences_per_sec,
            'batch_size': batch_size,
        }
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Model Inference Results")
        logger.info(f"{'='*70}")
        logger.info(f"Total inferences: {num_inferences:,}")
        logger.info(f"Mean time: {mean_time*1000:.2f}ms")
        logger.info(f"P95 time: {p95_time*1000:.2f}ms")
        logger.info(f"P99 time: {p99_time*1000:.2f}ms")
        logger.info(f"Inferences/sec: {inferences_per_sec:.2f}")
        
        return results
    
    def benchmark_gpu_utilization(self, duration_sec: int = 30) -> Dict[str, float]:
        """Benchmark GPU utilization during training simulation.
        
        Args:
            duration_sec: Duration to monitor in seconds
            
        Returns:
            Dictionary with GPU metrics
        """
        if self.device.type != 'cuda':
            logger.warning("GPU benchmarking only available with CUDA device")
            return {}
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Benchmarking GPU Utilization")
        logger.info(f"{'='*70}")
        logger.info(f"Duration: {duration_sec}s")
        
        # Create model and dummy data
        obs_size = 180
        action_dim = 8
        batch_size = 32
        
        model = ActorCriticNet(
            obs_size=obs_size,
            action_dim=action_dim,
            hidden_sizes=[1024, 512, 256],
            activation='elu',
            use_lstm=False
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        
        # Collect metrics
        gpu_utils = []
        memory_utils = []
        
        start_time = time.time()
        iteration = 0
        
        logger.info("Running training simulation...")
        while time.time() - start_time < duration_sec:
            # Simulate training step
            obs = torch.randn(batch_size, obs_size, device=self.device)
            
            # Forward pass
            actions, values = model(obs)
            
            # Dummy loss
            loss = actions.sum() + values.sum()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Get GPU stats
            gpu_stats = self.perf_monitor.get_gpu_utilization()
            if gpu_stats:
                if 'gpu_utilization_percent' in gpu_stats:
                    gpu_utils.append(gpu_stats['gpu_utilization_percent'])
                if 'memory_utilization_percent' in gpu_stats:
                    memory_utils.append(gpu_stats['memory_utilization_percent'])
            
            iteration += 1
            
            if iteration % 10 == 0:
                elapsed = time.time() - start_time
                logger.info(f"  Progress: {elapsed:.1f}/{duration_sec}s")
        
        # Calculate statistics
        results = {}
        if gpu_utils:
            results['mean_gpu_utilization_percent'] = np.mean(gpu_utils)
            results['max_gpu_utilization_percent'] = np.max(gpu_utils)
            results['min_gpu_utilization_percent'] = np.min(gpu_utils)
        
        if memory_utils:
            results['mean_memory_utilization_percent'] = np.mean(memory_utils)
        
        results['iterations'] = iteration
        results['iterations_per_sec'] = iteration / duration_sec
        
        logger.info(f"\n{'='*70}")
        logger.info(f"GPU Utilization Results")
        logger.info(f"{'='*70}")
        logger.info(f"Total iterations: {iteration}")
        logger.info(f"Iterations/sec: {iteration/duration_sec:.2f}")
        if 'mean_gpu_utilization_percent' in results:
            logger.info(f"Mean GPU util: {results['mean_gpu_utilization_percent']:.1f}%")
            logger.info(f"Max GPU util: {results['max_gpu_utilization_percent']:.1f}%")
            logger.info(f"Min GPU util: {results['min_gpu_utilization_percent']:.1f}%")
        if 'mean_memory_utilization_percent' in results:
            logger.info(f"Mean Memory util: {results['mean_memory_utilization_percent']:.1f}%")
        
        return results
    
    def run_full_diagnostics(self) -> Dict[str, Dict[str, float]]:
        """Run all diagnostic tests.
        
        Returns:
            Dictionary with all benchmark results
        """
        logger.info(f"\n{'#'*70}")
        logger.info(f"# RL-Bot Performance Diagnostics")
        logger.info(f"{'#'*70}\n")
        
        results = {}
        
        # System info
        logger.info("System Information:")
        logger.info(f"  PyTorch version: {torch.__version__}")
        logger.info(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"  CUDA version: {torch.version.cuda}")
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  Device: {self.device}")
        
        # Run benchmarks
        try:
            results['env_speed'] = self.benchmark_env_speed(num_steps=500, num_envs=8)
        except Exception as e:
            logger.error(f"Environment benchmark failed: {e}")
            results['env_speed'] = {'error': str(e)}
        
        try:
            results['inference'] = self.benchmark_model_inference(num_inferences=500, batch_size=8)
        except Exception as e:
            logger.error(f"Inference benchmark failed: {e}")
            results['inference'] = {'error': str(e)}
        
        if self.device.type == 'cuda':
            try:
                results['gpu'] = self.benchmark_gpu_utilization(duration_sec=20)
            except Exception as e:
                logger.error(f"GPU benchmark failed: {e}")
                results['gpu'] = {'error': str(e)}
        
        # Summary
        logger.info(f"\n{'#'*70}")
        logger.info(f"# Diagnostics Complete")
        logger.info(f"{'#'*70}\n")
        
        # Compare to baseline
        baseline_ticks_per_sec = 7.5  # From problem statement
        if 'env_speed' in results and 'ticks_per_sec' in results['env_speed']:
            actual = results['env_speed']['ticks_per_sec']
            logger.info(f"Performance vs Baseline:")
            logger.info(f"  Baseline: {baseline_ticks_per_sec:.1f} ticks/sec")
            logger.info(f"  Current: {actual:.1f} ticks/sec")
            logger.info(f"  Speedup: {actual/baseline_ticks_per_sec:.2f}x")
        
        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Performance diagnostics for RL-Bot',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/base.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'auto'],
        default='auto',
        help='Device to use for benchmarks'
    )
    
    parser.add_argument(
        '--test',
        type=str,
        choices=['all', 'env', 'inference', 'gpu'],
        default='all',
        help='Which test to run'
    )
    
    parser.add_argument(
        '--num-steps',
        type=int,
        default=500,
        help='Number of steps for environment benchmark'
    )
    
    parser.add_argument(
        '--num-envs',
        type=int,
        default=8,
        help='Number of parallel environments'
    )
    
    args = parser.parse_args()
    
    # Create diagnostics
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    diagnostics = PerformanceDiagnostics(config_path, device=args.device)
    
    # Run requested tests
    if args.test == 'all':
        diagnostics.run_full_diagnostics()
    elif args.test == 'env':
        diagnostics.benchmark_env_speed(num_steps=args.num_steps, num_envs=args.num_envs)
    elif args.test == 'inference':
        diagnostics.benchmark_model_inference(num_inferences=args.num_steps, batch_size=args.num_envs)
    elif args.test == 'gpu':
        diagnostics.benchmark_gpu_utilization(duration_sec=30)


if __name__ == '__main__':
    main()
