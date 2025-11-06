"""Performance monitoring utilities for training.

This module provides utilities to monitor training performance including
GPU utilization, training speed, and memory usage.
"""
import torch
import time
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor training performance metrics."""
    
    def __init__(self, device: torch.device):
        """Initialize performance monitor.
        
        Args:
            device: PyTorch device (cuda or cpu)
        """
        self.device = device
        self.is_cuda = device.type == "cuda"
        self.start_time = time.time()
        self.last_check_time = time.time()
        self.last_timestep = 0
        
        # Check if CUDA is available
        if self.is_cuda:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.nvml_available = True
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(device.index or 0)
            except (ImportError, Exception) as e:
                logger.warning(f"NVML not available for GPU monitoring: {e}")
                self.nvml_available = False
        else:
            self.nvml_available = False
    
    def get_gpu_utilization(self) -> Optional[Dict[str, float]]:
        """Get GPU utilization metrics.
        
        Returns:
            Dictionary with GPU metrics or None if unavailable
        """
        if not self.is_cuda:
            return None
        
        try:
            if self.nvml_available:
                import pynvml
                # Get utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                # Get memory info
                mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                
                return {
                    'gpu_utilization_percent': util.gpu,
                    'memory_utilization_percent': util.memory,
                    'memory_used_mb': mem.used / 1024**2,
                    'memory_total_mb': mem.total / 1024**2,
                }
            else:
                # Fallback to PyTorch memory stats
                if torch.cuda.is_available():
                    return {
                        'memory_allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                        'memory_reserved_mb': torch.cuda.memory_reserved() / 1024**2,
                    }
        except Exception as e:
            logger.debug(f"Failed to get GPU utilization: {e}")
        
        return None
    
    def get_training_speed(self, current_timestep: int) -> Optional[float]:
        """Calculate training speed in timesteps/sec.
        
        Args:
            current_timestep: Current training timestep
            
        Returns:
            Training speed in timesteps/sec or None
        """
        current_time = time.time()
        time_elapsed = current_time - self.last_check_time
        
        if time_elapsed > 0 and current_timestep > self.last_timestep:
            timesteps_elapsed = current_timestep - self.last_timestep
            speed = timesteps_elapsed / time_elapsed
            
            # Update for next calculation
            self.last_check_time = current_time
            self.last_timestep = current_timestep
            
            return speed
        
        return None
    
    def get_stats(self, current_timestep: int) -> Dict[str, any]:
        """Get all performance statistics.
        
        Args:
            current_timestep: Current training timestep
            
        Returns:
            Dictionary with performance metrics
        """
        stats = {}
        
        # Training speed
        speed = self.get_training_speed(current_timestep)
        if speed is not None:
            stats['training_speed_timesteps_per_sec'] = speed
        
        # GPU metrics
        gpu_stats = self.get_gpu_utilization()
        if gpu_stats:
            stats.update(gpu_stats)
        
        # Total elapsed time
        stats['total_elapsed_seconds'] = time.time() - self.start_time
        
        return stats
    
    def log_stats(self, current_timestep: int):
        """Log performance statistics.
        
        Args:
            current_timestep: Current training timestep
        """
        stats = self.get_stats(current_timestep)
        
        if 'training_speed_timesteps_per_sec' in stats:
            logger.info(f"[OK] Training speed: {stats['training_speed_timesteps_per_sec']:.1f} timesteps/sec")
        
        if 'gpu_utilization_percent' in stats:
            logger.info(
                f"[OK] GPU utilization: {stats['gpu_utilization_percent']:.1f}%, "
                f"Memory: {stats['memory_used_mb']:.0f}/{stats['memory_total_mb']:.0f} MB"
            )
    
    def __del__(self):
        """Cleanup NVML on destruction."""
        if self.nvml_available:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except:
                pass
