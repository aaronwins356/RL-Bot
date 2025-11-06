"""Performance profiler for monitoring inference time.

This module provides frame time monitoring with warnings for budget violations.
"""
import time
from typing import Dict, List, Optional
from collections import deque


class FrameProfiler:
    """Profiler for monitoring inference performance.
    
    Tracks inference time and warns when frame budget is exceeded.
    """
    
    def __init__(self, frame_budget_ms: float = 8.0, window_size: int = 100):
        """Initialize profiler.
        
        Args:
            frame_budget_ms: Target frame time budget in milliseconds
            window_size: Size of rolling window for statistics
        """
        self.frame_budget_ms = frame_budget_ms
        self.window_size = window_size
        
        # Timing data
        self.times = deque(maxlen=window_size)
        self.violations = 0
        self.total_frames = 0
        
        # Current frame
        self.start_time: Optional[float] = None
    
    def start_frame(self):
        """Start timing a frame."""
        self.start_time = time.time()
    
    def end_frame(self) -> float:
        """End timing a frame and record duration.
        
        Returns:
            Frame duration in milliseconds
        """
        if self.start_time is None:
            return 0.0
        
        duration_ms = (time.time() - self.start_time) * 1000.0
        self.times.append(duration_ms)
        self.total_frames += 1
        
        # Check for violation
        if duration_ms > self.frame_budget_ms:
            self.violations += 1
        
        self.start_time = None
        
        return duration_ms
    
    def check_budget(self) -> bool:
        """Check if last frame was within budget.
        
        Returns:
            True if within budget, False otherwise
        """
        if not self.times:
            return True
        
        return self.times[-1] <= self.frame_budget_ms
    
    def get_stats(self) -> Dict[str, float]:
        """Get profiling statistics.
        
        Returns:
            Dictionary with timing statistics
        """
        if not self.times:
            return {
                "mean_ms": 0.0,
                "max_ms": 0.0,
                "min_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
                "violation_rate": 0.0,
                "total_violations": 0
            }
        
        times_list = list(self.times)
        times_sorted = sorted(times_list)
        
        return {
            "mean_ms": sum(times_list) / len(times_list),
            "max_ms": max(times_list),
            "min_ms": min(times_list),
            "p95_ms": times_sorted[int(len(times_sorted) * 0.95)],
            "p99_ms": times_sorted[int(len(times_sorted) * 0.99)],
            "violation_rate": (
                self.violations / self.total_frames * 100
                if self.total_frames > 0 else 0.0
            ),
            "total_violations": self.violations
        }
    
    def reset(self):
        """Reset profiler statistics."""
        self.times.clear()
        self.violations = 0
        self.total_frames = 0
        self.start_time = None
