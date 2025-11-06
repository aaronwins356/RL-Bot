"""Telemetry logging for observations, actions, and rewards.

This module provides structured telemetry logging with ring buffer.
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from collections import deque
from datetime import datetime


class TelemetryLogger:
    """Logger for game telemetry data."""
    
    def __init__(
        self,
        save_path: Path,
        buffer_size: int = 100000,
        save_interval: int = 10000,
        enabled: bool = True
    ):
        """Initialize telemetry logger.
        
        Args:
            save_path: Path to save telemetry data
            buffer_size: Size of ring buffer
            save_interval: Save to disk every N samples
            enabled: Whether logging is enabled
        """
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        self.buffer_size = buffer_size
        self.save_interval = save_interval
        self.enabled = enabled
        
        # Ring buffer
        self.buffer = deque(maxlen=buffer_size)
        
        # Counters
        self.samples_logged = 0
        self.files_saved = 0
        
        # Current file
        self.current_file = None
    
    def log(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        info: Optional[Dict[str, Any]] = None
    ):
        """Log a telemetry sample.
        
        Args:
            observation: Observation vector
            action: Action vector
            reward: Reward value
            done: Episode done flag
            info: Additional info
        """
        if not self.enabled:
            return
        
        sample = {
            "timestamp": datetime.now().isoformat(),
            "observation": observation.tolist(),
            "action": action.tolist(),
            "reward": float(reward),
            "done": bool(done),
            "info": info or {}
        }
        
        self.buffer.append(sample)
        self.samples_logged += 1
        
        # Save periodically
        if self.samples_logged % self.save_interval == 0:
            self.flush()
    
    def flush(self):
        """Flush buffer to disk."""
        if not self.buffer:
            return
        
        # Create new file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"telemetry_{timestamp}_{self.files_saved}.jsonl"
        filepath = self.save_path / filename
        
        # Write buffer to file
        with open(filepath, "w") as f:
            for sample in self.buffer:
                f.write(json.dumps(sample) + "\n")
        
        self.files_saved += 1
        print(f"Telemetry: Saved {len(self.buffer)} samples to {filepath}")
        
        # Clear buffer after saving
        self.buffer.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get telemetry statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "samples_logged": self.samples_logged,
            "files_saved": self.files_saved,
            "buffer_size": len(self.buffer),
            "enabled": self.enabled
        }
