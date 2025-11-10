"""
Telemetry Dashboard - Live match telemetry and metrics.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from collections import deque
import time


class TelemetryDashboard:
    """
    Real-time telemetry monitoring for live matches.
    Tracks performance metrics, game state, and bot behavior.
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Args:
            max_history: Maximum number of data points to keep
        """
        self.max_history = max_history
        
        # Metric histories
        self.ball_speed_history = deque(maxlen=max_history)
        self.player_speed_history = deque(maxlen=max_history)
        self.boost_history = deque(maxlen=max_history)
        self.distance_to_ball_history = deque(maxlen=max_history)
        self.reward_history = deque(maxlen=max_history)
        
        # Current state
        self.current_state = {}
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'goals_scored': 0,
            'goals_conceded': 0,
            'ball_touches': 0,
            'boost_collected': 0,
            'demos_dealt': 0,
            'demos_received': 0
        }
        
        self.start_time = time.time()
    
    def update(self, state: Dict[str, Any]):
        """
        Update telemetry with new state.
        
        Args:
            state: Current game state dictionary
        """
        # Extract metrics
        ball_speed = np.linalg.norm(state.get('ball_velocity', [0, 0, 0]))
        player_speed = np.linalg.norm(state.get('player_velocity', [0, 0, 0]))
        boost = state.get('boost', 0.0)
        
        ball_pos = np.array(state.get('ball_position', [0, 0, 0]))
        player_pos = np.array(state.get('player_position', [0, 0, 0]))
        distance_to_ball = np.linalg.norm(ball_pos - player_pos)
        
        reward = state.get('reward', 0.0)
        
        # Update histories
        self.ball_speed_history.append(ball_speed)
        self.player_speed_history.append(player_speed)
        self.boost_history.append(boost)
        self.distance_to_ball_history.append(distance_to_ball)
        self.reward_history.append(reward)
        
        # Update current state
        self.current_state = {
            'ball_speed': ball_speed,
            'player_speed': player_speed,
            'boost': boost,
            'distance_to_ball': distance_to_ball,
            'reward': reward,
            'timestamp': time.time() - self.start_time
        }
        
        # Update statistics
        self.stats['total_frames'] += 1
        if state.get('goal_scored', False):
            self.stats['goals_scored'] += 1
        if state.get('goal_conceded', False):
            self.stats['goals_conceded'] += 1
        if state.get('ball_touched', False):
            self.stats['ball_touches'] += 1
        if state.get('boost_collected', False):
            self.stats['boost_collected'] += 1
        if state.get('demo_dealt', False):
            self.stats['demos_dealt'] += 1
        if state.get('demo_received', False):
            self.stats['demos_received'] += 1
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current telemetry metrics."""
        return self.current_state.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cumulative statistics."""
        stats = self.stats.copy()
        
        # Add derived statistics
        if stats['total_frames'] > 0:
            stats['goals_per_minute'] = stats['goals_scored'] / (stats['total_frames'] / 1800.0)  # 30 fps
            stats['touches_per_minute'] = stats['ball_touches'] / (stats['total_frames'] / 1800.0)
        
        # Add history summaries
        if len(self.ball_speed_history) > 0:
            stats['avg_ball_speed'] = np.mean(self.ball_speed_history)
            stats['max_ball_speed'] = np.max(self.ball_speed_history)
            stats['avg_player_speed'] = np.mean(self.player_speed_history)
            stats['avg_boost'] = np.mean(self.boost_history)
            stats['avg_distance_to_ball'] = np.mean(self.distance_to_ball_history)
        
        return stats
    
    def get_history(self, metric: str, last_n: Optional[int] = None) -> List[float]:
        """
        Get history for a specific metric.
        
        Args:
            metric: Metric name
            last_n: Return only last N points (None for all)
            
        Returns:
            history: List of metric values
        """
        history_map = {
            'ball_speed': self.ball_speed_history,
            'player_speed': self.player_speed_history,
            'boost': self.boost_history,
            'distance_to_ball': self.distance_to_ball_history,
            'reward': self.reward_history
        }
        
        history = history_map.get(metric, [])
        
        if last_n is not None:
            return list(history)[-last_n:]
        return list(history)
    
    def reset(self):
        """Reset all telemetry data."""
        self.ball_speed_history.clear()
        self.player_speed_history.clear()
        self.boost_history.clear()
        self.distance_to_ball_history.clear()
        self.reward_history.clear()
        
        self.current_state = {}
        self.stats = {k: 0 for k in self.stats}
        self.start_time = time.time()
    
    def export_data(self, output_path: str):
        """
        Export telemetry data to file.
        
        Args:
            output_path: Where to save data
        """
        import json
        
        data = {
            'statistics': self.get_statistics(),
            'histories': {
                'ball_speed': list(self.ball_speed_history),
                'player_speed': list(self.player_speed_history),
                'boost': list(self.boost_history),
                'distance_to_ball': list(self.distance_to_ball_history),
                'reward': list(self.reward_history)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Telemetry data exported to {output_path}")


class PerformanceMonitor:
    """Monitor training and inference performance metrics."""
    
    def __init__(self):
        self.fps_history = deque(maxlen=100)
        self.update_time_history = deque(maxlen=100)
        self.last_time = time.time()
        self.frame_count = 0
    
    def update(self):
        """Update performance metrics."""
        current_time = time.time()
        dt = current_time - self.last_time
        
        if dt > 0:
            fps = 1.0 / dt
            self.fps_history.append(fps)
        
        self.last_time = current_time
        self.frame_count += 1
    
    def record_update_time(self, update_time: float):
        """Record time taken for training update."""
        self.update_time_history.append(update_time)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        metrics = {
            'fps': np.mean(self.fps_history) if self.fps_history else 0.0,
            'avg_update_time': np.mean(self.update_time_history) if self.update_time_history else 0.0,
            'total_frames': self.frame_count
        }
        return metrics
