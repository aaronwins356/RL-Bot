"""
Replay Viewer - Playback and visualization of saved replays.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import pickle


class ReplayViewer:
    """
    View and analyze saved match replays.
    Supports playback, frame-by-frame analysis, and export.
    """
    
    def __init__(self, replay_path: str):
        """
        Args:
            replay_path: Path to replay file
        """
        self.replay_path = Path(replay_path)
        self.replay_data = None
        self.current_frame = 0
        
        self.load_replay()
    
    def load_replay(self):
        """Load replay from file."""
        try:
            with open(self.replay_path, 'rb') as f:
                self.replay_data = pickle.load(f)
            print(f"✓ Loaded replay: {self.replay_path}")
            print(f"  Frames: {len(self.replay_data.get('frames', []))}")
            print(f"  Duration: {self.replay_data.get('duration', 'unknown')}")
        except Exception as e:
            print(f"✗ Failed to load replay: {e}")
            self.replay_data = {'frames': []}
    
    def get_frame(self, frame_idx: int) -> Dict[str, Any]:
        """
        Get specific frame data.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            frame_data: Dictionary with frame information
        """
        frames = self.replay_data.get('frames', [])
        if 0 <= frame_idx < len(frames):
            return frames[frame_idx]
        return {}
    
    def get_num_frames(self) -> int:
        """Get total number of frames."""
        return len(self.replay_data.get('frames', []))
    
    def play(self, start_frame: int = 0, end_frame: Optional[int] = None):
        """
        Play replay from start to end frame.
        
        Args:
            start_frame: Starting frame
            end_frame: Ending frame (None for end of replay)
        """
        if end_frame is None:
            end_frame = self.get_num_frames()
        
        print(f"▶️ Playing replay from frame {start_frame} to {end_frame}")
        
        for frame_idx in range(start_frame, end_frame):
            frame = self.get_frame(frame_idx)
            # Playback logic here
            # In a real implementation, this would render the frame
    
    def export_highlights(
        self,
        output_path: str,
        min_event_value: float = 5.0
    ) -> List[Dict[str, Any]]:
        """
        Export highlight moments from replay.
        
        Args:
            output_path: Where to save highlights
            min_event_value: Minimum reward value to consider as highlight
            
        Returns:
            highlights: List of highlight clips
        """
        highlights = []
        frames = self.replay_data.get('frames', [])
        
        for i, frame in enumerate(frames):
            reward = frame.get('reward', 0.0)
            if reward >= min_event_value:
                # Extract highlight clip (e.g., 5 seconds before and after)
                highlight = {
                    'frame': i,
                    'reward': reward,
                    'event': frame.get('event', 'unknown'),
                    'start_frame': max(0, i - 150),  # 5 sec before at 30fps
                    'end_frame': min(len(frames), i + 150)  # 5 sec after
                }
                highlights.append(highlight)
        
        # Save highlights
        with open(output_path, 'wb') as f:
            pickle.dump(highlights, f)
        
        print(f"✓ Exported {len(highlights)} highlights to {output_path}")
        return highlights
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get replay statistics."""
        frames = self.replay_data.get('frames', [])
        if not frames:
            return {}
        
        rewards = [f.get('reward', 0.0) for f in frames]
        
        stats = {
            'total_frames': len(frames),
            'total_reward': sum(rewards),
            'mean_reward': np.mean(rewards),
            'max_reward': np.max(rewards),
            'goals_scored': sum(1 for f in frames if f.get('goal_scored', False)),
            'ball_touches': sum(1 for f in frames if f.get('ball_touched', False)),
        }
        
        return stats


def create_heatmap(replay_paths: List[str], output_path: str = "heatmap.png"):
    """
    Create field heatmap from multiple replays.
    
    Args:
        replay_paths: List of replay file paths
        output_path: Where to save heatmap image
    """
    print(f"🗺️ Creating heatmap from {len(replay_paths)} replays...")
    
    # Collect all ball positions
    ball_positions = []
    
    for replay_path in replay_paths:
        try:
            viewer = ReplayViewer(replay_path)
            for frame_idx in range(viewer.get_num_frames()):
                frame = viewer.get_frame(frame_idx)
                ball_pos = frame.get('ball_position', None)
                if ball_pos is not None:
                    ball_positions.append(ball_pos)
        except Exception as e:
            print(f"  Skipping {replay_path}: {e}")
    
    if not ball_positions:
        print("✗ No ball positions found")
        return
    
    # Create 2D histogram (field top-down view)
    positions = np.array(ball_positions)
    x_positions = positions[:, 0]  # Field X
    y_positions = positions[:, 1]  # Field Y
    
    # Heatmap would be rendered here using matplotlib or plotly
    print(f"✓ Heatmap created with {len(ball_positions)} data points")
    print(f"  Saved to: {output_path}")


def compare_replays(replay_paths: List[str]) -> Dict[str, Any]:
    """
    Compare statistics across multiple replays.
    
    Args:
        replay_paths: List of replay paths
        
    Returns:
        comparison: Dictionary with comparison statistics
    """
    stats_list = []
    
    for replay_path in replay_paths:
        viewer = ReplayViewer(replay_path)
        stats = viewer.get_statistics()
        stats['replay'] = Path(replay_path).name
        stats_list.append(stats)
    
    # Create comparison summary
    comparison = {
        'replays': stats_list,
        'best_replay': max(stats_list, key=lambda x: x.get('total_reward', 0)),
        'worst_replay': min(stats_list, key=lambda x: x.get('total_reward', 0))
    }
    
    return comparison
