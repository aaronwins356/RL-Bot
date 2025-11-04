import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, Rectangle

class Visualizer:
    def __init__(self, output_dir: str = "reports/figures"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
    def plot_shot_heatmap(self, shots: List[Dict], filename: str):
        """Plot heatmap of shot locations and outcomes"""
        field_length = 5120
        field_width = 4096
        
        # Create pitch outline
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.add_patch(Rectangle((-field_length, -field_width/2),
                             field_length*2, field_width,
                             fill=False, color='white'))
        
        # Plot shot locations
        x = [shot['position'][0] for shot in shots]
        y = [shot['position'][1] for shot in shots]
        colors = ['g' if shot['scored'] else 'r' for shot in shots]
        
        plt.scatter(x, y, c=colors, alpha=0.5)
        plt.title("Shot Placement Analysis")
        plt.xlabel("Field Length")
        plt.ylabel("Field Width")
        
        # Add goal locations
        goal_width = 892.755
        plt.plot([-field_length, -field_length], 
                [-goal_width/2, goal_width/2], 'y-', linewidth=2)
        plt.plot([field_length, field_length],
                [-goal_width/2, goal_width/2], 'y-', linewidth=2)
        
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        
    def plot_boost_paths(self, trajectories: List[Dict], filename: str):
        """Plot boost collection patterns"""
        # Plot field
        fig, ax = plt.subplots(figsize=(12, 8))
        self._draw_field(ax)
        
        # Plot boost pad locations
        boost_locs = self._get_boost_locations()
        small_pads = np.array([loc for loc, is_big in boost_locs if not is_big])
        big_pads = np.array([loc for loc, is_big in boost_locs if is_big])
        
        plt.scatter(small_pads[:, 0], small_pads[:, 1], 
                   c='white', alpha=0.3, s=20)
        plt.scatter(big_pads[:, 0], big_pads[:, 1],
                   c='yellow', alpha=0.5, s=100)
        
        # Plot trajectories
        for traj in trajectories:
            points = np.array(traj['points'])
            boost = np.array(traj['boost_amount'])
            
            # Color by boost amount
            plt.scatter(points[:, 0], points[:, 1],
                      c=boost, cmap='RdYlGn',
                      alpha=0.5, s=10)
            
        plt.colorbar(label='Boost Amount')
        plt.title("Boost Collection Patterns")
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        
    def plot_mechanic_transitions(self, transitions: List[Dict], filename: str):
        """Plot mechanic state transitions"""
        mechanics = sorted(list(set(
            [t['from'] for t in transitions] + 
            [t['to'] for t in transitions]
        )))
        
        # Create transition matrix
        n = len(mechanics)
        matrix = np.zeros((n, n))
        
        for t in transitions:
            i = mechanics.index(t['from'])
            j = mechanics.index(t['to'])
            matrix[i, j] += 1
            
        # Normalize
        row_sums = matrix.sum(axis=1)
        matrix = matrix / row_sums[:, np.newaxis]
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, fmt='.2f',
                   xticklabels=mechanics,
                   yticklabels=mechanics)
        plt.title("Mechanic Transition Probabilities")
        plt.xlabel("To State")
        plt.ylabel("From State")
        
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        
    def plot_training_curves(self, metrics: Dict[str, List], filename: str):
        """Plot multiple training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Training Progress")
        
        # Reward curve
        ax = axes[0, 0]
        ax.plot(metrics['step'], metrics['reward'])
        ax.set_title("Average Reward")
        ax.set_xlabel("Training Steps")
        
        # Success rate
        ax = axes[0, 1]
        ax.plot(metrics['step'], metrics['success_rate'])
        ax.set_title("Success Rate")
        ax.set_xlabel("Training Steps")
        
        # Loss curves
        ax = axes[1, 0]
        ax.plot(metrics['step'], metrics['policy_loss'], label='Policy')
        ax.plot(metrics['step'], metrics['value_loss'], label='Value')
        ax.set_title("Loss Values")
        ax.set_xlabel("Training Steps")
        ax.legend()
        
        # Entropy
        ax = axes[1, 1]
        ax.plot(metrics['step'], metrics['entropy'])
        ax.set_title("Policy Entropy")
        ax.set_xlabel("Training Steps")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        
    def _draw_field(self, ax):
        """Helper to draw Rocket League field"""
        field_length = 5120
        field_width = 4096
        
        # Field outline
        ax.add_patch(Rectangle((-field_length, -field_width/2),
                             field_length*2, field_width,
                             fill=False, color='white'))
                             
        # Center line and circle
        plt.plot([0, 0], [-field_width/2, field_width/2],
                'white', alpha=0.3)
        center_circle = Circle((0, 0), 500,
                             fill=False, color='white', alpha=0.3)
        ax.add_patch(center_circle)
        
    def _get_boost_locations(self) -> List[Tuple[np.ndarray, bool]]:
        """Return boost pad locations [(position, is_big_boost)]"""
        # Simplified boost layout
        return [
            (np.array([0, 0]), True),  # Example pad
            # Add actual boost locations
        ]