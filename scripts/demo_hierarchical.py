"""Demo script for hierarchical RL/IL controller.

This script demonstrates the hierarchical control system with opportunity detection,
skill programs, and low-level control.
"""

import yaml
import numpy as np
from pathlib import Path
from core.hierarchical_controller import HierarchicalController


def create_mock_observation() -> dict:
    """Create a mock observation for testing."""
    return {
        'frame': 0,
        'time': 0.0,
        'car_position': np.array([0.0, -4000.0, 20.0]),
        'car_velocity': np.array([0.0, 500.0, 0.0]),
        'car_orientation': np.eye(3),
        'angular_velocity': np.array([0.0, 0.0, 0.0]),
        'ball_position': np.array([0.0, 0.0, 800.0]),
        'ball_velocity': np.array([0.0, 0.0, -200.0]),
        'ball_radius': 91.25,
        'boost': 80,
        'wheels_on_ground': 1,
        'score_diff': 0,
        'time_remaining': 240.0,
        'is_last_defender': False,
        'our_goal_y': -5120.0,
        'opponent_goal_position': np.array([0.0, 5120.0, 0.0]),
        'possession_prob': 0.7,
        'own_eta_to_ball': 1.2,
        'opponent_closest_eta': 2.0,
    }


def main():
    """Run hierarchical controller demo."""
    print("=" * 70)
    print("Hierarchical RL/IL Controller Demo")
    print("=" * 70)
    
    # Load config
    config_path = Path(__file__).parent.parent / 'configs' / 'hierarchical_rl.yaml'
    
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\nLoaded config from: {config_path}")
    print(f"Curriculum stage: {config['curriculum']['stage']}")
    print(f"Flashy mechanics enabled: {config['features']['flashy_enabled']}")
    
    # Initialize controller
    print("\nInitializing hierarchical controller...")
    controller = HierarchicalController(config, device='cpu')
    print("✓ Controller initialized")
    
    # Print available skill programs
    print("\nAvailable Skill Programs:")
    for sp_name in controller.skill_programs.keys():
        print(f"  - {sp_name}")
    
    # Run simulation loop
    print("\n" + "=" * 70)
    print("Running Simulation (10 steps)")
    print("=" * 70)
    
    controller.reset()
    obs = create_mock_observation()
    
    for step in range(10):
        obs['frame'] = step
        obs['time'] = step * (1.0 / 120.0)
        
        # Get action from controller
        action = controller.get_action(obs)
        
        # Print step info
        metadata = action.pop('_metadata', {})
        print(f"\nStep {step}:")
        print(f"  SP: {metadata.get('sp_name', 'None')}")
        print(f"  Category: {metadata.get('category', 'Unknown')}")
        print(f"  Confidence: {metadata.get('confidence', 0.0):.3f}")
        print(f"  Risk Score: {metadata.get('risk_score', 0.0):.3f}")
        print(f"  Controls: throttle={action['throttle']:.2f}, "
              f"pitch={action['pitch']:.2f}, boost={action['boost']}")
        
        # Update observation (simple physics)
        obs['car_position'] += obs['car_velocity'] * (1.0 / 120.0)
        obs['ball_position'] += obs['ball_velocity'] * (1.0 / 120.0)
    
    # Print telemetry summary
    print("\n" + "=" * 70)
    print("Telemetry Summary")
    print("=" * 70)
    
    telemetry = controller.get_telemetry()
    
    print(f"\nSP Choices: {len(telemetry['sp_choices'])}")
    for choice in telemetry['sp_choices']:
        print(f"  Frame {choice['frame']}: {choice['sp']} "
              f"(category={choice['category']}, conf={choice['confidence']:.3f})")
    
    print(f"\nRisk Scores: min={min(telemetry['risk_scores']):.3f}, "
          f"max={max(telemetry['risk_scores']):.3f}, "
          f"mean={np.mean(telemetry['risk_scores']):.3f}")
    
    print("\n" + "=" * 70)
    print("Demo Complete")
    print("=" * 70)


if __name__ == '__main__':
    main()
