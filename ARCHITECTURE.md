# RL-Bot Architecture Documentation

## Overview

This document describes the modular architecture of RL-Bot, designed to combine learned reinforcement learning policies with hardcoded behaviors and heuristics inspired by top-tier Rocket League bots (Nexto, Necto, Ripple).

## Design Philosophy

The architecture follows these principles:

1. **Modularity**: Behaviors, rewards, and observations are modular and can be independently configured
2. **Extensibility**: Easy to add new behaviors, reward components, or observation features
3. **Performance**: Starts with strong baseline via heuristics, improves through training
4. **Simplicity**: Clean code structure, easy to understand and modify

## Core Components

### 1. Modular Behavior System (`rl_bot/core/behaviors.py`)

The behavior system allows hardcoded logic to override the learned policy when appropriate.

#### BehaviorModule (Base Class)

All behaviors inherit from `BehaviorModule` and implement:
- `should_activate()`: Determines if the behavior should take control
- `get_action()`: Returns the action to execute
- `reset()`: Resets internal state

#### Available Behaviors

**KickoffManager**
- **Purpose**: Execute fast, optimized kickoff routine (inspired by Nexto)
- **Activation**: Detects ball at center with minimal velocity
- **Actions**: Rush to ball with boost, execute flip at optimal distance
- **Configuration**: `behaviors.kickoff_enabled` in config.yaml

**RecoveryManager**
- **Purpose**: Recover from aerials and reorient wheels-down
- **Activation**: Car in air and not upright, or tumbling
- **Actions**: Pitch/roll to orient wheels downward
- **Configuration**: `behaviors.recovery_enabled` in config.yaml

**BoostManager** (Experimental)
- **Purpose**: Guide bot to collect boost when low
- **Activation**: Boost below threshold and ball far away
- **Actions**: Navigate toward nearest boost pad
- **Configuration**: `behaviors.boost_management_enabled` in config.yaml

#### BehaviorCoordinator

Manages multiple behaviors and decides which should take control:
- Behaviors are checked in priority order
- First behavior to activate gets control
- If no behavior activates, learned policy is used

```python
coordinator = BehaviorCoordinator(config)
behavior_action = coordinator.get_behavior_action(player, state, action_parser)
```

### 2. Agent Classes (`rl_bot/core/agent.py`)

#### ModularAgent

Combines learned policy with behavior overrides:
```python
agent = ModularAgent(
    model=actor_critic_model,
    action_parser=action_parser,
    device=device,
    config=config,
    use_behaviors=True
)

action, info = agent.predict(obs, player, state, deterministic=False)
```

Features:
- Checks behavior coordinator before using policy
- Tracks override statistics
- Returns metadata about decision source

#### HeuristicBaselineAgent

Uses only heuristics (no learned policy):
```python
agent = HeuristicBaselineAgent(
    action_parser=action_parser,
    config=config
)
```

Useful for:
- Baseline comparisons
- Testing behavior system
- Bootstrapping training

### 3. Ball Prediction (`rl_bot/core/ball_prediction.py`)

Physics-based ball trajectory prediction for planning aerials and positioning.

#### SimpleBallPredictor

Simulates ball physics:
```python
predictor = SimpleBallPredictor()
predictions = predictor.predict(
    ball_pos, ball_vel, ball_ang_vel,
    num_steps=120  # 1 second at 120Hz
)
```

Features:
- Gravity and drag simulation
- Wall and ground collision handling
- Landing prediction
- Configurable physics parameters

#### PredictionFeatureExtractor

Extracts features for agent observations:
```python
extractor = PredictionFeatureExtractor(predictor)
features = extractor.get_prediction_features(
    ball_pos, ball_vel, ball_ang_vel, player_pos,
    num_predictions=5
)
```

Also provides:
- Aerial opportunity detection
- Time-to-intercept calculations

### 4. Advanced Observations (`rl_bot/core/advanced_obs.py`)

#### TeamAwareObsBuilder

For 2v2 and 3v3 team play:
```python
obs_builder = TeamAwareObsBuilder(
    max_team_size=3,
    include_predictions=True,
    num_predictions=5
)
```

Includes:
- Own player state (20 dims)
- Ball state (9 dims)
- Relative ball info (6 dims)
- Teammate states with padding (9 * (max_team_size - 1) dims)
- Opponent states with padding (9 * max_team_size dims)
- Ball predictions (3 * num_predictions dims)

Total dimensions for 3v3 with predictions: 20 + 9 + 6 + 18 + 27 + 15 = 95 dims

#### CompactObsBuilder

Minimal observation for fast training:
```python
obs_builder = CompactObsBuilder(include_predictions=True)
```

Includes only essentials:
- Player position, velocity, forward vector, boost (11 dims)
- Ball position, velocity (6 dims)
- Relative ball info (6 dims)
- Ball predictions (15 dims)

Total: 38 dimensions

### 5. Advanced Reward Functions (`rl_bot/core/reward_functions.py`)

#### PositioningReward

Rewards good positioning relative to ball and goal:
- Defensive: Be between ball and own goal when ball is threatening
- Offensive: Be close to ball when attacking

#### RotationReward

Rewards proper rotation in team play:
- If teammate is closer to ball, reward rotating back
- If you're closest, reward engaging

Usage in config.yaml:
```yaml
rewards:
  positioning_weight: 0.1
  rotation_weight: 0.05
```

## Configuration

All features are controlled via `config.yaml`:

### Behavior System
```yaml
behaviors:
  enabled: true
  kickoff_enabled: true
  recovery_enabled: true
  boost_management_enabled: false
```

### Agent Settings
```yaml
agent:
  use_behaviors: true
  heuristic_baseline: false
```

### Environment
```yaml
environment:
  obs_builder: "simple"  # or "team_aware", "compact"
  include_predictions: true
  max_team_size: 3
```

## Training Pipeline Integration

The architecture integrates seamlessly with the existing PPO training loop:

1. **Environment Creation**: Observation builder selected based on config
2. **Agent Creation**: ModularAgent wraps policy model
3. **Rollout Collection**: Behaviors can override policy during rollouts
4. **Policy Update**: Standard PPO updates on collected experience

Key advantage: Behaviors provide strong exploration during early training, then gradually give way to learned policy as it improves.

## RLBot Deployment

The `run_bot.py` provides a wrapper for deploying trained agents in RLBot:

```python
agent = RLBotAgent(
    name="RL-Bot",
    team=0,
    index=0,
    checkpoint_path="checkpoints/best_model.pt",
    config_path="config.yaml"
)
```

Requires optional dependencies:
- `rlbot` (RLBot framework)
- `rlgym-compat` (compatibility layer)

## Performance Expectations

With this architecture:

- **Baseline (behaviors only)**: ~1200-1400 Elo (basic competence)
- **After 5M timesteps**: ~1400-1600 Elo (solid performance)
- **After 20M+ timesteps**: ~1600-1800+ Elo (high-level play)

Behaviors provide immediate competence, training improves from there.

## Future Extensions

The architecture is designed to support:

1. **Imitation Learning**: Pre-train policy on expert demonstrations
2. **Self-Play**: Train against copies of itself at various skill levels
3. **Curriculum Learning**: Gradually increase difficulty
4. **Multi-Modal Policies**: Different strategies for different game states
5. **Advanced Behaviors**: Dribbling, air dribbles, flip resets, etc.

## Code Examples

### Adding a New Behavior

```python
class DribbleManager(BehaviorModule):
    def should_activate(self, player: PlayerData, state: GameState) -> bool:
        # Check if ball is on top of car
        ball_pos = state.ball.position
        car_pos = player.car_data.position
        
        # Simple check: ball close and above car
        distance = np.linalg.norm(ball_pos[:2] - car_pos[:2])
        height_diff = ball_pos[2] - car_pos[2]
        
        return distance < 150 and 50 < height_diff < 200
    
    def get_action(self, player: PlayerData, state: GameState, action_parser):
        # Execute dribble: balance ball while moving forward
        # ... implementation ...
        return BehaviorAction(action_index=action_idx, confidence=0.9)
```

Add to coordinator:
```python
self.dribble_manager = DribbleManager(enabled=config.get('dribble_enabled', False))
self.behaviors.insert(2, self.dribble_manager)  # Priority after kickoff and recovery
```

### Using Ball Predictions in Custom Reward

```python
class AerialOpportunityReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.predictor = SimpleBallPredictor()
        self.extractor = PredictionFeatureExtractor(self.predictor)
    
    def get_reward(self, player: PlayerData, state: GameState, previous_action):
        has_opportunity, time_to_ball, pred_pos = self.extractor.get_aerial_opportunity(
            state.ball.position,
            state.ball.linear_velocity,
            state.ball.angular_velocity,
            player.car_data.position,
            player.car_data.linear_velocity
        )
        
        if has_opportunity:
            return 0.5  # Reward being in position for aerial
        return 0.0
```

## Summary

This architecture provides:
- ✅ Strong baseline performance via behaviors
- ✅ Seamless integration with learned policy
- ✅ Support for team play (2v2, 3v3)
- ✅ Ball prediction for tactical planning
- ✅ Advanced rewards for positioning and rotation
- ✅ Easy extensibility and configuration
- ✅ RLBot deployment capability

The result is a bot that starts competent and grows to high skill through training, combining the best of hardcoded logic and reinforcement learning.
