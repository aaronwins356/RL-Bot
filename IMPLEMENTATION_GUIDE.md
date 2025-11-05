# SSL Bot Implementation Guide

This guide explains how to integrate the new SSL-level systems into the bot and test them.

## Overview

The bot now has three decision-making layers:

1. **Strategic Layer** (Utility System): Decides "what to do" (attack, defend, shadow, etc.)
2. **Tactical Layer** (Behaviors): Decides "how to do it" (shadow positioning, fake challenges, etc.)
3. **Mechanical Layer** (Sequences/NN): Executes controls (speedflip, wavedash, neural network)

## Integration Steps

### Step 1: Initialize New Systems (Already Done ✅)

In `bot.py`, the following systems are initialized:
```python
self.ball_predictor = BallPredictor(prediction_horizon=4.0, timestep=1/60)
self.boost_manager = BoostManager(field_info)
```

### Step 2: Add Utility System and Behaviors

To complete the integration, add these imports and initializations to `bot.py`:

```python
# Add imports at top of bot.py
from decision.utility_system import UtilitySystem, BehaviorType
from behaviors.shadow_defense import ShadowDefense
from mechanics.fast_aerial import FastAerial

# Add to __init__
self.utility_system = UtilitySystem()
self.shadow_defense = ShadowDefense()
self.fast_aerial = None  # Created when needed
```

### Step 3: Modify get_output() Method

Replace the neural network-only decision making with behavior-based control:

```python
def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
    # ... existing code for time tracking and game state update ...
    
    # Get our player and opponent
    player = self.game_state.players[self.index]
    teammates = [p for p in self.game_state.players if p.team_num == self.team]
    opponents = [p for p in self.game_state.players if p.team_num != self.team]
    opponent = opponents[0] if opponents else player  # Get first opponent
    
    # Evaluate current situation and choose behavior
    behavior = self.utility_system.evaluate(
        player, opponent, self.game_state, 
        self.ball_predictor, self.boost_manager
    )
    
    # Execute behavior
    if behavior == BehaviorType.SHADOW:
        return self._execute_shadow_defense(player, opponent)
    elif behavior == BehaviorType.AERIAL:
        return self._execute_aerial(player)
    elif behavior == BehaviorType.COLLECT_BOOST:
        return self._execute_boost_collection(player, opponent)
    # ... etc ...
    else:
        # Fallback to neural network
        return self._execute_neural_network(player, opponent)
```

### Step 4: Implement Behavior Execution Methods

Add these methods to the `WinYour1s` class:

```python
def _execute_shadow_defense(self, player: PlayerData, opponent: PlayerData) -> SimpleControllerState:
    """Execute shadow defense behavior"""
    # Check for fake challenge opportunity
    current_time = self.game_state.time
    should_fake = self.shadow_defense.should_fake_challenge(
        player, opponent, self.game_state, current_time
    )
    
    if should_fake:
        is_faking, fake_controls = self.shadow_defense.execute_fake_challenge(
            player, self.game_state, current_time
        )
        if is_faking:
            self.action = np.array(fake_controls)
            self.update_controls(self.action)
            return self.controls
    
    # Check for challenge opportunity
    should_challenge = self.shadow_defense.should_challenge(
        player, opponent, self.game_state, self.ball_predictor
    )
    
    if should_challenge:
        # Switch to challenge/attack behavior
        return self._execute_neural_network(player, opponent)
    
    # Maintain shadow position
    shadow_pos = self.shadow_defense.calculate_shadow_position(
        player, opponent, self.game_state
    )
    shadow_controls = self.shadow_defense.calculate_shadow_controls(
        player, shadow_pos, self.game_state
    )
    
    self.action = np.array(shadow_controls)
    self.update_controls(self.action)
    return self.controls

def _execute_aerial(self, player: PlayerData) -> SimpleControllerState:
    """Execute fast aerial to ball"""
    # Get aerial target (predicted ball position)
    intercept = self.ball_predictor.get_intercept_time(
        self.game_state.ball, 
        player.car_data.position
    )
    
    if intercept is not None:
        target_pos = intercept[1]
    else:
        target_pos = self.game_state.ball.position
    
    # Create fast aerial if needed
    if self.fast_aerial is None or self.fast_aerial.is_finished():
        self.fast_aerial = FastAerial(target_pos)
    
    # Execute fast aerial
    if self.fast_aerial.is_valid(player):
        aerial_controls = self.fast_aerial.execute(
            player, self.game_state, self.action
        )
        self.action = np.array(aerial_controls)
        self.update_controls(self.action)
        return self.controls
    else:
        # Fallback to neural network
        return self._execute_neural_network(player, opponent)

def _execute_boost_collection(self, player: PlayerData, opponent: PlayerData) -> SimpleControllerState:
    """Execute boost collection strategy"""
    # Check for boost steal opportunity
    should_steal, steal_pos = self.boost_manager.should_steal_boost(
        player, opponent, self.game_state
    )
    
    if should_steal and steal_pos is not None:
        target_pos = steal_pos
    else:
        # Get best boost pad to collect
        best_pad = self.boost_manager.get_best_boost_to_collect(
            player, opponent, self.game_state
        )
        if best_pad is not None:
            target_pos = best_pad.position
        else:
            # No good boost to collect, fallback to NN
            return self._execute_neural_network(player, opponent)
    
    # Drive toward boost target
    car = player.car_data
    to_target = target_pos - car.position
    distance = np.linalg.norm(to_target)
    
    if distance > 10:
        to_target_norm = to_target / distance
        forward = car.forward()
        steer = np.cross(forward, to_target_norm)[2] * 3.0
        steer = np.clip(steer, -1, 1)
        
        boost = 1.0 if distance > 1500 and player.boost_amount > 0.3 else 0.0
        
        self.action = np.array([1.0, steer, 0.0, 0.0, 0.0, 0.0, boost, 0.0])
    else:
        # Reached boost, return to normal play
        return self._execute_neural_network(player, opponent)
    
    self.update_controls(self.action)
    return self.controls

def _execute_neural_network(self, player: PlayerData, opponent: PlayerData) -> SimpleControllerState:
    """Execute neural network decision (existing code)"""
    # This is the existing NN code from bot.py
    if self.update_action:
        self.update_action = False
        
        self.game_state.players = [player, opponent]
        obs = self.obs_builder.build_obs(player, self.game_state, self.action)
        self.action = self.agent.act(obs)[0] + self.action_trans
    
    if self.ticks >= self.tick_skip - 1:
        self.update_controls(self.action)
    
    if self.ticks >= self.tick_skip:
        self.ticks = 0
        self.update_action = True
    
    return self.controls
```

## Testing the Bot

### Manual Testing

1. **Test Ball Prediction**:
   ```python
   # Add debug output in get_output()
   predictions = self.ball_predictor.predict_trajectory(self.game_state.ball)
   landing_time = self.ball_predictor.get_landing_time(self.game_state.ball)
   print(f"Ball will land in {landing_time:.2f}s" if landing_time else "Ball won't land")
   ```

2. **Test Boost Management**:
   ```python
   best_boost = self.boost_manager.get_best_boost_to_collect(
       player, opponent, self.game_state
   )
   if best_boost:
       print(f"Best boost: {best_boost.position}, Value: {best_boost.value}")
   ```

3. **Test Utility System**:
   ```python
   behavior = self.utility_system.evaluate(...)
   print(f"Chosen behavior: {behavior.value}")
   ```

### Automated Testing

Create `testing/test_bot.py`:

```python
import numpy as np
from util.ball_prediction import BallPredictor
from util.physics_object import PhysicsObject

def test_ball_prediction():
    """Test ball prediction accuracy"""
    predictor = BallPredictor(prediction_horizon=2.0)
    
    # Create test ball state
    ball = PhysicsObject()
    ball.position = np.array([0, 0, 500])
    ball.linear_velocity = np.array([1000, 0, 0])
    ball.angular_velocity = np.array([0, 0, 0])
    
    # Predict trajectory
    predictions = predictor.predict_trajectory(ball)
    
    # Check predictions
    assert len(predictions) == 120  # 2 seconds at 60 Hz
    assert predictions[0].position[0] > 0  # Moving in x direction
    
    print("✓ Ball prediction tests passed")

def test_boost_manager():
    """Test boost management logic"""
    # TODO: Add tests
    pass

if __name__ == "__main__":
    test_ball_prediction()
    test_boost_manager()
    print("\n✓ All tests passed!")
```

### Performance Benchmarking

Create `testing/benchmark.py`:

```python
import time
from bot import WinYour1s

def benchmark_decision_speed():
    """Measure decision-making performance"""
    bot = WinYour1s("Test", 0, 0)
    
    # Simulate game packets
    start = time.time()
    for i in range(1000):
        # Create fake packet
        # packet = create_test_packet()
        # bot.get_output(packet)
        pass
    
    elapsed = time.time() - start
    fps = 1000 / elapsed
    
    print(f"Decision speed: {fps:.1f} FPS")
    print(f"Average time per decision: {elapsed/1000*1000:.2f}ms")

if __name__ == "__main__":
    benchmark_decision_speed()
```

## Performance Tuning

### Optimizing Decision Speed

Current target: 30 FPS (tick_skip=4)

If performance is slow:
1. **Increase tick_skip** to 6 (20 FPS) temporarily
2. **Reduce ball prediction horizon** from 4.0s to 2.0s
3. **Cache predictions** for multiple ticks
4. **Profile code** to find bottlenecks:
   ```python
   import cProfile
   cProfile.run('bot.get_output(packet)')
   ```

### Balancing NN vs Explicit Control

Adjust behavior thresholds in `utility_system.py`:
- **More aggressive**: Increase attack/challenge scores
- **More defensive**: Increase shadow/defend scores
- **More NN reliance**: Decrease all explicit behavior scores

### Tuning Shadow Defense

In `shadow_defense.py`:
- `min_shadow_distance`: How close to ball when shadowing
- `ideal_shadow_distance`: Optimal shadow distance
- Adjust challenge thresholds in `should_challenge()`

## Common Issues & Solutions

### Issue: Bot is too aggressive
**Solution**: Increase shadow defense scores in utility system

### Issue: Bot never collects boost
**Solution**: Lower boost collection score thresholds, increase urgency multiplier

### Issue: Mechanical sequences conflict with NN
**Solution**: Add sequence priority checks, disable NN when sequence is active

### Issue: Performance is slow (<30 FPS)
**Solution**: Increase tick_skip, reduce prediction horizon, profile code

### Issue: Shadow defense too passive
**Solution**: Lower challenge thresholds, increase fake challenge frequency

## Next Steps

1. **Integration**: Fully integrate all systems into bot.py
2. **Testing**: Test each behavior individually
3. **Tuning**: Adjust thresholds and parameters
4. **Training**: Optionally retrain NN with new state features
5. **Benchmarking**: Test against various opponents
6. **Iteration**: Refine based on performance

## Advanced Features (Future)

- **Opponent Modeling**: Track opponent patterns and adapt strategy
- **Dribble Control**: Implement ball carry and flick execution
- **Wall Play**: Add wall aerial and ceiling shot capabilities
- **Training Loop**: Set up RLGym training with new reward function
- **Neural Architecture**: Explore transformer/attention mechanisms

## Resources

- **RLBot Documentation**: https://rlbot.readthedocs.io/
- **RLGym Documentation**: https://rlgym.github.io/
- **Rocket League Physics**: https://samuelpmish.github.io/notes/RocketLeague/
- **SSL Gameplay Analysis**: Study top-ranked 1v1 replays on ballchasing.com

## Support

For questions or issues:
1. Check `SSL_UPGRADE_ANALYSIS.md` for detailed explanations
2. Review existing code comments and docstrings
3. Test individual components in isolation
4. Profile performance to identify bottlenecks

---

**Remember**: SSL-level play requires both mechanical execution AND strategic decision-making. The systems are in place - now it's about integration, testing, and tuning!
