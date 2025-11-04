# Development Documentation

## Project Structure

### Core Components

#### 1. Behavior Tree Framework
The behavior tree is a hierarchical decision-making system located in `rlbot_pro/planning/selector/`:

- `tree.py`: Core behavior tree components
  - `Node`: Abstract base class for all behavior tree nodes
  - `Selector`: Tries children until one succeeds
  - `Sequence`: Runs all children in sequence
  - `BehaviorTree`: Main tree class with configuration loading

- `nodes.py`: Tactical nodes implementing specific behaviors
  - `DefensiveRotation`: Handles defensive positioning
  - `Approach`: Ball approach and positioning
  - `FirstTouch`: Initial ball contact decisions
  - `CarryControl`: Ball carrying and dribbling
  - `FreestyleSelector`: Advanced mechanic selection
  - `FallbackClear`: Emergency clear when other options fail

Configuration is handled via `config/settings.yaml`, controlling:
- Mechanic enable/disable flags
- Aggression parameters
- Shot selection thresholds
- Safety fallback settings

#### 2. Training System

The training system uses RLGym for reinforcement learning:

- `training/envs/scenarios.py`: Custom training scenarios
  - Aerial intercept
  - Wall carry
  - Dribble control
  - Flip reset
  Each scenario includes:
  - Custom observation builder
  - Shaped reward function
  - Terminal conditions
  - State initialization

- `training/rewards/rewards.py`: Reward functions
  - Touch quality rewards
  - Shot placement rewards
  - Recovery efficiency
  - Style bonuses for advanced mechanics

- `training/curriculum/scheduler.py`: Progressive training
  - Stage-based progression
  - Performance thresholds
  - Mechanic enabling

#### 3. Evaluation System

Evaluation tools in `metrics/` and `reports/`:

- `metrics/evaluator.py`: Performance metrics
  - Per-mechanic success rates
  - Shot quality analysis
  - Recovery efficiency
  - Boost management

- `reports/generate_report.py`: Reporting
  - Markdown report generation
  - Performance visualizations
  - Training progress plots

### Development Workflow

1. Adding New Mechanics:
   ```python
   # 1. Create mechanic implementation
   class NewMechanic(Node):
       def tick(self, state: GameState) -> NodeStatus:
           # Implementation
           pass
           
   # 2. Add to FreestyleSelector options
   def _select_best_mechanic(self):
       mechanics = {
           'new_mechanic': NewMechanic()
           # ...
       }
           
   # 3. Add configuration
   mechanics:
     new_mechanic: true
   ```

2. Adding Training Scenarios:
   ```python
   class NewScenario:
       class ObsBuilder(ObsBuilder):
           def build_obs(self, player, state, previous_action):
               # Custom observations
               
       class Reward(RewardFunction):
           def get_reward(self, player, state, previous_action):
               # Shaped rewards
               
       class Terminal(TerminalCondition):
           def is_terminal(self, current_state):
               # Terminal conditions
               
       class State(StateSetter):
           def reset(self, state_wrapper):
               # Initial state setup
   ```

3. Adding Evaluation Metrics:
   ```python
   def _evaluate_new_metric(self, model):
       results = {}
       # Metric computation
       return results
   ```

### Testing

1. Unit Tests:
   - Behavior tree node tests
   - Reward function tests
   - State management tests

2. Integration Tests:
   - Full behavior tree execution
   - Training scenario completion
   - Evaluation pipeline

3. Performance Tests:
   - Training stability
   - Inference speed
   - Memory usage

### Best Practices

1. Code Organization:
   - Keep behavior tree nodes focused and single-purpose
   - Use composition over inheritance
   - Implement fallback behaviors

2. Training:
   - Start with simple scenarios
   - Gradually increase complexity
   - Monitor reward shaping carefully
   - Save checkpoints frequently

3. Testing:
   - Write tests for new mechanics
   - Verify reward functions
   - Test edge cases
   - Profile performance impact