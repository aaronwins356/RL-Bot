# Hierarchical Control + Curriculum RL/IL Pipeline

This implementation provides a comprehensive hierarchical control system for advanced Rocket League mechanics including ceiling shots, flip resets, musty flicks, breezi, double taps, and wall-to-air dribbles.

## Architecture Overview

The system consists of three layers:

### Layer A: Opportunity Detector (OD)
- **Model**: Bi-LSTM (hidden 256) or small transformer with 0.5s context window
- **Function**: Classifies game state into {Ground Play, Wall Play, Aerial Opportunity, Flashy Opportunity}
- **Selection**: Uses Thompson sampling bandit to select Skill Program (SP)
- **Risk-Aware**: Temperature-controlled sampling with floor on safe options when risk is high

### Layer B: Skill Programs (SP)
Modular micro-policies with timeouts and fallback sequences:

1. **SP_FastAerial** - Fast aerial with 2-jump pattern
2. **SP_AerialControl** - Orient body and damp angular velocity
3. **SP_WallRead** - Wall bounce prediction
4. **SP_BackboardRead** - Backboard bounce prediction and double-tap escalation
5. **SP_CeilingSetup** - Setup for ceiling shot
6. **SP_CeilingShot** - Execute ceiling shot
7. **SP_FlipReset** - Flip reset detection and execution
8. **SP_Musty** - Musty flick mechanic
9. **SP_Breezi** - Breezi with oscillatory air-roll
10. **SP_DoubleTap** - Backboard double tap
11. **SP_GroundToAirDribble** - Ground to air dribble

### Layer C: Low-Level Controller (LLC)
Converts SP targets to control surfaces:
- **PID Controllers** - Orientation and position control with anti-windup
- **Fast Aerial Helper** - 2-jump timing (10-12 frames between jumps)
- **Flip Reset Detector** - 4-wheel contact detection (20ms window)
- **Air Roll Stabilizer** - Angular velocity damping

## Configuration

See `configs/hierarchical_rl.yaml` for full configuration including:

- **Curriculum Settings** - Stage progression and promotion rules
- **Gate Thresholds** - Pass/fail criteria for each mechanic
- **Reward Shaping** - Contact quality, style bonuses, safety costs
- **Controller Parameters** - PID gains, timeouts, timing constraints
- **Training Settings** - IL→RL pipeline, data augmentation, self-play

### Key Configuration Sections

```yaml
curriculum:
  gates:
    fast_aerial:
      threshold: 0.88  # 88% success rate required
    flip_reset:
      threshold: 0.35  # 35% clean resets
      convert: 0.20    # 20% immediate conversions

rewards:
  contact: 1.0
  ceiling_bonus: 0.6
  flipreset_goal_bonus: 1.0
  boost_cost: -0.004

controllers:
  fast_aerial:
    inter_jump_frames: [10, 12]
    pitch_up: [0.6, 0.9]
  
  breezi:
    roll_freq_hz: [5, 9]
    roll_amp: [0.12, 0.25]
```

## Usage

### Basic Usage

```python
from core.hierarchical_controller import HierarchicalController
import yaml

# Load config
with open('configs/hierarchical_rl.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize controller
controller = HierarchicalController(config, device='cpu')

# Run control loop
controller.reset()
for step in range(1000):
    action = controller.get_action(obs)
    obs, reward, done, info = env.step(action)
```

### Demo Script

Run the demo to see the hierarchical controller in action:

```bash
python scripts/demo_hierarchical.py
```

### Training with Drills

```python
from core.training.drill_evaluator import DrillEvaluator

# Initialize evaluator
evaluator = DrillEvaluator(config['curriculum'])

# Evaluate all gates
results = evaluator.evaluate_all_gates(policy)

# Save results
evaluator.save_results(results, 'evaluation/drill_results.json')
```

## Reward Shaping

The system implements sophisticated reward shaping from §5:

### Hit Quality
- **Contact**: +1.0 for valid aerial/flashy contact
- **Speed Bonus**: +0.25 if ball speed increases > 800 uu/s toward goal
- **Target Cone**: +0.5 for shot within 10° cone
- **Goal Proximity**: +0.002 per unit projection toward goal

### Setup Fidelity
- **Fast Aerial Timing**: +0.2 for correct timing (10-12 frames)
- **Flip Reset Contact**: +0.4 for 4-wheel contact

### Style Bonuses (only when Flashy OK)
- **Ceiling Shot**: +0.6 on successful conversion
- **Musty/Breezi**: +0.5 if on-target and ball speed increases
- **Double Tap**: +0.7 for second contact in window
- **Flip Reset Goal**: +1.0 for scoring with flip reset

### Safety Costs
- **Boost Usage**: -0.004 per unit (scaled by game state)
- **Whiff**: -0.2 near own box
- **Risky Flashy**: -0.15 when high risk and attempting flashy
- **Bad Recovery**: -0.3 if recovery > 1.0s
- **Residual Angular Velocity**: -0.05 per 0.1 rad/s at landing

## Evaluation Gates

From §6, the system must pass strict drill batteries:

| Gate | Attempts | Threshold | Success Criteria |
|------|----------|-----------|------------------|
| Fast Aerial | 50 | 88% | Contact within 120uu, speed > 1500uu/s |
| Double Tap | 30 | 40% | Secure double-tap, on-target |
| Ceiling | 25 | 60% | Meaningful shot (on-target or rebound) |
| Flip Reset | 40 | 35% clean, 20% convert | 4-wheel contact, immediate conversion |
| Musty | 25 | 30% | On-target, ball speed > 1000uu/s |
| Breezi | 25 | 30% | On-target, ball speed > 1000uu/s |

### Match KPIs

For ranked self-play enablement:
- **Aerial Contest Win-Rate**: +12% over baseline
- **Flashy Attempt Precision**: ≥75% (attempt only when viable)
- **Flashy Conversion**: ≥25% of attempts
- **Defensive Turnover**: <5% increase over baseline
- **Elo Delta**: +50 over 500 games

## Attempt Selection Heuristics

From §8, flashy mechanics are attempted only when ALL conditions are met:

1. **Possession**: Probability ≥ 0.6
2. **Time Advantage**: Opponent ETA - Own ETA ≥ 150ms
3. **Boost**: ≥ 40 (≥ 30 for musty specifically)
4. **Game State**: Score diff ≥ 0 OR time remaining ≥ 45s
5. **Safety**: Never when last defender with ball in defensive third

Additional mechanic-specific feasibility checks:
- **Ceiling**: Ball height > 1500uu
- **Flip Reset**: Ball at 200-1800uu height
- **Musty**: Ball 30-150uu above car
- **Breezi**: Approach angle 20-35°
- **Double Tap**: Ball velocity toward backboard > 500uu/s

## Feature Flags

The system uses feature flags for safe deployment:

```python
# Flashy mechanics disabled by default
controller.flashy_enabled = False

# Auto-enable after passing gates for 3 consecutive eval cycles
if all_gates_passed and consecutive_passes >= 3:
    controller.enable_flashy_mechanics(True)
```

## Telemetry

The system logs comprehensive telemetry:

```python
telemetry = controller.get_telemetry()

# SP choices and outcomes
telemetry['sp_choices']       # When each SP was selected
telemetry['sp_successes']     # Success/failure of each SP
telemetry['sp_timeouts']      # Timeout events
telemetry['confidence_history']  # OD confidence over time
telemetry['risk_scores']      # Risk assessment over time
```

## Training Pipeline

The IL→RL pipeline (§4):

1. **Behavior Cloning (BC)** - Pretrain on curated clips (30k-80k per SP)
2. **DAgger** - Data aggregation during scripted rollouts
3. **RL Fine-Tuning** - PPO/SAC with shaped rewards in task arenas
4. **Mixed Self-Play** - 60% drills, 25% scrimmages, 15% matches

### Data Augmentation

Randomization for robustness:
- Ball speed, spawn height, wall normal jitter
- Car starting yaw/pitch/roll
- Boost pad states
- Latency injection (2-6 frames)

## Module Structure

```
core/
├── hierarchical_controller.py    # Main 3-layer controller
├── opportunity_detector/
│   ├── detector.py               # OD model and bandit
│   └── risk_scorer.py            # Risk assessment
├── skill_programs/
│   ├── base.py                   # SkillProgram base class
│   ├── fast_aerial.py
│   ├── aerial_control.py
│   ├── flip_reset.py
│   ├── ceiling_shot.py
│   ├── musty.py
│   ├── breezi.py
│   ├── double_tap.py
│   └── ...
├── llc/
│   └── __init__.py               # PID, helpers, detectors
└── training/
    ├── hierarchical_rewards.py   # Reward shaping
    └── drill_evaluator.py        # Evaluation harness
```

## Testing

Run unit tests for the hierarchical system:

```bash
# Test fast aerial timing constraints
pytest tests/test_hierarchical.py::test_fast_aerial_timing

# Test flip reset detector
pytest tests/test_hierarchical.py::test_flip_reset_detector

# Test musty trigger conditions
pytest tests/test_hierarchical.py::test_musty_conditions

# Test breezi oscillation amplitude
pytest tests/test_hierarchical.py::test_breezi_oscillation
```

## Success Criteria

From §11, the system is ready for deployment when:

1. ✅ All drill thresholds met (§6.1)
2. ✅ All match KPIs achieved (§6.2)
3. ✅ Aerial contest win-rate maintained for 3 eval windows
4. ✅ No defensive turnover regression (>5%)

## Future Work

- Complete IL data collection from expert replays
- Implement MPPI for snap turns > 90°
- Add vision-based state estimation
- Extend to multi-agent coordination (2v2, 3v3)
- Advanced opponent modeling

## References

This implementation is based on the comprehensive problem statement in §1-11, incorporating:
- Three-layer hierarchical control architecture
- Curriculum learning with promotion gates
- IL→RL training pipeline
- Sophisticated reward shaping
- Risk-aware attempt selection
- Strict evaluation criteria

## License

MIT License - See main repository LICENSE file.
