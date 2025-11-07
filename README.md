# RL-Bot - Advanced Rocket League AI

<div align="center">

**A modular, SSL-level Rocket League bot featuring hybrid decision-making, hierarchical control, and advanced mechanics**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![RLBot](https://img.shields.io/badge/RLBot-1.x-green.svg)](https://rlbot.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 🚀 Quick Start

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/aaronwins356/RL-Bot.git
cd RL-Bot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the bot
rlbot gui
# Then add bot.cfg in the RLBot GUI
```

### Training (One Command)

**Windows (PowerShell):**
```powershell
# Single line training with professional display
.\train.ps1

# With custom settings
.\train.ps1 -Timesteps 10000000 -Device cuda -Config configs/base.yaml
```

**Linux/Mac (Bash):**
```bash
# Single line training with professional display
./train.sh

# With custom settings
./train.sh -t 10000000 -d cuda -c configs/base.yaml
```

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Training](#training)
- [Configuration](#configuration)
- [Development](#development)
- [Performance](#performance)
- [Contributing](#contributing)

---

## 🎯 Overview

RL-Bot is an advanced Rocket League bot combining cutting-edge reinforcement learning with rule-based tactics to achieve SSL-level performance. The bot features:

- **Hybrid Policy System**: Intelligent routing between ML and rule-based policies
- **Hierarchical Control**: 3-layer architecture for advanced mechanics
- **11 Skill Programs**: Fast aerials, ceiling shots, flip resets, musty flicks, breezi, double taps, and more
- **Modular Architecture**: Clean, testable, and extensible codebase
- **Comprehensive Training**: 9-stage curriculum with performance-based progression

### Current Capabilities

✅ **Fast Reactions** - 30 FPS decision-making (tick_skip=4)  
✅ **Ball Prediction** - 4-second trajectory forecasting  
✅ **Boost Control** - Strategic collection and denial  
✅ **Advanced Mechanics** - Speedflips, wavedash, halfflips, fast aerials  
✅ **Team Play** - Rotation awareness and positioning  
✅ **Adaptive Strategy** - Utility-based decision making

---

## ✨ Features

### Hybrid Policy System

The bot intelligently routes between rule-based and ML policies based on context and confidence:

**Rule Policy** handles:
- Kickoffs (based on spawn position)
- Defensive positioning and saves
- Aerial detection and execution
- Boost management
- Safe rotation

**ML Policy** handles:
- General gameplay with high confidence
- Complex decision-making
- Learned behaviors

**Hybrid Routing**:
- Uses rules for kickoffs, low confidence, OOD detection
- Uses ML for high-confidence general play
- Smooth transitions with statistics tracking

### Hierarchical Control System

**Layer A - Opportunity Detector (OD)**:
- Bi-LSTM or small transformer with 0.5s context window
- Classifies game state: Ground Play, Wall Play, Aerial, Flashy Opportunity
- Thompson sampling for skill program selection
- Risk-aware temperature-controlled sampling

**Layer B - Skill Programs (11 total)**:
1. Fast Aerial - 2-jump pattern (10-12 frame timing)
2. Aerial Control - Body orientation and angular velocity damping
3. Wall Read - Wall bounce prediction
4. Backboard Read - Backboard bounce and double-tap escalation
5. Ceiling Setup - Setup for ceiling shot
6. Ceiling Shot - Execute ceiling shot
7. Flip Reset - 4-wheel detection and execution
8. Musty Flick - 60-110° nose angle mechanic
9. Breezi - 5-9 Hz oscillatory air-roll
10. Double Tap - Backboard double tap
11. Ground to Air Dribble - Dribble transition

**Layer C - Low-Level Controller (LLC)**:
- PID controllers with anti-windup
- Fast aerial helper
- Flip reset detector (20ms window)
- Air roll stabilizer

### Observation System

**180-dimensional feature vector** including:
- Car State (22 features): position, velocity, rotation, boost, flags
- Ball State (9 features): position, velocity, angular velocity
- Ball Relative (6 features): relative position and velocity
- Ball Prediction (4 features): intercept position and time
- Aerial Features (7 features): height buckets, opportunity flags, alignment
- Teammates (26 features): up to 2 teammates
- Opponents (39 features): up to 3 opponents
- Boost Pads (60 features): nearest 10 pads
- Game State (3 features): kickoff, time, score
- Phase Encoding (4 features): KICKOFF/OFFENSE/DEFENSE/NEUTRAL

---

## 🏗️ Architecture

### Project Structure

```
RL-Bot/
├── core/                          # Core modular components
│   ├── agents/                    # Policy implementations
│   │   ├── rule_policy.py         # Rule-based tactical AI
│   │   ├── ml_policy.py           # ML inference with confidence
│   │   ├── hybrid_policy.py       # Intelligent policy routing
│   │   └── intents.py             # High-level action intents
│   │
│   ├── hierarchical_controller.py # 3-layer hierarchical control
│   │
│   ├── opportunity_detector/      # Game state classification
│   │   ├── detector.py            # OD model with Thompson sampling
│   │   └── risk_scorer.py         # Risk assessment
│   │
│   ├── skill_programs/            # 11 modular micro-policies
│   │   ├── base.py                # SkillProgram base class
│   │   ├── fast_aerial.py         # Fast aerial
│   │   ├── ceiling_shot.py        # Ceiling shot
│   │   ├── flip_reset.py          # Flip reset
│   │   └── ...                    # More skill programs
│   │
│   ├── llc/                       # Low-level controller
│   │   └── __init__.py            # PID, helpers, detectors
│   │
│   ├── env/                       # Environment and wrappers
│   │   ├── rocket_sim_env.py      # Gym-compatible RL environment
│   │   └── wrappers.py            # Observation/reward wrappers
│   │
│   ├── features/                  # Feature engineering
│   │   └── encoder.py             # 180-feature encoder
│   │
│   ├── models/                    # Neural network architectures
│   │   ├── ppo.py                 # PPO implementation
│   │   └── nets.py                # MLP, CNN-LSTM networks
│   │
│   ├── training/                  # Training infrastructure
│   │   ├── train_loop.py          # Main training loop
│   │   ├── buffer.py              # Experience replay
│   │   ├── eval.py                # Elo rating and evaluation
│   │   ├── selfplay.py            # Self-play curriculum
│   │   └── hierarchical_rewards.py # Advanced reward shaping
│   │
│   └── infra/                     # Infrastructure utilities
│       ├── config.py              # YAML configuration
│       ├── logging.py             # TensorBoard logging
│       └── checkpoints.py         # Model checkpointing
│
├── configs/                       # Configuration files
│   ├── base.yaml                  # Training and network config
│   ├── rewards.yaml               # Reward shaping
│   └── hierarchical_rl.yaml       # Hierarchical system config
│
├── scripts/                       # Command-line scripts
│   ├── train.py                   # Training script
│   └── evaluate.py                # Evaluation script
│
├── tests/                         # Unit tests (39+ tests)
│   ├── test_encoder.py
│   ├── test_hybrid_policy.py
│   ├── hierarchical/
│   │   └── test_hierarchical.py   # 16 unit tests
│   └── ...
│
├── bot.py                         # RLBot integration
├── train.ps1                      # PowerShell training launcher (Windows)
├── train.sh                       # Bash training launcher (Linux/Mac)
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🎓 Training

### 9-Stage Progressive Curriculum

The training system uses performance-based transitions through 9 stages:

#### Stage 0: Basic 1v1 Ground Play (0-500k steps)
- Basic ball control and ground movement
- Opponent: Basic script (70% speed)

#### Stage 1: Boost Control & Management (500k-1.5M)
- Boost collection, conservation, and efficiency
- Opponent: Rule policy (90% speed)

#### Stage 2: Kickoff Micro-Strategy (1.5M-2.5M)
- Kickoff positioning, fast kickoffs, boost stealing
- Opponent: Rule policy (100% speed)

#### Stage 3: Aerial Basics & Defense (2.5M-4M)
- Basic aerial touches, defense, jump timing
- Opponent: Self-play

#### Stage 4: Advanced Aerial Play (4M-5.5M)
- Aerial shots, redirects, fast aerial mechanics
- Game Mode: 2v2, Self-play

#### Stage 5: 2v2 Rotation Focus (5.5M-7M)
- Team positioning, rotation mechanics, passing
- Heavy rotation penalties (0.6 weight)

#### Stage 6: 1v2 Defense Training (7M-8.5M)
- Outnumbered defense, shadow defense, buying time
- Game Mode: 1v2, Opponents at 110% speed

#### Stage 7: 3v3 Team Play (8.5M-10M)
- 3v3 positioning, team coordination, advanced rotations
- Game Mode: 3v3, Self-play

#### Stage 8: Pro-Level 3v3 Chaos (10M+)
- High-speed gameplay, complex team plays, pro mechanics
- Opponents at 115% speed, maximum difficulty

### Training Commands

```bash
# Basic training with defaults
python scripts/train.py --config configs/base.yaml

# Training with aerial curriculum
python scripts/train.py --config configs/base.yaml --aerial-curriculum

# Training with specific stage (for testing)
python scripts/train.py --config configs/base.yaml --curriculum-stage 3

# Debug mode (short run with verbose logging)
python scripts/train.py --debug --timesteps 1000

# With offline pretraining (behavioral cloning)
python scripts/train.py --config configs/base.yaml --offline-pretrain

# Custom settings
python scripts/train.py \
  --config configs/base.yaml \
  --timesteps 10000000 \
  --device cuda \
  --logdir logs/my_run
```

### Unified Training Scripts

**Windows (PowerShell):**
```powershell
# Simple one-line training
.\train.ps1

# With custom options
.\train.ps1 -Timesteps 10000000 -Device cuda -AerialCurriculum

# Force specific stage
.\train.ps1 -CurriculumStage 3 -Timesteps 500000

# Debug mode
.\train.ps1 -Debug -Timesteps 1000
```

**Linux/Mac (Bash):**
```bash
# Simple one-line training
./train.sh

# With custom options
./train.sh -t 10000000 -d cuda -a

# Force specific stage
./train.sh -s 3 -t 500000

# Debug mode
./train.sh -D -t 1000
```

### Performance-Based Transitions

Stages automatically advance when ALL criteria are met:
- **Win rate** ≥ 60% against current opponent
- **Elo rating** ≥ 1400
- **Games played** ≥ 100 in current stage
- **Minimum timesteps** ≥ 100k in current stage

### Evaluation

```bash
# Basic evaluation
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt

# Against multiple opponents with plots
python scripts/evaluate.py \
  --checkpoint checkpoints/best_model.pt \
  --opponents rule_policy baseline_ml nexto \
  --num-games 20 \
  --plot \
  --k-factor 32
```

---

## ⚙️ Configuration

### Main Configuration Files

#### `configs/base.yaml` - Core Training Settings

```yaml
training:
  algorithm: "ppo"              # or "sac"
  total_timesteps: 10000000
  batch_size: 4096
  learning_rate: 3.0e-4
  device: "cuda"                # or "cpu"
  
  curriculum:
    use_performance_transitions: true
    transition_win_rate: 0.6
    transition_elo: 1400

network:
  hidden_sizes: [512, 512, 256]
  activation: "relu"
  use_lstm: false
```

#### `configs/rewards.yaml` - Reward Shaping

```yaml
sparse_rewards:
  goal_scored: 10.0
  goal_conceded: -10.0
  demo: 2.0
  save: 3.0
  shot: 1.0

dense_rewards:
  ball_velocity_toward_goal: 0.2
  boost_pickup: 0.1
  good_rotation: 0.1
  touch_bonus: 0.1

aerial_rewards:
  aerial_touch: 0.5
  aerial_goal: 5.0
  aerial_positioning: 0.1
  missed_aerial: -0.3

penalties:
  double_commit: -0.5
  boost_waste: -0.02
  ball_chasing: -0.3
```

#### `configs/hierarchical_rl.yaml` - Hierarchical System

```yaml
curriculum:
  gates:
    fast_aerial:
      threshold: 0.88           # 88% success required
    flip_reset:
      threshold: 0.35           # 35% clean resets
      convert: 0.20             # 20% conversions

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

---

## 🔧 Development

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test categories
pytest tests/test_eval_elo.py -v          # Elo rating tests
pytest tests/test_curriculum.py -v        # Curriculum tests
pytest tests/test_encoder.py -v           # Encoder tests
pytest tests/hierarchical/ -v             # Hierarchical system tests

# With coverage
pytest tests/ --cov=core --cov-report=html
```

### Code Quality

```bash
# Format code
black core/ scripts/ tests/

# Lint code
flake8 core/ scripts/

# Type checking
mypy core/
```

### Adding New Features

1. Follow the modular architecture
2. Add comprehensive docstrings
3. Write unit tests
4. Update configuration files
5. Update this README

---

## 📊 Performance

### Expected Results

After training for **10M steps** with curriculum:

| Metric | Target | Typical Results |
|--------|--------|----------------|
| Elo Rating | 1600+ | 1550-1650 |
| Win Rate vs Rule Policy | 70%+ | 65-75% |
| Win Rate vs Baseline ML | 60%+ | 55-65% |
| Aerial Success Rate | 50%+ | 45-55% |
| Boost Efficiency | 70%+ | 65-75% |
| Training Time (GPU) | 12-18h | ~15h |
| Training Time (CPU) | 48-72h | ~60h |

### Curriculum Progression

Expected Elo by stage:
- **Stage 0 (1M)**: 1400-1450 - Learning basics
- **Stage 1 (3M)**: 1500-1550 - Tactical fundamentals
- **Stage 2 (5M)**: 1550-1600 - Team play emerging
- **Stage 3 (8M)**: 1600-1650 - Mechanical proficiency
- **Stage 4 (10M+)**: 1650-1700 - Advanced multi-agent play

### SSL-Level Features

What makes this bot SSL-level:

1. **Ball Reading** - 4 seconds ahead with bounce prediction
2. **Boost Control** - 70%+ time with >50 boost
3. **Smart Positioning** - Shadow defense, space control
4. **Mechanical Excellence** - Fast aerials, wavedash, halfflip, speedflip
5. **Adaptive Strategy** - Utility-based decision making

---

## 🤝 Contributing

Contributions are welcome! Areas of focus:

- Hyperparameter tuning
- Additional test coverage
- Documentation improvements
- Advanced mechanics (wall play, dribbling)
- Performance optimization

### Development Setup

```bash
# Clone repository
git clone https://github.com/aaronwins356/RL-Bot.git
cd RL-Bot

# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy

# Run tests
pytest tests/ -v

# Format code
black .
```

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Credits

- **Original Bot**: Based on RLGym training framework
- **RLBot Framework**: https://github.com/RLBot/RLBot
- **RLGym**: https://github.com/lucas-emery/rocket-league-gym

---

## 📚 Documentation

### Quick References

- **Installation**: See [Quick Start](#quick-start)
- **Training**: See [Training](#training) section
- **Configuration**: See [Configuration](#configuration) section
- **Development**: See [Development](#development) section

### Version History

See [Changelog.md](Changelog.md) for detailed version history and updates.

---

## 🎮 Usage Examples

### Using the Hybrid Policy

```python
from pathlib import Path
from core.agents.hybrid_policy import HybridPolicy
from core.features.encoder import RawObservation

# Initialize hybrid policy
policy = HybridPolicy(
    model_path=Path("checkpoints/best_model.pt"),
    config={"hybrid": {"confidence_threshold": 0.7}}
)

# Get action from game state
obs = RawObservation(...)  # Create from game packet
action = policy.get_action(obs)
```

### Training with Custom Environment

```python
from core.env.rocket_sim_env import RocketSimEnv
from core.env.wrappers import NormalizeObservation, FrameStack
from core.training.train_loop import TrainingLoop
from core.infra.config import load_config

# Create environment with wrappers
env = RocketSimEnv(
    reward_config_path=Path("configs/rewards.yaml"),
    enable_aerial_training=True
)
env = NormalizeObservation(env)
env = FrameStack(env, num_stack=4)

# Load config and train
config = load_config(Path("configs/base.yaml"))
trainer = TrainingLoop(config, log_dir="logs")
trainer.train(total_timesteps=10_000_000)
```

### Hierarchical Controller

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

---

## 🔍 Troubleshooting

### Common Issues

**Bot not moving:**
- Check RLBot GUI shows "Connected"
- Verify Rocket League is running
- Try starting match from RLBot GUI

**Import errors:**
```bash
pip install --upgrade rlbot numpy torch omegaconf
```

**Training too slow:**
- Reduce `num_envs` in config
- Use smaller batch_size
- Enable `use_amp` for GPU training

**CUDA errors:**
```bash
# Training automatically falls back to CPU
python scripts/train.py --device cpu
```

**Windows torch.compile error:**
- Automatically handled with eager backend
- You'll see: `[Windows] Triton not supported, using eager backend`

---

## 📈 Monitoring Training

### TensorBoard

```bash
tensorboard --logdir logs/
# Open http://localhost:6006
```

### Key Metrics to Watch

1. **Elo Progression** - Should steadily increase
2. **Training Speed** - Target: ≥15 timesteps/sec
3. **GPU Utilization** - Target: ≥40%
4. **Entropy Decay** - Should decrease from 0.02 to 0.005
5. **KL Divergence** - Should stay in 0.01-0.03 range

### Output Files

```
logs/{run_id}/
├── run_metadata.json          # Full config and git info
├── tensorboard/               # TensorBoard logs
├── checkpoints/
│   ├── checkpoint_50000.pt
│   ├── best_model.pt          # Best by Elo
│   └── latest_model.pt
└── evaluation/
    ├── eval_summary.csv       # Per-opponent summaries
    ├── game_by_game.csv       # Detailed game results
    └── elo_history.png        # Elo curve plot
```

---

<div align="center">

**Built with ❤️ for the Rocket League community**

[Report Bug](https://github.com/aaronwins356/RL-Bot/issues) · [Request Feature](https://github.com/aaronwins356/RL-Bot/issues)

</div>
