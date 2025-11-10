# 🚀 RocketMind - Next-Generation Rocket League AI

A comprehensive reinforcement learning bot system for Rocket League featuring modern PPO implementation, RLGym 2.0+ integration, RocketSim simulation, and interactive Streamlit dashboard.

**Built with:** RLGym 2.0+, Gymnasium, PyTorch 2.2+, RocketSim

## ✨ Highlights

- **✅ RLGym 2.0+ Compatible** - Fully migrated to the modern unified RLGym API
- **🏎️ RocketSim Engine** - Fast, physics-accurate Rocket League simulation
- **🎮 RLBot Integration** - Deploy trained models to actual Rocket League matches
- **📊 Streamlit Dashboard** - Real-time training visualization and control
- **⚡ Modern PPO** - With AMP, torch.compile(), adaptive LR, and more
- **🎯 Modular Design** - Easy to extend with custom rewards, observations, and behaviors

---

## 🎯 Features

### Core System
- **Modern PPO Implementation**: Clean, optimized PPO with GAE, clipping, and normalization
- **RLGym 2.0+ API**: Latest unified API with RocketSimEngine for fast simulation
- **Vectorized Training**: Parallel environments for faster training
- **Ball Prediction**: Physics-based trajectory prediction
- **Team Play Support**: 1v1, 2v2, 3v3 with teammate/opponent awareness
- **Advanced Rewards**: Positioning, rotation, aerials, demos, and game sense
- **Modular Behaviors**: Kickoff, recovery, and boost management
- **Automatic Device Detection**: Seamless CPU/GPU training
- **Checkpoint Management**: Auto-save with best model tracking
- **TensorBoard Integration**: Comprehensive logging and metrics

### 🚀 RocketMind System
Next-generation PPO bot with cutting-edge features:

**Advanced Training:**
- Automatic Mixed Precision (AMP) training
- torch.compile() optimization (PyTorch 2.x)
- Optional LSTM for temporal awareness
- Adaptive learning rate and entropy scheduling
- Gradient clipping and layer normalization
- Reward normalization and advantage scaling

**🎮 RLBot Integration:**
- Full RLBot Framework compatibility
- Deploy to live matches via RLBot GUI
- State parsing from GameTickPacket
- Action-to-controller conversion

**📊 Streamlit Dashboard:**
- Real-time training metrics
- Model comparison and evaluation
- Hyperparameter tuning interface
- Live replay visualization
- Training control (pause/resume)

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (optional, for GPU training)
- Git

### Quick Start

```bash
# Clone repository
git clone https://github.com/aaronwins356/RL-Bot.git
cd RL-Bot

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install RLGym and RocketSim
pip install rlgym-rocket-league[all]>=2.0.1

# Verify installation
python verify_rl_env.py
```

### Manual Installation (Alternative)

```bash
# Core dependencies
pip install torch>=2.2.0 numpy gymnasium>=0.29.0

# RLGym ecosystem
pip install rlgym-rocket-league[all]>=2.0.1
pip install rocketsim>=2.1

# Training and visualization
pip install tensorboard streamlit plotly wandb

# Optional: Development tools
pip install pytest black flake8
```

---

## 🚀 Quick Start

### 1. Basic Training

```bash
# Start training with default config
python main.py

# Or use RocketMind system
python -m rocketmind.main train

# With custom config
python main.py --config my_config.yaml

# Resume from checkpoint
python main.py --resume checkpoints/best_model.pt
```

### 2. Launch Dashboard

```bash
# Start Streamlit dashboard
python -m rocketmind.main dashboard

# Opens at http://localhost:8501
```

### 3. Evaluate Model

```bash
# Evaluate trained model
python rl_bot/eval.py --checkpoint checkpoints/best_model.pt --episodes 20
```

### 4. Deploy to RLBot

```bash
# Deploy to RLBot for live matches
python -m rocketmind.main deploy --checkpoint checkpoints/best_model.pt

# Then add bot through RLBot GUI using generated bot.cfg
```

---

## ⚙️ Configuration

### Training Configuration (`config.yaml`)

```yaml
# Training hyperparameters
training:
  total_timesteps: 10_000_000
  batch_size: 4096
  n_steps: 2048
  n_epochs: 10
  learning_rate: 3e-4
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 0.5
  use_adaptive_lr: true

# Environment settings
environment:
  num_envs: 4
  team_size: 1
  tick_skip: 8
  timeout_seconds: 300
  spawn_opponents: true
  obs_builder: 'simple'  # or 'team_aware', 'compact'

# Network architecture
network:
  hidden_sizes: [512, 512, 256]
  activation: 'relu'
  use_layer_norm: false

# Rewards (weights for different components)
rewards:
  goal_scored: 10.0
  goal_conceded: -10.0
  touch_ball: 0.5
  touch_ball_aerial: 1.0
  velocity_ball_to_goal: 0.5
  save: 3.0
  demo: 2.0
  boost_pickup: 0.1
  positioning_weight: 0.1
  rotation_weight: 0.05

# Logging
logging:
  log_interval: 10
  save_interval: 500
```

---

## 📚 Architecture

### Project Structure

```
RL-Bot/
├── rl_bot/                 # Core RL bot implementation
│   ├── core/               # Core components
│   │   ├── env_setup.py    # Environment creation (RLGym 2.0+)
│   │   ├── model.py        # PPO actor-critic network
│   │   ├── agent.py        # Modular agent with behaviors
│   │   ├── behaviors.py    # Kickoff, recovery, boost management
│   │   ├── reward_functions.py  # Custom reward components
│   │   ├── advanced_obs.py # Team-aware observation builders
│   │   └── utils.py        # Utilities and helpers
│   ├── train.py            # Main PPO training loop
│   └── eval.py             # Model evaluation
│
├── rocketmind/             # RocketMind system
│   ├── ppo_core/           # Advanced PPO implementation
│   │   ├── agent.py        # PPO agent
│   │   ├── trainer.py      # PPO trainer with AMP
│   │   ├── network.py      # Neural network architectures
│   │   ├── memory.py       # Rollout buffer
│   │   └── hyperopt.py     # Hyperparameter optimization
│   ├── rlbot_interface/    # RLBot integration
│   │   ├── rocket_agent.py # RLBot agent wrapper
│   │   ├── rlbot_adapter.py # State/action conversion
│   │   └── state_parser.py # GameTickPacket parsing
│   ├── visualization/      # Dashboard and plotting
│   ├── envs/               # Environment wrappers
│   ├── train.py            # Training script
│   ├── main.py             # CLI entry point
│   └── streamlit_app.py    # Dashboard application
│
├── rlgym_sim/              # Compatibility shim for legacy imports
├── main.py                 # Main training entry point
├── config.yaml             # Default configuration
├── requirements.txt        # Dependencies
└── verify_rl_env.py        # Environment verification script
```

### Key Components

#### 1. Environment Setup (`rl_bot/core/env_setup.py`)
- Uses RLGym 2.0+ API with RocketSimEngine
- Supports custom observation builders, action parsers, and rewards
- Vectorized environment wrapper for parallel training

#### 2. PPO Trainer (`rl_bot/train.py`, `rocketmind/ppo_core/trainer.py`)
- Modern PPO implementation with GAE
- Automatic mixed precision training
- Learning rate scheduling
- Reward normalization

#### 3. Neural Network (`rl_bot/core/model.py`)
- Actor-Critic architecture
- Customizable hidden layers
- Layer normalization support
- Optional LSTM for recurrent policies

#### 4. Reward Functions (`rl_bot/core/reward_functions.py`)
- Modular reward components
- Goal scoring, ball touching, aerial rewards
- Positioning, rotation, and game sense rewards
- Easy to add custom rewards

#### 5. Behaviors (`rl_bot/core/behaviors.py`)
- Kickoff manager for fast kickoffs
- Recovery manager for aerial recovery
- Boost management for efficient boost usage
- Can override learned policy when appropriate

---

## 🔧 Advanced Usage

### Custom Observation Builder

```python
from rlgym.api import ObsBuilder

class MyObsBuilder(ObsBuilder):
    def build_obs(self, agents, state, shared_info):
        obs = {}
        for agent_id in agents:
            # Build custom observation for each agent
            car = state.cars[agent_id]
            ball = state.ball
            
            obs[agent_id] = np.concatenate([
                car.position / 4096,
                car.linear_velocity / 2300,
                ball.position / 4096,
                ball.linear_velocity / 6000,
            ])
        return obs
```

### Custom Reward Function

```python
from rlgym.api import RewardFunction

class MyReward(RewardFunction):
    def compute_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        for agent_id in agents:
            car = state.cars[agent_id]
            ball = state.ball
            
            # Calculate reward based on distance to ball
            dist_to_ball = np.linalg.norm(ball.position - car.position)
            rewards[agent_id] = -dist_to_ball / 5000
        return rewards
```

### Multi-Agent Training

```python
# config.yaml
environment:
  team_size: 2  # 2v2
  spawn_opponents: true
  obs_builder: 'team_aware'  # Use team-aware observations
```

---

## 📊 Monitoring Training

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir logs

# Opens at http://localhost:6006
```

### Weights & Biases

```python
# In config.yaml
logging:
  use_wandb: true
  wandb_project: "rocketmind"
  wandb_entity: "your-username"
```

### Streamlit Dashboard

```bash
python -m rocketmind.main dashboard
```

Features:
- Real-time loss and reward plots
- Hyperparameter comparison
- Model evaluation metrics
- Training control interface

---

## 🎮 Deploying to RLBot

### 1. Train a Model

```bash
python main.py --timesteps 5000000
```

### 2. Deploy via RocketMind

```bash
python -m rocketmind.main deploy --checkpoint checkpoints/best_model.pt
```

### 3. Add to RLBot GUI
1. Open RLBot GUI
2. Add bot using generated `bot.cfg`
3. Start match!

### 4. Manual Deployment

Create `bot.cfg`:
```ini
[Bot Parameters]
name = RocketMind Bot
python_file = rocketmind/rlbot_interface/rocket_agent.py
```

Add to RLBot match configuration and launch.

---

## 🐛 Troubleshooting

### Environment Creation Fails

```bash
# Verify RLGym installation
python -c "from rlgym.rocket_league import make; print('OK')"

# Verify RocketSim
python -c "import RocketSim; print('OK')"

# Run full verification
python verify_rl_env.py
```

### CUDA Out of Memory

```yaml
# Reduce batch size in config.yaml
training:
  batch_size: 2048  # Reduced from 4096
  
environment:
  num_envs: 2  # Reduced from 4
```

### Slow Training

- Enable AMP (Automatic Mixed Precision)
- Use torch.compile() (PyTorch 2.x)
- Increase num_envs for parallel training
- Use GPU if available

### Import Errors

The project includes compatibility shims for legacy `rlgym_sim` imports. If you encounter import errors:

```bash
# Ensure you have the latest RLGym
pip install --upgrade rlgym-rocket-league[all]

# Check compatibility shim
ls -la rlgym_sim/
```

---

## 🔬 Development

### Running Tests

```bash
# Install dev dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/

# With coverage
pytest --cov=rl_bot --cov=rocketmind tests/
```

### Code Formatting

```bash
# Format code
black rl_bot/ rocketmind/

# Check linting
flake8 rl_bot/ rocketmind/
```

---

## 📈 Performance Tips

1. **Use GPU**: 3-5x faster than CPU
2. **Vectorized Envs**: Train with 4-8 parallel environments
3. **Batch Size**: Larger batches (4096-8192) often work better
4. **Learning Rate**: Start with 3e-4, decrease if unstable
5. **Early Stopping**: Monitor validation performance
6. **Curriculum Learning**: Start with simpler tasks, gradually increase complexity

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- **RLGym Team** - For the excellent RLGym 2.0+ framework
- **RocketSim** - For fast, accurate Rocket League simulation
- **RLBot Community** - For RLBot framework and support
- **PyTorch Team** - For the deep learning framework
- **OpenAI** - For PPO algorithm research

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/aaronwins356/RL-Bot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/aaronwins356/RL-Bot/discussions)
- **RLGym Discord**: [rlgym.org](https://rlgym.org)
- **RLBot Discord**: [rlbot.org](https://rlbot.org)

---

## 📚 Resources

- [RLGym Documentation](https://rlgym.org/)
- [RocketSim](https://github.com/ZealanL/RocketSim)
- [RLBot Documentation](https://rlbot.org/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [Rocket League Gym Guide](https://github.com/ZealanL/RLGym-PPO-Guide)

---

**Happy Training! 🚀⚽**
