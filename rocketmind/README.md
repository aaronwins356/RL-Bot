# 🚀 RocketMind - Next-Generation PPO Rocket League Bot

Advanced reinforcement learning bot for Rocket League built with Proximal Policy Optimization (PPO), featuring a modern Streamlit dashboard and full RLBot compatibility.

## ✨ Features

### 🧠 Core System
- **Advanced PPO Implementation** with modern optimizations
  - Automatic Mixed Precision (AMP) training
  - torch.compile() optimization (PyTorch 2.x)
  - Optional LSTM for temporal awareness
  - Adaptive learning rate and entropy scheduling
  - GAE (Generalized Advantage Estimation)
  - Gradient clipping and value function clipping

### 🎮 RLBot Integration
- **Full RLBot Framework compatibility**
  - Deploy trained bots in actual Rocket League matches
  - RLBot GUI launcher integration
  - State parsing from RLBot GameTickPacket
  - Action conversion to controller inputs

### 📊 Streamlit Dashboard
- **Interactive training visualization**
  - Real-time training metrics (reward, loss, KL divergence)
  - Live telemetry (ball speed, car position, boost %)
  - Performance monitoring (GPU util, FPS, throughput)
  - Skill progression tracker (radar chart)
  - Hyperparameter editor with live updates
  - Bot control panel (start/stop/launch)

### 🎬 Advanced Features
- **Replay system** with playback and analysis
- **Telemetry dashboard** for live match monitoring
- **WebSocket streaming** for external integrations
- **Field heatmaps** for position analysis
- **Adaptive reward sculpting** based on performance
- **Curriculum learning** support
- **Discord notifications** for training milestones
- **Distributed training** support (multi-GPU)

## 📁 Project Structure

```
rocketmind/
├── main.py                      # Main CLI entry point
├── train.py                     # Training script
├── streamlit_app.py             # Interactive dashboard
├── configs/
│   └── default.yaml             # Configuration file
├── ppo_core/                    # PPO implementation
│   ├── network.py               # Actor-Critic networks
│   ├── memory.py                # Rollout buffers
│   ├── trainer.py               # PPO trainer
│   ├── losses.py                # Loss functions
│   └── utils.py                 # Utilities
├── rlbot_interface/             # RLBot integration
│   ├── rocket_agent.py          # Main RLBot agent
│   ├── rlbot_adapter.py         # RLBot adapter
│   ├── state_parser.py          # State parsing
│   └── reward_functions.py      # Reward components
├── visualization/               # Visualization tools
│   ├── replay_viewer.py         # Replay playback
│   ├── telemetry_dashboard.py  # Live telemetry
│   └── rocket_stream.py         # WebSocket streaming
└── tests/
    └── test_ppo.py              # Unit tests
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/aaronwins356/RL-Bot.git
cd RL-Bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Start training with default configuration
python -m rocketmind.main train

# Train with custom config
python -m rocketmind.main train --config rocketmind/configs/custom.yaml

# Resume from checkpoint
python -m rocketmind.main train --resume checkpoints/rocketmind/checkpoint_1000000.pt
```

### Streamlit Dashboard

```bash
# Launch interactive dashboard
python -m rocketmind.main dashboard

# Open browser to http://localhost:8501
```

The dashboard provides:
- Real-time training metrics
- Live telemetry visualization
- Hyperparameter tuning
- Bot control panel
- Performance monitoring
- Skill progression tracking

### Deploy to RLBot

```bash
# Deploy trained bot to RLBot GUI
python -m rocketmind.main deploy --checkpoint checkpoints/rocketmind/best_model.pt

# This creates bot.cfg for RLBot
# Add the bot through RLBot GUI
```

## ⚙️ Configuration

Edit `rocketmind/configs/default.yaml` to customize:

```yaml
training:
  total_timesteps: 10_000_000
  batch_size: 4096
  learning_rate: 3.0e-4
  use_amp: true              # Automatic mixed precision
  
network:
  hidden_sizes: [512, 512, 256]
  use_lstm: false            # Optional recurrent policy
  use_torch_compile: true    # PyTorch 2.x optimization
  
environment:
  num_envs: 4
  team_size: 1               # 1v1, 2v2, or 3v3
  
rewards:
  goal_scored: 10.0
  ball_touch: 0.5
  aerial_touch: 1.0
  adaptive_rewards: true     # Adaptive reward sculpting
  
dashboard:
  enabled: true
  port: 8501
  enable_wandb: false        # Weights & Biases integration
```

## 📊 Monitoring Training

### TensorBoard

```bash
tensorboard --logdir logs/rocketmind
```

### Streamlit Dashboard

The dashboard auto-refreshes during training and displays:
- Training curves (reward, loss, entropy)
- Performance metrics (FPS, GPU usage)
- Live telemetry (if in spectate mode)
- Skill progression
- Replay viewer

## 🎯 Advanced Usage

### Custom Reward Functions

Add custom rewards in `rocketmind/rlbot_interface/reward_functions.py`:

```python
class MyCustomReward(RewardFunction):
    def calculate(self, state, prev_state):
        # Your reward logic
        return reward * self.weight
```

### Hyperparameter Tuning

Use the Streamlit dashboard's hyperparameter editor to adjust:
- Learning rate
- Clip range
- Entropy coefficient
- Reward weights

Changes apply on next training update.

### WebSocket Streaming

Enable live telemetry streaming:

```python
from rocketmind.visualization import RocketStream

stream = RocketStream(host="localhost", port=8765)
stream.start()
```

Connect clients to `ws://localhost:8765` for real-time data.

### Discord Notifications

Get training updates in Discord:

```python
from rocketmind.visualization import DiscordNotifier

notifier = DiscordNotifier(webhook_url="YOUR_WEBHOOK_URL")
await notifier.notify_training_milestone(timesteps, reward, checkpoint_path)
```

## 🧪 Testing

```bash
# Run unit tests
python -m pytest rocketmind/tests/

# Or use unittest
python rocketmind/tests/test_ppo.py
```

## 📈 Expected Performance

After ~10M timesteps (12-24 hours on GPU):
- **Win Rate**: 65-75% vs default opponents
- **Elo Rating**: 1200-1500
- **Skills**: Decent aerials, dribbling, positioning

## 🔧 Troubleshooting

### GPU Out of Memory
- Reduce `batch_size` in config
- Reduce `num_envs`
- Disable `use_amp`

### Training Too Slow
- Increase `num_envs` (if memory allows)
- Enable `use_amp`
- Enable `use_torch_compile`
- Increase `tick_skip`

### Dashboard Not Loading
```bash
pip install streamlit plotly
python -m rocketmind.main dashboard
```

### RLBot Integration Issues
```bash
# Install RLBot dependencies
pip install rlbot rlgym-compat
```

## 🏗️ Architecture

### PPO Core
- Modern PPO implementation with clipping
- Supports discrete and continuous actions
- GAE for advantage estimation
- Optional LSTM for recurrent policies

### RLBot Interface
- Converts RLBot state to observations
- Maps discrete actions to controller inputs
- Handles reward calculation
- Compatible with RLBot GUI

### Visualization
- Streamlit for interactive dashboard
- Plotly for charts and graphs
- WebSocket for real-time streaming
- Replay system for analysis

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional reward functions
- Self-play training
- Multi-agent support
- Advanced visualizations
- Curriculum learning strategies

## 📄 License

MIT License - See LICENSE file

## 🙏 Acknowledgments

- [rlgym](https://github.com/lucas-emery/rocket-league-gym) - Rocket League Gym
- [RLBot](https://github.com/RLBot/RLBot) - Rocket League Bot Framework
- [PyTorch](https://pytorch.org/) - Deep Learning Framework
- [Streamlit](https://streamlit.io/) - Dashboard Framework

## 📞 Support

- Open an issue on GitHub
- Check existing documentation
- Join the community discussions

---

**Built with ❤️ for the Rocket League AI community**

*Train smart. Play hard. Score goals.* 🚗⚽
