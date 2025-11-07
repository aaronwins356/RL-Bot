# RL-Bot - Modern Rocket League AI

A clean, modular reinforcement learning bot for Rocket League built with modern ML libraries (rlgym >= 2.0.1, gymnasium, PyTorch >= 2.2).

This is a complete rebuild focused on simplicity, maintainability, and ease of extension.

## 🎯 Features

- **Simple PPO Implementation**: Clean, readable PPO training loop
- **Modular Reward System**: Easy-to-customize reward components
- **Modern Dependencies**: Latest versions of rlgym, gymnasium, and PyTorch
- **Automatic Device Detection**: Seamlessly trains on CPU or GPU
- **Comprehensive Logging**: TensorBoard integration and detailed logs
- **Checkpoint Management**: Automatic saving and resuming
- **Elo Rating System**: Track performance over time
- **Vectorized Training**: Parallel environments for faster training

## 📁 Project Structure

```
rl_bot/
├── core/
│   ├── env_setup.py          # rlgym environment configuration
│   ├── model.py               # PyTorch policy and value networks
│   ├── reward_functions.py    # Modular reward components
│   └── utils.py               # Logging, device management, checkpointing
├── train.py                   # PPO training loop
├── eval.py                    # Evaluation and Elo tracking
├── main.py                    # Entry point
├── config.yaml                # Training configuration
└── requirements.txt           # Dependencies
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/aaronwins356/RL-Bot.git
cd RL-Bot

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Training

```bash
# Train with default settings
python main.py

# Train with custom settings
python main.py --timesteps 5000000 --device cuda --num-envs 8

# Resume from checkpoint
python main.py --resume checkpoints/checkpoint_1000000.pt
```

### 3. Monitor Training

```bash
# Launch TensorBoard
tensorboard --logdir logs

# Open in browser: http://localhost:6006
```

### 4. Evaluate Performance

```bash
# Run evaluation script (after training)
python -c "
from rl_bot.core.utils import load_config, get_device
from rl_bot.core.env_setup import make_vec_env
from rl_bot.eval import evaluate_checkpoint

config = load_config('config.yaml')
env = make_vec_env(config, num_envs=1)
metrics = evaluate_checkpoint('checkpoints/best_model.pt', env, config, num_episodes=20, plot_elo=True, save_dir='results')
print(f'Win Rate: {metrics[\"win_rate\"]:.1%}')
print(f'Elo Rating: {metrics[\"elo_rating\"]:.0f}')
"
```

## ⚙️ Configuration

All training parameters are configured in `config.yaml`:

```yaml
training:
  total_timesteps: 10_000_000  # Total training steps
  batch_size: 4096              # PPO batch size
  learning_rate: 3.0e-4         # Learning rate
  # ... more parameters

rewards:
  goal_scored: 10.0             # Goal reward
  touch_ball: 0.5               # Ball touch reward
  # ... more rewards

environment:
  num_envs: 4                   # Parallel environments
  team_size: 1                  # 1v1, 2v2, or 3v3
  tick_skip: 8                  # Physics ticks per action
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `total_timesteps` | 10,000,000 | Total training steps |
| `batch_size` | 4096 | PPO batch size |
| `learning_rate` | 3e-4 | Initial learning rate |
| `num_envs` | 4 | Parallel environments |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `clip_range` | 0.2 | PPO clip range |

## 🎮 Reward System

The bot uses modular reward components that can be easily customized:

### Sparse Rewards (Event-based)
- **Goal Scored**: +10.0
- **Goal Conceded**: -10.0
- **Save**: +3.0
- **Demo**: +2.0

### Dense Rewards (Continuous)
- **Ball Velocity to Goal**: Scaled by ball speed toward opponent goal
- **Ball Touch**: +0.5 for touching the ball
- **Aerial Touch**: +1.0 for aerial ball touches
- **Boost Pickup**: +0.1 for collecting boost pads

### Advanced Mechanics
- **Flick Attempt**: +0.3 for attempting flicks
- **Bump Attempt**: +0.2 for bumping opponents

All rewards can be customized in `config.yaml` under the `rewards` section.

## 📊 Training Progress

### Expected Performance

After ~10M timesteps (12-24 hours on GPU):
- **Win Rate**: 60-70% vs default opponent
- **Elo Rating**: 1200-1400
- **Average Reward**: 5-10 per episode

### Monitoring Metrics

Track these key metrics in TensorBoard:
1. **Mean Reward**: Should steadily increase
2. **Policy Loss**: Should decrease initially, then stabilize
3. **Value Loss**: Should decrease over time
4. **Entropy**: Should gradually decrease (exploration → exploitation)
5. **KL Divergence**: Should stay low (< 0.05)

## 🔧 Customization

### Adding New Reward Components

1. Create a new reward class in `rl_bot/core/reward_functions.py`:

```python
class MyCustomReward(RewardFunction):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
    
    def reset(self, initial_state: GameState):
        # Reset any tracking variables
        pass
    
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # Calculate your custom reward
        reward = 0.0
        # ... your logic here
        return reward * self.weight
```

2. Add it to the `create_reward_function` factory:

```python
components = [
    # ... existing components
    ('my_custom', MyCustomReward(weight=config.get('my_custom_weight', 1.0)))
]
```

3. Configure the weight in `config.yaml`:

```yaml
rewards:
  my_custom_weight: 0.5
```

### Modifying Network Architecture

Edit `config.yaml`:

```yaml
network:
  hidden_sizes: [512, 512, 256]  # Change layer sizes
  activation: "relu"              # Change activation: relu, tanh, elu
  use_layer_norm: false          # Enable layer normalization
```

### Changing Environment Settings

Edit `config.yaml`:

```yaml
environment:
  num_envs: 8          # More parallel environments (faster but more memory)
  team_size: 2         # Change to 2v2 or 3v3
  tick_skip: 12        # Fewer actions per second (faster training)
  timeout_seconds: 180 # Shorter episodes
```

## 🐛 Troubleshooting

### Training is slow
- Reduce `num_envs` if running out of memory
- Increase `tick_skip` to reduce action frequency
- Use `--device cuda` if you have a GPU

### GPU out of memory
- Reduce `batch_size` in config.yaml
- Reduce `num_envs` in config.yaml
- Use smaller network: `hidden_sizes: [256, 256]`

### ImportError: No module named 'rlgym_sim'
```bash
pip install rlgym-sim>=2.0.1
```

### Training not improving
- Check reward function weights - they may be too small/large
- Increase training time (10M+ timesteps recommended)
- Adjust learning rate (try 1e-4 or 1e-3)
- Enable reward normalization in config

## 📈 Advanced Features

### Resume Training

```bash
python main.py --resume checkpoints/checkpoint_5000000.pt
```

### Custom Training Script

```python
from rl_bot.core.utils import load_config, get_device, setup_logging, set_seed
from rl_bot.core.env_setup import make_vec_env
from rl_bot.train import train

# Load config and setup
config = load_config('config.yaml')
set_seed(42)
device = get_device('auto')
logger = setup_logging('logs')

# Create environment
env = make_vec_env(config, num_envs=4)

# Train
model = train(
    env=env,
    config=config,
    total_timesteps=10_000_000,
    device=device,
    logger=logger
)
```

### Evaluation and Comparison

```python
from rl_bot.eval import compare_checkpoints
from rl_bot.core.env_setup import make_vec_env
from rl_bot.core.utils import load_config

config = load_config('config.yaml')
env = make_vec_env(config, num_envs=1)

# Compare multiple checkpoints
checkpoints = [
    'checkpoints/checkpoint_1000000.pt',
    'checkpoints/checkpoint_5000000.pt',
    'checkpoints/best_model.pt'
]

results = compare_checkpoints(checkpoints, env, config, num_episodes=20, save_dir='comparison')
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional reward components
- Advanced training algorithms (SAC, TD3)
- Curriculum learning
- Self-play
- RLBot framework integration
- Documentation improvements

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- [rlgym](https://github.com/lucas-emery/rocket-league-gym) - Rocket League Gym environment
- [RLBot](https://github.com/RLBot/RLBot) - Rocket League bot framework
- [PyTorch](https://pytorch.org/) - Deep learning framework

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues for solutions

---

**Happy Training! 🚗⚽**
