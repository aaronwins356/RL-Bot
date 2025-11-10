# RL-Bot Quick Reference

## Installation
```bash
git clone https://github.com/aaronwins356/RL-Bot.git
cd RL-Bot
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Training Commands

### Basic Training
```bash
python main.py
```

### Custom Settings
```bash
# Train for 5M timesteps
python main.py --timesteps 5000000

# Use GPU
python main.py --device cuda

# Use 8 parallel environments
python main.py --num-envs 8

# All together
python main.py --timesteps 5000000 --device cuda --num-envs 8
```

### Resume Training
```bash
python main.py --resume checkpoints/checkpoint_1000000.pt
```

## Monitoring

### TensorBoard
```bash
tensorboard --logdir logs
# Open: http://localhost:6006
```

## Configuration

### Key Parameters (config.yaml)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `total_timesteps` | 10,000,000 | Total training steps |
| `batch_size` | 4096 | PPO batch size |
| `learning_rate` | 3e-4 | Initial learning rate |
| `num_envs` | 4 | Parallel environments |
| `team_size` | 1 | 1v1, 2v2, or 3v3 |
| `tick_skip` | 8 | Physics ticks per action |

### Reward Weights
```yaml
rewards:
  goal_scored: 10.0
  touch_ball: 0.5
  aerial_touch: 1.0
  save: 3.0
  boost_pickup: 0.1
```

## Project Structure
```
rl_bot/
├── core/
│   ├── env_setup.py       # Environment configuration
│   ├── model.py           # Neural networks
│   ├── reward_functions.py # Reward components
│   └── utils.py           # Utilities
├── train.py               # PPO training loop
├── eval.py                # Evaluation system
└── main.py                # Entry point
```

## File Locations

| Item | Location |
|------|----------|
| Checkpoints | `checkpoints/` |
| Logs | `logs/` |
| TensorBoard | `logs/tensorboard/` |
| Config | `config.yaml` |

## Common Issues

### Out of Memory
```yaml
# Reduce in config.yaml:
training:
  batch_size: 2048  # From 4096
environment:
  num_envs: 2       # From 4
```

### Slow Training
```yaml
# Increase in config.yaml:
environment:
  tick_skip: 12     # From 8 (fewer decisions)
```

### Dependencies Missing
```bash
pip install -r requirements.txt --force-reinstall
```

## Evaluation

### Quick Eval
```python
from rl_bot.core.utils import load_config
from rl_bot.core.env_setup import make_vec_env
from rl_bot.eval import evaluate_checkpoint

config = load_config('config.yaml')
env = make_vec_env(config, num_envs=1)
metrics = evaluate_checkpoint('checkpoints/best_model.pt', env, config)
print(f"Win Rate: {metrics['win_rate']:.1%}")
```

## Expected Performance

| Timesteps | Elo | Win Rate | Time (GPU) |
|-----------|-----|----------|------------|
| 1M | 1100-1200 | 40-50% | 1-2 hours |
| 5M | 1200-1300 | 50-60% | 5-10 hours |
| 10M | 1300-1400 | 60-70% | 10-20 hours |

## Key Metrics to Monitor

1. **Mean Reward** - Should increase
2. **Win Rate** - Target 60%+ 
3. **Elo Rating** - Target 1200+
4. **Policy Loss** - Should decrease then stabilize
5. **KL Divergence** - Should stay < 0.05

## Customization

### Add New Reward
```python
# In rl_bot/core/reward_functions.py
class MyReward(RewardFunction):
    def get_reward(self, player, state, action):
        # Your logic here
        return reward_value
```

### Modify Network
```yaml
# In config.yaml
network:
  hidden_sizes: [512, 512, 256]  # Change layer sizes
  activation: "relu"              # relu, tanh, elu
```

## Resources

- **Full Documentation**: README.md
- **Setup Guide**: SETUP_GUIDE.md
- **Examples**: EXAMPLES.md
- **TensorBoard**: http://localhost:6006

## Support

- GitHub Issues: https://github.com/aaronwins356/RL-Bot/issues
- Check existing issues first
- Include error messages and config when reporting

---

**Quick Start**: `python main.py` 🚀
