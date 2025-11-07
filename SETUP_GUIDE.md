# RL-Bot Setup Guide

Complete step-by-step guide to set up and train the Rocket League AI bot.

## Prerequisites

- **Python 3.10+** (required)
- **8GB+ RAM** (16GB recommended for parallel training)
- **GPU with CUDA support** (optional but recommended for faster training)
- **20GB+ free disk space** (for checkpoints and logs)

## Step 1: System Setup

### Windows

```powershell
# Check Python version (must be 3.10+)
python --version

# If Python is not installed or version is < 3.10:
# Download from https://www.python.org/downloads/
# Make sure to check "Add Python to PATH" during installation
```

### Linux/Mac

```bash
# Check Python version
python3 --version

# If Python 3.10+ is not installed:
# Ubuntu/Debian
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip

# macOS (using Homebrew)
brew install python@3.10
```

### Verify CUDA (for GPU training)

```bash
# Check if NVIDIA GPU is available
nvidia-smi

# If command works, you have CUDA available
# PyTorch will automatically use it
```

## Step 2: Clone Repository

```bash
# Clone the repository
git clone https://github.com/aaronwins356/RL-Bot.git
cd RL-Bot

# Verify files
ls
# Should see: rl_bot/, config.yaml, requirements.txt, main.py, README.md
```

## Step 3: Create Virtual Environment

**Highly recommended to avoid dependency conflicts**

### Windows

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate

# You should see (venv) in your command prompt
```

### Linux/Mac

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# You should see (venv) in your terminal prompt
```

## Step 4: Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# This will install:
# - rlgym >= 2.0.1
# - rlgym-tools, rlgym-sim
# - gymnasium
# - torch >= 2.2.0
# - tensorboard, pyyaml, tqdm, matplotlib, etc.

# Installation may take 5-10 minutes depending on your internet speed
```

### Verify Installation

```bash
# Test imports
python -c "import torch; import rlgym_sim; import gymnasium; print('All imports successful!')"

# Check PyTorch device
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Step 5: Configure Training

### Review Configuration

Open `config.yaml` and adjust parameters if needed:

```yaml
# For faster testing (lower quality):
training:
  total_timesteps: 1_000_000  # Reduced from 10M for testing
  batch_size: 2048             # Reduced for less memory usage

environment:
  num_envs: 2                  # Fewer parallel envs for testing

# For production training:
training:
  total_timesteps: 10_000_000  # Full training
  batch_size: 4096

environment:
  num_envs: 4                  # Or more if you have powerful hardware
```

### Memory Considerations

| Configuration | RAM Required | Recommended For |
|---------------|--------------|-----------------|
| `num_envs: 2, batch_size: 2048` | 4-6 GB | Testing, laptops |
| `num_envs: 4, batch_size: 4096` | 8-12 GB | Standard training |
| `num_envs: 8, batch_size: 8192` | 16-24 GB | High-end systems |

## Step 6: Launch Training

### Basic Training

```bash
# Start training with default settings
python main.py

# Training will display:
# - Configuration summary
# - Progress bar with reward/loss metrics
# - Checkpoints saved every N steps
```

### Training with Custom Settings

```bash
# Train for 5M timesteps
python main.py --timesteps 5000000

# Use GPU (if available)
python main.py --device cuda

# Use 8 parallel environments
python main.py --num-envs 8

# Combine options
python main.py --timesteps 5000000 --device cuda --num-envs 8

# Verbose logging
python main.py --verbose
```

### Monitor Training

**Option 1: Terminal Output**
- Real-time progress bar showing:
  - Current step
  - Mean reward
  - Mean value
  - Loss

**Option 2: TensorBoard (Recommended)**

```bash
# In a new terminal (keep training running)
# Activate virtual environment first if needed
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Launch TensorBoard
tensorboard --logdir logs

# Open browser to: http://localhost:6006
```

TensorBoard shows:
- Reward curves
- Loss metrics
- Learning rate schedule
- Policy entropy
- KL divergence

## Step 7: Training Process

### What to Expect

**First 100K steps (30 min - 2 hours):**
- Random exploration
- Reward may be negative
- High loss values
- Learning to hit the ball

**100K - 1M steps (2-8 hours):**
- Basic ball control emerging
- Reward becoming positive
- Some goals scored

**1M - 5M steps (8-24 hours):**
- Consistent ball touches
- Strategic positioning
- 40-60% win rate

**5M - 10M steps (24-48 hours):**
- Advanced mechanics
- 60-70% win rate
- Elo rating 1200-1400

### Training Time Estimates

| Hardware | 1M Steps | 10M Steps |
|----------|----------|-----------|
| CPU (8 cores) | 4-6 hours | 40-60 hours |
| GPU (RTX 3060) | 1-2 hours | 10-20 hours |
| GPU (RTX 4090) | 30-60 min | 5-10 hours |

### Checkpoints

Checkpoints are automatically saved:

```
checkpoints/
├── checkpoint_500000.pt   # Every 500K steps
├── checkpoint_1000000.pt
├── best_model.pt          # Best performing model
└── final_model.pt         # Final model after training
```

## Step 8: Interrupting and Resuming

### Graceful Interruption

```bash
# Press Ctrl+C to stop training
# The current state will be saved automatically

# Resume from last checkpoint
python main.py --resume checkpoints/checkpoint_5000000.pt
```

### Resume Training

```bash
# Resume from specific checkpoint
python main.py --resume checkpoints/checkpoint_1000000.pt

# Resume with different settings
python main.py --resume checkpoints/best_model.pt --timesteps 15000000
```

## Step 9: Evaluate Performance

### Quick Evaluation

```python
# Create eval.py in project root
from rl_bot.core.utils import load_config, get_device
from rl_bot.core.env_setup import make_vec_env
from rl_bot.eval import evaluate_checkpoint

# Load config
config = load_config('config.yaml')

# Create single environment for evaluation
env = make_vec_env(config, num_envs=1)

# Evaluate best checkpoint
metrics = evaluate_checkpoint(
    'checkpoints/best_model.pt',
    env,
    config,
    num_episodes=20,
    plot_elo=True,
    save_dir='eval_results'
)

# Print results
print(f"Mean Reward: {metrics['mean_reward']:.2f}")
print(f"Win Rate: {metrics['win_rate']:.1%}")
print(f"Elo Rating: {metrics['elo_rating']:.0f}")
```

```bash
# Run evaluation
python eval.py
```

### Compare Checkpoints

```python
# compare.py
from rl_bot.core.utils import load_config
from rl_bot.core.env_setup import make_vec_env
from rl_bot.eval import compare_checkpoints

config = load_config('config.yaml')
env = make_vec_env(config, num_envs=1)

checkpoints = [
    'checkpoints/checkpoint_1000000.pt',
    'checkpoints/checkpoint_5000000.pt',
    'checkpoints/best_model.pt',
]

results = compare_checkpoints(
    checkpoints,
    env,
    config,
    num_episodes=20,
    save_dir='comparison_results'
)
```

## Step 10: Troubleshooting

### Common Issues

#### 1. Import Error: No module named 'rlgym_sim'

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Or install manually
pip install rlgym-sim>=2.0.1
```

#### 2. CUDA Out of Memory

```bash
# Reduce batch size and num_envs
# Edit config.yaml:
training:
  batch_size: 2048  # Reduced from 4096

environment:
  num_envs: 2       # Reduced from 4
```

#### 3. Training Very Slow on CPU

```bash
# Reduce complexity
# Edit config.yaml:
environment:
  tick_skip: 12     # Increased from 8 (fewer decisions)
  num_envs: 2       # Fewer environments

network:
  hidden_sizes: [256, 256]  # Smaller network
```

#### 4. Reward Not Improving

- **Train Longer**: Try 5-10M timesteps minimum
- **Check Rewards**: Verify reward weights in config.yaml aren't too small
- **Learning Rate**: Try adjusting (1e-4 or 1e-3)
- **Enable Normalization**: Set `normalize_rewards: true` in config

#### 5. TensorBoard Not Starting

```bash
# Make sure you're in the project directory
cd /path/to/RL-Bot

# Specify full path
tensorboard --logdir ./logs

# Try different port
tensorboard --logdir logs --port 6007
```

## Step 11: Next Steps

### Customize Rewards

1. Edit `rl_bot/core/reward_functions.py`
2. Add new reward components
3. Update `config.yaml` with weights
4. Retrain

### Advanced Training

```bash
# Longer training for better results
python main.py --timesteps 20000000

# Self-play (experimental)
# Edit config.yaml:
environment:
  self_play: true
```

### Deploy to RLBot (Future Work)

The trained model can be integrated with RLBot framework for actual Rocket League gameplay (requires additional integration code).

## Support

If you encounter issues:

1. Check this guide thoroughly
2. Review error messages carefully
3. Check TensorBoard for training metrics
4. Open an issue on GitHub with:
   - Python version
   - OS and hardware specs
   - Error message and full traceback
   - Config file (if modified)

---

**Good luck with training! 🚀**
