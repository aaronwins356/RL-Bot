# 🚀 RocketMind Setup and Usage Guide

## Complete Installation and Setup

### Step 1: Environment Setup

Run the automatic environment setup script to install all dependencies:

```bash
python setup_env.py
```

This script will:
- Remove old/broken packages (rlgym, rlgym-sim)
- Install PyTorch >= 2.2 with CUDA support (if available)
- Install Gymnasium >= 0.29
- Install RLGym-Rocket-League from GitHub
- Install RLBot and RocketSim
- Install Streamlit and visualization packages
- Install optional packages (W&B, TensorBoard, etc.)
- Check CUDA/GPU availability
- Verify all installations

**Note**: If RLBot or RocketSim installation fails, it's non-critical. Core training will still work.

### Step 2: Verify Installation

Verify that everything is set up correctly:

```bash
python verify_rl_env.py
```

This verification script will:
- Test all package imports
- Check hardware (GPU/CPU)
- Create a test environment
- Run simulation steps
- Test neural network creation
- Display comprehensive diagnostics

Expected output:
```
✅ RocketMind Environment Healthy
✅ PPO Core Operational
✅ Simulation Running Correctly
```

### Step 3: Choose Your Training Method

You have two options for training:

#### Option A: Streamlit Dashboard (Recommended)

Launch the interactive dashboard:

```bash
python -m rocketmind.main dashboard
```

Then open your browser to `http://localhost:8501`

Features:
- Real-time training metrics and graphs
- Live GPU monitoring
- Hyperparameter tuning interface
- Model comparison tools
- Replay viewer with heatmaps
- Live simulation viewer

#### Option B: Command-Line Training

Start training directly:

```bash
python -m rocketmind.main train
```

With custom config:

```bash
python -m rocketmind.main train --config rocketmind/configs/custom.yaml
```

Resume from checkpoint:

```bash
python -m rocketmind.main train --resume checkpoints/rocketmind/checkpoint_1000000.pt
```

## 📊 Monitoring Training

### TensorBoard

View detailed training metrics:

```bash
tensorboard --logdir logs/rocketmind
```

Open `http://localhost:6006` in your browser.

### Weights & Biases (Optional)

Enable W&B integration in `rocketmind/configs/default.yaml`:

```yaml
dashboard:
  enable_wandb: true
  wandb_project: "rocketmind"
```

Then run:

```bash
wandb login
python -m rocketmind.main train
```

## ⚙️ Configuration

### Key Configuration Files

**Main Config**: `rocketmind/configs/default.yaml`

Key parameters to adjust:

```yaml
training:
  total_timesteps: 10_000_000  # Training duration
  batch_size: 4096             # PPO batch size
  learning_rate: 3.0e-4        # Initial learning rate
  use_amp: true                # Automatic mixed precision
  use_adaptive_lr: true        # Adaptive learning rate

network:
  hidden_sizes: [512, 512, 256]  # Network architecture
  use_lstm: false                # Enable temporal awareness
  use_torch_compile: true        # PyTorch 2.x optimization

environment:
  num_envs: 4          # Parallel environments
  team_size: 1         # 1v1, 2v2, or 3v3
  tick_skip: 8         # Actions per second control
  
curriculum:
  enabled: false       # Progressive difficulty training
```

## 🎯 Advanced Features

### 1. Adaptive Hyperparameters

The system automatically adjusts:
- **Entropy coefficient**: Reduces as performance improves (exploration → exploitation)
- **Learning rate**: Adapts based on training stability and progress
- **KL penalty**: Auto-tuned to maintain target KL divergence
- **Clip range**: Adjusts based on policy update stability

Enable in config:
```yaml
training:
  use_adaptive_lr: true
rewards:
  adaptive_rewards: true
```

### 2. Curriculum Learning

Progressive difficulty training (inspired by Nexto/Ripple):

```yaml
curriculum:
  enabled: true
  stages:
    - name: "basic"
      timesteps: 2_000_000
      opponent_skill: 0.3
    - name: "intermediate"  
      timesteps: 5_000_000
      opponent_skill: 0.6
    - name: "advanced"
      timesteps: 10_000_000
      opponent_skill: 1.0
```

### 3. Domain Randomization

Add variability to training for better generalization:

```yaml
environment:
  domain_randomization: true
  gravity_mult_range: [0.9, 1.1]
  boost_spawn_rate_range: [0.8, 1.2]
```

### 4. Self-Play (Experimental)

Train against your own agent:

```yaml
environment:
  self_play: true
```

### 5. Smart Checkpointing

The checkpoint manager automatically:
- Keeps top-3 models by reward
- Saves best model as `best_model.pt`
- Tracks last good checkpoint for rollback
- Auto-deletes redundant checkpoints

Manual checkpoint management:

```python
from rocketmind.checkpoint_manager import CheckpointManager

manager = CheckpointManager(
    checkpoint_dir="checkpoints/rocketmind",
    keep_top_k=3,
    auto_cleanup=True
)

# Save checkpoint
manager.save_checkpoint(
    model=model,
    optimizer=optimizer,
    timestep=1000000,
    metrics={'mean_reward': 8.5, 'win_rate': 0.72},
    model_config=config,
    training_config=config,
    force=True
)

# Load best checkpoint
checkpoint = manager.load_checkpoint(load_best=True)

# Rollback to last good checkpoint (recovery)
checkpoint = manager.rollback_to_last_good()
```

## 🎮 Deployment to RLBot

### Deploy Trained Model

```bash
python -m rocketmind.main deploy --checkpoint checkpoints/rocketmind/best_model.pt
```

This will:
1. Export model to RLBot-compatible format
2. Create bot configuration file
3. Launch in RLBot GUI (if `auto_launch: true`)

### Manual RLBot Setup

1. Export model:
```python
from rocketmind.checkpoint_manager import CheckpointManager

manager = CheckpointManager()
model_path = manager.export_to_rlbot(
    checkpoint_path="checkpoints/rocketmind/best_model.pt",
    output_dir="rlbot_models/rocketmind"
)
```

2. Add bot through RLBot GUI using the generated configuration

## 🧪 Testing and Validation

### Run Unit Tests

```bash
python -m rocketmind.tests.test_ppo
```

### Run Stress Tests

```bash
python -m rocketmind.tests.test_stress
```

This runs comprehensive tests:
- Environment stability (10 episodes)
- Vectorized environment
- Checkpoint save/load
- Top-K retention
- Agent interface
- Hyperparameter adaptation
- Curriculum learning

### Manual Validation

Test environment:
```bash
python verify_rl_env.py
```

## 🔧 Troubleshooting

### Import Errors

**Issue**: `ImportError: No module named 'rlgym_rocket_league'`

**Solution**:
```bash
pip install git+https://github.com/RLGym/rlgym-rocket-league.git
```

### GPU Not Detected

**Issue**: Training uses CPU instead of GPU

**Solution**:
1. Check CUDA installation: `nvidia-smi`
2. Reinstall PyTorch with CUDA:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory

**Issue**: `CUDA out of memory`

**Solution**: Reduce memory usage in config:
```yaml
training:
  batch_size: 2048  # Reduce from 4096
environment:
  num_envs: 2  # Reduce from 4
network:
  hidden_sizes: [256, 256]  # Reduce from [512, 512, 256]
```

### Training Not Improving

**Issue**: Reward curve is flat

**Solutions**:
1. Check reward weights - may be too small
2. Increase training time (10M+ timesteps)
3. Adjust learning rate:
```yaml
training:
  learning_rate: 1.0e-3  # Increase from 3e-4
```
4. Enable adaptive rewards:
```yaml
rewards:
  adaptive_rewards: true
```

### Streamlit Dashboard Won't Start

**Issue**: `streamlit: command not found`

**Solution**:
```bash
pip install streamlit>=1.30
```

## 📈 Performance Tips

### Maximize Training Speed

1. **Use GPU**: Ensure CUDA is available
2. **Increase parallel envs**: 
```yaml
environment:
  num_envs: 8  # More if you have RAM
```
3. **Enable AMP**:
```yaml
training:
  use_amp: true
```
4. **Enable torch.compile**:
```yaml
network:
  use_torch_compile: true  # PyTorch 2.x only
```
5. **Increase tick_skip**:
```yaml
environment:
  tick_skip: 12  # Less frequent actions = faster training
```

### Improve Learning Quality

1. **Use curriculum learning**
2. **Enable adaptive hyperparameters**
3. **Tune reward functions** in config
4. **Increase training duration**
5. **Use domain randomization** for generalization

## 🎓 Learning Resources

### Understanding the Codebase

Key modules:
- `rocketmind/ppo_core/`: PPO implementation
  - `network.py`: Actor-Critic architecture
  - `trainer.py`: PPO training loop
  - `agent.py`: Agent interface
  - `hyperopt.py`: Adaptive hyperparameters
  - `memory.py`: Rollout buffer with GAE

- `rocketmind/envs/`: Environment wrappers
  - `rocket_env.py`: RLGym-RL wrapper with domain randomization

- `rocketmind/checkpoint_manager.py`: Smart checkpoint management

- `rocketmind/streamlit_app.py`: Interactive dashboard

### Customization Examples

**Add custom reward**:
```python
# In rocketmind/rlbot_interface/reward_functions.py
class MyCustomReward:
    def __init__(self, weight=1.0):
        self.weight = weight
    
    def reset(self, initial_state):
        pass
    
    def get_reward(self, player, state, previous_action):
        reward = 0.0
        # Your logic here
        return reward * self.weight
```

**Custom network architecture**:
```yaml
network:
  hidden_sizes: [1024, 512, 256, 128]  # Deeper network
  activation: "elu"  # Different activation
  use_layer_norm: true  # Add normalization
```

## 🤝 Contributing

Found a bug or want to add a feature? Contributions welcome!

## 📄 License

MIT License - See LICENSE file for details

---

**Built with ❤️ for the Rocket League AI community**

For issues: https://github.com/aaronwins356/RL-Bot/issues
